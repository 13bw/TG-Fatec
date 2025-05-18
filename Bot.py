import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import pickle
import re
from unidecode import unidecode
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sqlite3
import datetime

load_dotenv()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

token = os.getenv("token")
mod_ids = [int(id) for id in os.getenv("mod_ids").split(",")]
model_path = os.getenv("MODEL_PATH", "best_model_final_rmsprop_binary_crossentropy_lr0.001_20250516_134013.keras")
tokenizer_path = os.getenv("TOKENIZER_PATH", "tokenizer.pickle")
threshold = float(os.getenv("THRESHOLD", "0.7"))
db_path = os.getenv("DB_PATH", "discord_messages.db")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
intents.members = True
intents.presences = True

bot = commands.Bot(command_prefix='!', intents=intents)

# Configuração do banco de dados SQLite
def setup_database():
    """Configura o banco de dados SQLite para armazenar mensagens"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Cria tabela de mensagens se não existir
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY,
        message_id TEXT UNIQUE,
        guild_id TEXT,
        guild_name TEXT,
        channel_id TEXT,
        channel_name TEXT,
        author_id TEXT,
        author_name TEXT,
        content TEXT,
        timestamp TEXT,
        is_hate_speech INTEGER,
        hate_probability REAL,
        confidence REAL
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Banco de dados configurado em: {db_path}")

def save_message(message, is_hate=False, hate_probability=0.0, confidence=0.0):
    """Salva a mensagem no banco de dados SQLite"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.datetime.now().isoformat()
        
        cursor.execute('''
        INSERT OR IGNORE INTO messages 
        (message_id, guild_id, guild_name, channel_id, channel_name, 
        author_id, author_name, content, timestamp, is_hate_speech, 
        hate_probability, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(message.id),
            str(message.guild.id) if message.guild else None,
            message.guild.name if message.guild else None,
            str(message.channel.id),
            message.channel.name if hasattr(message.channel, 'name') else 'DM',
            str(message.author.id),
            str(message.author),
            message.content,
            timestamp,
            1 if is_hate else 0,
            hate_probability,
            confidence
        ))
        
        conn.commit()
        conn.close()
        print(f"Mensagem {message.id} salva no banco de dados")
        return True
    except Exception as e:
        print(f"ERRO ao salvar mensagem no banco de dados: {str(e)}")
        return False

try:
    model = tf.keras.models.load_model(model_path)
    print(f"Modelo carregado de: {model_path}")
    
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print(f"Tokenizer carregado de: {tokenizer_path}")
except Exception as e:
    print(f"ERRO ao carregar modelo ou tokenizer: {str(e)}")
    print("O bot será iniciado, mas a detecção de discurso de ódio não funcionará corretamente.")
    model = None
    tokenizer = None

def preprocess_text(text):
    """Pré-processa o texto para análise"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = unidecode(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('portuguese'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def detect_hate_speech(input_text, max_len=150):
    """Detecta discurso de ódio no texto fornecido"""

    if model is None or tokenizer is None:
        return {"error": "Modelo ou tokenizer não disponível", "eh_discurso_odio": False, "confianca": 0.0}
    
    processed_text = preprocess_text(input_text)
    
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    
    prediction = model.predict(padded_sequence)[0][0]
    
    is_hate_speech = prediction > 0.5
    confidence = prediction if is_hate_speech else 1 - prediction
    
    return {
        "texto_original": input_text,
        "texto_processado": processed_text,
        "probabilidade": float(prediction),
        "eh_discurso_odio": is_hate_speech,
        "confianca": float(confidence)
    }

@bot.event
async def on_ready():
    print(f'Bot {bot.user} está online!')
    print(f'Monitorando mensagens em {len(bot.guilds)} servidores')
    
    # Configura o banco de dados ao iniciar
    setup_database()
    
    if model is not None:
        print("Sistema de detecção de discurso de ódio está ativo")
        print(f"Usando limiar de confiança: {threshold}")
    else:
        print("AVISO: Sistema de detecção de discurso de ódio NÃO está ativo")

async def get_available_mod():
    """Retorna o primeiro moderador online encontrado"""
    for mod_id in mod_ids:
        try:
            mod = await bot.fetch_user(mod_id)
    
            for guild in bot.guilds:
                member = guild.get_member(mod_id)
                if member and member.status != discord.Status.offline:
                    return mod
        except discord.NotFound:
            continue
    return await bot.fetch_user(mod_ids[0])

async def is_hate_speech(message):
    """Verifica se a mensagem contém discurso de ódio"""
    texto = message.content.lower()
    
    if model is None or tokenizer is None:
        print("Aviso: Modelo não disponível, mensagem não verificada")
        return False
    
    if len(texto.split()) < 3:
        return False
        
    result = detect_hate_speech(texto)
    print(result)

    return result["eh_discurso_odio"] and result["confianca"] >= threshold

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    # Processa as mensagens em DM sem análise ou salvamento
    if not message.guild:
        await bot.process_commands(message)
        return

    # Verifica se é discurso de ódio
    result = detect_hate_speech(message.content)
    is_hate = result["eh_discurso_odio"] and result["confianca"] >= threshold
    
    # Salva a mensagem no banco de dados
    save_message(
        message,
        is_hate=is_hate,
        hate_probability=result["probabilidade"],
        confidence=result["confianca"]
    )
    
    # Se for detectado discurso de ódio, notifica um moderador
    if is_hate:
        mod = await get_available_mod()
        
        if mod:
            prediction_info = (
                f"Probabilidade: {result['probabilidade']:.2%}\n"
                f"Confiança: {result['confianca']:.2%}\n"
            )
            
            report = (
                f"🚨 **Alerta de Discurso de Ódio Detectado**\n"
                f"Servidor: {message.guild.name}\n"
                f"Canal: #{message.channel.name}\n"
                f"Autor: {message.author} (ID: {message.author.id})\n"
                f"Mensagem: {message.content}\n"
                f"{prediction_info}"
                f"Link: {message.jump_url}"
            )
            
            try:
                await mod.send(report)
                print(f"Alerta enviado para moderador {mod.name} sobre mensagem de {message.author}")
            except discord.Forbidden:
                print(f"Não foi possível enviar mensagem para o moderador {mod.id}")

    await bot.process_commands(message)

@bot.command(name="testar")
async def test_detection(ctx, *, text: str = None):
    """Testa a detecção de discurso de ódio em um texto"""

    if ctx.author.id not in mod_ids:
        await ctx.send("Você não tem permissão para usar este comando.")
        return
    
    if text is None:
        messages = [message async for message in ctx.channel.history(limit=2)]
        if len(messages) > 1:
            text = messages[1].content
        else:
            await ctx.send("Por favor, forneça um texto para análise ou use o comando após uma mensagem.")
            return
    
    if model is not None and tokenizer is not None:
        result = detect_hate_speech(text)
        
        response = (
            f"📊 **Análise de Texto**\n"
            f"Texto original: {result['texto_original']}\n"
            f"Texto processado: {result['texto_processado']}\n"
            f"Probabilidade de discurso de ódio: {result['probabilidade']:.2%}\n"
            f"Classificação: {'🔴 DISCURSO DE ÓDIO' if result['eh_discurso_odio'] else '🟢 TEXTO NORMAL'}\n"
            f"Confiança na classificação: {result['confianca']:.2%}\n"
            f"Limiar configurado: {threshold:.2%}\n"
        )
        
        await ctx.send(response)
    else:
        await ctx.send("❌ Modelo de detecção não está disponível.")

@bot.command(name="estatisticas")
async def message_stats(ctx):
    """Exibe estatísticas das mensagens armazenadas"""
    
    if ctx.author.id not in mod_ids:
        await ctx.send("Você não tem permissão para usar este comando.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Total de mensagens
        cursor.execute("SELECT COUNT(*) FROM messages")
        total_messages = cursor.fetchone()[0]
        
        # Total de mensagens de ódio
        cursor.execute("SELECT COUNT(*) FROM messages WHERE is_hate_speech = 1")
        total_hate = cursor.fetchone()[0]
        
        # Mensagens por servidor (top 5)
        cursor.execute("""
        SELECT guild_name, COUNT(*) as count 
        FROM messages 
        WHERE guild_name IS NOT NULL 
        GROUP BY guild_name 
        ORDER BY count DESC 
        LIMIT 5
        """)
        top_guilds = cursor.fetchall()
        
        # Usuários com mais mensagens de ódio (top 5)
        cursor.execute("""
        SELECT author_name, COUNT(*) as count 
        FROM messages 
        WHERE is_hate_speech = 1 
        GROUP BY author_id 
        ORDER BY count DESC 
        LIMIT 5
        """)
        top_hate_users = cursor.fetchall()
        
        conn.close()
        
        # Formata a resposta
        stats = (
            f"📈 **Estatísticas do Banco de Dados**\n"
            f"Total de mensagens: {total_messages}\n"
            f"Mensagens com discurso de ódio: {total_hate} ({(total_hate/total_messages)*100:.2f}% do total)\n\n"
        )
        
        if top_guilds:
            stats += "**Top 5 Servidores por Volume de Mensagens:**\n"
            for guild, count in top_guilds:
                stats += f"- {guild}: {count} mensagens\n"
            stats += "\n"
        
        if top_hate_users:
            stats += "**Top 5 Usuários com Mensagens de Ódio:**\n"
            for user, count in top_hate_users:
                stats += f"- {user}: {count} mensagens\n"
        
        await ctx.send(stats)
    except Exception as e:
        await ctx.send(f"❌ Erro ao obter estatísticas: {str(e)}")

@bot.command(name="limpar_db")
async def clear_database(ctx, dias: int = 30):
    """Remove mensagens antigas do banco de dados"""
    
    if ctx.author.id not in mod_ids:
        await ctx.send("Você não tem permissão para usar este comando.")
        return
    
    try:
        # Calcula a data limite para exclusão
        data_limite = (datetime.datetime.now() - datetime.timedelta(days=dias)).isoformat()
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Conta quantas mensagens serão removidas
        cursor.execute("SELECT COUNT(*) FROM messages WHERE timestamp < ?", (data_limite,))
        count = cursor.fetchone()[0]
        
        # Remove as mensagens antigas
        cursor.execute("DELETE FROM messages WHERE timestamp < ?", (data_limite,))
        conn.commit()
        conn.close()
        
        await ctx.send(f"✅ {count} mensagens com mais de {dias} dias foram removidas do banco de dados.")
    except Exception as e:
        await ctx.send(f"❌ Erro ao limpar o banco de dados: {str(e)}")

@bot.command(name="buscar")
async def search_messages(ctx, *, termo: str):
    """Busca mensagens no banco de dados que contenham o termo especificado"""
    
    if ctx.author.id not in mod_ids:
        await ctx.send("Você não tem permissão para usar este comando.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Busca mensagens que contenham o termo
        cursor.execute("""
        SELECT message_id, guild_name, channel_name, author_name, content, 
               is_hate_speech, hate_probability 
        FROM messages 
        WHERE content LIKE ? 
        ORDER BY timestamp DESC 
        LIMIT 10
        """, (f"%{termo}%",))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            await ctx.send(f"Nenhuma mensagem encontrada contendo '{termo}'.")
            return
        
        response = f"🔍 **Resultados da busca por '{termo}'** (10 mais recentes):\n\n"
        
        for msg_id, guild, channel, author, content, is_hate, probability in results:
            hate_label = "🔴 ÓDIO" if is_hate else "🟢 OK"
            prob_formatted = f"{probability:.1%}"
            
            result_text = (
                f"**Servidor:** {guild} | **Canal:** #{channel}\n"
                f"**Autor:** {author} | **Classificação:** {hate_label} ({prob_formatted})\n"
                f"**Mensagem:** {content[:100]}{'...' if len(content) > 100 else ''}\n\n"
            )
            
            # Verifica se a adição deste resultado excederia o limite de caracteres
            if len(response) + len(result_text) > 1900:
                response += "...(mais resultados omitidos)..."
                break
                
            response += result_text
        
        await ctx.send(response)
    except Exception as e:
        await ctx.send(f"❌ Erro ao buscar mensagens: {str(e)}")

bot.run(token)