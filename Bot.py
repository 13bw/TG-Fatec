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

load_dotenv()

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

token = os.getenv("token")
mod_ids = [int(id) for id in os.getenv("mod_ids").split(",")]
model_path = os.getenv("MODEL_PATH", "best_model_final.h5")
tokenizer_path = os.getenv("TOKENIZER_PATH", "tokenizer.pickle")
threshold = float(os.getenv("THRESHOLD", "0.7")) 

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
intents.members = True
intents.presences = True

bot = commands.Bot(command_prefix='!', intents=intents)

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
    

    return result["eh_discurso_odio"] and result["confianca"] >= threshold

@bot.event
async def on_message(message):

    if message.author == bot.user:
        return
    

    if not message.guild:
        await bot.process_commands(message)
        return


    is_hate = await is_hate_speech(message)
    
    if is_hate:
    
        mod = await get_available_mod()
        
        if mod:
        
            result = detect_hate_speech(message.content)
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

bot.run(token)