import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os

# --- INTERFACE MODERNA ---
st.set_page_config(page_title="Kalimba AI Studio", page_icon="🎵")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 20px; background: linear-gradient(45deg, #00dbde, #fc00ff); color: white; border: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎵 Kalimba AI Studio")
st.write("Converta arquivos ou links do YouTube para o som de uma Kalimba.")

# --- FUNÇÃO DE DOWNLOAD CORRIGIDA ---
def download_youtube(url):
    if os.path.exists("yt_audio.wav"):
        os.remove("yt_audio.wav")
        
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'yt_audio',
        'noplaylist': True,
        'quiet': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "yt_audio.wav"

# --- INTERFACE ---
aba1, aba2 = st.tabs(["📁 Enviar Arquivo", "🎥 Link do YouTube"])

audio_path = None

with aba1:
    file = st.file_uploader("Selecione um MP3 ou WAV", type=["mp3", "wav"])
    if file:
        audio_path = file

with aba2:
    url = st.text_input("Cole o link do YouTube (Ex: https://youtu.be/...)")
    if url:
        if st.button("PROCESSAR LINK"):
            with st.spinner("Extraindo áudio..."):
                try:
                    audio_path = download_youtube(url)
                    st.success("Áudio preparado!")
                except Exception as e:
                    st.error("O YouTube bloqueou o download automático. Tente enviar o arquivo MP3 manualmente na outra aba.")

# --- CONVERSOR ---
if audio_path:
    if st.button("✨ TRANSFORMAR EM KALIMBA"):
        with st.spinner("A IA está criando a versão Kalimba..."):
            try:
                y, sr = librosa.load(audio_path, sr=22050)
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                out_audio = np.zeros_like(y)
                
                for i in range(0, pitches.shape[1], 4):
                    idx = magnitudes[:, i].argmax()
                    p = pitches[idx, i]
                    if 80 < p < 1200:
                        t = np.linspace(0, 0.4, int(0.4 * sr))
                        note = np.sin(2 * np.pi * p * t) * np.exp(-9 * t)
                        start = i * 512
                        end = min(start + len(note), len(out_audio))
                        out_audio[start:end] += note[:end-start]

                out_audio = librosa.util.normalize(out_audio)
                sf.write("kalimba_final.wav", out_audio, sr)
                
                st.audio("kalimba_final.wav")
                st.download_button("📥 Baixar Kalimba", open("kalimba_final.wav", "rb"), "kalimba.wav")
                st.balloons()
            except Exception as e:
                st.error(f"Erro no processamento: {e}")
