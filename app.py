import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os

# --- DESIGN E CONFIGURAÇÃO ---
st.set_page_config(page_title="Kalimba AI Pro", page_icon="✨", layout="wide")

st.markdown("""
    <style>
    .main { background: #0e1117; color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1e212b; border-radius: 10px; padding: 10px 20px; color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("💎 Kalimba AI Studio")
st.write("Transforme links ou arquivos em melodias de Kalimba.")

# --- FUNÇÕES DE DOWNLOAD ---
def download_youtube(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'yt_audio.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'wav','preferredquality': '192',}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "yt_audio.wav"

# --- INTERFACE POR ABAS ---
tab1, tab2, tab3 = st.tabs(["📁 Arquivo Local", "🎥 YouTube", "🎧 Spotify"])

audio_source = None

with tab1:
    uploaded_file = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav"])
    if uploaded_file:
        audio_source = uploaded_file

with tab2:
    yt_url = st.text_input("Cole o link do vídeo do YouTube")
    if yt_url:
        if st.button("Extrair Áudio do YouTube"):
            with st.spinner("Baixando do YouTube..."):
                audio_source = download_youtube(yt_url)
                st.success("Áudio pronto para conversão!")

with tab3:
    st.info("O Spotify exige chaves de API. Para este protótipo, use o link do YouTube da mesma música.")
    st.text_input("Cole o link da música do Spotify")

# --- MOTOR DE CONVERSÃO (O MESMO REFINADO) ---
if audio_source:
    if st.button("🪄 CONVERTER PARA KALIMBA"):
        with st.spinner("IA Processando... Isso pode levar 1 minuto."):
            # Carregamento
            y, sr = librosa.load(audio_source, sr=22050)
            
            # Análise de Melodia
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            out_audio = np.zeros_like(y)
            
            # Síntese de Kalimba (Simulada)
            for i in range(0, pitches.shape[1], 5):
                index = magnitudes[:, i].argmax()
                pitch = pitches[index, i]
                if pitch > 100:
                    t = np.linspace(0, 0.4, int(0.4 * sr))
                    env = np.exp(-7 * t)
                    note = np.sin(2 * np.pi * pitch * t) * env
                    
                    start = i * 512
                    end = min(start + len(note), len(out_audio))
                    out_audio[start:end] += note[:end-start]

            # Normalização e Resultado
            out_audio = librosa.util.normalize(out_audio)
            sf.write("resultado_kalimba.wav", out_audio, sr)
            
            st.audio("resultado_kalimba.wav")
            st.download_button("Baixar Música", open("resultado_kalimba.wav", "rb"), "kalimba.wav")
