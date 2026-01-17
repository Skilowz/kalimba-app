import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os
from pydub import AudioSegment

# --- DESIGN PROFISSIONAL ---
st.set_page_config(page_title="Piano Lullaby Pro", page_icon="🎹", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stButton>button { 
        background: linear-gradient(135deg, #1f6feb, #094193); 
        color: white; border-radius: 8px; border: none; padding: 12px;
        font-weight: 600; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎹 Piano Lullaby Studio Pro")
st.write("Tecnologia de Modelagem Acústica para Conversão de Alta Fidelidade.")

# --- ENGINE DE SÍNTESE ACÚSTICA ---
def synthesize_piano_note(freq, duration, intensity, sr=22050):
    t = np.linspace(0, duration, int(sr * duration))
    # Harmônicos complexos para som amadeirado
    sound = np.sin(2 * np.pi * freq * t)
    sound += 0.4 * np.sin(2 * np.pi * freq * 2.001 * t) * np.exp(-t * 1.5)
    sound += 0.2 * np.sin(2 * np.pi * freq * 3.002 * t) * np.exp(-t * 3.0)
    
    # Envelope de toque suave (Soft Hammer)
    env = np.exp(-2.5 * t) * (1 - np.exp(-60 * t))
    return sound * env * intensity

def create_arrangement(path):
    # Carregamento seguro
    y, sr = librosa.load(path, sr=22050)
    
    # 1. Desaceleração (70% da velocidade original)
    y_slow = librosa.effects.time_stretch(y, rate=0.7)
    
    # 2. Separação Harmônica Agressiva (Isola a melodia da distorção)
    y_harm, _ = librosa.effects.hpss(y_slow, margin=3.0)
    
    # 3. Análise Espectral
    hop_length = 512
    cqt = np.abs(librosa.cqt(y_harm, sr=sr, hop_length=hop_length))
    
    out_audio = np.zeros_like(y_slow)
    
    # 4. Transcrição Humana (Máximo 2 notas simultâneas)
    for t in range(0, cqt.shape[1], 5):
        top_indices = np.argsort(cqt[:, t])[-2:]
        for idx in top_indices:
            mag = cqt[idx, t]
            if mag > np.max(cqt) * 0.15:
                freq = librosa.cqt_frequencies(cqt.shape[0], fmin=librosa.note_to_hz('C1'))[idx]
                # Dinâmica de volume baseada na energia original
                intensity = (mag / np.max(cqt)) * 0.6
                note = synthesize_piano_note(freq, 1.2, intensity, sr)
                
                start = t * hop_length
                end = min(start + len(note), len(out_audio))
                out_audio[start:end] += note[:end-start]

    return librosa.util.normalize(out_audio), sr

# --- INTERFACE ---
col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload da música", type=["mp3", "wav"])
    yt_url = st.text_input("Ou link do YouTube")

with col2:
    if st.button("GERAR MASTER EM PIANO"):
        input_path = "temp_input.wav"
        try:
            with st.spinner("Processando..."):
                if yt_url:
                    ydl_opts = {'format': 'bestaudio/best', 'outtmpl': 'temp_yt', 'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'wav'}]}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([yt_url])
                    input_path = "temp_yt.wav"
                elif uploaded:
                    # CORREÇÃO DO ERRO: Salva o arquivo corretamente antes de ler
                    with open(input_path, "wb") as f:
                        f.write(uploaded.getbuffer())
                
                audio_res, sr_res = create_arrangement(input_path)
                sf.write("output.wav", audio_res, sr_res)
                
                st.audio("output.wav")
                st.success("Masterização concluída!")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")
