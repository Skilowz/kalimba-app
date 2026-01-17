import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os

st.set_page_config(page_title="Kalimba AI Studio v2", page_icon="🎵")

# --- INTERFACE ---
st.title("🎵 Kalimba AI Studio - Refinado")
st.write("Versão com correção de tempo e fidelidade melódica.")

def download_youtube(url):
    if os.path.exists("yt_audio.wav"): os.remove("yt_audio.wav")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'yt_audio',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'wav','preferredquality': '192'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "yt_audio.wav"

aba1, aba2 = st.tabs(["📁 Arquivo Local", "🎥 Link do YouTube"])
audio_path = None

with aba1:
    file = st.file_uploader("Upload MP3/WAV", type=["mp3", "wav"])
    if file: audio_path = file
with aba2:
    url = st.text_input("Link do YouTube")
    if url and st.button("EXTRAIR ÁUDIO"):
        try:
            audio_path = download_youtube(url)
            st.success("Áudio extraído!")
        except: st.error("Erro no download.")

# --- MOTOR DE CONVERSÃO REFINADO ---
if audio_path:
    if st.button("✨ GERAR VERSÃO KALIMBA"):
        with st.spinner("Analisando métrica e harmonia..."):
            # 1. Carrega o áudio e mantém a taxa de amostragem padrão
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 2. Análise espectral mais lenta para ser mais precisa
            hop_length = 512
            S = np.abs(librosa.stft(y, hop_length=hop_length))
            pitches, magnitudes = librosa.piptrack(S=S, sr=sr, hop_length=hop_length)
            
            # Criamos um silêncio do mesmo tamanho da música original
            out_audio = np.zeros_like(y)
            
            # 3. Processamento respeitando o tempo original
            for t_frame in range(pitches.shape[1]):
                # Só processa se houver um som forte o suficiente (filtra o bug de ruído)
                index = magnitudes[:, t_frame].argmax()
                magnitude = magnitudes[index, t_frame]
                
                if magnitude > 20: # Limiar de volume (Threshold)
                    pitch = pitches[index, t_frame]
                    
                    if 100 < pitch < 1200: # Range da Kalimba
                        # Gera a nota
                        dur_nota = 0.5 
                        t_nota = np.linspace(0, dur_nota, int(dur_nota * sr))
                        
                        # Timbre: Harmônico leve para tirar o som de "apito"
                        onda = np.sin(2 * np.pi * pitch * t_nota) 
                        ataque_suave = np.exp(-10 * t_nota) # O "Pluck"
                        som_nota = onda * ataque_suave
                        
                        # O SEGREDO DO TEMPO: Coloca a nota exatamente onde ela começa na música
                        pos_original = t_frame * hop_length
                        fim_nota = pos_original + len(som_nota)
                        
                        if fim_nota < len(out_audio):
                            out_audio[pos_original:fim_nota] += som_nota * 0.5

            # 4. Finalização
            out_audio = librosa.util.normalize(out_audio)
            sf.write("kalimba_v2.wav", out_audio, sr)
            
            st.audio("kalimba_v2.wav")
            st.download_button("Baixar Versão Sincronizada", open("kalimba_v2.wav", "rb"), "kalimba_fixed.wav")
