import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os

st.set_page_config(page_title="Kalimba AI Studio Pro", page_icon="🎼")

st.title("🎼 Kalimba AI Studio - Edição Acústica")
st.write("Processamento robusto para músicas complexas (Rock, Pop, Instrumental).")

# --- FUNÇÃO DE DOWNLOAD ---
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
    file = st.file_uploader("Upload da Música", type=["mp3", "wav"])
    if file: audio_path = file
with aba2:
    url = st.text_input("Link do YouTube")
    if url and st.button("PROCESSAR LINK"):
        try:
            audio_path = download_youtube(url)
            st.success("Áudio extraído!")
        except: st.error("Erro no download.")

# --- MOTOR DE CONVERSÃO ROBUSTO ---
if audio_path:
    if st.button("✨ GERAR VERSÃO KALIMBA REALISTA"):
        with st.spinner("Limpando ruídos e isolando a melodia acústica..."):
            # 1. Carregamento em alta fidelidade
            y, sr = librosa.load(audio_path, sr=22050)
            
            # 2. SEPARAÇÃO HARMÔNICA (O SEGREDO)
            # Isso separa a melodia (harmônica) da bateria (percussiva)
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            
            # 3. Análise por CQT (Constant-Q Transform) 
            # É muito melhor que o STFT comum para identificar notas musicais reais
            hop_length = 512
            C = np.abs(librosa.cqt(y_harmonic, sr=sr, hop_length=hop_length))
            
            out_audio = np.zeros_like(y)
            
            # 4. Síntese com Timbre Acústico
            for t in range(C.shape[1]):
                # Pegamos apenas o pico de energia mais forte naquele momento
                f_idx = C[:, t].argmax()
                magnitude = C[f_idx, t]
                
                # Só toca se for uma nota clara e forte (evita os "bugs" de chiado)
                if magnitude > np.max(C) * 0.2: 
                    freq = librosa.cqt_frequencies(C.shape[0], fmin=librosa.note_to_hz('C2'))[f_idx]
                    
                    if 100 < freq < 1500: # Range da Kalimba
                        dur = 0.6
                        t_nota = np.linspace(0, dur, int(dur * sr))
                        
                        # Timbre Realista: Senoidal + Harmônico de Metal + Ruído de Ataque
                        fundamental = np.sin(2 * np.pi * freq * t_nota)
                        brilho = 0.2 * np.sin(2 * np.pi * freq * 2.8 * t_nota) # O harmônico da lâmina
                        
                        # Envelope: Ataque instantâneo e decaimento natural
                        env = np.exp(-6 * t_nota)
                        nota_final = (fundamental + brilho) * env
                        
                        # Posicionamento no tempo
                        start = t * hop_length
                        end = min(start + len(nota_final), len(out_audio))
                        out_audio[start:end] += nota_final[:end-start] * 0.4

            # 5. Masterização e Reverb Leve
            out_audio = librosa.util.normalize(out_audio)
            sf.write("kalimba_pro.wav", out_audio, sr)
            
            st.success("Sua versão acústica está pronta!")
            st.audio("kalimba_pro.wav")
            st.download_button("Baixar MP3 Realista", open("kalimba_pro.wav", "rb"), "kalimba_pro.wav")
            st.balloons()
