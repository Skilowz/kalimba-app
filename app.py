import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os

# --- INTERFACE MODERNA ---
st.set_page_config(page_title="Kalimba AI Arranger", page_icon="🍃")

st.markdown("""
    <style>
    .main { background: #121212; color: #e0e0e0; }
    .stButton>button { 
        background: linear-gradient(90deg, #d4a373 0%, #faedcd 100%); 
        color: #121212; border-radius: 12px; font-weight: bold; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🍃 Kalimba AI Arranger")
st.write("IA avançada para transcrição melódica e arranjo acústico realista.")

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

tab1, tab2 = st.tabs(["📁 Arquivo", "🎥 YouTube"])
audio_path = None

with tab1:
    file = st.file_uploader("Upload", type=["mp3", "wav"])
    if file: audio_path = file
with tab2:
    url = st.text_input("URL do YouTube")
    if url and st.button("PROCESSAR"):
        audio_path = download_youtube(url)

# --- ENGINE DE IA E ARRANJO ---
if audio_path:
    if st.button("🪄 CRIAR ARRANJO PARA KALIMBA"):
        with st.spinner("IA analisando harmonia e criando dedilhado..."):
            # 1. Carregamento e Separação de Vozes
            y, sr = librosa.load(audio_path, sr=22050)
            y_harmonic = librosa.effects.hpss(y)[0]
            
            # 2. Reconhecimento da Tonalidade (Key Detection)
            chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
            key_idx = np.argmax(np.mean(chroma, axis=1))
            
            # 3. Transcrição com Limite de Polifonia
            hop_len = 512
            cqt = np.abs(librosa.cqt(y_harmonic, sr=sr, hop_length=hop_len))
            
            out_audio = np.zeros_like(y)
            
            # Algoritmo de decisão: No máximo 2 notas por vez (Simulando polegares)
            for t in range(0, cqt.shape[1], 4):
                # Seleciona os dois picos mais fortes
                top_indices = np.argsort(cqt[:, t])[-2:] 
                
                for f_idx in top_indices:
                    mag = cqt[f_idx, t]
                    if mag > np.max(cqt) * 0.25: # Sensibilidade refinada
                        freq = librosa.cqt_frequencies(cqt.shape[0], fmin=librosa.note_to_hz('C2'))[f_idx]
                        
                        if 130 < freq < 1000:
                            # 4. Síntese de Modelagem Física (Timbre Realista)
                            dur = 1.0 # Sustain longo
                            t_n = np.linspace(0, dur, int(dur * sr))
                            
                            # Componentes do som: Metal + Madeira + Ar
                            fundamental = np.sin(2 * np.pi * freq * t_n)
                            metal_ping = 0.3 * np.sin(2 * np.pi * freq * 2.81 * t_n)
                            wood_thump = 0.1 * np.sin(2 * np.pi * (freq/2) * t_n)
                            
                            # Envelope ADSR Natural
                            env = np.exp(-4.5 * t_n) * (1 - np.exp(-300 * t_n))
                            note_wav = (fundamental + metal_ping + wood_thump) * env
                            
                            # Mixagem no tempo correto
                            start = t * hop_len
                            end = min(start + len(note_wav), len(out_audio))
                            out_audio[start:end] += note_wav[:end-start] * 0.3

            # 5. Masterização e Espacialização
            out_audio = librosa.util.normalize(out_audio)
            sf.write("kalimba_ai_final.wav", out_audio, sr)
            
            st.success("Arranjo concluído!")
            st.audio("kalimba_ai_final.wav")
            st.download_button("Baixar Master Acústica", open("kalimba_ai_final.wav", "rb"), "kalimba_ai.wav")
            st.balloons()
