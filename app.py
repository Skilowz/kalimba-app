import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os

# --- INTERFACE MODERNA E CALMA ---
st.set_page_config(page_title="Lullaby AI Studio", page_icon="🌙", layout="centered")

st.markdown("""
    <style>
    .main { background: linear-gradient(180deg, #1a2a6c 0%, #b21f1f 100%); color: white; }
    .stButton>button { 
        background-color: #fdbb2d; color: #1a2a6c; border-radius: 20px; 
        font-weight: bold; border: none; width: 100%; height: 50px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: rgba(255,255,255,0.1); border-radius: 10px; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌙 Lullaby AI Studio")
st.write("Transforme qualquer música em uma doce canção de ninar com sons de sinos.")

# --- FUNÇÃO DE DOWNLOAD ---
def download_youtube(url):
    if os.path.exists("input_audio.wav"): os.remove("input_audio.wav")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'input_audio',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'wav','preferredquality': '192'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "input_audio.wav"

tab1, tab2 = st.tabs(["📁 Enviar Arquivo", "🎥 Link do YouTube"])
audio_path = None

with tab1:
    file = st.file_uploader("Escolha uma música", type=["mp3", "wav"])
    if file: audio_path = file

with tab2:
    url = st.text_input("Cole o link do YouTube")
    if url and st.button("PREPARAR MÚSICA"):
        with st.spinner("Buscando áudio..."):
            audio_path = download_youtube(url)

# --- MOTOR DE CANÇÃO DE NINAR ---
if audio_path:
    if st.button("✨ GERAR MÚSICA DE NINAR"):
        with st.spinner("Criando arranjo relaxante..."):
            # 1. Carregamento e Desaceleração (Ritmo de Ninar)
            y, sr = librosa.load(audio_path, sr=22050)
            y_slow = librosa.effects.time_stretch(y, rate=0.8) # 20% mais lenta
            
            # 2. Isolamento Melódico (Remoção de Bateria e Ruídos)
            y_harm = librosa.effects.hpss(y_slow)[0]
            
            # 3. Transcrição Tonal
            hop_len = 512
            cqt = np.abs(librosa.cqt(y_harm, sr=sr, hop_length=hop_len))
            
            out_audio = np.zeros_like(y_slow)
            
            # 4. Síntese de Sinos Celestiais
            for t in range(0, cqt.shape[1], 6): # Intervalos maiores para maior calma
                f_idx = cqt[:, t].argmax()
                mag = cqt[f_idx, t]
                
                if mag > np.max(cqt) * 0.2:
                    freq = librosa.cqt_frequencies(cqt.shape[0], fmin=librosa.note_to_hz('C2'))[f_idx]
                    
                    if freq < 2000: # Limite de agudos para segurança do bebê
                        dur = 1.5 # Notas longas e flutuantes
                        t_n = np.linspace(0, dur, int(dur * sr))
                        
                        # Timbre de Sino: Senoidal Pura + Harmônico Cristalino
                        bell = np.sin(2 * np.pi * freq * t_n)
                        shimmer = 0.3 * np.sin(2 * np.pi * freq * 2.0 * t_n)
                        
                        # Envelope de Caixinha de Música (Decaimento Suave)
                        env = np.exp(-3.0 * t_n) * (1 - np.exp(-100 * t_n))
                        note_wav = (bell + shimmer) * env
                        
                        start = t * hop_len
                        end = min(start + len(note_wav), len(out_audio))
                        out_audio[start:end] += note_wav[:end-start] * 0.3

            # 5. Masterização Suave
            out_audio = librosa.util.normalize(out_audio)
            sf.write("lullaby_master.wav", out_audio, sr)
            
            st.success("Canção de ninar concluída!")
            st.audio("lullaby_master.wav")
            st.download_button("📥 Baixar Música de Ninar", open("lullaby_master.wav", "rb"), "lullaby_baby.wav")
            st.balloons()
