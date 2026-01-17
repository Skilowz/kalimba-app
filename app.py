import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os

# --- INTERFACE ACOLHEDORA ---
st.set_page_config(page_title="Piano de Ninar AI", page_icon="🎹", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f0f4f8; color: #2c3e50; }
    .stButton>button { 
        background-color: #a8dadc; color: #1d3557; border-radius: 25px; 
        font-weight: bold; border: none; width: 100%; height: 50px;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #457b9d; color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: white; border-radius: 10px; color: #1d3557; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎹 Piano de Ninar AI")
st.write("Transforme qualquer música em um solo de piano doce e relaxante para bebês.")

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
    if url and st.button("PREPARAR ÁUDIO"):
        with st.spinner("Buscando música..."):
            audio_path = download_youtube(url)

# --- MOTOR DE PIANO DE NINAR ---
if audio_path:
    if st.button("✨ GERAR VERSÃO EM PIANO"):
        with st.spinner("A IA está compondo o arranjo em piano solo..."):
            # 1. Carregamento e Desaceleração (Ritmo de Ninar)
            y, sr = librosa.load(audio_path, sr=22050)
            y_slow = librosa.effects.time_stretch(y, rate=0.75) # Mais lento para calma total
            
            # 2. HPSS para focar na melodia harmônica
            y_harm = librosa.effects.hpss(y_slow)[0]
            
            # 3. Transcrição com CQT (Melhor para notas de Piano)
            hop_len = 512
            cqt = np.abs(librosa.cqt(y_harm, sr=sr, hop_length=hop_len))
            
            out_audio = np.zeros_like(y_slow)
            
            # 4. Síntese de Piano Solo (Soft Piano)
            for t in range(0, cqt.shape[1], 4):
                f_idx = cqt[:, t].argmax()
                mag = cqt[f_idx, t]
                
                if mag > np.max(cqt) * 0.18:
                    freq = librosa.cqt_frequencies(cqt.shape[0], fmin=librosa.note_to_hz('C2'))[f_idx]
                    
                    if freq < 1500: # Evita notas estridentes
                        dur = 1.2 # Sustain do pedal de piano
                        t_n = np.linspace(0, dur, int(dur * sr))
                        
                        # Timbre de Piano: Senóide + Harmônicos de corda (Pares)
                        # O segredo do piano são os harmônicos sutis
                        p1 = np.sin(2 * np.pi * freq * t_n)
                        p2 = 0.2 * np.sin(2 * np.pi * freq * 2 * t_n)
                        p3 = 0.1 * np.sin(2 * np.pi * freq * 3 * t_n)
                        
                        # Envelope de Piano (Ataque percussivo porém doce)
                        # Ataque ligeiramente mais lento que a Kalimba para ser "soft"
                        env = np.exp(-3.5 * t_n) * (1 - np.exp(-60 * t_n))
                        note_wav = (p1 + p2 + p3) * env
                        
                        start = t * hop_len
                        end = min(start + len(note_wav), len(out_audio))
                        # Sobreposição suave (Crossfade natural)
                        out_audio[start:end] += note_wav[:end-start] * 0.25

            # 5. Masterização com Soft Clipping
            out_audio = librosa.util.normalize(out_audio)
            sf.write("piano_ninar_master.wav", out_audio, sr)
            
            st.success("Sua canção de ninar em piano está pronta!")
            st.audio("piano_ninar_master.wav")
            st.download_button("📥 Baixar Piano de Ninar", open("piano_ninar_master.wav", "rb"), "piano_bebe.wav")
            st.balloons()
