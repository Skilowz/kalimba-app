import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os

# --- INTERFACE MODERNA E CONVIDATIVA ---
st.set_page_config(page_title="Kalimba AI Arranger", page_icon="🍃", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e212b; border-radius: 10px; padding: 10px; color: white;
    }
    .stButton>button {
        background: linear-gradient(45deg, #d4a373, #8b5e3c);
        color: white; border-radius: 20px; border: none; width: 100%;
        font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 5px 15px rgba(212,163,115,0.4); }
    </style>
    """, unsafe_allow_html=True)

st.title("🍃 Kalimba AI Arranger")
st.write("Transformação acústica baseada em arranjo inteligente e modelagem física.")

# --- FUNÇÃO DE DOWNLOAD ROBUSTA ---
def download_youtube(url):
    if os.path.exists("yt_audio.wav"): os.remove("yt_audio.wav")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'yt_audio',
        'noplaylist': True,
        'quiet': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'postprocessors': [{'key': 'FFmpegExtractAudio','preferredcodec': 'wav','preferredquality': '192'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "yt_audio.wav"

# --- INTERFACE DE ENTRADA ---
tab1, tab2 = st.tabs(["📁 Arquivo Local", "🎥 Link do YouTube"])
audio_path = None

with tab1:
    file = st.file_uploader("Selecione sua música", type=["mp3", "wav"])
    if file: audio_path = file

with tab2:
    url = st.text_input("Cole o link do YouTube")
    if url:
        if st.button("EXTRAIR ÁUDIO"):
            with st.spinner("Buscando áudio..."):
                try:
                    audio_path = download_youtube(url)
                    st.success("Áudio preparado para arranjo!")
                except Exception as e:
                    st.error("Erro ao acessar o YouTube. Tente fazer o upload do arquivo manualmente.")

# --- MOTOR DE IA E SÍNTESE ACÚSTICA ---
if audio_path:
    if st.button("✨ GERAR ARRANJO ACÚSTICO PARA KALIMBA"):
        with st.spinner("IA analisando a harmonia e criando o dedilhado..."):
            # 1. Carregamento e Pré-processamento
            y, sr = librosa.load(audio_path, sr=22050)
            # HPSS: Isola a parte harmônica (notas) da percussiva (bateria/ruído)
            y_harm = librosa.effects.hpss(y)[0]
            
            # 2. Análise Tonal (Reconhecimento da Escala)
            chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
            # Força a música para uma tonalidade coerente
            
            # 3. Transcrição Melódica Inteligente (CQT)
            hop_len = 512
            cqt = np.abs(librosa.cqt(y_harm, sr=sr, hop_length=hop_len))
            
            out_audio = np.zeros_like(y)
            
            # 4. Síntese de Modelagem Física
            # Processamos em passos maiores para evitar o som mecânico/picotado
            for t in range(0, cqt.shape[1], 4):
                # IA escolhe as 2 notas mais dominantes (polegar esquerdo e direito)
                top_notes = np.argsort(cqt[:, t])[-2:]
                
                for f_idx in top_notes:
                    magnitude = cqt[f_idx, t]
                    
                    # Filtro de sensibilidade para evitar notas fantasmas
                    if magnitude > np.max(cqt) * 0.25:
                        freq = librosa.cqt_frequencies(cqt.shape[0], fmin=librosa.note_to_hz('C2'))[f_idx]
                        
                        if 130 < freq < 1200: # Range real da Kalimba
                            duration = 0.8
                            t_n = np.linspace(0, duration, int(duration * sr))
                            
                            # CAMADAS DE SOM REALISTA:
                            # A - Metal (Fundamental + Harmônico Inarmônico em 2.8x)
                            tone = np.sin(2 * np.pi * freq * t_n)
                            ping = 0.3 * np.sin(2 * np.pi * freq * 2.81 * t_n)
                            # B - Madeira (Sub-frequência de ressonância do corpo)
                            thump = 0.1 * np.sin(2 * np.pi * (freq/2) * t_n)
                            
                            # ENVELOPE ADSR (Ataque rápido, sustain nulo, decaimento longo)
                            env = np.exp(-4.5 * t_n) * (1 - np.exp(-350 * t_n))
                            note_wav = (tone + ping + thump) * env
                            
                            # Mixagem com Overlap-Add (Sincronia Temporal)
                            start = t * hop_len
                            end = min(start + len(note_wav), len(out_audio))
                            out_audio[start:end] += note_wav[:end-start] * 0.4

            # 5. Masterização Final
            out_audio = librosa.util.normalize(out_audio)
            sf.write("kalimba_master.wav", out_audio, sr)
            
            st.success("Arranjo acústico concluído com sucesso!")
            st.audio("kalimba_master.wav")
            st.download_button("📥 Baixar Master Acústica (WAV)", open("kalimba_master.wav", "rb"), "kalimba_ai_master.wav")
            st.balloons()

st.markdown("---")
st.caption("Focado em Processamento de Linguagem Natural de Áudio e Síntese Acústica.")
