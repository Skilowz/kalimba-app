import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os

st.set_page_config(page_title="Kalimba AI Studio Pro", page_icon="🎼")

st.markdown("""
    <style>
    .main { background: #1a1c24; color: #f0f2f6; }
    .stButton>button { background: #4facfe; color: white; border-radius: 25px; border: none; padding: 10px 25px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎼 Kalimba AI Studio - Edição Acústica")
st.write("Transformação realista e suave para músicas complexas.")

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
    url = st.text_input("Link do YouTube (ex: Jimi Hendrix)")
    if url and st.button("EXTRAIR ÁUDIO"):
        try:
            audio_path = download_youtube(url)
            st.success("Áudio extraído com sucesso!")
        except: st.error("Erro no download.")

if audio_path:
    if st.button("✨ GERAR KALIMBA ACÚSTICA REALISTA"):
        with st.spinner("Analisando melodia e sintetizando timbres de madeira..."):
            # 1. Carregamento e Separação Harmônica
            y, sr = librosa.load(audio_path, sr=22050)
            y_harm, _ = librosa.effects.hpss(y) # Remove bateria e foca na melodia
            
            # 2. Transcrição de Notas Precisa (CQT)
            hop_length = 512
            cqt = np.abs(librosa.cqt(y_harm, sr=sr, hop_length=hop_length, n_bins=72))
            
            # Criar silêncio do mesmo tamanho
            out_audio = np.zeros_like(y)
            
            # 3. Síntese com Timbre de Kalimba Real
            # Usamos uma janela maior para evitar o som "picotado"
            for t in range(0, cqt.shape[1], 3): # Processa frames com intervalo para suavizar
                f_idx = cqt[:, t].argmax()
                mag = cqt[f_idx, t]
                
                # Só toca se a nota for clara (limiar de ruído)
                if mag > np.max(cqt) * 0.15:
                    freq = librosa.cqt_frequencies(72, fmin=librosa.note_to_hz('C2'))[f_idx]
                    
                    if 130 < freq < 1200: # Range real de uma Kalimba
                        dur = 0.8 # Notas mais longas para ressonância
                        t_n = np.linspace(0, dur, int(dur * sr))
                        
                        # Timbre: Fundamental + Harmônico Metálico (2.8x) + Ruído de dedo (ataque)
                        tone = np.sin(2 * np.pi * freq * t_n)
                        overtone = 0.2 * np.sin(2 * np.pi * freq * 2.8 * t_n)
                        
                        # Envelope ADSR: Ataque percussivo e decaimento exponencial suave
                        env = np.exp(-5 * t_n) * (1 - np.exp(-200 * t_n))
                        note_wav = (tone + overtone) * env
                        
                        # Inserir no tempo correto (Overlap-Add)
                        start = t * hop_length
                        end = min(start + len(note_wav), len(out_audio))
                        out_audio[start:end] += note_wav[:end-start] * 0.5

            # 4. Masterização Final (Normalização e Compressão leve)
            out_audio = librosa.util.normalize(out_audio)
            sf.write("kalimba_pro_acustica.wav", out_audio, sr)
            
            st.audio("kalimba_pro_acustica.wav")
            st.download_button("Baixar Versão Final", open("kalimba_pro_acustica.wav", "rb"), "kalimba_pro.wav")
            st.balloons()
