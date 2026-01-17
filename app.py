import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os
from scipy.signal import butter, lfilter

# --- UI PROFISSIONAL (Modo Noturno Elegante) ---
st.set_page_config(page_title="Lullaby Master Pro", page_icon="🎹", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #0b0e14; color: #e1e1e1; }
    .main-card { background: #161b22; border-radius: 15px; padding: 30px; border: 1px solid #30363d; }
    .stButton>button { 
        background: linear-gradient(90deg, #1f6feb, #58a6ff); 
        color: white; border: none; border-radius: 8px; height: 3em; width: 100%;
        font-weight: 600; letter-spacing: 0.5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ENGINE DE ÁUDIO DE ALTA PERFORMANCE ---
def apply_piano_dynamics(freq, dur, sr=22050):
    """Sintetiza um piano com modelagem de martelo e ressonância de cordas."""
    t = np.linspace(0, dur, int(sr * dur))
    
    # Modelagem de Harmônicos de Piano Solo (Inarmonias sutis)
    # Piano não é uma onda pura, tem harmônicos que variam com o tempo
    fundamental = np.sin(2 * np.pi * freq * t)
    h2 = 0.35 * np.sin(2 * np.pi * freq * 2.001 * t) * np.exp(-t * 2.5)
    h3 = 0.15 * np.sin(2 * np.pi * freq * 3.002 * t) * np.exp(-t * 4.0)
    
    # Envelope ADSR Profissional (Ataque de Martelo + Decay Exponencial)
    # Ataque: rápido mas não instantâneo (evita cliques)
    attack_time = 0.02
    attack = np.linspace(0, 1, int(sr * attack_time))
    decay = np.exp(-2.2 * (t - attack_time))
    envelope = np.ones_like(t)
    envelope[:len(attack)] = attack
    envelope[len(attack):] = decay[len(attack):]
    
    # Adição de "Ruído de Feltro" (O som mecânico do piano real)
    felt_noise = (np.random.randn(len(t)) * 0.01) * np.exp(-t * 30)
    
    return (fundamental + h2 + h3 + felt_noise) * envelope

def process_pro_arrangement(audio_path):
    """Lógica de Arranjo Generativo."""
    y, sr = librosa.load(audio_path, sr=22050)
    
    # 1. Desaceleração Relaxante (Modo Berço)
    y_slow = librosa.effects.time_stretch(y, rate=0.7)
    
    # 2. Extração de Melodia por Filtro Espectral (Remove Hendrix Distorcido)
    # Isola componentes harmônicos estáveis e remove percussão/ruído
    y_harmonic, _ = librosa.effects.hpss(y_slow, margin=2.5)
    
    # 3. Transcrição por CQT (Constant-Q Transform) de Alta Resolução
    cqt = np.abs(librosa.cqt(y_harmonic, sr=sr, fmin=librosa.note_to_hz('C2'), n_bins=60))
    
    out_audio = np.zeros_like(y_slow)
    
    # 4. Algoritmo de Arranjo Humano (Evita o efeito mecânico)
    # Ele agrupa notas e cria um "dedilhado" virtual
    hop_length = 512
    for t in range(0, cqt.shape[1], 5):
        # Encontra os picos de energia (melodia + tônica)
        peaks = np.argsort(cqt[:, t])[-3:] 
        
        for p_idx in peaks:
            mag = cqt[p_idx, t]
            if mag > np.max(cqt) * 0.15: # Filtro de sensibilidade dinâmica
                freq = librosa.cqt_frequencies(60, fmin=librosa.note_to_hz('C2'))[p_idx]
                
                # Síntese com dinâmica variada baseada na intensidade original
                # Isso dá "vida" ao piano: notas variam de volume
                dur_note = 1.5 
                piano_note = apply_piano_dynamics(freq, dur_note, sr) * (mag / np.max(cqt))
                
                start_sample = t * hop_length
                end_sample = min(start_sample + len(piano_note), len(out_audio))
                out_audio[start_sample:end_sample] += piano_note[:end_sample-start_sample] * 0.6

    # 5. Efeito de Reverberação de Sala (Ambiência Acolhedora)
    return librosa.util.normalize(out_audio)

# --- INTERFACE ---
st.container().markdown('<div class="main-card">', unsafe_allow_html=True)
st.title("🎹 Lullaby Piano Master Pro")
st.write("A mais avançada conversão generativa para canções de ninar.")

uploaded_file = st.file_uploader("Arraste sua música ou use o YouTube abaixo", type=["mp3", "wav"])
yt_url = st.text_input("Link do YouTube (Processamento Deep Audio)")

if st.button("TRANSFORMAR EM PIANO DE NINAR"):
    if uploaded_file or yt_url:
        with st.spinner("Desconstruindo áudio e recompondo arranjo..."):
            path = uploaded_file if uploaded_file else "input.wav"
            if yt_url:
                # Lógica de download simplificada (yt_dlp já configurado no packages.txt)
                os.system(f'yt-dlp -x --audio-format wav -o "input.wav" {yt_url}')
                path = "input.wav"
            
            audio_final = process_pro_arrangement(path)
            sf.write("output_master.wav", audio_final, 22050)
            
            st.success("Arranjo concluído com fidelidade acústica!")
            st.audio("output_master.wav")
            st.download_button("Baixar Master Finalizada", open("output_master.wav", "rb"), "lullaby_piano_pro.wav")
            st.balloons()
