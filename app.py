import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import yt_dlp
import os
import pretty_midi
import fluidsynth

# ================= UI =================
st.set_page_config(
    page_title="Piano Lullaby AI Studio",
    page_icon="🎹",
    layout="wide"
)

st.title("🎹 Piano Lullaby AI Studio")
st.write("Transformando músicas complexas em versões de ninar, com inteligência musical real.")

# ================= HELPERS =================

def download_youtube(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'input',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav'
        }]
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "input.wav"

def analyze_music(path):
    y, sr = librosa.load(path, sr=22050)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chords_energy = chroma.mean(axis=1)

    key_index = np.argmax(chords_energy)
    key = librosa.midi_to_note(60 + key_index)

    return {
        "tempo": tempo,
        "key": key,
        "chroma": chroma
    }

# ================= IA DE REDUÇÃO MUSICAL =================

BABY_SAFE_INTERVALS = [0, 3, 4, 7, 12]  # uníssono, terça, quinta, oitava

def reduce_harmony(chroma):
    notes = []
    for t in range(0, chroma.shape[1], 12):
        frame = chroma[:, t]
        root = np.argmax(frame)

        for interval in BABY_SAFE_INTERVALS[:2]:
            notes.append(root + interval)

    return notes

# ================= ARRANJO LULLABY =================

def create_lullaby_score(notes, base_tempo):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    time = 0.0
    tempo = max(55, min(70, base_tempo * 0.65))
    beat = 60 / tempo

    for n in notes:
        pitch = 60 + (n % 12)
        note = pretty_midi.Note(
            velocity=40,
            pitch=pitch,
            start=time,
            end=time + beat * 2
        )
        piano.notes.append(note)
        time += beat * 2

    midi.instruments.append(piano)
    midi.write("lullaby.mid")

# ================= RENDERIZAÇÃO REAL =================

def render_piano():
    fs = fluidsynth.Synth()
    fs.start(driver="alsa" if os.name != "nt" else "dsound")

    fs.sfload("piano_felt.sf2", reset_presets=True)
    fs.program_select(0, 0, 0, 0)

    fs.midi_to_audio("lullaby.mid", "output.wav")
    fs.delete()

# ================= STREAMLIT =================

uploaded = st.file_uploader("Upload MP3 ou WAV", type=["mp3", "wav"])
yt_url = st.text_input("Ou cole um link do YouTube")

if st.button("🎼 GERAR LULLABY PROFISSIONAL"):
    try:
        with st.spinner("Analisando música com IA musical..."):
            if yt_url:
                path = download_youtube(yt_url)
            else:
                path = "input.wav"
                with open(path, "wb") as f:
                    f.write(uploaded.getbuffer())

            analysis = analyze_music(path)
            notes = reduce_harmony(analysis["chroma"])
            create_lullaby_score(notes, analysis["tempo"])
            render_piano()

            st.audio("output.wav")
            st.success("Versão de ninar criada com sucesso!")

    except Exception as e:
        st.error(str(e))
