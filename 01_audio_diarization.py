import os
import subprocess
import pandas as pd
import numpy as np
import shutil
import opensmile
from pydub import AudioSegment, effects
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from huggingface_hub import login

# --- CONFIG ---
load_dotenv()  # Cargar variables de entorno desde .env
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("⚠️ VARIABLE HF_TOKEN NO CONFIGURADA. Crea un archivo .env con: HF_TOKEN=tu_token_aqui")
OUTPUT_DIR = "audio_output"

def normalize_audio_rms(audio_segment, target_rms=-20.0):
    """
    Normaliza audio por RMS (Root Mean Square) para machine learning.
    Todos los audios quedarán con el mismo volumen promedio (.
    
    target_rms: volumen objetivo en dBFS (default -20 dB es estándar para ML)
    """
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    
    # Calcular RMS actual
    current_rms = np.sqrt(np.mean(samples**2))
    if current_rms == 0:
        return audio_segment
    
    # Calcular ganancia necesaria para alcanzar target_rms
    target_linear = 10 ** (target_rms / 20.0)
    gain = target_linear / current_rms
    
    # Aplicar ganancia
    normalized_samples = samples * gain
    
    # Prevenir clipping (distorsión por recorte)
    max_val = np.max(np.abs(normalized_samples))
    if max_val > 32767:
        normalized_samples = normalized_samples * (32767 / max_val)
    
    # Convertir de vuelta a AudioSegment
    normalized_samples = np.int16(normalized_samples)
    return AudioSegment(
        normalized_samples.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=audio_segment.sample_width,
        channels=audio_segment.channels
    )

def calculate_snr(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    mean_sq = np.mean(samples**2)
    noise_sq = np.mean(np.sort(samples**2)[:int(len(samples)*0.1)])
    if noise_sq == 0: return 50
    return 10 * np.log10(mean_sq / noise_sq)

#------------------------------------------------------------------
# DIARIZATION AND SNR CALCULATION
#------------------------------------------------------------------

print("Diarizating...")

# Crear las carpetas de salida si no existen
speaker_out_1_dir = os.path.join(OUTPUT_DIR, "speaker_out_1")
speaker_out_2_dir = os.path.join(OUTPUT_DIR, "speaker_out_2")
invalid_dir = os.path.join(OUTPUT_DIR, "invalid")
marginal_dir = os.path.join(OUTPUT_DIR, "marginal")
valid_dir = os.path.join(OUTPUT_DIR, "valid")

for dir_path in [speaker_out_1_dir, speaker_out_2_dir, invalid_dir, marginal_dir, valid_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Autenticarse con Hugging Face
login(token=HF_TOKEN, add_to_git_credential=False)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

for f in os.listdir(OUTPUT_DIR):
    if f.endswith(".wav"):
        path = os.path.join(OUTPUT_DIR, f)
        diarization = pipeline(path, max_speakers=2)

        # Contar el número de hablantes únicos
        speakers = set()
        durations = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            durations[speaker] = durations.get(speaker, 0) + (turn.end - turn.start)

        # Solo diarizar si hay más de un hablante
        if len(speakers) <= 1:
            print(f"Saltando {f}: Solo {len(speakers)} hablante(s) detectado(s)")
            continue

        # Procesar audios recortados para cada hablante
        audio = AudioSegment.from_wav(path)
        speakers_audio = {speaker: AudioSegment.empty() for speaker in speakers}
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers_audio[speaker] += audio[turn.start*1000 : turn.end*1000]

        # Normalizar y guardar audios recortados
        speakers_list = sorted(speakers)
        for idx, speaker in enumerate(speakers_list[:2], 1):  # Máximo 2 hablantes
            speaker_audio = normalize_audio_rms(speakers_audio[speaker])
            output_path = os.path.join(
                speaker_out_1_dir if idx == 1 else speaker_out_2_dir,
                f"{os.path.splitext(f)[0]}_speaker{idx}.wav"
            )
            speaker_audio.export(output_path, format="wav")

        # Calcular SNR del audio original y clasificar
        snr = calculate_snr(audio)
        status = "valid"
        if 10 <= snr < 15:
            status = "marginal"
            destination_dir = marginal_dir
        elif snr < 10:
            status = "invalid"
            destination_dir = invalid_dir
        else:
            destination_dir = valid_dir

        # Mover archivo a la carpeta correspondiente
        destination_path = os.path.join(destination_dir, f)
        shutil.move(path, destination_path)
        print(f"{f} -> {status} (SNR: {snr:.2f} dB)")
