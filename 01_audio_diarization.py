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
    Returns audio segment unchanged (scale invariant for ML - not needed for Random Forest)
    """
    return audio_segment

def calculate_snr(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if len(samples) == 0:
        return 0
    
    mean_sq = np.mean(samples**2)
    noise_sq = np.mean(np.sort(samples**2)[:int(len(samples)*0.1)])
    
    if noise_sq == 0 or mean_sq == 0:
        return 0 if mean_sq == 0 else 50
    
    ratio = mean_sq / noise_sq
    if ratio <= 0:
        return 0
    
    snr = 10 * np.log10(ratio)
    return snr if not np.isnan(snr) else 0

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
        if not os.path.isfile(path):
            continue
            
        print(f"Processing: {f}")

        diarization = pipeline(path, max_speakers=2)

        # Contar el número de hablantes únicos
        speakers = set()
        durations = {}
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            durations[speaker] = durations.get(speaker, 0) + (turn.end - turn.start)

        # Solo diarizar si hay más de un hablante
        if len(speakers) <= 1:
            print(f"  Skipping: Only {len(speakers)} speaker(s) detected")
            continue

        # Procesar audios recortados para cada hablante
        audio = AudioSegment.from_wav(path)
        print(f"  Audio loaded: {len(audio)}ms ({len(audio)/1000:.2f}s)")
        speakers_audio = {speaker: AudioSegment.empty() for speaker in speakers}
        
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            segment = audio[start_ms:end_ms]
            speakers_audio[speaker] += segment

        print(f"  Speakers: {sorted(speakers)}")
        for speaker in sorted(speakers):
            print(f"    {speaker}: {len(speakers_audio[speaker])}ms")

        # Normalizar y guardar audios recortados
        speakers_list = sorted(speakers)
        for idx, speaker in enumerate(speakers_list[:2], 1):  # Máximo 2 hablantes
            original_audio = speakers_audio[speaker]
            
            # Debug: Check audio before normalization
            if len(original_audio) == 0:
                print(f"  WARNING: Speaker {speaker} has 0ms of audio!")
                continue
            
            samples_before = np.array(original_audio.get_array_of_samples())
            rms_before = np.sqrt(np.mean(samples_before**2))
            print(f"  Before norm - {speaker}: {len(original_audio)}ms, RMS={rms_before:.2f}")
            
            speaker_audio = normalize_audio_rms(original_audio)
            
            samples_after = np.array(speaker_audio.get_array_of_samples())
            rms_after = np.sqrt(np.mean(samples_after**2))
            print(f"  After norm - {speaker}: {len(speaker_audio)}ms, RMS={rms_after:.2f}")
            
            output_path = os.path.join(
                speaker_out_1_dir if idx == 1 else speaker_out_2_dir,
                f"{os.path.splitext(f)[0]}_speaker{idx}.wav"
            )
            speaker_audio.export(output_path, format="wav")
            print(f"  Exported: {output_path}")

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
