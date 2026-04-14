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
load_dotenv()  # Load environment variables from .env
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("⚠️ HF_TOKEN VARIABLE NOT CONFIGURED. Create a .env file with: HF_TOKEN=your_token_here")
OUTPUT_DIR = "audio_output"

def normalize_audio_rms(audio_segment, target_rms=-20.0):
    """
    Returns audio segment unchanged (scale invariant for ML - not needed for Random Forest)
    """
    return audio_segment

def calculate_snr(audio_segment):
    """
    Calculate SNR using standard formula: 20*log10(RMS_signal / RMS_noise)
    Noise floor is estimated from the quietest 10% of the signal
    """
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    if len(samples) == 0:
        return 0
    
    # Calculate overall RMS (signal power)
    rms_signal = np.sqrt(np.mean(samples**2))
    
    if rms_signal == 0:
        return 0  # Silent audio
    
    # Estimate noise floor from quietest 10% of absolute samples
    abs_samples = np.abs(samples)
    threshold = np.percentile(abs_samples, 10)  # 10th percentile
    
    # Use samples below threshold as noise estimate
    noise_samples = abs_samples[abs_samples <= threshold]
    
    if len(noise_samples) == 0 or np.max(noise_samples) == 0:
        # If noise estimation fails, use a minimum threshold
        rms_noise = rms_signal * 0.01  # Assume noise is 1% of signal
    else:
        rms_noise = np.sqrt(np.mean(noise_samples**2))
    
    # Avoid division by zero
    if rms_noise == 0:
        rms_noise = 1  # Minimum noise floor
    
    # Calculate SNR in dB
    snr = 20 * np.log10(rms_signal / rms_noise)
    
    return snr if not np.isnan(snr) and not np.isinf(snr) else 0

#------------------------------------------------------------------
# DIARIZATION AND SNR CALCULATION
#------------------------------------------------------------------

print("Diarizating...")

# Create output folders if they don't exist
speaker_out_1_dir = os.path.join(OUTPUT_DIR, "speaker_out_1")
speaker_out_2_dir = os.path.join(OUTPUT_DIR, "speaker_out_2")
invalid_dir = os.path.join(OUTPUT_DIR, "invalid")
marginal_dir = os.path.join(OUTPUT_DIR, "marginal")
valid_dir = os.path.join(OUTPUT_DIR, "valid")

for dir_path in [speaker_out_1_dir, speaker_out_2_dir, invalid_dir, marginal_dir, valid_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Authenticate with Hugging Face
login(token=HF_TOKEN, add_to_git_credential=False)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

for f in os.listdir(OUTPUT_DIR):
    if f.endswith(".wav"):
        path = os.path.join(OUTPUT_DIR, f)
        if not os.path.isfile(path):
            continue
            
        print(f"Processing: {f}")

        diarization = pipeline(path, max_speakers=2)

        # Count the number of unique speakers
        speakers = set()
        durations = {}
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            durations[speaker] = durations.get(speaker, 0) + (turn.end - turn.start)

        # Load audio for later use
        audio = AudioSegment.from_wav(path)
        print(f"  Audio loaded: {len(audio)}ms ({len(audio)/1000:.2f}s)")
        
        # Process speaker extraction only for multi-speaker files
        if len(speakers) > 1:
            speakers_audio = {speaker: AudioSegment.empty() for speaker in speakers}
            
            for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                segment = audio[start_ms:end_ms]
                speakers_audio[speaker] += segment

            print(f"  Speakers: {sorted(speakers)}")
            for speaker in sorted(speakers):
                print(f"    {speaker}: {len(speakers_audio[speaker])}ms")

            # Normalize and save trimmed audio
            speakers_list = sorted(speakers)
            for idx, speaker in enumerate(speakers_list[:2], 1):  # Maximum 2 speakers
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
        else:
            print(f"  Single speaker detected - skipping speaker extraction")

        # Calculate SNR of original audio and classify (ALWAYS happens)
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

        # Move file to corresponding folder
        destination_path = os.path.join(destination_dir, f)
        shutil.move(path, destination_path)
        print(f"{f} -> {status} (SNR: {snr:.2f} dB)")
