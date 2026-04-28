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

#------------------------------------------------------------------
# CONFIGURATION
#------------------------------------------------------------------
load_dotenv()  # Load environment variables from .env
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    raise ValueError("⚠️ HF_TOKEN VARIABLE NOT CONFIGURED. Create a .env file with: HF_TOKEN=your_token_here")
OUTPUT_DIR = "audio_output"

#------------------------------------------------------------------
# HELPER FUNCTIONS
#------------------------------------------------------------------

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
# SNR CLASSIFICATION AND DIARIZATION
#------------------------------------------------------------------

print("Processing audio files...")

# Create output folders for each classification level
invalid_dir = os.path.join(OUTPUT_DIR, "invalid")
marginal_dir = os.path.join(OUTPUT_DIR, "marginal")
valid_dir = os.path.join(OUTPUT_DIR, "valid")

# Create subdirectories for speaker outputs within each classification
invalid_speaker1_dir = os.path.join(invalid_dir, "speaker_1")
invalid_speaker2_dir = os.path.join(invalid_dir, "speaker_2")
marginal_speaker1_dir = os.path.join(marginal_dir, "speaker_1")
marginal_speaker2_dir = os.path.join(marginal_dir, "speaker_2")
valid_speaker1_dir = os.path.join(valid_dir, "speaker_1")
valid_speaker2_dir = os.path.join(valid_dir, "speaker_2")

# Create all directories
for dir_path in [invalid_dir, marginal_dir, valid_dir, 
                 invalid_speaker1_dir, invalid_speaker2_dir,
                 marginal_speaker1_dir, marginal_speaker2_dir,
                 valid_speaker1_dir, valid_speaker2_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Authenticate with Hugging Face
login(token=HF_TOKEN, add_to_git_credential=False)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

for f in os.listdir(OUTPUT_DIR):
    if f.endswith(".wav"):
        path = os.path.join(OUTPUT_DIR, f)
        if not os.path.isfile(path):
            continue
            
        print(f"\nProcessing: {f}")
        
        # Load audio
        audio = AudioSegment.from_wav(path)
        print(f"  Audio loaded: {len(audio)}ms ({len(audio)/1000:.2f}s)")
        
        # Calculate SNR first and classify
        snr = calculate_snr(audio)
        
        if snr < 10:
            status = "invalid"
            base_dir = invalid_dir
        elif 10 <= snr < 15:
            status = "marginal"
            base_dir = marginal_dir
        else:
            status = "valid"
            base_dir = valid_dir
        
        print(f"  SNR: {snr:.2f} dB -> Classification: {status}")
        
        # Perform diarization
        diarization = pipeline(path, max_speakers=2)
        
        # Count unique speakers
        speakers = set()
        for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
            speakers.add(speaker)
        
        print(f"  Speakers detected: {sorted(speakers)}")
        
        # Process speaker extraction for multi-speaker files
        if len(speakers) > 1:
            speakers_audio = {speaker: AudioSegment.empty() for speaker in speakers}
            
            for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                segment = audio[start_ms:end_ms]
                speakers_audio[speaker] += segment
            
            # Save each speaker to the appropriate classification folder
            speakers_list = sorted(speakers)
            for idx, speaker in enumerate(speakers_list[:2], 1):  # Maximum 2 speakers
                speaker_audio = speakers_audio[speaker]
                
                if len(speaker_audio) == 0:
                    print(f"  WARNING: Speaker {speaker} has 0ms of audio!")
                    continue
                
                # Determine output path within classification folder
                speaker_dir = os.path.join(base_dir, f"speaker_{idx}")
                output_path = os.path.join(speaker_dir, f"{os.path.splitext(f)[0]}_speaker{idx}.wav")
                
                speaker_audio.export(output_path, format="wav")
                print(f"  Exported to {status}/speaker_{idx}/: {os.path.basename(output_path)}")
        
        # Move original file to classification folder (single speaker or as backup)
        destination_path = os.path.join(base_dir, f)
        shutil.move(path, destination_path)
        print(f"  Moved to: {status}/")
