import os
import subprocess
import pandas as pd
import numpy as np
import opensmile

#------------------------------------------------------------------
# CONFIGURATION
#------------------------------------------------------------------
INPUT_DIR = "audio_input"
OUTPUT_DIR = "audio_output"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#------------------------------------------------------------------
# FORMATTING WITH FFMPEG
#------------------------------------------------------------------

print("Converting to WAV (8kHz, 16bit, Mono)...\n")
extensions = ('.mp4', '.mpeg', '.ogg')
if os.path.exists(INPUT_DIR):
    files_found = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(extensions)]
    print(f"Found {len(files_found)} audio files to convert\n")
    
    for f in files_found:
        input_path = os.path.join(INPUT_DIR, f)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(f)[0] + ".wav")
        print(f"Converting: {f} -> {output_path}")
        try:
            subprocess.run(["ffmpeg", '-i', input_path, '-ar', '8000', '-ac', '1', '-sample_fmt', 's16', '-y', output_path], check=True, capture_output=True)
            if os.path.exists(output_path):
                print(f"✓ Successfully saved: {output_path}\n")
            else:
                print(f"✗ Error: File was not created\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ FFmpeg error converting {f}: {e}\n")
        except FileNotFoundError:
            print(f"✗ FFmpeg not found. Make sure FFmpeg is installed and in PATH\n")
            break
else:
    print(f"✗ Input directory '{INPUT_DIR}' not found!\n")

print("Conversion completed.\n")