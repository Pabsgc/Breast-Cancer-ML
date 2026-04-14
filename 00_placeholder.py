# 2. DIARIZATION AND CLEANUP
print("2. Running diarization...")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    
snr_results = []
processed_files = []

for f in os.listdir(OUTPUT_DIR):
    if f.endswith(".wav"):
        path = os.path.join(OUTPUT_DIR, f)
        diarization = pipeline(path, max_speakers=2)

        durations = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            durations[speaker] = durations.get(speaker, 0) + (turn.end - turn.start)

        if not durations: continue
        patient_speaker = max(durations, key=durations.get)

        audio = AudioSegment.from_wav(path)
        patient_audio = AudioSegment.empty()
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker == patient_speaker:
                patient_audio += audio[turn.start*1000 : turn.end*1000]

        patient_audio = normalize_audio(patient_audio)
        patient_audio.export(path, format="wav")

        snr = calculate_snr(patient_audio)
        status = "valid"
        if 10 <= snr < 15: status = "marginal"
        elif snr < 10: status = "invalid"

        snr_results.append({"file": f, "snr_db": snr, "status": status})
        if status != "invalid": processed_files.append(path)

if snr_results:
    pd.DataFrame(snr_results).to_csv(SNR_LOG, index=False)

# 4. EXTRACTION WITH OPENSMILE
print("4. Extracting eGeMAPS features...")
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPS,
    feature_level=opensmile.FeatureLevel.Functionals,
)

all_features = []
for path in processed_files:
    features = smile.process_file(path)
    all_features.append(features)

if all_features:
    final_df = pd.concat(all_features)
    final_df.to_csv(DATASET)
    print(f"Proceso completado. Resultados en {DATASET} y {SNR_LOG}")
else:
    print("No se procesaron archivos válidos para extracción.")
