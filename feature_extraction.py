import os
import csv
import librosa
import numpy as np

dataset_path = r"D:\RocinanteAI\Data\Training\genres_original"
output_csv = r"D:\RocinanteAI\Data\Training\gtzan_features_rich.csv"

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

genres = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

# Build a comprehensive header
header = ["filename"]

# 20 MFCCs (Mean + Variance = 40 features)
for i in range(1, 21):
    header.append(f"mfcc_{i}_mean")
    header.append(f"mfcc_{i}_var")

# 12 Chroma features (Mean + Variance = 24 features)
for i in range(1, 13):
    header.append(f"chroma_{i}_mean")
    header.append(f"chroma_{i}_var")

# Spectral & Energy features
header.extend([
    "spectral_centroid_mean", "spectral_centroid_var",
    "spectral_bandwidth_mean", "spectral_bandwidth_var",
    "spectral_rolloff_mean", "spectral_rolloff_var",
    "spectral_contrast_mean", "spectral_contrast_var",
    "rms_mean", "rms_var",
    "zero_crossing_rate_mean", "zero_crossing_rate_var",
    "tempo",
    "genre"
])

print(f"Starting rich feature extraction across {len(genres)} genres... This may take a few minutes.")

with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for genre in genres:
        folder = os.path.join(dataset_path, genre)
        songs = [s for s in os.listdir(folder) if s.lower().endswith(".wav")]
        
        print(f"Processing genre: {genre} ({len(songs)} tracks)")
        
        for song in songs:
            file_path = os.path.join(folder, song)
            try:
                # Load audio (duration=30 ensures standard length for GTZAN)
                y, sr = librosa.load(file_path, duration=30)
                
                # 1. MFCCs (20 coefficients)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                mfcc_features = []
                for e in mfccs:
                    mfcc_features.extend([np.mean(e), np.var(e)])
                
                # 2. Chroma Stft (12 pitch classes)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                chroma_features = []
                for e in chroma:
                    chroma_features.extend([np.mean(e), np.var(e)])
                
                # 3. Spectral & Temporal Features
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)[0]
                rms = librosa.feature.rms(y=y)[0]
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                
                # Tempo (BPM estimation)
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
                
                spectral_features = [
                    np.mean(spec_cent), np.var(spec_cent),
                    np.mean(spec_bw), np.var(spec_bw),
                    np.mean(spec_roll), np.var(spec_roll),
                    np.mean(spec_contrast), np.var(spec_contrast),
                    np.mean(rms), np.var(rms),
                    np.mean(zcr), np.var(zcr),
                    tempo
                ]
                
                # Assemble row
                row = [song] + mfcc_features + chroma_features + spectral_features + [genre]
                writer.writerow(row)
                
            except Exception as e:
                print(f"  [Error] Skipping {song}: {e}")

print(f"\nFeature extraction complete! Rich CSV saved to: {output_csv}")
