import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

dataset_path = r"D:\RocinanteAI\Data\Training\genres_original"
output_dir = r"D:\RocinanteAI\Data\Training\spectrogram_images"

os.makedirs(output_dir, exist_ok=True)
genres = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

for genre in genres:
    input_folder = os.path.join(dataset_path, genre)
    output_genre_dir = os.path.join(output_dir, genre)
    os.makedirs(output_genre_dir, exist_ok=True)
    
    songs = [s for s in os.listdir(input_folder) if s.lower().endswith(".wav")]
    print(f"Processing genre for PNGs: {genre} ({len(songs)} tracks)")
    
    for song in songs:
        file_path = os.path.join(input_folder, song)
        base_name = os.path.splitext(song)[0]
        png_path = os.path.join(output_genre_dir, f"{base_name}.png")
        
        # Skip if already generated
        if os.path.exists(png_path):
            continue
            
        try:
            y, sr = librosa.load(file_path, duration=30)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Plot without axes, frames, or white space to make a pure image matrix
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            ax.set_axis_off()
            librosa.display.specshow(log_mel_spec, sr=sr, ax=ax)
            plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
        except Exception as e:
            print(f"  [Error] Skipping {song}: {e}")

print(f"\nPNG Spectrogram generation complete! Saved to: {output_dir}")
