import os

dataset_path = r"D:\RocinanteAI\Data\Training\genres_original"

genres = os.listdir(dataset_path)

for genre in genres:
    folder = os.path.join(dataset_path, genre)

    if os.path.isdir(folder):
        songs = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".wav")
        ]

        print(genre, ":", len(songs), "songs")
