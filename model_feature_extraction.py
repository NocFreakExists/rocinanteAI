"""Feature extraction that matches the audio and spectrogram inputs used by rocinanteAI.pkcls."""
import argparse
import json
from pathlib import Path
import tempfile

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Exact order used in Data/Training/gtzan_features.csv and model.original_domain.
FEATURE_NAMES = tuple(
    [f"mfcc_{i}_mean" for i in range(1, 21)]
    + ["spectral_centroid", "spectral_rolloff", "zero_crossing_rate"]
)
SAMPLE_DURATION_SECONDS = 30


def extract_model_inputs(audio_path):
    """Return the 23 acoustic features and a training-style mel spectrogram.

    The Orange worker turns the spectrogram into the model's 1,000 SqueezeNet
    image features, then combines them with this ordered 23-value vector.
    """
    y, sr = librosa.load(str(audio_path), duration=SAMPLE_DURATION_SECONDS)
    if y.size == 0:
        raise ValueError("The audio file is empty or could not be decoded.")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features = [float(np.mean(row)) for row in mfcc]
    features.extend([
        float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
        float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
        float(np.mean(librosa.feature.zero_crossing_rate(y))),
    ])
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} audio features, got {len(features)}.")

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image_path = Path(tmp.name)

    fig = None
    try:
        # Training images are 432 x 288 px (6 x 4 in at 72 dpi), with default
        # plot margins retained and an explicit -40 to 0 dB color range.
        fig, ax = plt.subplots(figsize=(6, 4), dpi=72)
        ax.set_axis_off()
        librosa.display.specshow(log_mel, sr=sr, ax=ax, vmin=-40, vmax=0)
        fig.savefig(image_path)
    except Exception:
        image_path.unlink(missing_ok=True)
        raise
    finally:
        if fig is not None:
            plt.close(fig)

    return features, image_path


def main():
    parser = argparse.ArgumentParser(description="Extract model-aligned RocinanteAI audio inputs.")
    parser.add_argument("audio", help="Path to an audio file")
    args = parser.parse_args()
    values, image_path = extract_model_inputs(args.audio)
    print(json.dumps({
        "feature_names": FEATURE_NAMES,
        "features": values,
        "spectrogram": str(image_path),
    }, indent=2))


if __name__ == "__main__":
    main()
