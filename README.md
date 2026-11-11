# rocinanteAI

A music genre classifier with a record-player web interface. Drop in an audio file and the app predicts its genre from a saved [Orange](https://orangedatamining.com/) model (`rocinanteAI.pkcls`).

The model was trained on the [GTZAN](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset using two kinds of input per track:

- **23 acoustic features** extracted with librosa
- **1,000 image features** from a SqueezeNet embedding of the track's mel spectrogram

Together these give the model **1,023 inputs** per track.

## How it works

```
audio file ──► FastAPI (/predict)
                  │
                  ├─ librosa: 23 acoustic features + mel spectrogram PNG
                  │
                  └─► Orange worker (subprocess)
                         ├─ SqueezeNet embeds the spectrogram (1,000 values)
                         ├─ concatenates 23 + 1,000 = 1,023 inputs
                         └─ rocinanteAI.pkcls predicts the genre label
```

**Acoustic features (in model order):** the mean of each of the 20 MFCCs, then spectral centroid, spectral rolloff and zero-crossing rate, computed on the first 30 seconds of audio.

**Spectrogram:** a 128-band log-mel spectrogram (fmax 8 kHz, −40 to 0 dB range), rendered as a 432 × 288 px PNG with no axes.

The API server keeps the Orange model loaded in a long-running worker process, so only the first start is slow.

## Project structure

| File | Purpose |
| --- | --- |
| `rocinante_server.py` | FastAPI app: serves the web page and the `/predict` endpoint |
| `rocinante_orange_worker.py` | Runs inside Orange's Python: loads the model, embeds the spectrogram with SqueezeNet, returns the genre |
| `model_feature_extraction.py` | Extracts the 23 acoustic features and the spectrogram image used at prediction time. It can also be run on its own for debugging |
| `rocinanteAI.pkcls` | The trained Orange model |
| `rocinanteAI.html` | Front-end (single page, served at `/`) |
| `requirements-rocinante.txt` | Python dependencies for the API server |
| `feature_extraction.py` | Training-time script: builds a richer feature CSV from GTZAN (not used by the server) |
| `image_extraction.py` | Training-time script: renders GTZAN spectrogram PNGs (not used by the server) |
| `file_verification.py` | Small helper that counts the `.wav` files per genre in the dataset |
| `dqjs.html`, `dqjs.png`, `dqjs.webp`, `dsjq-audio.mp3` | Assets for a hidden easter egg in the UI |
| `favicon.webp` | Site icon |

The training scripts use hard-coded Windows paths (`D:\RocinanteAI\Data\Training\...`). Edit them to point at your copy of GTZAN before running.

## Requirements

- **Windows** with [Orange](https://orangedatamining.com/download/) installed, plus its **Image Analytics** add-on (Options → Add-ons). The server launches Orange's bundled `python.exe` to run the model, and it looks for it at `%LOCALAPPDATA%\Programs\Orange\python.exe` by default.
- **Python 3.10+** for the API server.

## Setup

```bash
pip install -r requirements-rocinante.txt
```

If Orange is installed somewhere else, point the server at it:

```bash
# Windows (PowerShell)
$env:ORANGE_PYTHON = "C:\path\to\Orange\python.exe"
```

## Run

```bash
python rocinante_server.py
```

Then open <http://localhost:5000>. Or run it with uvicorn:

```bash
uvicorn rocinante_server:app --host 0.0.0.0 --port 5000
```

## API

### `POST /predict`

Upload an audio file as multipart form data in a field named `audio`.

```bash
curl -X POST http://localhost:5000/predict -F "audio=@song.mp3"
```

```json
{ "label": "jazz" }
```

Errors come back as `{"error": "..."}` with an appropriate status code. Uploads are limited to 200 MB.

### `GET /health`

Returns `{"status": "ok"}`.

### Other routes

`GET /` serves the web interface, and the remaining routes serve its static assets.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `PORT` | `5000` | Port used when starting with `python rocinante_server.py` |
| `ORANGE_PYTHON` | auto-detected | Full path to Orange's `python.exe` |
| `ORANGE_INSTALL` | `%LOCALAPPDATA%\Programs\Orange` | Orange install folder |
| `ROCI_ALLOWED_ORIGINS` | *(empty)* | Comma-separated list of origins allowed to call the API from a separately hosted front-end (enables CORS) |

## Using the feature extractor on its own

```bash
python model_feature_extraction.py path/to/song.mp3
```

This prints the 23 acoustic feature names and values and the path to the generated spectrogram image.

## Notes

- The model is pickled with Orange, so it needs a compatible Orange install to load. Loading it with a different scikit-learn version can trigger `InconsistentVersionWarning` and may change results.
- The SqueezeNet embedding is computed by Orange's own `SqueezenetModel`. Re-implementing it elsewhere (for example in PyTorch) may not reproduce the same values.
- Because the server depends on a local Orange installation, it is not a drop-in fit for typical Linux hosting such as Render without extra setup.
- Genre predictions come from a model trained on GTZAN (30-second clips of ten genres), so accuracy on other music may vary.
