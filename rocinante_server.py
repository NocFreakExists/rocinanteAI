from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import threading

from model_feature_extraction import extract_model_inputs
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
USER_LOCAL_APP_DATA = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "AppData" / "Local"
ORANGE_ROOT = Path(os.environ.get("ORANGE_INSTALL", Path(os.environ.get("LOCALAPPDATA", str(USER_LOCAL_APP_DATA))) / "Programs" / "Orange"))
ORANGE_CANDIDATES = [ORANGE_ROOT / "python.exe", USER_LOCAL_APP_DATA / "Programs" / "Orange" / "python.exe"]
ORANGE_PYTHON = Path(os.environ.get("ORANGE_PYTHON", next((str(candidate) for candidate in ORANGE_CANDIDATES if candidate.is_file()), str(ORANGE_CANDIDATES[0]))))

WORKER = ROOT / "rocinante_orange_worker.py"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
logger = logging.getLogger("rocinante")


class OrangeWorker:
    """Keep Orange and the saved classifier loaded between predictions."""

    def __init__(self):
        if not ORANGE_PYTHON.is_file():
            raise FileNotFoundError(f"Orange Python was not found at {ORANGE_PYTHON}")
        self.process = subprocess.Popen(
            [str(ORANGE_PYTHON), str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.lock = threading.Lock()
        ready = self.process.stdout.readline()
        if not ready:
            self.process.wait(timeout=2)
            raise RuntimeError("Orange model worker failed to start; see the Orange worker error above.")
        status = json.loads(ready)
        if not status.get("ready"):
            raise RuntimeError(status.get("error", "Orange model worker failed to start."))

    def predict(self, audio_features, image_path):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError("Orange model worker has stopped.")
            self.process.stdin.write(json.dumps({"audio": audio_features, "image": str(image_path)}) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Orange model worker stopped before returning a prediction.")
        result = json.loads(line)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result["label"]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()


def classify_audio(audio_path, model_worker):
    features, image_path = extract_model_inputs(audio_path)
    try:
        return model_worker.predict(features, image_path)
    finally:
        image_path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.orange_worker = OrangeWorker()
    try:
        yield
    finally:
        app.state.orange_worker.close()


app = FastAPI(title="RocinanteAI API", lifespan=lifespan)

# Enable cross-origin calls only when a separate static frontend (for example,
# a Netlify site) is configured. The default local page is served by this API.
allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ROCI_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"] ,
    )


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.url.path == "/predict":
        length = request.headers.get("content-length")
        if length:
            try:
                if int(length) > MAX_UPLOAD_BYTES:
                    return JSONResponse({"error": "Audio file exceeds the 200 MB upload limit."}, status_code=413)
            except ValueError:
                return JSONResponse({"error": "Invalid content length."}, status_code=400)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(
        ROOT / "rocinanteAI.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/dqjs.html", include_in_schema=False)
def dqjs_page():
    return FileResponse(
        ROOT / "dqjs.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/dqjs.png", include_in_schema=False)
def dqjs_image():
    return FileResponse(ROOT / "dqjs.png", media_type="image/png")


@app.get("/dsjq-audio.mp3", include_in_schema=False)
def dqjs_audio():
    return FileResponse(ROOT / "dsjq-audio.mp3", media_type="audio/mpeg")


@app.get("/favicon.webp", include_in_schema=False)
def favicon():
    return FileResponse(ROOT / "favicon.webp", media_type="image/webp")


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request, audio: UploadFile = File(...)):
    audio_path = None
    try:
        suffix = Path(audio.filename or "audio").suffix or ".audio"
        total = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio_path = Path(tmp.name)
            while chunk := await audio.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    return JSONResponse({"error": "Audio file exceeds the 200 MB upload limit."}, status_code=413)
                tmp.write(chunk)
        if total == 0:
            return JSONResponse({"error": "The audio file is empty."}, status_code=400)
        label = await run_in_threadpool(
            classify_audio, audio_path, request.app.state.orange_worker
        )
        return {"label": label}
    except Exception as exc:
        logger.exception("Prediction failed")
        return JSONResponse({"error": f"Could not classify this audio: {exc}"}, status_code=500)
    finally:
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)
        await audio.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        reload=False,
    )




