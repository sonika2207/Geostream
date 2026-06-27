Deployment
==========

This backend is prepared for production deployment with Docker or direct Python.

Quick start (local):

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Quick start (Docker):

```bash
docker build -t geostream-backend .
docker run -p 8000:8000 geostream-backend
```

Environment variables:
- `CORS_ALLOW_ORIGINS` — comma-separated list or `*` (default `*`).
- `RIFE_WEIGHTS_DIR` — directory where RIFE `train_log` lives. Defaults to `backend/rife/ECCV2022-RIFE/RIFE_trained_v6/train_log`.
- `RIFE_WEIGHTS_GDRIVE_ID` — Google Drive file id to download if weights are missing.
- `RIFE_OUTPUT_DIR` — output directory for interpolated frames.
- `VIDEO_DIR` — directory to write generated video.

The server exposes a health check at `/health` and a status endpoint at `/status`.

Notes:
- The backend will automatically download and extract the RIFE pretrained weights if they are missing. Progress is shown in the container logs.
- FFmpeg must be available on PATH; the Docker image installs `ffmpeg` system package.
