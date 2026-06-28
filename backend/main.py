"""
GeoStream — Main FastAPI Server
Full pipeline: Fetch Himawari-8 frames → RIFE interpolation → Real-ESRGAN enhancement → Video
"""

import os
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch

# Limit PyTorch CPU thread allocation to save memory on server hosting platforms (e.g. Render)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from goes_fetcher import fetch_frames
from rife_interpolator import interpolate_frames
from esrgan_enhancer import enhance_frames
from video_generator import generate_video
from frame_analyzer import run_full_analysis

app = FastAPI(title="GeoStream", version="1.0")

# Single-worker executor: prevents concurrent RIFE+ESRGAN model loads that would OOM the container
_executor = ThreadPoolExecutor(max_workers=1)

# Logging
import logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("geostream")

# CORS (configurable via env)
_allowed_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*")
if _allowed_origins == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in _allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths and configurable directories
BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Fallback for Docker container if frontend is under /app/frontend and backend is directly under /app
if not FRONTEND_DIR.exists():
    docker_frontend = BACKEND_DIR / "frontend"
    if docker_frontend.exists():
        FRONTEND_DIR = docker_frontend

VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", BACKEND_DIR / "data" / "videos"))
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_PATH = VIDEO_DIR / os.environ.get("VIDEO_NAME", "satellite_video.mp4")
FRAMES_DIR = Path(os.environ.get("FRAMES_DIR", BACKEND_DIR / "data" / "fetched_frames"))
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# Ensure RIFE output directory exists (same as used by rife_interpolator default)
RIFE_OUT = Path(os.environ.get("RIFE_OUTPUT_DIR", BACKEND_DIR / "data" / "interpolated_frames"))
RIFE_OUT.mkdir(parents=True, exist_ok=True)

# Mount static assets
if FRONTEND_DIR.exists() and FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
else:
    LOG.warning("Frontend directory not found at '%s'. Static files will not be served.", FRONTEND_DIR)

# FFmpeg detection helper
import shutil

def check_ffmpeg():
    ff = shutil.which("ffmpeg")
    if not ff:
        LOG.error("FFmpeg not found in PATH. Please install FFmpeg and ensure it's on PATH.")
        return False
    LOG.info("FFmpeg found: %s", ff)
    return True

# Health endpoint
@app.get("/health")
def health():
    """Basic health check for readiness. Verifies essential components."""
    ff = check_ffmpeg()
    return {
        "ok": True,
        "ffmpeg": bool(ff),
    }

# Pipeline status tracking
pipeline_status = {
    "running": False,
    "stage": "idle",
    "progress": 0,
    "message": "",
    "error": None,
}

# Latest analysis result
latest_analysis = {
    "result": None,
    "running": False,
    "error": None,
}


def _update_status(stage: str, progress: int, message: str):
    pipeline_status["stage"] = stage
    pipeline_status["progress"] = progress
    pipeline_status["message"] = message


# ──────────────── Routes ────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/result.html", response_class=HTMLResponse)
def result():
    return (FRONTEND_DIR / "result.html").read_text(encoding="utf-8")


@app.get("/analysis-report.html", response_class=HTMLResponse)
def analysis_report_page():
    return (FRONTEND_DIR / "analysis-report.html").read_text(encoding="utf-8")


@app.get("/status")
def status():
    return pipeline_status


@app.post("/generate")
async def generate(req: Request):
    if pipeline_status["running"]:
        return JSONResponse({"error": "Pipeline already running"}, status_code=409)

    data = await req.json()
    date_str = data.get("date")
    from_time = data.get("from_time")
    to_time = data.get("to_time")
    use_esrgan = data.get("enhance", True)
    rife_exp = data.get("rife_exp", 1)

    if not all([date_str, from_time, to_time]):
        return JSONResponse({"error": "Missing date, from_time, or to_time"}, status_code=400)

    # Set state immediately to prevent frontend race conditions during polling
    pipeline_status["running"] = True
    pipeline_status["stage"] = "idle"
    pipeline_status["message"] = ""
    pipeline_status["error"] = None

    # Run pipeline in background using bounded executor (max_workers=1 prevents OOM on constrained hosts)
    asyncio.get_event_loop().run_in_executor(_executor, _run_pipeline, date_str, from_time, to_time, use_esrgan, rife_exp)

    return {"status": "started"}


def _run_pipeline(date_str: str, from_time: str, to_time: str, use_esrgan: bool, rife_exp: int):
    """Execute the full satellite video pipeline."""
    pipeline_status["running"] = True
    pipeline_status["error"] = None

    try:
        # ── Step 1: Fetch ──
        _update_status("fetching", 10, "Fetching Himawari-8 satellite frames...")
        fetched = fetch_frames(date_str, from_time, to_time)
        if not fetched:
            raise RuntimeError("No frames fetched — the satellite may not have data for this time range.")

        # ── Step 2: RIFE Interpolation ──
        _update_status("interpolating", 35, f"Interpolating {len(fetched)} frames with RIFE...")
        interpolated = interpolate_frames(fetched, exp=rife_exp)

        # Free RIFE memory immediately to prevent concurrent model memory overhead
        try:
            import rife_interpolator
            rife_interpolator.unload_model()
        except Exception as e:
            LOG.warning("Failed to unload RIFE model: %s", e)

        # ── Step 3: Real-ESRGAN Enhancement ──
        if use_esrgan:
            _update_status("enhancing", 60, f"Enhancing {len(interpolated)} frames with Real-ESRGAN...")
            final_frames = enhance_frames(interpolated, scale=4)
            # Free Real-ESRGAN memory immediately after use
            try:
                import esrgan_enhancer
                esrgan_enhancer.unload_model()
            except Exception as e:
                LOG.warning("Failed to unload Real-ESRGAN model: %s", e)
        else:
            final_frames = interpolated

        # ── Step 4: Generate Video ──
        _update_status("encoding", 85, "Encoding video...")
        generate_video(final_frames, fps=24)

        _update_status("done", 100, "Video ready!")
        print("✅ Pipeline complete!")

    except Exception as e:
        traceback.print_exc()
        pipeline_status["error"] = str(e)
        _update_status("error", 0, f"Error: {e}")

    finally:
        pipeline_status["running"] = False


@app.get("/video")
def video():
    if not VIDEO_PATH.exists():
        return JSONResponse(
            {"error": "No video generated yet"},
            status_code=404
        )

    return FileResponse(
        path=str(VIDEO_PATH),
        media_type="video/mp4"
    )


# ──────────────── Analysis Endpoints ────────────────

@app.post("/analyze")
async def analyze(req: Request):
    """Run satellite frame analysis for disaster monitoring."""
    if latest_analysis["running"]:
        return JSONResponse({"error": "Analysis already running"}, status_code=409)

    data = await req.json()
    region = data.get("region", "Himawari-8 Full Disk")
    date_str = data.get("date", "")
    from_time = data.get("from_time", "")
    to_time = data.get("to_time", "")
    fetch_new = data.get("fetch_new", False)

    latest_analysis["running"] = True
    latest_analysis["error"] = None
    latest_analysis["result"] = None

    # Run analysis in background using bounded executor
    asyncio.get_event_loop().run_in_executor(
        _executor, _run_analysis, region, date_str, from_time, to_time, fetch_new
    )

    return {"status": "started"}


def _run_analysis(region: str, date_str: str, from_time: str, to_time: str, fetch_new: bool):
    """Execute frame analysis pipeline."""
    try:
        if fetch_new and all([date_str, from_time, to_time]):
            print("📡 Fetching fresh frames for analysis...")
            fetched = fetch_frames(date_str, from_time, to_time)
            if not fetched:
                raise RuntimeError("No frames fetched for analysis")
            frame_paths = fetched
        else:
            # Use already-fetched frames
            frames_dir = str(FRAMES_DIR)
            if not os.path.isdir(frames_dir):
                raise RuntimeError("No fetched frames found. Generate a video first or enable fetch_new.")
            frame_paths = sorted([
                os.path.join(frames_dir, f)
                for f in os.listdir(frames_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            if len(frame_paths) < 2:
                raise RuntimeError("Need at least 2 frames for analysis. Generate a video first.")

        print(f"🔬 Running analysis on {len(frame_paths)} frames...")
        report = run_full_analysis(
            frame_paths,
            region=region,
            date=date_str or None,
            start_time=from_time or None,
            end_time=to_time or None,
        )
        latest_analysis["result"] = report
        print("✅ Analysis complete!")

    except Exception as e:
        traceback.print_exc()
        latest_analysis["error"] = str(e)

    finally:
        latest_analysis["running"] = False


@app.get("/analysis-status")
def analysis_status():
    """Check analysis progress."""
    return {
        "running": latest_analysis["running"],
        "error": latest_analysis["error"],
        "has_result": latest_analysis["result"] is not None,
    }


@app.get("/analysis-result")
def analysis_result():
    """Get the latest analysis report."""
    if latest_analysis["result"] is None:
        return JSONResponse({"error": "No analysis result available"}, status_code=404)
    return latest_analysis["result"]
