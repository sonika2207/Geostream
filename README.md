<img width="1873" height="1028" alt="image" src="https://github.com/user-attachments/assets/efacb876-8fdd-49d0-a680-7318b736194e" /># GeoStream

Turns raw Himawari-8 satellite imagery into smooth, upscaled timelapse video. Fetches frames for a given date/time range, interpolates between them with RIFE for fluid motion, enhances detail with Real-ESRGAN, and stitches the result into an MP4 — plus a frame-analysis mode for spotting changes across a sequence.

## 🌐 Live Demo

**Demo:** https://geostream-zhsb.onrender.com
> ⚠️ **Heads up about the live demo:** this is deployed on Render's free tier, which spins the instance down after inactivity. The first request after a period of idle can take 30–60s to wake up, and may briefly show a `502` or stay unresponsive until you refresh once or twice. If it doesn't recover, see [Known Limitation](#known-limitation-free-tier-memory-ceiling) below for why — short version: the ML stack this app runs is genuinely too heavy for a 512MB container. The video below shows the full pipeline running successfully end-to-end.

---

## Demo

Demo video:
<video controls width="100%">
  <source src="https://raw.githubusercontent.com/sonika2207/Geostream/main/demo/geostream-demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Home:
<img width="1875" height="1020" alt="image" src="https://github.com/user-attachments/assets/0fc2f339-19e3-41d9-8717-7ac491e48440" />

Generating Process:
Fetching:
<img width="1872" height="1022" alt="image" src="https://github.com/user-attachments/assets/75d11cd6-ce48-4a84-b33c-d7f68972011c" />

RIFE Interpolation:
<img width="1868" height="1020" alt="image" src="https://github.com/user-attachments/assets/eae8873b-9246-41e9-a80b-f2b9fb16e508" />

 Real-ESRGAN Enhancement:
<img width="1873" height="1028" alt="image" src="https://github.com/user-attachments/assets/34b46e87-3dc4-415d-b106-d93bcdf87bb6" />


Satellite Video:
<img width="1877" height="1021" alt="image" src="https://github.com/user-attachments/assets/25c39716-20a4-4062-9d25-91b630bfecc5" />

Frame Analysis:
<img width="1860" height="1012" alt="image" src="https://github.com/user-attachments/assets/765eb0ca-904b-4993-a4ce-0711a70f39ec" />










---

## Architecture

```
Client (browser)
   │
   ▼
FastAPI app (main.py)
   │
   ├─ POST /generate  ──► background job on a 1-worker ThreadPoolExecutor
   │                          │
   │                          ├─ 1. fetch_frames()        — pull Himawari-8 frames for the time range
   │                          ├─ 2. interpolate_frames()  — RIFE: generate in-between frames
   │                          ├─ 3. enhance_frames()      — Real-ESRGAN: 4x upscale / detail enhancement
   │                          └─ 4. generate_video()       — ffmpeg encode to MP4
   │
   ├─ GET  /status    ──► poll job progress (stage, %, message)
   ├─ GET  /video      ──► serve the finished MP4
   │
   └─ POST /analyze    ──► separate background job: run_full_analysis()
       GET  /analysis-status / /analysis-result
```

**Why a background job instead of handling it inline on the request:** RIFE + Real-ESRGAN inference takes far longer than any reasonable HTTP timeout. `/generate` kicks the actual pipeline off on a `ThreadPoolExecutor` and returns immediately; the frontend polls `/status` to track progress.

**Why `max_workers=1` on the executor:** RIFE and Real-ESRGAN are both substantial models. Letting two pipeline runs execute concurrently would mean two full copies of that model stack in memory at once — affordable on a beefy machine, not on a constrained host. Capping the executor at one worker means at most one run's models are ever loaded simultaneously.

**Why frame resolution is capped (`MAX_DIM`) before interpolation:** full-resolution Himawari-8 frames are large enough that running them through RIFE directly risked exceeding available memory on its own, independent of everything else. Downscaling before interpolation keeps peak memory bounded.

---

## The deployment debugging story

This app runs fine locally. Deployed to Render's free tier, it returned `502 Bad Gateway` on `/status`. Chasing that down turned into a useful lesson in how a healthy app and a starved one can look identical from the outside — and where to actually look to tell them apart.

### 1. First guess: blocking code — wrong, but worth ruling out

The obvious suspect for a `/status` hang is something blocking the event loop — a synchronous call inside an `async def` route that stalls every other request behind it. Checked first, ruled out quickly: `/generate` already offloads the real work to a `ThreadPoolExecutor`, so the event loop stays free to keep answering `/status`.

### 2. The dashboard had the actual answer

Render's own dashboard for the service stated it plainly: **"Ran out of memory (used over 512MB)."** Not a code logic bug — a resource ceiling.

### 3. The crash was happening at *boot*, before any request arrived

The OOM wasn't triggered by load — it was happening during container startup, before a single request could be served. The cause: `main.py` imported `torch`, `opencv`, `basicsr`, and `realesrgan` at module level. In Python, importing these isn't free — `torch` alone routinely costs 150–300MB+ of memory just to load its runtime, before any model or tensor exists. With several of these heavy libraries all importing eagerly at the top of the file, the process could exceed 512MB before `uvicorn` even finished binding to a port.

### 4. Fixes applied, in order

- **Resolution cap (`MAX_DIM`)** on frames before RIFE interpolation — full-size Himawari-8 frames were likely too large to run inference on safely within 512MB.
- **Lighter dependencies**: switched `torch`/`torchvision` to CPU-only wheels (the default install pulls bloated CUDA builds that are irrelevant on a CPU-only host), swapped `opencv-python` for `opencv-python-headless` (drops GUI dependencies the server never needs), and removed an unused `gfpgan` dependency.
- **Lazy imports**: moved `torch` (and friends) from module-level imports to function-level imports, deferred until the first time the pipeline actually runs. This fixed the boot-time crash — the server now starts cleanly every time.
- **A live config bug, found along the way**: the Dockerfile's `CMD` was running `--workers 2`, despite a comment right above it saying `--workers 1`. Two workers means two full independent copies of the entire ML stack loaded in memory at once — quietly doubling peak usage. Fixed to match the comment's intent.

### Known limitation: free-tier memory ceiling

Even with all of the above fixed, triggering `/generate` on the deployed instance can still run into memory pressure. At that point the conclusion isn't a remaining code bug — it's that **torch + torchvision + basicsr + realesrgan, loaded together and running real inference, structurally needs more memory than Render's free 512MB tier provides.** That's not something fixable through code changes alone; it would need a host with a larger memory allowance.

Worth being precise about how confident that conclusion actually is: the app boots and runs correctly locally, and breaks only after deploying, which is consistent with a memory-ceiling issue and fits the 512MB limit that Render's own dashboard already confirmed once for the boot-time crash. Render's free tier also spins the instance down after inactivity, which independently produces `502`s on wake-up that look superficially similar but are a different, expected behavior (cold start, not a crash). Direct confirmation of *which* of these is occurring at any given moment — Render's infrastructure-level Events log, rather than the application's own logs — wasn't exhaustively checked on every occurrence, since the OS-level OOM killer terminates a process from outside before it gets the chance to log its own error. So: strong circumstantial evidence, consistent with the known constraint, not independently re-confirmed every single time.

### Alternatives considered for hosting

| Option | Verdict |
|---|---|
| Fly.io | Free tier is legacy-only / no longer available for new apps |
| PythonAnywhere | No ASGI support — incompatible with FastAPI/uvicorn |
| Google Cloud Run | Best fit — full Docker support, configurable memory (e.g. `--memory 2Gi`, `--no-cpu-throttling`), generous free quota — but exceeding that quota requires billing enabled |

### Why this is still deployed on the free tier

This project is a portfolio piece, not a production service. Rather than pay to host a memory-hungry ML pipeline indefinitely, the chosen tradeoff is: keep the free-tier deployment up as a live demo (with the known cold-start/memory caveats above), and rely on the included **demo video** as the reliable, always-working proof that the full pipeline works end-to-end — independent of whatever mood Render's free tier is in on a given day.

---

## Tech stack

- **Backend**: FastAPI, served with `uvicorn`
- **Frame interpolation**: RIFE
- **Super-resolution**: Real-ESRGAN (`basicsr`, `realesrgan`)
- **Video encoding**: ffmpeg
- **Deployment**: Docker, Render

## Project structure

```
backend/
  main.py                # FastAPI app, routes, pipeline orchestration
  goes_fetcher.py         # Himawari-8 frame fetching
  rife_interpolator.py    # RIFE frame interpolation
  esrgan_enhancer.py       # Real-ESRGAN enhancement
  video_generator.py       # ffmpeg video encoding
  frame_analyzer.py        # frame-sequence analysis
frontend/
  index.html, result.html, analysis-report.html
Dockerfile
requirements.txt
```
