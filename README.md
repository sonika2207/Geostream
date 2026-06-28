# 🛰️ GeoStream

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue?logo=opencv)
![Render](https://img.shields.io/badge/Deployment-Render-purple)

GeoStream transforms raw **Himawari-8 satellite imagery** into smooth, AI-enhanced timelapse videos.

The application fetches satellite frames for a user-selected date and time range, generates intermediate frames using **RIFE Frame Interpolation**, enhances image quality with **Real-ESRGAN**, and stitches everything into a browser-compatible MP4 video. It also includes a **frame-analysis mode** for identifying changes across a sequence for weather and disaster monitoring.

---

# ✨ Features

- 📡 Fetches Himawari-8 satellite imagery
- 🎞️ AI-powered frame interpolation using RIFE
- ✨ Image enhancement using Real-ESRGAN
- 🎥 Generates smooth MP4 timelapse videos
- 🔬 Satellite frame analysis for weather/disaster monitoring
- 📈 Live pipeline progress tracking
- ⚡ Background processing with FastAPI
- 🌐 Interactive web interface

---

# 🌐 Live Demo

**Demo:** https://geostream-zhsb.onrender.com

> ⚠️ **Note:** The application is hosted on Render's free tier. The service sleeps after inactivity, so the first request may take **30–60 seconds** to wake up and may briefly return a **502 Bad Gateway** until the instance starts again.

---

# 🎥 Demo

Demo Video:

https://github.com/user-attachments/assets/0472b2fa-e37f-453b-99f1-4f66a91538aa

---

# 📸 Screenshots

## Home

<img width="1875" alt="Home" src="https://github.com/user-attachments/assets/0fc2f339-19e3-41d9-8717-7ac491e48440" />

---

## Generating Pipeline

### 📡 Fetching Frames

<img width="1872" alt="Fetching" src="https://github.com/user-attachments/assets/75d11cd6-ce48-4a84-b33c-d7f68972011c" />

### 🔀 RIFE Frame Interpolation

<img width="1868" alt="Interpolation" src="https://github.com/user-attachments/assets/eae8873b-9246-41e9-a80b-f2b9fb16e508" />

### ✨ Real-ESRGAN Enhancement

<img width="1873" alt="Enhancement" src="https://github.com/user-attachments/assets/34b46e87-3dc4-415d-b106-d93bcdf87bb6" />

### 🎬 Generated Satellite Video

<img width="1877" alt="Video" src="https://github.com/user-attachments/assets/25c39716-20a4-4062-9d25-91b630bfecc5" />

### 🔬 Frame Analysis

<img width="1860" alt="Analysis" src="https://github.com/user-attachments/assets/765eb0ca-904b-4993-a4ce-0711a70f39ec" />

---

# 🏗️ Architecture

```text
Client (Browser)
      │
      ▼
 FastAPI Backend
      │
      ├─────────────── POST /generate
      │
      ▼
Fetch Himawari Frames
      │
      ▼
RIFE Frame Interpolation
      │
      ▼
Real-ESRGAN Enhancement
      │
      ▼
FFmpeg Video Encoding
      │
      ▼
Generated MP4 Video

Frontend polls:

GET /status
GET /video

Analysis:

POST /analyze
GET /analysis-status
GET /analysis-result
```

### Why background jobs?

Running RIFE and Real-ESRGAN inference takes much longer than a typical HTTP request timeout.

Instead of blocking the request, `/generate` immediately starts a background task using a `ThreadPoolExecutor`. The frontend polls `/status` to display progress while the pipeline executes asynchronously.

### Why only one worker?

Each pipeline execution loads multiple large machine learning models into memory.

Running multiple workers would duplicate these models in RAM, dramatically increasing memory usage.

Using a single worker guarantees that only one inference pipeline runs at a time.

### Why downscale frames before interpolation?

Native Himawari-8 frames are extremely large.

Running interpolation directly on full-resolution images risks exhausting available memory, especially on low-memory deployments.

Frames are resized before RIFE inference to keep memory usage bounded.

---

# 🤖 AI Models Used

| Model | Purpose |
|---------|----------|
| Himawari-8 | Satellite imagery |
| RIFE (ECCV2022) | Frame interpolation |
| Real-ESRGAN | Super resolution |
| FFmpeg | MP4 encoding |

---

# 🚧 Deployment Challenges & Lessons Learned

This application works flawlessly on a local machine but exposed several interesting deployment challenges when hosted on Render's free tier.

## 1. Initial suspicion: blocking code

The first hypothesis was that synchronous work inside an `async` endpoint was blocking FastAPI's event loop.

This was ruled out because `/generate` already offloads the heavy pipeline onto a `ThreadPoolExecutor`, allowing `/status` to remain responsive.

---

## 2. Render dashboard revealed the real issue

Render reported:

> **Ran out of memory (used over 512MB)**

This indicated that the issue wasn't application logic—it was resource exhaustion.

---

## 3. Boot-time memory exhaustion

The process exceeded Render's memory limit before serving any requests.

The root cause was eager imports of heavy machine learning libraries:

- torch
- torchvision
- basicsr
- realesrgan
- opencv

Simply importing these libraries consumes hundreds of megabytes before any inference begins.

---

## 4. Improvements applied

Several optimizations significantly reduced memory usage:

- Added a maximum frame resolution before interpolation.
- Replaced GPU-enabled PyTorch wheels with CPU-only builds.
- Switched `opencv-python` to `opencv-python-headless`.
- Removed unused dependencies.
- Converted eager imports into lazy imports.
- Fixed Docker configuration to use a single worker.

These changes eliminated boot-time crashes.

---

## Remaining limitation

Although startup memory usage was greatly reduced, the complete inference pipeline still requires more memory than Render's free 512 MB tier can provide.

This limitation stems from simultaneously loading:

- PyTorch
- RIFE
- Real-ESRGAN
- OpenCV

during inference.

The application therefore serves as a functional live demonstration, while the included demo video showcases the complete pipeline without infrastructure limitations.

---

# 🚀 Local Setup

Clone the repository:

```bash
git clone https://github.com/sonika2207/Geostream.git
```

Navigate into the backend:

```bash
cd Geostream/backend
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn main:app --reload
```

Visit:

```
http://127.0.0.1:8000
```

---

# 📁 Project Structure

```text
Geostream/

backend/
│
├── main.py
├── goes_fetcher.py
├── rife_interpolator.py
├── esrgan_enhancer.py
├── frame_analyzer.py
├── video_generator.py
├── requirements.txt
└── Dockerfile

frontend/
│
├── index.html
├── result.html
├── analysis-report.html
├── style.css
└── script.js
```

---

# 🛠️ Tech Stack

### Backend

- FastAPI
- Uvicorn

### AI

- PyTorch
- RIFE
- Real-ESRGAN

### Computer Vision

- OpenCV
- Pillow

### Video Processing

- FFmpeg

### Deployment

- Docker
- Render

---

# 📄 License

This project is intended for educational and portfolio purposes.

---

# 👩‍💻 Author

**Sonika**

GitHub: https://github.com/sonika2207
