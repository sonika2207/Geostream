FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better Docker cache)
COPY backend/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project (backend + frontend)
COPY . .

# Switch to backend directory
WORKDIR /app/backend

# Environment variables
ENV PYTHONUNBUFFERED=1

# Render provides PORT automatically
EXPOSE 8000

# Start FastAPI with extended timeouts to handle long RIFE interpolation jobs
# --timeout-keep-alive: keep HTTP connection alive up to 300s
# --workers 1: single worker to minimize memory usage on constrained hosts
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 300"]