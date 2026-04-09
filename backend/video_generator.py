"""
Video Generator
Stitches a list of frame images into a browser-compatible H.264 MP4 video.
"""

import os
import subprocess
import cv2


VIDEO_DIR = os.path.join(os.path.dirname(__file__), "data", "videos")

# Minimum video duration in seconds — frames will be repeated to fill this
MIN_DURATION_SECONDS = 10


def generate_video(frames: list[str], output: str | None = None, fps: int = 24) -> str:
    """
    Generate an MP4 video from a list of frame image paths.

    Args:
        frames: List of frame image file paths (in order).
        output: Output video file path. Defaults to data/videos/satellite_video.mp4.
        fps:    Frames per second (default 24 for smooth playback).

    Returns:
        Path to the generated video file.
    """
    if not frames:
        raise RuntimeError("No frames provided for video generation")

    os.makedirs(VIDEO_DIR, exist_ok=True)
    if output is None:
        output = os.path.join(VIDEO_DIR, "satellite_video.mp4")

    # Read first frame to get dimensions
    first = cv2.imread(frames[0])
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frames[0]}")
    h, w = first.shape[:2]

    # Limit max dimension to 3840 (4K-ish) for H.264 compatibility
    max_dim = 3840
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        w = int(w * scale)
        h = int(h * scale)

    # Ensure dimensions are always even (required by most H.264 encoders)
    w = w - (w % 2)
    h = h - (h % 2)

    # Calculate how many times each frame should repeat for minimum duration
    total_frames_needed = fps * MIN_DURATION_SECONDS  # e.g. 24 * 10 = 240
    if len(frames) < total_frames_needed:
        repeat_count = max(1, total_frames_needed // len(frames))
    else:
        repeat_count = 1

    # Write temporary avi file, then re-encode to H.264
    temp_output = output + ".tmp.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter")

    written = 0
    for path in frames:
        img = cv2.imread(path)
        if img is None:
            continue
        # Resize if dimensions don't match
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        # Write the frame multiple times for longer duration
        for _ in range(repeat_count):
            writer.write(img)
            written += 1

    writer.release()

    # Re-encode to H.264 with ffmpeg for browser compatibility
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", temp_output,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output,
            ],
            capture_output=True,
            check=True,
            timeout=300,
        )
        # Remove temp file on success
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"🎬 Video created: {output}  ({written} frames @ {fps} FPS, H.264)")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        # ffmpeg not available or failed — rename temp file as fallback
        print(f"⚠️  ffmpeg re-encode failed ({e}), falling back to MJPG AVI")
        if os.path.exists(temp_output):
            if os.path.exists(output):
                os.remove(output)
            os.rename(temp_output, output)
        print(f"🎬 Video created: {output}  ({written} frames @ {fps} FPS)")

    return output
