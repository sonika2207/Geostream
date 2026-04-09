"""
GeoStream — Satellite Frame Analyzer
AI-powered disaster monitoring & environmental intelligence module.

Analyzes satellite video frames for:
  - Cloud movement patterns (direction, speed, density)
  - Atmospheric anomalies (cyclone formation, heavy accumulation, brightness shifts)
  - Risk classification (Low / Medium / High)
  - Event prediction (Rainfall, Storm, Cyclone, No significant event)
"""

import os
import cv2
import numpy as np
from datetime import datetime


# ── Core Analysis Functions ──

def analyze_frames(frame1: np.ndarray, frame2: np.ndarray) -> tuple[float, float, float]:
    """
    Compare two consecutive satellite frames.

    Returns:
        motion_score   – mean pixel-level difference (movement intensity)
        cloud_density  – average brightness as a proxy for cloud coverage (0–1)
        edge_density   – edge intensity (structural features / cyclone hints) (0–1)
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Absolute difference → motion detection
    diff = cv2.absdiff(gray1, gray2)
    motion_score = float(np.mean(diff))

    # Cloud density (brightness proxy)
    cloud_density = float(np.mean(gray2)) / 255.0

    # Edge detection → structural features / cyclone hint
    edges = cv2.Canny(gray2, 50, 150)
    edge_density = float(np.mean(edges)) / 255.0

    return motion_score, cloud_density, edge_density


def classify_risk(motion: float, cloud: float, edge: float) -> tuple[str, str]:
    """
    Classify risk level based on motion, cloud density, and edge density.

    Returns:
        (risk_level, reason)
    """
    if cloud > 0.7 and motion > 25:
        return "High", "Possible storm or cyclone"
    elif cloud > 0.7 and edge > 0.15:
        return "High", "Dense cloud with spiral/structural patterns detected"
    elif cloud > 0.5 and motion > 20:
        return "Medium-High", "Significant cloud movement with moderate density"
    elif cloud > 0.4:
        return "Medium", "Cloud accumulation"
    elif motion > 15:
        return "Medium", "Moderate atmospheric movement"
    else:
        return "Low", "Stable weather"


def estimate_cloud_direction(frame1: np.ndarray, frame2: np.ndarray) -> dict:
    """
    Estimate cloud movement direction and speed using optical flow.

    Returns:
        dict with keys: direction, speed, dx, dy
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Average flow vectors
    avg_dx = float(np.mean(flow[..., 0]))
    avg_dy = float(np.mean(flow[..., 1]))
    speed = float(np.sqrt(avg_dx ** 2 + avg_dy ** 2))

    # Determine direction based on dominant flow
    angle = np.degrees(np.arctan2(-avg_dy, avg_dx)) % 360

    directions = [
        (337.5, 360, "East"), (0, 22.5, "East"),
        (22.5, 67.5, "Northeast"),
        (67.5, 112.5, "North"),
        (112.5, 157.5, "Northwest"),
        (157.5, 202.5, "West"),
        (202.5, 247.5, "Southwest"),
        (247.5, 292.5, "South"),
        (292.5, 337.5, "Southeast"),
    ]

    direction = "Stationary"
    for low, high, name in directions:
        if low <= angle < high:
            direction = name
            break

    if speed < 0.3:
        direction = "Stationary"

    return {
        "direction": direction,
        "speed": round(speed, 3),
        "dx": round(avg_dx, 3),
        "dy": round(avg_dy, 3),
        "angle": round(angle, 1),
    }


def detect_spiral_pattern(frame: np.ndarray) -> dict:
    """
    Detect spiral/circular patterns that may indicate cyclone formation.
    Uses Hough circle detection on edge maps.

    Returns:
        dict with is_spiral, circle_count, confidence
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blurred, 30, 100)

    # Hough circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=50,
        param1=100,
        param2=40,
        minRadius=20,
        maxRadius=200
    )

    circle_count = 0 if circles is None else len(circles[0])

    # Contour-based rotation detection
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [c for c in contours if cv2.contourArea(c) > 500]

    # Check for nested/concentric structures (spiral proxy)
    nested_score = min(len(large_contours) / 20.0, 1.0)

    is_spiral = circle_count >= 2 or nested_score > 0.5
    confidence = min((circle_count * 15 + nested_score * 50), 100)

    return {
        "is_spiral": is_spiral,
        "circle_count": circle_count,
        "nested_score": round(nested_score, 3),
        "confidence": round(confidence, 1),
    }


def detect_brightness_anomaly(frames: list[np.ndarray]) -> dict:
    """
    Detect sudden changes in brightness across consecutive frames,
    which may indicate temperature shifts or convective events.

    Returns:
        dict with brightness_values, max_change, anomaly_detected
    """
    brightness_values = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_values.append(float(np.mean(gray)))

    changes = []
    for i in range(1, len(brightness_values)):
        changes.append(abs(brightness_values[i] - brightness_values[i - 1]))

    max_change = max(changes) if changes else 0
    avg_change = np.mean(changes) if changes else 0

    # A sudden brightness change > 15 units is notable
    anomaly_detected = max_change > 15

    return {
        "brightness_values": [round(b, 2) for b in brightness_values],
        "changes": [round(c, 2) for c in changes],
        "max_change": round(max_change, 2),
        "avg_change": round(avg_change, 2),
        "anomaly_detected": anomaly_detected,
    }


def predict_event(risk_level: str, motion: float, cloud: float, edge: float, spiral: dict, brightness: dict) -> dict:
    """
    Predict possible weather event based on all analysis metrics.

    Returns:
        dict with event, confidence, details
    """
    confidence = 0
    events = []

    if risk_level == "High" and spiral.get("is_spiral"):
        events.append("Cyclone")
        confidence = max(confidence, 75 + spiral.get("confidence", 0) * 0.25)
    
    if risk_level == "High" and motion > 30:
        events.append("Storm")
        confidence = max(confidence, 70)
    
    if cloud > 0.6 and motion > 10:
        events.append("Rainfall")
        confidence = max(confidence, 55 + cloud * 20)
    
    if cloud > 0.5 and not events:
        events.append("Rainfall")
        confidence = max(confidence, 40 + cloud * 15)

    if brightness.get("anomaly_detected"):
        if "Storm" not in events:
            events.append("Storm")
        confidence = min(confidence + 10, 100)

    if not events:
        events.append("No significant event")
        confidence = max(90 - motion * 2 - cloud * 30, 40)

    primary_event = events[0]
    confidence = min(round(confidence, 1), 100)

    return {
        "primary_event": primary_event,
        "all_events": events,
        "confidence": confidence,
    }


def generate_frame_differences(frames: list[np.ndarray]) -> list[dict]:
    """
    Compare consecutive frames and highlight major visual differences.

    Returns:
        List of dicts with pair indices, motion_score, description
    """
    differences = []
    for i in range(len(frames) - 1):
        motion, cloud, edge = analyze_frames(frames[i], frames[i + 1])
        
        desc_parts = []
        if motion > 25:
            desc_parts.append("significant cloud movement detected")
        elif motion > 10:
            desc_parts.append("moderate cloud shifting")
        else:
            desc_parts.append("minimal change")

        if cloud > 0.7:
            desc_parts.append("dense cloud cover")
        elif cloud > 0.4:
            desc_parts.append("partial cloud cover")

        if edge > 0.15:
            desc_parts.append("strong structural features visible")

        differences.append({
            "pair": f"Frame {i} → Frame {i + 1}",
            "motion_score": round(motion, 2),
            "cloud_density": round(cloud, 4),
            "edge_density": round(edge, 4),
            "description": "; ".join(desc_parts),
        })

    return differences


# ── Main Analysis Pipeline ──

def run_full_analysis(
    frame_paths: list[str],
    region: str = "Auto-detected satellite region",
    date: str = None,
    start_time: str = None,
    end_time: str = None,
) -> dict:
    """
    Run the complete satellite frame analysis pipeline.

    Args:
        frame_paths:  List of image file paths to analyze.
        region:       Region name/description.
        date:         Date of observation (YYYY-MM-DD).
        start_time:   Start time (HH:MM).
        end_time:     End time (HH:MM).

    Returns:
        Full structured analysis report as a dictionary.
    """
    if not frame_paths:
        return {"error": "No frames provided for analysis"}

    # Load all frames
    frames = []
    for path in frame_paths:
        img = cv2.imread(path)
        if img is not None:
            frames.append(img)

    if len(frames) < 2:
        return {"error": "Need at least 2 valid frames for analysis"}

    # ── Run all analyses ──

    # 1. Pairwise frame analysis (aggregate metrics)
    all_motion = []
    all_cloud = []
    all_edge = []
    for i in range(len(frames) - 1):
        m, c, e = analyze_frames(frames[i], frames[i + 1])
        all_motion.append(m)
        all_cloud.append(c)
        all_edge.append(e)

    avg_motion = float(np.mean(all_motion))
    avg_cloud = float(np.mean(all_cloud))
    avg_edge = float(np.mean(all_edge))
    max_motion = float(np.max(all_motion))

    # 2. Cloud movement direction (use first and last frame)
    cloud_movement = estimate_cloud_direction(frames[0], frames[-1])

    # 3. Spiral / cyclone detection (check the densest frame)
    densest_idx = int(np.argmax([np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]))
    spiral = detect_spiral_pattern(frames[densest_idx])

    # 4. Brightness anomaly
    brightness = detect_brightness_anomaly(frames)

    # 5. Risk classification
    risk_level, risk_reason = classify_risk(avg_motion, avg_cloud, avg_edge)

    # 6. Event prediction
    prediction = predict_event(risk_level, avg_motion, avg_cloud, avg_edge, spiral, brightness)

    # 7. Frame-by-frame differences
    frame_diffs = generate_frame_differences(frames)

    # ── Build description ──
    if date and start_time and end_time:
        time_desc = f"{date} from {start_time} to {end_time} UTC"
    else:
        time_desc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    description = (
        f"Satellite observation over {region} at {time_desc}. "
        f"Analyzed {len(frames)} frames. "
        f"Average cloud density: {avg_cloud:.1%}, "
        f"average motion intensity: {avg_motion:.1f}, "
        f"cloud movement: {cloud_movement['direction']} at relative speed {cloud_movement['speed']:.2f}."
    )

    observation_parts = []
    if avg_cloud > 0.6:
        observation_parts.append("Heavy cloud coverage detected across the observation period")
    elif avg_cloud > 0.3:
        observation_parts.append("Moderate cloud activity observed")
    else:
        observation_parts.append("Relatively clear skies with sparse cloud distribution")

    if avg_motion > 20:
        observation_parts.append("Rapid cloud movement suggests active atmospheric conditions")
    elif avg_motion > 10:
        observation_parts.append("Steady cloud migration patterns noted")

    if spiral["is_spiral"]:
        observation_parts.append("Circular/spiral cloud structures detected — potential cyclonic activity")

    if brightness["anomaly_detected"]:
        observation_parts.append(f"Brightness anomaly detected (max shift: {brightness['max_change']:.1f} units)")

    # Include risk description in observation
    observation_parts.append(f"Risk assessment: {risk_level} — {risk_reason}")

    observation = ". ".join(observation_parts) + "."

    # ── Assemble final report ──
    report = {
        "region": region,
        "time": time_desc,
        "frames_analyzed": len(frames),
        "description": description,
        "observation": observation,
        "cloud_movement": {
            "direction": cloud_movement["direction"],
            "speed": cloud_movement["speed"],
            "angle": cloud_movement["angle"],
            "description": f"Clouds moving {cloud_movement['direction']} at relative speed {cloud_movement['speed']:.3f}",
        },
        "metrics": {
            "avg_motion_score": round(avg_motion, 2),
            "max_motion_score": round(max_motion, 2),
            "avg_cloud_density": round(avg_cloud, 4),
            "avg_edge_density": round(avg_edge, 4),
        },
        "anomalies": {
            "spiral_detection": spiral,
            "brightness_anomaly": brightness,
        },
        "frame_differences": frame_diffs,
        "risk_level": risk_level,
        "risk_reason": risk_reason,
        "prediction": prediction,
        "confidence": prediction["confidence"],
        "summary": {
            "observation": observation,
            "risk_level": f"{risk_level} — {risk_reason}",
            "prediction": f"{prediction['primary_event']} ({', '.join(prediction['all_events'])})",
            "confidence_score": f"{prediction['confidence']}%",
        },
    }

    return report
