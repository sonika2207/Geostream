import requests
from PIL import Image
from io import BytesIO
import os
import urllib3
import sys
from datetime import datetime, timedelta

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Fix Windows console encoding for emoji output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LATEST_TIME_URL = "https://himawari8-dl.nict.go.jp/himawari8/img/D531106/latest.json"
BASE_IMAGE_URL = "https://himawari8-dl.nict.go.jp/himawari8/img/D531106/1d/550"

FRAME_DIR = os.path.join(os.path.dirname(__file__), "data", "fetched_frames")

def fetch_frames(date_str=None, from_time=None, to_time=None, zoom=1) -> list[str]:
    os.makedirs(FRAME_DIR, exist_ok=True)
    
    # Clear previous frames
    for f in os.listdir(FRAME_DIR):
        os.remove(os.path.join(FRAME_DIR, f))
        
    saved_paths = []
    print("📡 Fetching latest 1-hour Himawari-8 satellite frames...")
    
    try:
        if date_str and from_time and to_time:
            # User provided a specific time range
            start_dt = datetime.strptime(f"{date_str} {from_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{date_str} {to_time}", "%Y-%m-%d %H:%M")
            
            # Ensure start_dt is before end_dt
            if start_dt > end_dt:
                start_dt, end_dt = end_dt, start_dt

            # Retrieve the latest available timestamp to avoid fetching future times
            try:
                latest_response = requests.get(LATEST_TIME_URL, verify=False, timeout=10)
                latest_response.raise_for_status()
                latest_dt = datetime.strptime(latest_response.json()["date"], "%Y-%m-%d %H:%M:%S")
                
                # If requested start time is in the future (commonly due to local time vs UTC input)
                # Fallback to the latest available hour
                if start_dt > latest_dt:
                    print(f"Requested time {start_dt} is in the future (max {latest_dt}). Falling back to last hour.")
                    start_dt = latest_dt - timedelta(minutes=50) # Last 6 frames
                    end_dt = latest_dt
                elif end_dt > latest_dt:
                    print(f"Capping end time to latest available UTC time: {latest_dt}")
                    end_dt = latest_dt
            except Exception as e:
                print(f"Warning: Could not fetch latest time for bounds check: {e}")
                
            # Align start_dt to nearest 10-minute interval (Himawari-8 schedule)
            minute = (start_dt.minute // 10) * 10
            current_dt = start_dt.replace(minute=minute, second=0, microsecond=0)
            
            frame_times = []
            while current_dt <= end_dt:
                frame_times.append(current_dt)
                current_dt += timedelta(minutes=10)
            
            # Cap maximum frames to 24 to prevent overwhelming the server/pipeline
            if len(frame_times) > 24:
                frame_times = frame_times[-24:]
                print(f"Capped frames to {len(frame_times)} recently requested frames.")
        else:
            # Get latest timestamp
            response = requests.get(LATEST_TIME_URL, verify=False, timeout=10)
            response.raise_for_status()
            latest_time = response.json()["date"]

            latest_dt = datetime.strptime(latest_time, "%Y-%m-%d %H:%M:%S")

            frame_times = []
            # Himawari updates every 10 minutes → 6 frames = 1 hour
            for i in range(6):
                frame_times.append(latest_dt - timedelta(minutes=10 * (5 - i)))

        frame_index = 0

        for frame_time in frame_times:
            date_path = frame_time.strftime("%Y/%m/%d/%H%M%S")
            image_url = f"{BASE_IMAGE_URL}/{date_path}_0_0.png"

            print(f"  Fetching frame {frame_index}: {image_url}")

            try:
                img_response = requests.get(image_url, verify=False, timeout=15)
                img_response.raise_for_status()

                image = Image.open(BytesIO(img_response.content))
                file_path = os.path.join(FRAME_DIR, f"frame_{frame_index:03d}.png")
                image.save(file_path)

                print("Saved:", file_path)
                saved_paths.append(file_path)
                frame_index += 1

            except Exception as e:
                print("Skipped frame:", e)
                
        print(f"🎉 Fetched {len(saved_paths)} frames")
        return saved_paths

    except Exception as e:
        print(f"Error fetching latest frames: {e}")
        return saved_paths

