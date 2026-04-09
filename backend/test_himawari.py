import requests
from PIL import Image
from io import BytesIO
import os
import urllib3
from datetime import datetime, timedelta
import concurrent.futures

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SAVE_DIR = "satellite_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

LATEST_TIME_URL = "https://himawari8-dl.nict.go.jp/himawari8/img/D531106/latest.json"
BASE_IMAGE_URL = "https://himawari8-dl.nict.go.jp/himawari8/img/D531106/1d/550"

def download_frame(frame_index, image_url):
    print(f"Fetching frame {frame_index}: {image_url}")
    try:
        # Added a timeout to avoid hanging requests
        img_response = requests.get(image_url, verify=False, timeout=15)
        img_response.raise_for_status()

        image = Image.open(BytesIO(img_response.content))
        file_path = os.path.join(SAVE_DIR, f"frame_{frame_index:03d}.png")
        image.save(file_path)

        print("Saved:", file_path)
        return True
    except Exception as e:
        print(f"Skipped frame {frame_index}:", e)
        return False

def fetch_last_hour_frames():
    # Get latest timestamp
    print("Fetching latest timestamp...")
    response = requests.get(LATEST_TIME_URL, verify=False, timeout=10)
    response.raise_for_status()
    latest_time = response.json()["date"]

    latest_dt = datetime.strptime(latest_time, "%Y-%m-%d %H:%M:%S")
    print(f"Latest timestamp: {latest_dt}")

    tasks = []

    # Himawari updates every 10 minutes → 6 frames = 1 hour
    for i in range(6):
        frame_time = latest_dt - timedelta(minutes=10 * (5 - i))
        date_path = frame_time.strftime("%Y/%m/%d/%H%M%S")
        image_url = f"{BASE_IMAGE_URL}/{date_path}_0_0.png"
        
        # We store the frame index (i) and the url
        tasks.append((i, image_url))

    print("Starting parallel downloads...")
    # Use ThreadPoolExecutor to download all 6 frames at the same time
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(download_frame, index, url) for index, url in tasks]
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)
        
    print("All downloads completed.")

if __name__ == "__main__":
    import time
    start_time = time.time()
    fetch_last_hour_frames()
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")
