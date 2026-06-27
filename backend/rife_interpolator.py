import os
import sys
import cv2
import torch
import numpy as np

# Limit PyTorch CPU thread allocation to save memory on server hosting platforms (e.g. Render)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Add RIFE model directory to path
RIFE_DIR = os.path.join(os.path.dirname(__file__), "rife", "ECCV2022-RIFE")
sys.path.insert(0, RIFE_DIR)

# Configurable paths via environment variables
WEIGHTS_DIR = os.environ.get("RIFE_WEIGHTS_DIR") or os.path.join(RIFE_DIR, "RIFE_trained_v6", "train_log")
# Allow overriding Google Drive id for the model archive
RIFE_GDRIVE_ID = os.environ.get("RIFE_WEIGHTS_GDRIVE_ID")

OUTPUT_DIR = os.environ.get("RIFE_OUTPUT_DIR") or os.path.join(os.path.dirname(__file__), "data", "interpolated_frames")

# Logging
import logging
from model_downloader import ensure_rife_weights
LOG = logging.getLogger("rife_interpolator")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None


def unload_model():
    """Unload model from memory and trigger garbage collection to free RAM."""
    global _model
    if _model is not None:
        _model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOG.info("RIFE model unloaded from memory.")


def _load_model():
    """Load RIFE model (cached singleton). Ensures weights are present first."""
    global _model
    if _model is not None:
        return _model

    # Ensure the weights exist (download if necessary)
    try:
        ensured = ensure_rife_weights(WEIGHTS_DIR, gdrive_id=RIFE_GDRIVE_ID)
        LOG.info("Using RIFE weights at %s", ensured)
    except Exception as e:
        LOG.exception("Failed to ensure RIFE weights: %s", e)
        raise

    # Dynamically import model based on what is available in the weights directory
    from pathlib import Path
    try:
        parent_dir = str(Path(ensured).parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from train_log.RIFE_HDv3 import Model
        LOG.info("Loaded v3.x HD model from train_log.")
    except Exception as e:
        LOG.warning("Failed to load v3.x HD model: %s. Falling back to ECCV2022-RIFE model.", e)
        from model.RIFE import Model
        LOG.info("Loaded ECCV2022-RIFE model.")

    _model = Model()
    _model.load_model(str(ensured), -1)
    _model.eval()
    _model.device()
    LOG.info("RIFE model loaded from %s (device: %s)", ensured, device)
    return _model


def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 numpy array to NCHW float32 tensor on device."""
    t = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(device)


def _tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """Convert NCHW float32 tensor to HWC uint8 numpy array."""
    img = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return img


def _pad_to_multiple(img: np.ndarray, multiple: int = 32) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad image so H and W are multiples of `multiple`. Returns padded image and original (H,W)."""
    h, w, c = img.shape
    new_h = ((h - 1) // multiple + 1) * multiple
    new_w = ((w - 1) // multiple + 1) * multiple
    padded = np.zeros((new_h, new_w, c), dtype=img.dtype)
    padded[:h, :w, :] = img
    return padded, (h, w)


def interpolate_frames(input_frames: list[str], exp: int = 1) -> list[str]:
    """
    Interpolate between consecutive frames using RIFE.

    Args:
        input_frames: List of paths to input frame images.
        exp:          Interpolation exponent. Generates 2^exp - 1 intermediate frames
                      per pair (exp=1 → 1 mid-frame, exp=2 → 3 mid-frames, etc.)

    Returns:
        Sorted list of all output frame paths (original + interpolated).
    """
    if len(input_frames) < 2:
        print("Warning: Need at least 2 frames for interpolation")
        return input_frames

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Clear previous output
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    model = _load_model()
    all_output = []
    idx = 0

    print(f"Interpolating {len(input_frames)} frames (exp={exp})...")

    for i in range(len(input_frames)):
        # Always save the original frame
        img = cv2.imread(input_frames[i])
        out_path = os.path.join(OUTPUT_DIR, f"frame_{idx:06d}.png")
        cv2.imwrite(out_path, img)
        all_output.append(out_path)
        idx += 1

        # Interpolate between this frame and next
        if i < len(input_frames) - 1:
            img0 = cv2.imread(input_frames[i])
            img1 = cv2.imread(input_frames[i + 1])

            # Resize to common dimensions
            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]
            target_h = min(h0, h1)
            target_w = min(w0, w1)
            if (h0, w0) != (target_h, target_w):
                img0 = cv2.resize(img0, (target_w, target_h))
            if (h1, w1) != (target_h, target_w):
                img1 = cv2.resize(img1, (target_w, target_h))

            # Convert BGR → RGB
            img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

            # Pad to multiple of 32
            img0_padded, (oh, ow) = _pad_to_multiple(img0_rgb, 32)
            img1_padded, _ = _pad_to_multiple(img1_rgb, 32)

            t0 = _img_to_tensor(img0_padded)
            t1 = _img_to_tensor(img1_padded)

            # Generate intermediate frames
            n_mid = 2 ** exp - 1
            for m in range(1, n_mid + 1):
                timestep = m / (n_mid + 1)
                with torch.no_grad():
                    mid = model.inference(t0, t1, timestep=timestep)

                mid_img = _tensor_to_img(mid)[:oh, :ow, :]  # crop padding
                mid_bgr = cv2.cvtColor(mid_img, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(OUTPUT_DIR, f"frame_{idx:06d}.png")
                cv2.imwrite(out_path, mid_bgr)
                all_output.append(out_path)
                idx += 1

    print(f"Interpolation complete: {len(all_output)} total frames")
    return all_output
