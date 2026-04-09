"""
Real-ESRGAN Frame Enhancement Module
Uses the lightweight realesr-animevideov3 model for fast video-frame enhancement.
"""

import os
import sys
import cv2
import numpy as np
import torch

# ── Compatibility shim ──
# basicsr references torchvision.transforms.functional_tensor which was
# removed in torchvision >= 0.18.  Alias it so the import succeeds.
import torchvision.transforms.functional as _F
sys.modules.setdefault("torchvision.transforms.functional_tensor", _F)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "enhanced_frames")

_upsampler = None


def _load_model(scale: int = 2):
    """Load the lightweight Real-ESRGAN video model (cached singleton)."""
    global _upsampler
    if _upsampler is not None:
        return _upsampler

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    # ── Use the COMPACT video model (SRVGGNetCompact) ──
    # ~10x faster than RealESRGAN_x4plus on CPU
    try:
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_conv=16, upscale=4, act_type='prelu'
        )
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        print("📦 Using fast compact model (realesr-animevideov3)")
    except ImportError:
        # Fallback to standard RRDB model if SRVGGNetCompact not available
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=6, num_grow_ch=32, scale=4
        )
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        print("📦 Using RealESRNet fallback model")

    _upsampler = RealESRGANer(
        scale=4,
        model_path=model_url,
        model=model,
        tile=0,         # No tiling for small 550x550 images -> HUGE speedup
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False, # Faster on GPU
        gpu_id=0 if torch.cuda.is_available() else None
    )

    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"✅ Real-ESRGAN loaded (device: {device_name})")
    return _upsampler


def enhance_frames(input_frames: list[str], scale: int = 2, target_size: tuple[int, int] | None = None) -> list[str]:
    """
    Enhance frames using Real-ESRGAN (fast compact model).

    Args:
        input_frames:  List of input frame paths.
        scale:         Upscale factor (default 2 for speed).
        target_size:   Optional (width, height) to resize output.

    Returns:
        List of enhanced frame paths.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Clear previous output
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    upsampler = _load_model(scale)
    enhanced_paths = []
    total = len(input_frames)

    print(f"🔬 Enhancing {total} frames with Real-ESRGAN (scale={scale})...")

    for i, frame_path in enumerate(input_frames):
        img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  ⚠️  Could not read {frame_path}, skipping")
            continue

        # Downscale very large inputs for speed, but keep good detail
        h, w = img.shape[:2]
        max_dim = 2048  # Keep inputs at a decent size for quality output
        if max(h, w) > max_dim:
            ratio = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

        try:
            output, _ = upsampler.enhance(img, outscale=scale)
        except Exception as e:
            print(f"  ⚠️  Error enhancing frame {i}: {e}, copying original")
            output = img

        # Optionally resize to target
        if target_size is not None:
            output = cv2.resize(output, target_size, interpolation=cv2.INTER_LANCZOS4)

        out_path = os.path.join(OUTPUT_DIR, f"enhanced_{i:06d}.png")
        cv2.imwrite(out_path, output)
        enhanced_paths.append(out_path)

        if (i + 1) % 5 == 0 or i == total - 1:
            print(f"  ✅ Enhanced {i + 1}/{total}")

    print(f"🎉 Enhancement complete: {len(enhanced_paths)} frames")
    return enhanced_paths
