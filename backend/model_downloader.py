"""
Model downloader utilities.
Provides a cross-platform downloader for the ECCV2022-RIFE pretrained weights
and helpers to ensure the weights exist before model loading.
"""
import os
import sys
import time
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

LOG = logging.getLogger("model_downloader")


DEFAULT_RIFE_GDRIVE_ID = "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg"  # Practical-RIFE v4.25 (mirror from README)


def _download_stream(url: str, dest: Path, max_retries: int = 3, timeout: int = 30):
    """Download a file with streaming and a progress bar.
    Retries on transient network errors.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            return
        except Exception as e:
            last_exc = e
            LOG.warning("Download attempt %d/%d failed: %s", attempt, max_retries, e)
            time.sleep(2 ** attempt)
    raise last_exc


def _download_from_gdrive(file_id: str, dest: Path):
    """Download a file from Google Drive handling the confirmation for large files.
    Returns path to downloaded file.
    """
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    # If we get a confirm token, the cookie-driven confirmation is needed
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v
            break
    else:
        confirm_token = None

    if confirm_token:
        params = {"id": file_id, "confirm": confirm_token}
        response = session.get(URL, params=params, stream=True)

    # Save to dest
    _download_stream(response.url, dest)


def _extract_archive(archive_path: Path, target_dir: Path):
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(target_dir)
                LOG.info("Extracted zip to %s", target_dir)
                return
        # fallback: try tar
        import tarfile

        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as t:
                t.extractall(target_dir)
                LOG.info("Extracted tar to %s", target_dir)
                return
    except Exception as e:
        LOG.exception("Failed to extract archive: %s", e)
        raise
    raise RuntimeError("Unsupported archive format for %s" % archive_path)


def ensure_rife_weights(weights_dir: str | Path, gdrive_id: Optional[str] = None, force: bool = False) -> Path:
    """Ensure RIFE weights exist under `weights_dir`.

    If missing, downloads the archive from Google Drive (default id) and extracts
    the contents into the provided directory. Returns the final weights directory path.
    """
    weights_path = Path(weights_dir)
    weights_path = weights_path.resolve()
    sentinel = weights_path / ".rife_downloaded"

    # Quick check: any .pkl/.pth/.pt files present
    def _has_weights(p: Path) -> bool:
        if not p.exists():
            return False
        for ext in ("*.pkl", "*.pth", "*.pt"):
            if any(p.rglob(ext)):
                return True
        # also check for model python files
        if any(p.rglob("*.py")):
            return True
        return False

    if not force and sentinel.exists() and _has_weights(weights_path):
        LOG.info("RIFE weights already present at %s (sentinel found)", weights_path)
        return weights_path

    if _has_weights(weights_path) and not force:
        LOG.info("RIFE weights detected at %s", weights_path)
        # create sentinel
        weights_path.mkdir(parents=True, exist_ok=True)
        sentinel.write_text("ok")
        return weights_path

    # Not present — download
    LOG.info("RIFE weights missing at %s — starting download...", weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="rife_download_"))
    try:
        archive_path = tmp_dir / "rife_weights.zip"
        file_id = gdrive_id or DEFAULT_RIFE_GDRIVE_ID

        try:
            _download_from_gdrive(file_id, archive_path)
        except Exception:
            LOG.exception("Failed to download from Google Drive id=%s", file_id)
            raise

        # Extract
        _extract_archive(archive_path, tmp_dir)

        # Many public archives contain a top-level folder; find candidate train_log
        candidate = None
        for p in tmp_dir.rglob("train_log"):
            if p.is_dir():
                candidate = p
                break

        # If archive already contains train_log, move contents
        if candidate:
            LOG.info("Found train_log inside archive at %s", candidate)
            if weights_path.exists():
                shutil.rmtree(weights_path)
            shutil.move(str(candidate), str(weights_path))
        else:
            # Otherwise, move all files from tmp_dir into weights_path
            weights_path.mkdir(parents=True, exist_ok=True)
            for item in tmp_dir.iterdir():
                if item.name == archive_path.name:
                    continue
                target = weights_path / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))

        sentinel.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        LOG.info("RIFE weights downloaded and placed at %s", weights_path)
        return weights_path

    finally:
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
