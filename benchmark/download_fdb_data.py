"""
Download and extract the official Full-Duplex-Bench v3 released dataset.

Downloads from the official Google Drive link if not already present:
https://drive.google.com/file/d/1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz/view?usp=sharing
"""

import time
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
EXTRACT_DIR = DATA_DIR / "fdb_v3_data_released"
ZIP_PATH = DATA_DIR / "fdb_v3_data_released.zip"
GDRIVE_FILE_ID = "1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz"
DOWNLOAD_URL = f"https://drive.usercontent.google.com/download?id={GDRIVE_FILE_ID}&export=download&confirm=t"


def is_already_extracted() -> bool:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        # Quick check for at least one input.wav
        wav_files = list(EXTRACT_DIR.glob("*/input.wav"))
        if len(wav_files) > 0:
            print(f"Dataset already present at {EXTRACT_DIR} with {len(wav_files)} examples.")
            return True
    return False


def download_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if is_already_extracted():
        return EXTRACT_DIR

    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 700 * 1024 * 1024:
        print(f"Zip archive already exists at {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.1f} MB). Skipping download.")
    else:
        print(f"Downloading official FDB-v3 dataset from Google Drive ({DOWNLOAD_URL})...")
        req = urllib.request.Request(DOWNLOAD_URL, headers={"User-Agent": "Mozilla/5.0"})
        start_time = time.time()
        with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024 * 2  # 2MB chunks
            last_report = 0.0

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report > 5.0:
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    pct = (downloaded / total_size * 100) if total_size else 0
                    speed = mb_down / (now - start_time)
                    print(f"  Downloaded: {mb_down:.1f}/{mb_total:.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s")
                    last_report = now

        duration = time.time() - start_time
        print(f"Download complete: {ZIP_PATH.stat().st_size / (1024*1024):.1f} MB in {duration:.1f}s.")

    print(f"Extracting {ZIP_PATH} to {DATA_DIR}...")
    start_unzip = time.time()
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Extraction complete in {time.time() - start_unzip:.1f}s.")

    wav_count = len(list(EXTRACT_DIR.glob("*/input.wav")))
    print(f"Verified dataset: {wav_count} input.wav files ready in {EXTRACT_DIR}.")
    return EXTRACT_DIR


if __name__ == "__main__":
    download_dataset()
