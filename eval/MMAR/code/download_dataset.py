import os
import tarfile
import shutil
from huggingface_hub import hf_hub_download

# --- Configuration ---
REPO_ID = "BoJack/MMAR"
FILENAME = "mmar-audio.tar.gz"
TARGET_DIR = "./"  # Where you want the 'audio' folder to appear

# 1. Download the tarball
print(f"Downloading {FILENAME} from {REPO_ID}...")
tar_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    repo_type="dataset",
    local_dir=TARGET_DIR
)
