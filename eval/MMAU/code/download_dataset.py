"""Download MMAU test-mini audio from the official HF dataset (gamma-lab-umd/MMAU-test-mini).

Writes ./test-mini-audios/<id>.wav, matching `audio_id` in mmau-test-mini.json.
Run from eval/MMAU:  python code/download_dataset.py
"""

import json
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

parquet_path = hf_hub_download(
    repo_id="gamma-lab-umd/MMAU-test-mini",
    filename="test_mini.parquet",
    repo_type="dataset",
)

out_dir = "test-mini-audios"
os.makedirs(out_dir, exist_ok=True)
for row in tqdm(pq.read_table(parquet_path).to_pylist(), desc="Writing audio"):
    clip_id = json.loads(row["other_attributes"])["id"]
    with open(os.path.join(out_dir, f"{clip_id}.wav"), "wb") as f:
        f.write(row["context"]["bytes"])
print(f"Saved audio to {out_dir}/")
