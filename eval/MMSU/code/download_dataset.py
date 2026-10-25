"""Download MMSU audio from the official HF dataset (ddwang2000/MMSU).

Writes <output_dir>/audio/<id>.wav, matching `audio_path` in mmsu-test.json (the paper version of the
metadata, shipped in this repo). The upstream metadata has since been revised (answer options reshuffled);
see eval/README.md.

Run from eval/MMSU:  python code/download_dataset.py --output_dir download
"""

import argparse

from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="download")
args = parser.parse_args()

snapshot_download(
    repo_id="ddwang2000/MMSU",
    repo_type="dataset",
    allow_patterns="audio/*",
    local_dir=args.output_dir,
)
print(f"Saved audio to {args.output_dir}/audio/")
