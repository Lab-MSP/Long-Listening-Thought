"""Build the AVQA GRPO training set (ms-swift format) used for the optional GRPO stage.

Takes the official AVQA train annotations (downloaded from the AVQA repository by default) and the
8k question ids used in the paper (avqa_grpo_8k_ids.json), rewrites the video-centric wording to audio,
and formats each question with the LLT system prompt and MCQ prompt.

Audio is not downloaded: AVQA videos come from VGGSound (YouTube), so obtain them yourself and extract
16 kHz mono FLAC as <audio_dir>/<video_name>.flac (as in the paper).

Usage:
    python build_avqa_grpo.py --audio_dir /path/to/vggsound_audio
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../.."))
from prompts import SYSTEM_PROMPT, mcq_prompt  # noqa: E402

# Official AVQA annotations (github.com/AlyssaYoung/AVQA), pinned to the commit used for the paper
AVQA_TRAIN_QA_URL = (
    "https://raw.githubusercontent.com/AlyssaYoung/AVQA/"
    "a5056c1d82849aa0981f50fa4c5a57bd16658c52/data/annotation/train_qa.json"
)


def format_choices(choices: List[str]) -> str:
    """Format choices with labels (A), (B), ... if not already present."""
    filtered = [c for c in choices if c and c.strip()]
    label_pattern = re.compile(r"^\s*(\([A-Z]\)|[A-Z]\.)\s")

    formatted = []
    for i, choice in enumerate(filtered):
        if label_pattern.match(choice):
            formatted.append(choice.strip())
        else:
            formatted.append(f"({chr(ord('A') + i)}) {choice.strip()}")
    return "\n".join(formatted)


def avqa_to_grpo(entry: dict, audio_dir: str) -> dict:
    question = entry["question_text"].replace("video", "audio").replace("frame", "segment")
    prompt = mcq_prompt(question, format_choices(entry["multi_choice"]))
    answer_text = f"({'ABCD'[entry['answer']]})"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<audio> {prompt}"},
        ],
        "audios": [os.path.join(os.path.abspath(audio_dir), entry["video_name"] + ".flac")],
        "solution": answer_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avqa_train_qa",
        default="../../workdir/avqa/train_qa.json",
        help="AVQA train_qa.json (downloaded from the official repository if missing)",
    )
    parser.add_argument("--audio_dir", required=True, help="Directory of <video_name>.flac")
    parser.add_argument(
        "--ids", default=os.path.join(SCRIPT_DIR, "avqa_grpo_8k_ids.json")
    )
    parser.add_argument("--output", default="../../workdir/avqa_8k.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.avqa_train_qa):
        os.makedirs(os.path.dirname(os.path.abspath(args.avqa_train_qa)), exist_ok=True)
        print(f"Downloading AVQA train_qa.json to {args.avqa_train_qa}")
        urllib.request.urlretrieve(AVQA_TRAIN_QA_URL, args.avqa_train_qa)

    by_id = {entry["id"]: entry for entry in json.load(open(args.avqa_train_qa))}
    ids = json.load(open(args.ids))
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} ids not found in {args.avqa_train_qa}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for i in ids:
            row = avqa_to_grpo(by_id[i], args.audio_dir)
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(ids)} examples to {args.output}")


if __name__ == "__main__":
    main()
