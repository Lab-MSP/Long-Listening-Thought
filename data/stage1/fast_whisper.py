"""
Optimized Whisper transcription with timestamps using faster-whisper.
Maintains the same interface and output format as the HuggingFace version.
"""

import argparse
import json
import logging
import os
import time

import torch

# Required: pip install faster-whisper
from faster_whisper import BatchedInferencePipeline, WhisperModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


def whisper_json_to_timeline_text(whisper_result, ndigits=2):
    """
    Convert Whisper-style JSON with `chunks` into timeline text.
    Output format: [0.0–3.0s] "Okay."
    """
    lines = []

    for chunk in whisper_result.get("chunks", []):
        ts = chunk.get("timestamp")
        text = chunk.get("text", "").strip()

        if not ts or len(ts) != 2 or not text:
            continue

        start, end = ts
        start = round(float(start), ndigits)
        end = round(float(end), ndigits)

        lines.append(f'[{start}–{end}s] "{text}"')

    return "\n".join(lines)


class FasterWhisperRunner:
    def __init__(
        self,
        model_name="large-v3",
        device="cuda",
        torch_dtype=None,
        batch_size=128,
    ):
        # faster-whisper uses 'cuda' instead of 'cuda:0'
        self.device = "cuda" if "cuda" in device else "cpu"
        self.batch_size = batch_size

        # Map torch_dtype to compute_type
        if torch_dtype == torch.bfloat16:
            compute_type = "bfloat16"
        elif torch_dtype == torch.float16:
            compute_type = "float16"
        else:
            compute_type = "float16" if torch.cuda.is_available() else "float32"

        print(
            f"Loading Faster-Whisper model: {model_name} on {self.device} ({compute_type})"
        )

        # Load the base model
        self.model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=compute_type,
        )

        # Wrap for batched processing (similar to chunk_length_s=30 in HF)
        self.batched_model = BatchedInferencePipeline(self.model)

        print("Faster-Whisper pipeline initialized.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimized Faster-Whisper Transcription"
    )
    # In faster-whisper, "large-v3" is a standard name.
    # If using HF names, it will download and convert them automatically.
    parser.add_argument("--model_name", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../examples/afthink_metadata_sample.json",
        help="Input JSON file containing audio file paths.",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default="",
        help="Root directory of the AF-Think audio; metadata wav_path is relative to it",
    )
    parser.add_argument(
        "--save_path",
        "-o",
        type=str,
        default="../../workdir/whisper_fast",
        help="Path to save transcription files",
    )
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--id_key", type=str, default="id")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    args = parse_args()

    # Load input data
    with open(args.input, "r") as f:
        input_data = json.load(f)

    os.makedirs(args.save_path, exist_ok=True)

    # Prepare tasks - skip already processed files
    tasks = []
    skipped = 0

    for key, data in tqdm(
        list(input_data.items()), desc="Checking existing results", dynamic_ncols=True
    ):
        if "wav_path" in data:
            # Check if result file already exists
            result_file = os.path.join(args.save_path, f"{key}.txt")

            if os.path.exists(result_file):
                skipped += 1
                continue
            tasks.append((key, os.path.join(args.audio_root, data["wav_path"]), data["duration"]))
        else:
            print(f"Warning: No wav_path found for {key}")

    print(f"Total files: {len(input_data)}")
    print(f"Already processed: {skipped}")
    print(f"To process: {len(tasks)}")

    if len(tasks) == 0:
        print("All files already processed!")
    else:
        # Initialize Runner
        runner = FasterWhisperRunner(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
        )

        # Process tasks
        processed = 0
        errors = 0
        pbar = tqdm(
            tasks, desc="Transcribing (Faster)", unit="file", dynamic_ncols=True
        )

        for item_id, audio_path, duration in pbar:
            try:
                full_audio_path = audio_path

                if not os.path.exists(full_audio_path):
                    pbar.write(f"Audio not found: {full_audio_path}")
                    errors += 1
                    continue

                # Transcribe with internal batching
                segments, info = runner.batched_model.transcribe(
                    full_audio_path,
                    batch_size=args.batch_size,
                    language=args.language,
                    without_timestamps=False,
                    vad_filter=True,
                )

                # Format the output to match the original helper function's expectation
                chunks = []
                for segment in segments:
                    if segment.start >= duration:
                        continue
                    end_time = min(segment.end, duration)
                    chunks.append(
                        {"timestamp": (segment.start, end_time), "text": segment.text}
                    )

                formatted_result = {"chunks": chunks}
                result_txt = whisper_json_to_timeline_text(formatted_result)

                # Save results
                result_file = os.path.join(args.save_path, f"{item_id}.txt")
                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(result_txt)

                processed += 1
            except Exception as e:
                pbar.write(f"Error processing {item_id}: {e}")
                errors += 1

        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Processed: {len(tasks)}")
        print(f"Successful: {processed}")
        print(f"Errors: {errors}")
        print(f"Skipped (already done): {skipped}")
        print(f"Results saved to: {args.save_path}")
        print(f"Total time: {elapsed/60:.2f} minutes")
        print(f"{'='*50}")
