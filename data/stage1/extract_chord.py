"""
Batch music temporal structure extraction using allin1 model.
Processes multiple datasets with the same interface as qwen_omni_caption.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import autochord
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChordExtractor:
    """
    Wrapper class for music structure extraction using allin1.
    """

    def post_process_chord(
        self, extracted_chord: List[Tuple[float, float, str]]
    ) -> str:
        """
        Post-process the extracted chord information:
        1. Exclude 'N' (No chord).
        2. Merge consecutive identical chords (unless separated by 'N').
        3. Format into a readable timeline string.
        """
        if not extracted_chord:
            return ""

        merged_chords = []
        current_start = None
        current_end = None
        current_label = None

        for start, end, label in extracted_chord:
            # 1. Handle 'N' (No chord) - we skip these and break the merging chain
            if label == "N":
                if current_label is not None:
                    # Save the tracking chord before the gap
                    merged_chords.append((current_start, current_end, current_label))
                    current_label = None
                continue

            # 2. Merge logic for identical consecutive chords
            if current_label == label:
                # Same chord continues, just update the end time
                current_end = end
            else:
                # If a different chord starts, save the previous one first
                if current_label is not None:
                    merged_chords.append((current_start, current_end, current_label))

                # Start tracking the new chord
                current_start = start
                current_end = end
                current_label = label

        # 3. Add the final remaining chord segment after the loop
        if current_label is not None:
            merged_chords.append((current_start, current_end, current_label))

        # 4. Format into timeline text: [start–ends] Chord
        lines = [f"[{s:.2f}–{e:.2f}s] {c}" for s, e, c in merged_chords]

        return "\n".join(lines)

    def process_batch(
        self, audio_path: Union[str, Path], item_id: str, save_path: str
    ) -> Dict[str, Any]:

        try:
            extracted_chord = autochord.recognize(audio_path, lab_fn="chords.lab")
            result_txt = self.post_process_chord(extracted_chord)

            save_result = os.path.join(save_path, f"{item_id}.txt")
            os.makedirs(os.path.dirname(save_result), exist_ok=True)

            with open(save_result, "w") as f:
                f.write(result_txt)
            result = {"chords": extracted_chord, "text": result_txt}
        except Exception as e:
            print(f"Error processing {item_id}: {e}")
            result = {"error": str(e)}

        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Chord extraction with autochord")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../examples/afthink_metadata_sample_music_filtered.json",
        help="Input JSON file containing audio metadata",
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
        default="../../workdir/chord",
        help="Path to save temporal structure files",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    args = parse_args()

    # Load metadata
    with open(args.input, "r") as f:
        input_data = json.load(f)

    os.makedirs(args.save_path, exist_ok=True)

    # Initialize extractor
    extractor = ChordExtractor()

    print(f"\n{'='*60}")
    print(f"Processing metadata: {args.input}")
    print(f"{'='*60}")

    # Prepare tasks - skip already processed files
    tasks = []
    skipped = 0

    # Handle both dict and list metadata formats
    if isinstance(input_data, dict):
        items = list(input_data.items())  # [(key, metadata), ...]
    else:
        items = [
            (idx, item) for idx, item in enumerate(input_data)
        ]  # [(idx, item), ...]

    for item_id, item_data in tqdm(
        items, desc="Checking existing results", dynamic_ncols=True
    ):
        # Get wav_path from item_data
        if isinstance(item_data, dict) and "wav_path" in item_data:
            audio_path = os.path.join(args.audio_root, item_data["wav_path"])
        else:
            print(f"Warning: No wav_path found for {item_id}")
            continue

        # Skip if already processed
        result_file = os.path.join(args.save_path, f"{item_id}.txt")
        if os.path.exists(result_file):
            skipped += 1
            continue

        tasks.append((item_id, audio_path))

    print(f"\nTotal files: {len(items)}")
    print(f"Already processed: {skipped}")
    print(f"To process: {len(tasks)}")

    if len(tasks) == 0:
        print("All files already processed!")
    else:
        # Process tasks one by one
        processed = 0
        errors = []

        for item_id, audio_path in tqdm(
            tasks, desc="Processing audio files", dynamic_ncols=True
        ):
            result = extractor.process_batch(audio_path, item_id, args.save_path)
            if result.get("error"):
                errors.append(f"{item_id}: {result['error']}")
                continue
            else:
                processed += 1

        # Print summary
        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"Total samples: {len(items)}")
        print(f"Processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Errors: {len(errors)}")

        if errors:
            error_file = os.path.join(args.save_path, "errors.txt")
            with open(error_file, "w") as f:
                f.write("\n".join(errors))
            print(f"Error details saved to {error_file}")

        elapsed = time.time() - start_time
        print(f"Total time: {elapsed/60:.2f} minutes")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
