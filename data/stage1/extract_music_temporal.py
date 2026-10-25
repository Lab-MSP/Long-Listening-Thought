"""
Batch music temporal structure extraction using allin1 model.
Processes multiple datasets with the same interface as qwen_omni_caption.
"""

import argparse
import json
import logging
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import allin1
from tqdm import tqdm

logger = logging.getLogger(__name__)


logging.getLogger("natten.functional").setLevel(logging.ERROR)
logging.getLogger("natten").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", message=".*natten.functional.natten2dav.*")
warnings.filterwarnings("ignore", message=".*na2d_av.*")

FUNCTIONAL_LABELS = {
    "intro",
    "outro",
    "verse",
    "chorus",
    "bridge",
    "instrumental",
    "inst",
    "solo",
    "break",
    "drop",
    "breakdown",
    "buildup",
    "pre-chorus",
    "post-chorus",
    "hook",
    "refrain",
    "interlude",
    "fade",
}

# Map allin1's Harmonix labels to standard labels
HARMONIX_LABEL_MAP = {
    "start": "intro",
    "end": "outro",
    "intro": "intro",
    "outro": "outro",
    "verse": "verse",
    "chorus": "chorus",
    "bridge": "bridge",
    "inst": "instrumental",
    "solo": "solo",
    "break": "break",
}


def normalize_section_label(label: str, section_index: int = 0) -> str:
    """
    Normalize section label to standard vocabulary.

    - If label is in whitelist -> use it directly
    - If label is A/B/C or unknown -> use section_A, section_B, etc.
    """
    if not label:
        return f"section_{chr(65 + section_index % 26)}"

    label_lower = label.strip().lower()

    if label_lower in HARMONIX_LABEL_MAP:
        return HARMONIX_LABEL_MAP[label_lower]

    if label_lower in FUNCTIONAL_LABELS:
        return label_lower

    if len(label_lower) == 1 and label_lower.isalpha():
        return f"section_{label_lower.upper()}"

    return label_lower


def _compute_beat_stats(beats: List[float]) -> Dict[str, Any]:
    """Compute statistics for beats."""
    if not beats:
        return {"count": 0}

    if len(beats) < 2:
        return {
            "count": len(beats),
            "first": round(beats[0], 2) if beats else None,
        }

    intervals = [beats[i + 1] - beats[i] for i in range(len(beats) - 1)]
    avg_interval = sum(intervals) / len(intervals)

    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
    std = math.sqrt(variance)
    rel_std = std / avg_interval if avg_interval > 0 else 0

    # Stability description
    if rel_std < 0.05:
        stability = "very_steady"
    elif rel_std < 0.15:
        stability = "steady"
    else:
        stability = "flexible"

    return {
        "count": len(beats),
        "first": round(beats[0], 2),
        "interval_mean": round(avg_interval, 3),
        "interval_std": round(std, 3),
        "stability": stability,
    }


def _collapse_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse adjacent sections with the same label into one.
    """
    if not sections:
        return []

    # Sort by start time
    sorted_sections = sorted(sections, key=lambda x: x.get("start", 0))

    collapsed = []
    for sec in sorted_sections:
        start = sec.get("start", 0)
        end = sec.get("end", 0)
        label = sec.get("label", "section")

        # Skip invalid
        if start >= end:
            continue

        # If same label as previous, extend previous
        if collapsed and collapsed[-1]["label"] == label:
            collapsed[-1]["end"] = end
        else:
            collapsed.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "label": label,
                }
            )

    return collapsed


class MusicStructureExtractor:
    """
    Wrapper class for music structure extraction using allin1.
    """

    def __init__(
        self,
        device: str = "cpu",
        model: str = "harmonix-all",
        demix_dir: str = "../../workdir/demix_cache_full",
        spec_dir: str = "../../workdir/spec_cache_full",
    ):
        """
        Initialize the extractor.
        Args:
            device: Device to run inference on ("cpu" or "cuda")
            model: Model name for allin1
        """
        self.device = device
        self.model = model
        self.demix_dir = demix_dir
        self.spec_dir = spec_dir

    def _merge_short_segments(
        self,
        segments: List[Dict[str, Any]],
        min_duration: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Merge short segments into adjacent segments instead of discarding.

        Strategy:
        - If short segment has same label as neighbor -> merge
        - Otherwise merge into the longer neighbor
        """
        if not segments:
            return []

        # First pass: identify segments to merge
        result = []
        i = 0
        while i < len(segments):
            seg = segments[i].copy()
            seg_duration = seg["end"] - seg["start"]

            if seg_duration >= min_duration:
                result.append(seg)
                i += 1
                continue

            # Short segment - merge with neighbor
            if not result:
                # No previous segment, try to merge with next
                if i + 1 < len(segments):
                    # Extend next segment's start
                    segments[i + 1]["start"] = seg["start"]
                else:
                    # Only segment, keep it despite being short
                    result.append(seg)
                i += 1
                continue

            # Merge with previous segment (extend its end)
            prev = result[-1]

            # If same label, just extend
            if prev["label"] == seg["label"]:
                prev["end"] = seg["end"]
            else:
                # Different label - merge into longer one
                prev_duration = prev["end"] - prev["start"]
                if i + 1 < len(segments):
                    next_seg = segments[i + 1]
                    next_duration = next_seg["end"] - next_seg["start"]
                    if next_duration > prev_duration:
                        # Merge into next
                        segments[i + 1]["start"] = seg["start"]
                    else:
                        # Merge into prev
                        prev["end"] = seg["end"]
                else:
                    # No next segment, merge into prev
                    prev["end"] = seg["end"]

            i += 1

        return result

    def _convert_segments(
        self,
        result: Any,
    ) -> List[Dict[str, Any]]:
        """
        Convert allin1 segment output to our standard schema.
        """
        if not hasattr(result, "segments") or result.segments is None:
            return []

        raw_sections = []
        for idx, seg in enumerate(result.segments):
            start = float(seg.start)
            end = float(seg.end)

            # Skip invalid segments
            if start >= end:
                continue

            label = normalize_section_label(seg.label, idx)

            raw_sections.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "label": label,
                }
            )

        # Sort by start time
        raw_sections.sort(key=lambda x: x["start"])

        # Merge short segments
        sections = self._merge_short_segments(raw_sections, min_duration=2.0)

        return sections

    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        item_ids: List[str],
        save_path: str = None,
    ) -> None:
        """
        Process a batch of audio files and extract temporal structure.
        Saves results immediately without returning.

        Args:
            audio_paths: List of audio file paths
            item_ids: List of corresponding item IDs
            save_path: Path to save temporal structure files
        """
        try:
            results = allin1.analyze(
                audio_paths,
                model=self.model,
                device=self.device,
                demix_dir=self.demix_dir,
                spec_dir=self.spec_dir,
                keep_byproducts=True,
            )
        except Exception as e:
            print(f"Error in batch analysis: {e}")
            return

        # allin1 returns results sorted by (resolved) path, not in input order, so match
        # them back to the items by path
        result_by_path = {Path(r.path): r for r in results}
        results = [
            result_by_path.get(Path(p).expanduser().resolve()) for p in audio_paths
        ]

        # Process each result and save immediately
        for item_id, analysis_result in zip(item_ids, results):
            result = {}
            if analysis_result is None:
                result = {"error": "No analysis result"}
            else:
                try:
                    # Extract tempo
                    tempo_bpm = None
                    if (
                        hasattr(analysis_result, "bpm")
                        and analysis_result.bpm is not None
                    ):
                        tempo_bpm = float(analysis_result.bpm)

                    # Extract beats
                    beats = []
                    if (
                        hasattr(analysis_result, "beats")
                        and analysis_result.beats is not None
                    ):
                        beats = [round(float(b), 3) for b in analysis_result.beats]

                    # Extract downbeats
                    downbeats = []
                    if (
                        hasattr(analysis_result, "downbeats")
                        and analysis_result.downbeats is not None
                    ):
                        downbeats = [
                            round(float(d), 3) for d in analysis_result.downbeats
                        ]

                    # Extract sections
                    sections = self._convert_segments(analysis_result)

                    # Compute beat stats
                    beat_stats = _compute_beat_stats(beats)
                    downbeat_stats = _compute_beat_stats(downbeats)
                    sections_collapsed = _collapse_sections(sections)

                    result = {
                        "tempo_bpm": round(tempo_bpm, 1) if tempo_bpm else None,
                        "beats_count": beat_stats.get("count", 0),
                        "downbeats_count": downbeat_stats.get("count", 0),
                        "beat_start_s": beat_stats.get("first"),
                        "downbeat_start_s": downbeat_stats.get("first"),
                        "beat_interval_mean_s": beat_stats.get("interval_mean"),
                        "beat_interval_std_s": beat_stats.get("interval_std"),
                        "beat_stability": beat_stats.get("stability"),
                        "sections": sections_collapsed,
                    }
                except Exception as e:
                    result = None

            # Save immediately
            if save_path and result is not None:
                result_file = os.path.join(save_path, f"{item_id}.json")
                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Music temporal structure extraction using allin1"
    )
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
        default="../../workdir/music_temporal",
        help="Path to save temporal structure files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )
    parser.add_argument(
        "--demix_dir",
        type=str,
        default="../../workdir/demix_cache_full",
        help="Demucs stem cache (use the --output dir of data/third_party/demucs_batch/demucs_batch.py)",
    )
    parser.add_argument(
        "--spec_dir",
        type=str,
        default="../../workdir/spec_cache_full",
        help="Spectrogram cache for allin1",
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
    extractor = MusicStructureExtractor(
        device=args.device, demix_dir=args.demix_dir, spec_dir=args.spec_dir
    )

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
        result_file = os.path.join(args.save_path, f"{item_id}.json")
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
        batch_size = len(tasks)  # Process all files at once

        for batch_start in tqdm(
            range(0, len(tasks), batch_size),
            desc="Processing batches",
            dynamic_ncols=True,
        ):
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]

            # Collect batch data
            batch_audio_paths = []
            batch_item_ids = []

            for item_id, audio_path in batch_tasks:
                if not os.path.exists(audio_path):
                    errors.append(f"{item_id}: Audio file not found: {audio_path}")
                    continue

                batch_audio_paths.append(audio_path)
                batch_item_ids.append(item_id)

            if not batch_audio_paths:
                continue

            # Process batch and save immediately
            try:
                extractor.process_batch(
                    batch_audio_paths, batch_item_ids, args.save_path
                )
                processed += len(batch_item_ids)
            except Exception as e:
                for item_id in batch_item_ids:
                    errors.append(f"{item_id}: {str(e)}")

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
