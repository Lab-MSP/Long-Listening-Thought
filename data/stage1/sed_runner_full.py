"""
Pretrained Sound Event Detection (SED) Runner using BEATs.
Batch processing of multiple audio files with 10-second segment inference.
"""

import argparse
import json
import logging
import os
import sys
import time

import librosa
import torch
from tqdm import tqdm

# PretrainedSED (git submodule) is imported from its directory
SED_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../third_party/PretrainedSED"
)
sys.path.append(SED_ROOT)

import models.prediction_wrapper as prediction_wrapper  # noqa: E402
from data_util import audioset_classes  # noqa: E402
from helpers.decode import batched_decode_preds  # noqa: E402
from helpers.encode import ManyHotEncoder  # noqa: E402
from models.beats.BEATs_wrapper import BEATsWrapper  # noqa: E402
from models.prediction_wrapper import PredictionsWrapper  # noqa: E402

# PretrainedSED resolves checkpoints relative to the cwd ("resources"); point it at the
# submodule's resources dir so the script works from any directory.
prediction_wrapper.RESOURCES_FOLDER = os.path.join(SED_ROOT, "resources")
os.makedirs(prediction_wrapper.RESOURCES_FOLDER, exist_ok=True)

logger = logging.getLogger(__name__)


def format_detections_to_text(decoded_predictions, thresholds, ndigits=2):
    """
    Format decoded predictions with multiple thresholds into a single text output.

    Args:
        decoded_predictions: Dictionary mapping threshold -> DataFrame with detections
        thresholds: List of threshold values
        ndigits: Number of decimal places to round to

    Returns:
        Formatted text string combining all threshold results
    """
    output_lines = []

    for th in thresholds:
        if th not in decoded_predictions:
            continue

        df = decoded_predictions[th].sort_values(by="onset")
        output_lines.append(f"Sound event detection decoded with Threshold: {th}")

        if len(df) == 0:
            output_lines.append("  No events detected")
        else:
            for _, row in df.iterrows():
                onset = round(float(row["onset"]), ndigits)
                offset = round(float(row["offset"]), ndigits)
                label = str(row["event_label"])
                output_lines.append(f"  [{onset}–{offset}s] {label}")

        output_lines.append("")  # Empty line between thresholds

    return "\n".join(output_lines)


class PretrainedSEDRunner:
    def __init__(
        self,
        device="cuda",
        batch_size=8,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.sample_rate = 16000  # BEATs standard sample rate
        self.segment_duration = 10  # Models trained on 10-second segments
        self.segment_samples = self.segment_duration * self.sample_rate

        print(f"Loading BEATs model on {self.device}")

        # Initialize BEATs model
        beats = BEATsWrapper()
        self.model = PredictionsWrapper(beats, checkpoint="BEATs_strong_1")
        self.model.eval()
        self.model.to(self.device)

        # Load AudioSet labels
        self.labels = audioset_classes.as_strong_train_classes
        print(f"BEATs pipeline initialized with {len(self.labels)} classes.")

    def load_data(self, data_file):
        with open(data_file, "r") as f:
            self.data = json.load(f)

    def preprocess_audio(self, audio_path):
        """Load audio and resample to 16kHz mono"""
        try:
            wav, sr = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
            return torch.from_numpy(wav).float()
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            return None

    def _segment_audio(self, wav, audio_id, audio_len):
        """
        Split audio into 10-second segments.

        Returns:
            List of tuples: (audio_id, segment_idx, waveform, segment_audio_len)
        """
        waveform_len = wav.shape[0]
        num_chunks = waveform_len // self.segment_samples + (
            waveform_len % self.segment_samples != 0
        )

        segments = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.segment_samples
            end_idx = min((chunk_idx + 1) * self.segment_samples, waveform_len)
            waveform_chunk = wav[start_idx:end_idx]

            # Pad the last chunk if needed
            if waveform_chunk.shape[0] < self.segment_samples:
                pad_size = self.segment_samples - waveform_chunk.shape[0]
                waveform_chunk = torch.nn.functional.pad(waveform_chunk, (0, pad_size))

            # Calculate segment audio length for encoder
            segment_samples = end_idx - start_idx
            segment_audio_len = segment_samples / self.sample_rate

            segments.append((audio_id, chunk_idx, waveform_chunk, segment_audio_len))

        return segments

    def run_inference(
        self,
        tasks,
        save_path,
        thresholds=(0.1, 0.2, 0.5),
        median_filter=7,
    ):
        """
        Run inference on a list of (audio_id, audio_path) tasks.

        Args:
            tasks: List of (audio_id, audio_path) tuples
            save_path: Path to save results
            thresholds: Detection thresholds
            median_filter: Median filter size
        """
        os.makedirs(save_path, exist_ok=True)

        # Process in batches of files
        processed = 0
        errors = 0

        for batch_start_idx in tqdm(
            range(0, len(tasks), self.batch_size), desc="SED Batch Processing"
        ):
            batch_end_idx = min(batch_start_idx + self.batch_size, len(tasks))
            batch_files = tasks[batch_start_idx:batch_end_idx]

            # Load all files in batch and segment them
            all_segments = (
                []
            )  # List of (audio_id, segment_idx, waveform, segment_audio_len)
            audio_lens = {}  # Map audio_id -> total audio length in seconds

            for audio_id, audio_path in batch_files:
                wav = self.preprocess_audio(audio_path)
                if wav is None:
                    errors += 1
                    continue

                audio_len = wav.shape[0] / self.sample_rate
                audio_lens[audio_id] = audio_len
                segments = self._segment_audio(wav, audio_id, audio_len)
                all_segments.extend(segments)

            if not all_segments:
                continue

            # Prepare batch: stack all segment waveforms
            segment_waveforms = torch.stack([seg[2] for seg in all_segments])
            segment_waveforms = segment_waveforms.to(self.device)

            # Run inference on all segments at once
            with torch.no_grad():
                mel = self.model.mel_forward(segment_waveforms)
                y_strong, _ = self.model(mel)

            # Apply sigmoid to get probabilities
            y_strong = torch.sigmoid(y_strong)

            # Group predictions back by audio file
            predictions_by_audio = {}
            for pred_idx, (audio_id, segment_idx, _, segment_audio_len) in enumerate(
                all_segments
            ):
                if audio_id not in predictions_by_audio:
                    predictions_by_audio[audio_id] = []
                predictions_by_audio[audio_id].append(y_strong[pred_idx : pred_idx + 1])

            # Process each audio file: concatenate its segments and decode
            for audio_id, segment_predictions in predictions_by_audio.items():
                try:
                    # Concatenate all segments for this audio
                    y_strong_audio = torch.cat(segment_predictions, dim=2)
                    audio_len = audio_lens[audio_id]

                    # Encode and decode predictions with multiple thresholds
                    encoder = ManyHotEncoder(self.labels, audio_len=audio_len)

                    (scores_unprocessed, scores_postprocessed, decoded_predictions) = (
                        batched_decode_preds(
                            y_strong_audio.float(),
                            [audio_id],
                            encoder,
                            median_filter=median_filter,
                            thresholds=list(thresholds),
                        )
                    )

                    # Format results using the format function
                    result_txt = format_detections_to_text(
                        decoded_predictions, thresholds
                    )

                    # Save results
                    result_file = os.path.join(save_path, f"{audio_id}.txt")
                    os.makedirs(os.path.dirname(result_file), exist_ok=True)
                    with open(result_file, "w", encoding="utf-8") as f:
                        f.write(result_txt)

                    processed += 1
                except Exception as e:
                    logger.error(f"Error processing {audio_id}: {e}")
                    errors += 1

        return processed, errors


def parse_args():
    parser = argparse.ArgumentParser(description="PretrainedSED BEATs Batch Inference")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
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
        default="../../workdir/sed_beats",
        help="Path to save transcription files",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.2, 0.5))
    parser.add_argument("--median_filter", type=float, default=9)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    args = parse_args()

    runner = PretrainedSEDRunner(device=args.device, batch_size=args.batch_size)

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
            tasks.append((key, os.path.join(args.audio_root, data["wav_path"])))
        else:
            print(f"Warning: No wav_path found for {key}")

    print(f"Total files: {len(input_data)}")
    print(f"Already processed: {skipped}")
    print(f"To process: {len(tasks)}")

    if len(tasks) == 0:
        print("All files already processed!")
    else:
        processed, errors = runner.run_inference(
            tasks=tasks,
            save_path=args.save_path,
            thresholds=args.thresholds,
            median_filter=args.median_filter,
        )

        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Processed: {len(tasks)}")
        print(f"Successful: {processed}")
        print(f"Errors: {errors}")
        print(f"Skipped (already done): {skipped}")
        print(f"Results saved to: {args.save_path}")
        print(f"Total time: {elapsed/60:.2f} minutes")
        print(f"{'='*50}")
