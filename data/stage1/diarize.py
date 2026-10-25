"""
Pyannote Speaker Diarization Runner for af_think metadata format.
Maintains the same core logic as analysis/stage1/diarize.py but uses metadata.json input.
"""

import argparse
import json
import logging
import os
import time
import warnings

import torch
import torchaudio
import torchaudio.transforms as T

# Required: pip install pyannote.audio
from pyannote.audio import Pipeline
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


logger = logging.getLogger(__name__)


def pyannote_to_timeline_merged_text(diarization_result, ndigits=2):
    """
    Convert Pyannote diarization result into timeline text,
    merging consecutive segments from the same speaker.
    """
    lines = []

    current_speaker = None
    current_start = None
    current_end = None

    for turn, speaker in diarization_result:
        # Initialize the first segment
        if current_speaker is None:
            current_speaker = speaker
            current_start = turn.start
            current_end = turn.end
            continue

        # If the same speaker continues, just update the end time (Merge)
        if speaker == current_speaker:
            current_end = turn.end
        else:
            # If the speaker changes, append the previous consolidated segment
            start_val = round(float(current_start), ndigits)
            end_val = round(float(current_end), ndigits)
            lines.append(f"[{start_val}–{end_val}s] {current_speaker}")

            # Reset tracking for the new speaker
            current_speaker = speaker
            current_start = turn.start
            current_end = turn.end

    # Append the final remaining segment after the loop ends
    if current_speaker is not None:
        start_val = round(float(current_start), ndigits)
        end_val = round(float(current_end), ndigits)
        lines.append(f"[{start_val}–{end_val}s] {current_speaker}")

    return "\n".join(lines)


class PyannoteRunner:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-community-1",
        device="cuda",
        hf_token=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading Pyannote model: {model_name} on {self.device}")

        if hf_token is None:
            raise ValueError(
                "Pyannote requires a Hugging Face token (hf_token). "
                f"Please accept terms at https://huggingface.co/{model_name}"
            )

        # Load pipeline
        self.pipeline = Pipeline.from_pretrained(
            model_name, token=hf_token
        )
        self.pipeline.to(self.device)

        print("Pyannote Diarization pipeline initialized.")

    def load_data(self, data_file):
        with open(data_file, "r") as f:
            self.data = json.load(f)

    def run_inference(
        self,
        save_path,
        audio_path_key="wav_path",
        audio_root="",
        id_key="id",
        num_speakers=None,  # Optional hint
    ):
        os.makedirs(save_path, exist_ok=True)

        skipped = 0
        to_process_items = []

        # 1) Filter files
        if isinstance(self.data, dict):
            # Metadata format (key: value pairs)
            items = list(self.data.items())
        else:
            # List format
            items = [(idx, item) for idx, item in enumerate(self.data)]

        for item_id, item in tqdm(
            items, desc="Checking existing results", dynamic_ncols=True
        ):
            result_file = os.path.join(save_path, f"{item_id}.txt")

            if os.path.exists(result_file):
                skipped += 1
                continue

            audio_path = os.path.join(audio_root, item.get(audio_path_key, ""))

            if not os.path.exists(audio_path):
                logger.error(f"Audio not found: {audio_path}")
                continue

            to_process_items.append((item_id, audio_path))

        print(
            f"Total: {len(self.data)} | Already processed: {skipped} | To process: {len(to_process_items)}"
        )

        if not to_process_items:
            return

        # 2) Run inference
        processed = 0
        pbar = tqdm(to_process_items, desc="Diarizing", unit="file", dynamic_ncols=True)

        target_sample_rate = 16000

        for item_id, audio_path in pbar:
            try:
                # Pyannote batch processing is slightly different; here we do sequential with GPU acceleration
                # Note: Diarization is typically slower than ASR
                waveform, sample_rate = torchaudio.load(audio_path, channels_first=True)
                waveform = waveform.to(self.device)  # Move to H100 early

                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                if sample_rate != target_sample_rate:
                    resampler = T.Resample(
                        orig_freq=sample_rate, new_freq=target_sample_rate
                    ).to(self.device)
                    waveform = resampler(waveform)

                audio_input = {"waveform": waveform, "sample_rate": target_sample_rate}

                # output = self.pipeline(audio_path, num_speakers=num_speakers)
                output = self.pipeline(audio_input, num_speakers=num_speakers)

                # Format results
                result_txt = pyannote_to_timeline_merged_text(
                    output.speaker_diarization
                )
                # Save results
                result_file = os.path.join(save_path, f"{item_id}.txt")
                os.makedirs(os.path.dirname(result_file), exist_ok=True)
                with open(result_file, "w", encoding="utf-8") as f:
                    f.write(result_txt)

                processed += 1
            except Exception as e:
                pbar.write(f"Error processing {item_id}: {e}")

        print(f"Successfully processed: {processed}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pyannote Speaker Diarization")
    parser.add_argument(
        "--model_name", type=str, default="pyannote/speaker-diarization-community-1"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face User Access Token (defaults to $HF_TOKEN)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../examples/afthink_metadata_sample.json",
        help="Input JSON file containing audio metadata.",
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
        default="../../workdir/pyannote",
        help="Path to save diarization results",
    )
    parser.add_argument(
        "--audio_path_key",
        type=str,
        default="wav_path",
        help="Key in metadata for audio file path",
    )
    parser.add_argument(
        "--id_key",
        type=str,
        default="id",
        help="Key in metadata for item ID",
    )
    parser.add_argument(
        "--num_workers",
        "-w",
        type=int,
        default=1,
        help="Number of worker threads (Note: Pyannote processes sequentially)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    args = parse_args()

    print("=" * 50)
    print("PYANNOTE SPEAKER DIARIZATION")
    print("=" * 50)

    # Initialize Runner
    runner = PyannoteRunner(
        model_name=args.model_name, device=args.device, hf_token=args.hf_token
    )

    print(f"\nProcessing: {args.input}")

    runner.load_data(args.input)
    print(f"Loaded {len(runner.data)} items")

    runner.run_inference(
        save_path=args.save_path,
        audio_path_key=args.audio_path_key,
        audio_root=args.audio_root,
        id_key=args.id_key,
    )

    elapsed = time.time() - start_time
    print(f"\nProcessing complete in {elapsed / 60:.2f} minutes.")
    print(f"Results saved to: {args.save_path}")
    print("=" * 50)
