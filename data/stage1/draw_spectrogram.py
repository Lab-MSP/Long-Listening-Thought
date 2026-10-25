import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def plot_linear_spectrogram(z, sr, save_path=None):
    duration = len(z) / sr

    # Adjust figure size dynamically -> Should modify it based on preferred length & scale
    height = 6
    width = max(8, duration)
    width = min(30, width)
    n_fft = int(sr * 0.025)
    hop_length = int(sr * 0.01)

    # STFT power spectrum
    spec = np.abs(librosa.stft(z, n_fft=n_fft, hop_length=hop_length, win_length=n_fft))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(width, height))
    try:
        img = librosa.display.specshow(
            spec_db,
            sr=sr,
            hop_length=hop_length,
            x_axis="time",
            y_axis="linear",
            cmap="magma",
            ax=ax,
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set(title="Linear Spectrogram", xlabel="Time (s)", ylabel="Frequency (Hz)")

        if duration < 5:
            # Use default ticks for short durations under 5s
            x_ticks = ax.get_xticks()
            x_ticks = x_ticks[x_ticks < duration]

            if len(x_ticks) > 0:
                tick_interval = (
                    x_ticks[-1] - x_ticks[-2] if len(x_ticks) > 1 else x_ticks[-1]
                )
                min_gap = tick_interval * 0.5

                # Remove last tick if too close to duration
                if abs(x_ticks[-1] - duration) < min_gap:
                    x_ticks = x_ticks[:-1]
            # Add duration as the last tick
            x_ticks = np.append(x_ticks, duration)
        else:
            # Apply adaptive tick intervals for durations over 5 seconds
            if duration <= 10:
                tick_interval = 0.5
            elif duration <= 30:
                tick_interval = 1.0
            # I generated spectrograms up to 30s
            else:
                tick_interval = 2.0

            x_ticks = np.arange(0, duration + tick_interval / 2, tick_interval)

            # If the duration is too close to the last tick, remove the last tick
            min_gap = tick_interval * 0.5
            if len(x_ticks) > 1 and abs(x_ticks[-1] - duration) < min_gap:
                x_ticks = x_ticks[:-1]

            # Always add the duration as the last tick so the VLM knows the total duration of the file
            x_ticks = np.append(x_ticks, duration)

        ax.set_xticks(x_ticks)

        # Set decimal places for tick labels
        if duration < 1:
            decimal_places = 3
        elif duration < 10:
            decimal_places = 2
        else:
            decimal_places = 1

        ax.set_xticklabels([f"{tick:.{decimal_places}f}" for tick in x_ticks])
        ax.set_xlim(0, duration)

        max_freq = int(sr / 2)  # Nyquist frequency
        y_ticks = ax.get_yticks()
        if max_freq not in y_ticks:
            y_ticks = np.append(y_ticks[y_ticks < max_freq], max_freq)
        ax.set_yticks(y_ticks)
        ax.set_ylim(0, max_freq)

        # Add grid lines for better readability for VLMs
        ax.grid(True, alpha=0.5, linestyle="-", linewidth=1, color="white")

        plt.tight_layout()

        # Should adjust dpi or reshape figure based on desired quality and resolution
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, format="jpg", dpi="figure", bbox_inches="tight")
    except Exception as e:
        print(f"Error generating spectrogram for {save_path}: {e}")
    finally:
        # Always close the figure, even if an error occurs
        plt.close(fig)

    return save_path


def process_single_audio(key, input_data, save_dir, audio_root=""):
    """Process a single audio file and generate spectrogram"""
    try:
        # Check if spectrogram already exists
        save_path = os.path.join(save_dir, f"{key}.jpg")
        if os.path.exists(save_path):
            return key, save_path, None

        wav_path = os.path.join(audio_root, input_data[key]["wav_path"])
        z, sr = librosa.load(wav_path, sr=None, mono=True)
        plot_linear_spectrogram(z, sr, save_path=save_path)
        return key, save_path, None
    except Exception as e:
        return key, None, str(e)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Omni Captioner Inference")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../examples/afthink_metadata_sample.json",
        help="Input JSON file containing audio files to generate caption.",
    )
    parser.add_argument(
        "--audio_root",
        type=str,
        default="",
        help="Root directory of the AF-Think audio; metadata wav_path is relative to it",
    )
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        default="../../workdir/spectrograms",
        help="Path to save the generated description.",
    )
    parser.add_argument(
        "--num_workers",
        "-n",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    input_data = json.load(open(args.input, "r"))

    # Prepare data for multiprocessing - skip already processed files
    keys = []
    skipped = 0
    for key in tqdm(
        list(input_data.keys()),
        desc="Checking existing spectrograms",
        dynamic_ncols=True,
    ):
        save_path = os.path.join(args.save_path, f"{key}.jpg")
        if os.path.exists(save_path):
            skipped += 1
        else:
            keys.append(key)

    print(f"Total files: {len(input_data)}")
    print(f"Already processed: {skipped}")
    print(f"To process: {len(keys)}")

    if len(keys) == 0:
        print("All files already processed!")
        return

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_audio,
        input_data=input_data,
        save_dir=args.save_path,
        audio_root=args.audio_root,
    )

    # Process with multiprocessing
    errors = []
    with Pool(processes=args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, keys, chunksize=100),
                total=len(keys),
                desc="Generating spectrograms",
            )
        )

    # Update input_data with results
    for key, save_path, error in results:
        if error:
            errors.append(f"{key}: {error}")
            print(f"Error processing {key}: {error}")
        else:
            input_data[key]["spectrogram_path"] = save_path

    # Print summary
    print(f"\nProcessed: {len(keys)}")
    print(f"Successful: {len(keys) - len(errors)}")
    print(f"Errors: {len(errors)}")
    print(f"Skipped (already done): {skipped}")

    if errors:
        error_file = "spectrogram_errors.txt"
        with open(error_file, "w") as f:
            f.write("\n".join(errors))
        print(f"Error details saved to {error_file}")


if __name__ == "__main__":
    main()
