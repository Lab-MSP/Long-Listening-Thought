"""
Filter datasets to keep only samples with 'music' in SED results.
Creates filtered metadata files with _music_filtered suffix.
"""

import argparse
import json
import os

from tqdm import tqdm

# AudioSet labels related to Music (including instruments, genres, and singing)
music_keywords = [
    # Core & General
    "music",
    "singing",
    "instrument",
    "melody",
    "strum",
    "lullaby",
    "a capella",
    # Instruments (High Confidence)
    "guitar",
    "piano",
    "drum",
    # "keyboard", -> to avoid computer keyboard
    "violin",
    "cello",
    "flute",
    "saxophone",
    "trumpet",
    "orchestra",
    "choir",
    "synthesizer",
    "organ",
    "accordion",
    "harmonica",
    "harp",
    "clarinet",
    "trombone",
    "percussion",
    "hi-hat",
    "cymbal",
    # Genres (Definite Music)
    "jazz",
    "blues",
    "disco",
    "techno",
    "funk",
    "reggae",
    "opera",
    "symphony",
    "metal",
    "rapping",
    "beatboxing",
    "rock",
    "dubstep",
    "electronica",
    "afrobeat",
    "ska",
    "dub",
    # Vocal Music
    "humming",
    "chant",
    "yodeling",
]


def extract_music_samples(data_file, sed_path, id_key="id"):
    """
    Extract samples that contain 'music' in their SED results.
    Supports both dict (key: metadata) and list metadata formats.

    Args:
        data_file: Path to metadata JSON file
        sed_path: Path to directory containing SED result .txt files
        id_key: Key to use for item ID in list format (ignored for dict format)

    Returns:
        Tuple of (filtered_data, music_count, total_count)
    """
    # Load metadata
    with open(data_file, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {data_file}")

    # Determine if data is dict or list format
    is_dict_format = isinstance(data, dict)

    if is_dict_format:
        items = list(data.items())  # [(key, metadata), ...]
    else:
        items = [(idx, item) for idx, item in enumerate(data)]  # [(idx, item), ...]

    # Filter samples with music in SED results
    filtered_data = [] if not is_dict_format else {}
    music_count = 0

    for item_id, item_data in tqdm(
        items, desc="Checking SED results", dynamic_ncols=True
    ):
        sed_file = os.path.join(sed_path, f"{item_id}.txt")

        if not os.path.exists(sed_file):
            continue

        # Read SED result
        try:
            with open(sed_file, "r") as f:
                sed_content = f.read().lower()

            # Filter based on events > 0.5 confidence for certain presence of music
            sed_content = sed_content.split("threshold: 0.5")[-1]

            # Check if music keywords is in the SED result (case-insensitive)
            if any(keyword in sed_content for keyword in music_keywords):
                if is_dict_format:
                    filtered_data[item_id] = item_data
                else:
                    filtered_data.append(item_data)
                music_count += 1
        except Exception as e:
            print(f"Error reading {sed_file}: {e}")
            continue

    return filtered_data, music_count, len(data)


def save_filtered_data(data, output_file):
    """Save filtered data to output file."""
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} filtered samples to {output_file}")


def process_dataset(data_file, sed_path, id_key="id"):
    """
    Process a single dataset and save filtered version.

    Returns:
        Tuple of (output_file, music_count, total_count)
    """
    # Generate output file path
    base_name = os.path.splitext(data_file)[0]
    output_file = f"{base_name}_music_filtered.json"

    # Extract music samples
    filtered_data, music_count, total_count = extract_music_samples(
        data_file, sed_path, id_key
    )

    # Save filtered data
    if filtered_data:
        save_filtered_data(filtered_data, output_file)
    else:
        print(f"No music samples found in {data_file}")
        output_file = None

    return output_file, music_count, total_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter datasets to keep only samples with music in SED results"
    )
    parser.add_argument(
        "--data_files",
        "-d",
        type=str,
        nargs="+",
        default=[
            "../examples/afthink_metadata_sample.json",
        ],
        help="Path(s) to metadata JSON file(s)",
    )
    parser.add_argument(
        "--sed_paths",
        "-s",
        type=str,
        nargs="+",
        default=[
            "../../workdir/sed_beats",
        ],
        help="Path(s) to SED results directory",
    )
    parser.add_argument(
        "--id_key",
        type=str,
        default="id",
        help="Key to use for item ID in list metadata format",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate argument counts
    num_datasets = len(args.data_files)
    if len(args.sed_paths) != num_datasets:
        raise ValueError(
            f"Number of sed_paths ({len(args.sed_paths)}) must match data_files ({num_datasets})"
        )

    print("=" * 50)
    print("MUSIC FILTERING")
    print("=" * 50)

    total_music = 0
    total_samples = 0

    # Process each dataset
    for i, (data_file, sed_path) in enumerate(zip(args.data_files, args.sed_paths)):
        print(f"\n[{i+1}/{num_datasets}] Processing: {data_file}")
        print(f"SED path: {sed_path}")

        try:
            output_file, music_count, total_count = process_dataset(
                data_file, sed_path, args.id_key
            )

            if output_file:
                print(
                    f"Music samples: {music_count}/{total_count} ({100*music_count/total_count:.1f}%)"
                )
                total_music += music_count
                total_samples += total_count
            else:
                print(f"No output file created (no music samples found)")

        except Exception as e:
            print(f"Error processing {data_file}: {e}")
            continue

    # Print final summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Total samples: {total_samples}")
    print(f"Total music samples: {total_music}")
    if total_samples > 0:
        print(f"Music percentage: {100*total_music/total_samples:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
