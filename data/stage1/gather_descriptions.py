"""
Gather and combine descriptions from multiple sources.
Works with af_think metadata.json format (key: metadata_dict pairs).
Reads descriptions from semantic, acoustic, and temporal sources.
"""

import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

HEADER_NOTE = ""


def decode_music_structure(data):
    """
    Decodes music structure JSON into a readable text format.
    Only includes fields that are not None to avoid misleading information.
    """
    if not data or not isinstance(data, dict):
        return ""

    lines = []

    # Metadata fields
    metrics = [
        ("tempo_bpm", "Tempo (BPM)", ":.1f", ""),
        ("beats_count", "Beats Count", "", ""),
        ("downbeats_count", "Downbeats Count", "", ""),
        ("beat_start_s", "Beat Start Time", ":.2f", "s"),
        ("downbeat_start_s", "Downbeat Start Time", ":.2f", "s"),
        ("beat_interval_mean_s", "Average Beat Interval", ":.3f", "s"),
        ("beat_interval_std_s", "Beat Interval Std Dev", ":.3f", "s"),
    ]

    for key, label, fmt, suffix in metrics:
        val = data.get(key)
        if val is not None:
            formatted_val = format(val, fmt[1:]) if fmt else str(val)
            lines.append(f"{label}: {formatted_val}{suffix}")

    # Beat Stability
    stability = data.get("beat_stability")
    if stability is not None:
        lines.append(f"Beat Stability: {str(stability).replace('_', ' ')}")

    # Music Sections
    sections = data.get("sections")
    if sections and isinstance(sections, list):
        section_lines = []
        for s in sections:
            if not isinstance(s, dict):
                continue

            start = s.get("start")
            end = s.get("end")
            label = s.get("label")

            if start is not None and end is not None:
                section_lines.append(
                    f"  [{float(start):.1f}–{float(end):.1f}s] {label or 'section'}"
                )

        if section_lines:
            lines.append("Music Sections:")
            lines.extend(section_lines)

    return "\n".join(lines)


def read_description_file(file_path, is_json=False):
    """
    Read a description file.
    Returns None if file doesn't exist or is empty.
    """
    if not os.path.exists(file_path):
        return None

    try:
        if is_json:
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and not data:
                    return None
                return data
        else:
            with open(file_path, "r") as f:
                content = f.read().strip()
                return content if content else None
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")
        return None


def build_acoustic_description(acoustic_data):
    """Convert acoustic JSON to formatted text."""
    if not acoustic_data:
        return None

    try:
        lines = []
        for key, item in acoustic_data.items():
            if isinstance(item, str) and item.strip():
                lines.append(f"**{key}**\n{item}\n")
        return "\n".join(lines).strip() if lines else None
    except Exception as e:
        logger.warning(f"Error processing acoustic data: {e}")
        return None


def build_temporal_description(temporal_data):
    """Combine all temporal data into a single description."""
    lines = []

    if temporal_data.get("sed_beats"):
        lines.append(f"**Sound Event Detection**\n{temporal_data['sed_beats']}\n")

    if temporal_data.get("whisper_fast"):
        lines.append(
            f"**Speech Transcription with Timestamps**\n{temporal_data['whisper_fast']}\n"
        )

    if temporal_data.get("pyannote"):
        lines.append(f"**Speaker Diarization**\n{temporal_data['pyannote']}\n")

    if temporal_data.get("music_temporal"):
        lines.append(
            f"**Music Structure Analysis**\n{temporal_data['music_temporal']}\n"
        )

    if temporal_data.get("chord"):
        lines.append(f"**Chord Recognition**\n{temporal_data['chord']}\n")

    return "\n".join(lines).strip() if lines else None


class DescriptionGatherer:
    def __init__(self, base_paths, dataset_name, output_path):
        """
        Initialize the gatherer.

        Args:
            base_paths: Dict mapping description type to path
                       e.g. {"semantic": "...", "acoustic": "...", "temporal": "..."}
            dataset_name: Name of dataset (for logging)
            output_path: Path to save combined descriptions
        """
        self.base_paths = base_paths
        self.dataset_name = dataset_name
        self.output_path = output_path
        self.logger = logger

        # Statistics
        self.total_items = 0
        self.successful = 0
        self.skipped = 0
        self.incomplete = 0
        self.errors = []
        self.missing_descriptions = {}

    def _required_headers(self):
        """Sections every gathered description must contain (temporal parts are optional)."""
        headers = []
        if "semantic" in self.base_paths:
            headers.append("# Semantic Description:")
        if "acoustic" in self.base_paths:
            headers.append("# Acoustic Description:")
        return headers

    def _is_complete(self, output_file):
        if not os.path.exists(output_file):
            return False
        text = read_description_file(output_file) or ""
        return all(h in text for h in self._required_headers())

    def process_item(self, item_id, skip_existing=True):
        """
        Process a single item and gather its descriptions.

        Args:
            item_id: ID of the item
            skip_existing: Skip if output file already exists

        Returns:
            Dict with all descriptions, or None if skipped or incomplete
        """
        output_file = os.path.join(self.output_path, f"{item_id}.txt")

        # Skip only complete outputs, so incomplete ones are regenerated
        if skip_existing and self._is_complete(output_file):
            self.skipped += 1
            return None

        descriptions = {}
        missing_parts = []

        # Read semantic description
        if "semantic" in self.base_paths:
            semantic_path = os.path.join(self.base_paths["semantic"], f"{item_id}.txt")
            semantic_desc = read_description_file(semantic_path)
            descriptions["semantic"] = semantic_desc
            if not semantic_desc:
                missing_parts.append("semantic")

        # Read acoustic description
        if "acoustic" in self.base_paths:
            acoustic_path = os.path.join(self.base_paths["acoustic"], f"{item_id}.txt")
            acoustic_desc = read_description_file(acoustic_path)
            # Try to parse as JSON first
            if acoustic_desc:
                try:
                    acoustic_json = json.loads(acoustic_desc)
                    acoustic_desc = build_acoustic_description(acoustic_json)
                except:
                    # Not JSON, use as-is
                    pass
            descriptions["acoustic"] = acoustic_desc
            if not acoustic_desc:
                missing_parts.append("acoustic")

        # Read temporal descriptions (multiple parts)
        temporal_dict = {}
        if "temporal" in self.base_paths:
            temporal_base = self.base_paths["temporal"]

            # Try to read individual temporal components
            for component in [
                "sed_beats",
                "whisper_fast",
                "pyannote",
                "music_temporal",
                "chord",
            ]:
                if component == "music_temporal":
                    component_path = os.path.join(
                        temporal_base, "music_temporal", f"{item_id}.json"
                    )
                else:
                    component_path = os.path.join(
                        temporal_base, component, f"{item_id}.txt"
                    )
                if os.path.exists(component_path):
                    if component == "music_temporal":  # allin1
                        try:
                            component_data = read_description_file(
                                component_path, is_json=True
                            )
                            component_data = decode_music_structure(component_data)
                        except:
                            pass
                    else:
                        component_data = read_description_file(component_path)
                    temporal_dict[component] = component_data
                else:
                    temporal_dict[component] = None

            temporal_desc = build_temporal_description(temporal_dict)
            descriptions["temporal"] = temporal_desc

        # Semantic and acoustic descriptions are required: do not save incomplete items,
        # so a later run picks them up once the missing parts exist.
        if missing_parts:
            self.missing_descriptions[item_id] = missing_parts
            self.incomplete += 1
            return None

        return descriptions

    def save_descriptions(self, item_id, descriptions):
        """Save combined descriptions to file."""
        try:
            output_file = os.path.join(self.output_path, f"{item_id}.txt")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Build final description text
            final_lines = []

            if descriptions.get("semantic"):
                final_lines.append("# Semantic Description:\n")
                final_lines.append(descriptions["semantic"])

            if descriptions.get("acoustic"):
                final_lines.append("\n# Acoustic Description:\n")
                final_lines.append(descriptions["acoustic"])

            if descriptions.get("temporal"):
                final_lines.append("\n# Temporal Description:\n")
                final_lines.append(descriptions["temporal"])

            final_text = "\n".join(final_lines)

            with open(output_file, "w") as f:
                f.write(final_text)

            self.successful += 1
            return True

        except Exception as e:
            error_msg = f"{item_id}: {str(e)}"
            self.errors.append(error_msg)
            self.logger.error(f"Error saving {item_id}: {e}")
            return False

    def process_dataset(self, item_ids, skip_existing=True):
        """
        Process all items in dataset.

        Args:
            item_ids: List of item IDs to process
            skip_existing: Skip items that already have output files
        """
        self.total_items = len(item_ids)

        for item_id in tqdm(
            item_ids, desc=f"Processing {self.dataset_name}", dynamic_ncols=True
        ):
            descriptions = self.process_item(item_id, skip_existing=skip_existing)

            if descriptions is None:
                continue  # Skipped or incomplete

            self.save_descriptions(item_id, descriptions)

    def print_summary(self):
        """Print processing summary."""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"Total items: {self.total_items}")
        self.logger.info(f"Successful: {self.successful}")
        self.logger.info(f"Incomplete (not saved): {self.incomplete}")
        self.logger.info(f"Skipped (already done): {self.skipped}")
        self.logger.info(f"Errors: {len(self.errors)}")

        if self.missing_descriptions:
            self.logger.info(
                f"\nItems with missing descriptions: {len(self.missing_descriptions)}"
            )
            for item_id, missing_parts in list(self.missing_descriptions.items())[:5]:
                self.logger.info(f"  {item_id}: missing {', '.join(missing_parts)}")
            if len(self.missing_descriptions) > 5:
                self.logger.info(f"  ... and {len(self.missing_descriptions) - 5} more")

        if self.errors:
            self.logger.error(f"\nErrors ({len(self.errors)}):")
            for error in self.errors[:5]:
                self.logger.error(f"  {error}")
            if len(self.errors) > 5:
                self.logger.error(f"  ... and {len(self.errors) - 5} more")

        self.logger.info(f"{'='*50}\n")


def verify_descriptions(base_paths, metadata, metadata_music_filetered):
    """
    Verify that descriptions exist for all items in metadata.
    Logs missing descriptions and returns a dict of item_id to missing parts.
    """
    missing_descriptions = {}

    for item_id in tqdm(
        metadata.keys(), desc="Verifying descriptions", dynamic_ncols=True
    ):
        missing_parts = []

        # Check semantic description
        semantic_path = os.path.join(base_paths["semantic"], f"{item_id}.txt")
        if not os.path.exists(semantic_path):
            missing_parts.append("semantic")

        # Check acoustic description
        acoustic_path = os.path.join(base_paths["acoustic"], f"{item_id}.txt")
        if not os.path.exists(acoustic_path):
            missing_parts.append("acoustic")

        # Check temporal descriptions
        temporal_base = base_paths["temporal"]

        # Non-music components
        for component in [
            "sed_beats",
            "whisper_fast",
            "pyannote",
        ]:
            component_path = os.path.join(temporal_base, component, f"{item_id}.txt")
            if not os.path.exists(component_path):
                missing_parts.append(f"temporal_{component}")

        for component in [
            "music_temporal",
            "chord",
        ]:
            if item_id not in metadata_music_filetered:
                continue  # Skip music-specific components for non-music items

            if component == "music_temporal":
                component_path = os.path.join(
                    temporal_base, "music_temporal", f"{item_id}.json"
                )
            else:
                component_path = os.path.join(
                    temporal_base, component, f"{item_id}.txt"
                )
            if not os.path.exists(component_path):
                missing_parts.append(f"temporal_{component}")

        if missing_parts:
            missing_descriptions[item_id] = missing_parts

    return missing_descriptions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gather and combine descriptions from multiple sources using af_think metadata format"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../examples/afthink_metadata_sample.json",
        help="Path to metadata.json file (af_think format: key-dict pairs)",
    )

    parser.add_argument(
        "--semantic_path",
        "-s",
        type=str,
        default="../../workdir/semantic_descriptions",
        help="Path to semantic descriptions directory",
    )

    parser.add_argument(
        "--acoustic_path",
        "-a",
        type=str,
        default="../../workdir/acoustic_descriptions",
        help="Path to acoustic descriptions directory",
    )

    parser.add_argument(
        "--temporal_path",
        "-t",
        type=str,
        default="../../workdir",
        help="Path to temporal descriptions base directory",
    )

    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="../../workdir/descriptions",
        help="Path to save gathered descriptions",
    )

    parser.add_argument(
        "--skip_existing",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "Skip items whose output already has the semantic and acoustic sections. "
            "Temporal parts are optional, so use 0 to merge temporal results added later."
        ),
    )

    parser.add_argument(
        "--check_description",
        type=int,
        default=0,
        choices=[0, 1],
        help="Check for missing descriptions without processing (1) or gather descriptions (0)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Input metadata: {args.input}")
    logger.info(f"Semantic path: {args.semantic_path}")
    logger.info(f"Acoustic path: {args.acoustic_path}")
    logger.info(f"Temporal path: {args.temporal_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Skip existing: {bool(args.skip_existing)}")

    # Load metadata
    try:
        with open(args.input, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return

    if not isinstance(metadata, dict):
        logger.error(f"Metadata must be dict format, got {type(metadata)}")
        return

    item_ids = list(metadata.keys())
    logger.info(f"Loaded {len(item_ids)} items from metadata: {args.input}")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Build base paths dict
    base_paths = {
        "semantic": args.semantic_path,
        "acoustic": args.acoustic_path,
        "temporal": args.temporal_path,
    }

    # Check that all description exists:
    if args.check_description:
        music_filtered = args.input.replace(".json", "_music_filtered.json")
        with open(music_filtered, "r") as f:
            metadata_music_filtered = json.load(f)
        missing_items = verify_descriptions(
            base_paths, metadata, metadata_music_filtered
        )
        if missing_items:
            logger.warning(
                f"Found {len(missing_items)} items with missing descriptions:"
            )
            for item_id, missing_parts in list(missing_items.items())[:5]:
                logger.warning(f"  {item_id}: missing {', '.join(missing_parts)}")
            with open("missing_descriptions.json", "w") as f:
                json.dump(missing_items, f, indent=2)

            print(
                "Please address the missing descriptions before running the gatherer."
            )
        else:
            print("No missing items")
        sys.exit(0)

    # Initialize gatherer
    gatherer = DescriptionGatherer(
        base_paths=base_paths,
        dataset_name="af_think",
        output_path=args.output_path,
    )

    # Process dataset
    gatherer.process_dataset(item_ids, skip_existing=bool(args.skip_existing))

    # Print summary
    gatherer.print_summary()


if __name__ == "__main__":
    main()
