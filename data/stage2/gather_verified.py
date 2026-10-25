import argparse
import json
import os

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gather verified questions and answers from the dataset"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="../examples/afthink_metadata_sample.json",
        help="Input JSON file containing metadata with questions and responses.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="../../workdir/llt_hard_metadata.json",
        help="Path to save the gathered verified questions and answers.",
    )
    parser.add_argument(
        "--generated_questions_path",
        "-gq",
        type=str,
        default="../../workdir/questions_hard",
        help="Path to generated question files (JSON format)",
    )
    parser.add_argument(
        "--verification_path",
        "-v",
        type=str,
        default="../../workdir/stage2_verification",
        help="Path to save verification results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_data = json.load(open(args.input, "r"))
    print(f"Loaded {len(input_data)} items from {args.input}")

    skipped = 0
    not_verified = 0
    processed = 0
    verified_data = {}
    for key, data in tqdm(
        list(input_data.items()),
        desc="Gathering verified questions and answers",
        dynamic_ncols=True,
    ):
        generated_question = os.path.join(args.generated_questions_path, f"{key}.json")
        if not os.path.exists(generated_question):
            skipped += 1
            continue

        verification_result = os.path.join(args.verification_path, f"{key}.json")
        if not os.path.exists(verification_result):
            skipped += 1
            continue

        with open(verification_result, "r") as f:
            verification_data = json.load(f)

        audio_based_solvable = verification_data.get("audio_based_solvable", False)
        answer_correct = verification_data.get("answer_correct", False)
        reasoning_valid = verification_data.get("reasoning_valid", False)
        if not audio_based_solvable or not answer_correct or not reasoning_valid:
            not_verified += 1
            continue

        with open(generated_question, "r") as f:
            question_data = json.load(f)
        question = question_data.get("question", "")
        answer = question_data.get("answer", "")
        response = question_data.get("response", "")

        new_key = key + "_hard"
        if "_hard" in key:
            print(f"Warning: Key {key} already contains '_hard'.")

        choices = question_data.get("choices", [])
        entry = {
            "wav_path": data.get("wav_path", ""),
            "duration": data.get("duration", 0),
            "original_key": key,
            "question": question,
            "response": response,
            "answer": answer,
            "is_mcq": bool(choices),
            "choices": choices,
        }

        verified_data[new_key] = entry
        processed += 1

    print(f"Skipped: {skipped}, Not Verified: {not_verified}, Processed: {processed}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(verified_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
