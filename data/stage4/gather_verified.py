"""Gather the reasoning traces into `llt_full_reasoning_traces.json`.

For every metadata entry this collects
    o   the original short correct trace          (key = <key>)
    x   the stage3 incorrect short trace          (key = <key>_x)
    oo  the verified long trace continued from o  (key = <key>_oo)
    xo  the verified long trace continued from x  (key = <key>_xo)
in the format read by `../build_training_data.py`. Long traces are kept only when the answer is correct
and the reasoning is valid; a format-invalid trace is kept only if the verifier returned a fixed version.

Run it once per metadata file (AF-Think metadata and stage2 `llt_hard_metadata.json`), or pass both:
    python gather_verified.py -i ../examples/afthink_metadata_sample.json ../../workdir/llt_hard_metadata.json
"""

import argparse
import json
import os
import re

from tqdm import tqdm


def has_mcq_letters(text: str) -> bool:
    """True if the question lists choices as (A), (B), ..."""
    return len(re.findall(r"\([A-Z]\)", text)) >= 2


def normalize_answer(answer: str, data: dict, question: str) -> str:
    """Match the answer to the question format: (A) when choices are lettered, choice text otherwise."""
    if not data.get("is_mcq"):
        return answer

    is_letter = len(answer) == 1 and answer in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choices = data.get("choices") or []

    if has_mcq_letters(question):
        if is_letter:
            return f"({answer})"
        for i, choice in enumerate(choices):
            if choice.lower().strip() == answer.lower().strip():
                return f"({chr(ord('A') + i)})"
        return answer

    if is_letter:
        index = ord(answer) - ord("A")
        return choices[index] if index < len(choices) else answer
    return answer


def parse_long_cot(content: str):
    """Split a long CoT into (answer, reasoning before the <answer> tag)."""
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        reasoning = content[: match.start()].strip()
    else:
        answer = ""
        reasoning = content
    reasoning = reasoning.replace("<reason>", "").replace("</reason>", "").strip()
    return answer, reasoning


def load_long_trace(key: str, previous_reasoning: str, mode: str, args):
    """Return (answer, full reasoning) for a verified long trace, or None if it is filtered out."""
    verified_file = os.path.join(args.verified_path, mode, f"{key}.json")
    long_cot_file = os.path.join(args.long_cot_path, mode, f"{key}.txt")
    if not os.path.exists(verified_file) or not os.path.exists(long_cot_file):
        return None

    with open(verified_file, "r") as f:
        verified = json.load(f)
    if verified.get("answer_correct") is False or verified.get("reasoning_valid") is False:
        return None
    fixed = verified.get("fixed")
    if verified.get("format_valid") is False and not fixed:
        return None

    long_cot = fixed if fixed else open(long_cot_file, "r").read()
    answer, reasoning = parse_long_cot(long_cot)
    if answer == "" or reasoning == "":
        return None

    # The continuation sometimes repeats the previous reasoning; keep only what follows it
    previous = previous_reasoning.strip()
    start = reasoning.lower().find(previous.lower())
    if start != -1:
        reasoning = reasoning[start + len(previous) :].strip()
    if reasoning == "":
        return None

    return answer, previous + " " + reasoning


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gather o / x / oo / xo reasoning traces into one file"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",
        default=["../examples/afthink_metadata_sample.json"],
        help="Metadata JSON file(s): AF-Think metadata and/or stage2 llt_hard_metadata.json",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="../../workdir/llt_full_reasoning_traces.json",
        help="Path to save the gathered traces",
    )
    parser.add_argument(
        "--wrong_cot",
        "-wc",
        type=str,
        default="../../workdir/wrong_cot",
        help="Path to stage3 wrong CoT files",
    )
    parser.add_argument(
        "--long_cot_path",
        type=str,
        default="../../workdir/long_cot",
        help="Path to long CoT files ({oo,xo}/<key>.txt)",
    )
    parser.add_argument(
        "--verified_path",
        type=str,
        default="../../workdir/cot_verified",
        help="Path to long CoT verification results ({oo,xo}/<key>.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    traces = {}
    cnt = {"o": 0, "x": 0, "oo": 0, "xo": 0}
    for input_file in args.input:
        metadata = json.load(open(input_file, "r"))
        print(f"Loaded {len(metadata)} items from {input_file}")

        for key, data in tqdm(
            list(metadata.items()), desc="Gathering traces", dynamic_ncols=True
        ):
            question = data.get("question", "")
            base = {"key": key, "wav_path": data.get("wav_path", ""), "question": question}
            if "original_key" in data:
                base["original_key"] = data["original_key"]

            # o: original short trace
            answer = normalize_answer(str(data.get("answer", "")).strip(), data, question)
            if answer and data.get("response"):
                traces[key] = {**base, "reasoning": data["response"], "answer": answer, "type": "o"}
                cnt["o"] += 1
                long_trace = load_long_trace(key, data["response"], "oo", args)
                if long_trace:
                    answer, reasoning = long_trace
                    traces[f"{key}_oo"] = {**base, "reasoning": reasoning, "answer": answer, "type": "oo"}
                    cnt["oo"] += 1

            # x: stage3 incorrect short trace
            wrong_cot_file = os.path.join(args.wrong_cot, f"{key}.json")
            if not os.path.exists(wrong_cot_file):
                continue
            with open(wrong_cot_file, "r") as f:
                wrong_cot = json.load(f)
            wrong_response = wrong_cot.get("wrong_response", "")
            wrong_answer = normalize_answer(
                str(wrong_cot.get("wrong_answer", "")).strip(), data, question
            )
            if not wrong_response or not wrong_answer:
                continue
            traces[f"{key}_x"] = {**base, "reasoning": wrong_response, "answer": wrong_answer, "type": "x"}
            cnt["x"] += 1

            long_trace = load_long_trace(key, wrong_response, "xo", args)
            if long_trace:
                answer, reasoning = long_trace
                traces[f"{key}_xo"] = {**base, "reasoning": reasoning, "answer": answer, "type": "xo"}
                cnt["xo"] += 1

    print(cnt)
    print(f"Total traces: {len(traces)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
