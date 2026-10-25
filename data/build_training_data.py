"""Example: build SFT / DPO training files from the released reasoning traces.

`llt_full_reasoning_traces.json` maps `<key>[_oo|_xo|_x]` to
    {"key", "wav_path", "question", "reasoning", "answer", "type"[, "original_key"]}
where "key" is the metadata key of the question (stage2 questions also carry "original_key")
and type is one of
    o   original short correct trace          (key = <key>)
    oo  long trace continued from o           (key = <key>_oo)
    xo  corrected long trace continued from x (key = <key>_xo)
    x   incorrect short trace (stage3)        (key = <key>_x)

SFT uses o + oo (as in the paper). DPO pairs a correct trace (o / oo / xo) with the incorrect trace x
of the same question, `--n_per_type` pairs per type. The released `llt_sft_sharegpt.jsonl` and
`llt_dpo_60k.jsonl` are the exact files used for the paper; this script shows how to build similar
files (e.g. other mixtures or sizes), not a byte-exact reproduction of the paper's random selection.
The traces file is ~3 GB and is loaded into memory at once (expect tens of GB of RAM).

    python build_training_data.py --traces llt_full_reasoning_traces.json \
        --sft_output ../workdir/my_sft.jsonl --dpo_output ../workdir/my_dpo.jsonl
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from prompts import SYSTEM_PROMPT  # noqa: E402


def response(trace: dict) -> str:
    return f"<think> {trace['reasoning']} </think> <answer> {trace['answer']} </answer>"


def to_sharegpt(trace: dict) -> dict:
    """LLaMA-Factory ShareGPT format (SFT)."""
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": f"<audio> {trace['question']}"},
            {"from": "gpt", "value": response(trace)},
        ],
        "audios": [trace["wav_path"]],
    }


def to_swift_dpo(win: dict, lose: dict) -> dict:
    """ms-swift DPO format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<audio> {win['question']}"},
            {"role": "assistant", "content": response(win)},
        ],
        "rejected_response": response(lose),
        "audios": [win["wav_path"]],
    }


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} rows to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, help="llt_full_reasoning_traces.json")
    parser.add_argument("--sft_output", default="../workdir/llt_sft_example.jsonl")
    parser.add_argument("--dpo_output", default="../workdir/llt_dpo_example.jsonl")
    parser.add_argument("--sft_types", nargs="+", default=["o", "oo"])
    parser.add_argument("--dpo_types", nargs="+", default=["o", "xo", "oo"])
    parser.add_argument("--n_per_type", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    traces = json.load(open(args.traces))
    by_type = {}
    for key, trace in traces.items():
        by_type.setdefault(trace["type"], {})[key] = trace
    print({t: len(v) for t, v in by_type.items()})

    # SFT
    sft = [to_sharegpt(t) for typ in args.sft_types for t in by_type.get(typ, {}).values()]
    random.shuffle(sft)
    write_jsonl(sft, args.sft_output)

    # DPO: correct trace > incorrect trace (x) of the same question
    incorrect = {t["key"]: t for t in by_type.get("x", {}).values()}
    pairs = []
    for typ in args.dpo_types:
        candidates = [
            (t, incorrect[t["key"]])
            for t in by_type.get(typ, {}).values()
            if t["key"] in incorrect
        ]
        random.shuffle(candidates)
        chosen = candidates[: args.n_per_type]
        print(f"{typ} > x: {len(chosen):,} pairs (from {len(candidates):,} candidates)")
        pairs += [to_swift_dpo(win, lose) for win, lose in chosen]
    random.shuffle(pairs)
    write_jsonl(pairs, args.dpo_output)


if __name__ == "__main__":
    main()
