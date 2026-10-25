# Evaluation (MMAR / MMAU / MMSU)

```bash
cd eval
bash fetch_scorers.sh                               # official MMAR / MMSU scorers (pinned commits)
(cd MMAR && python code/download_dataset.py)        # -> MMAR/mmar-audio.tar.gz (extract in place -> MMAR/audio/)
(cd MMAU && python code/download_dataset.py)        # -> MMAU/test-mini-audios/ (from HF gamma-lab-umd/MMAU-test-mini)
(cd MMSU && python code/download_dataset.py --output_dir download)   # -> MMSU/download/audio/
MODEL=/path/to/checkpoint MMAR_AUDIO=$PWD/MMAR MMAU_AUDIO=$PWD/MMAU MMSU_AUDIO=$PWD/MMSU/download bash eval.sh
```

`run_inference_vllm.py` runs greedy decoding with the LLT system prompt (`prompts.py`) and extracts the
option from `<answer>...</answer>`. `eval.sh` then scores the predictions with the **official, unmodified**
scorers of each benchmark.

## Benchmark versions (paper setting)

The metadata files in this folder are the versions **used for the numbers reported in the paper**. The
benchmarks have been updated since then:

| Benchmark      | File here                  | Upstream today                                                                                                    |
| -------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| MMAR           | `MMAR/MMAR-meta.json`      | [`MMAR-meta.jsonl`](https://github.com/ddlBoJack/MMAR): 6 of 1,000 items changed (question / choices / answer)    |
| MMAU test-mini | `MMAU/mmau-test-mini.json` | identical to [upstream](https://github.com/Sakshi113/MMAU)                                                        |
| MMSU           | `MMSU/mmsu-test.json`      | [HF `ddwang2000/MMSU`](https://huggingface.co/datasets/ddwang2000/MMSU): answer options reshuffled for most items |

To evaluate on the latest versions, replace these metadata files with the upstream ones (and note that the
numbers are no longer comparable to the paper). The download scripts fetch audio only; the current upstream
MMSU metadata is in the parquet files of [ddwang2000/MMSU](https://huggingface.co/datasets/ddwang2000/MMSU).

## Scorers

| Benchmark | Scorer                         | Source                                                                           |
| --------- | ------------------------------ | -------------------------------------------------------------------------------- |
| MMAR      | `MMAR/code/evaluation.py`      | fetched by `fetch_scorers.sh` from ddlBoJack/MMAR @ `3bce090`                    |
| MMAU      | `MMAU/code/evaluation.py`      | copied verbatim from Sakshi113/MMAU @ `110127f` (Apache-2.0, see `MMAU/LICENSE`) |
| MMSU      | `MMSU/code/mmsu_evaluation.py` | fetched by `fetch_scorers.sh` from dingdongwang/MMSU @ `217d1d3`                 |

MMSU follows the official rule: a non-empty response that cannot be resolved to A/B/C/D is dropped from the
denominator.
