# LLT data pipeline

Scripts that build the LLT reasoning data from AF-Think questions. The final SFT / DPO data and
intermediate metadata are released (see below), so you can regenerate or
extend the data based on the pipeline.

```
examples/     small metadata samples (input format of every stage)
build_training_data.py  example: SFT / DPO files from the released traces
stage1/       multi-aspect dense audio descriptions
stage2/       challenging-question generation + verification
stage3/       incorrect short reasoning traces
stage4/       long reasoning traces (thought continuation) + verification
grpo/         AVQA GRPO data builder (optional GRPO stage)
third_party/  PretrainedSED, demucs_batch (git submodules)
```

## Released data

| File | Content |
|---|---|
| `afthink_metadata.json` | 368,860 AF-Think questions (audio ≤ 30 s) with normalized answers |
| `llt_hard_metadata.json` | 323,400 stage2 questions (`<key>_hard`, `original_key` points to the source clip) |
| `llt_full_reasoning_traces.json` | all reasoning traces: `o` (original short), `oo` / `xo` (verified long), `x` (stage3 incorrect) |
| `llt_sft_sharegpt.jsonl` | final SFT data (LLaMA-Factory ShareGPT format) |
| `llt_dpo_60k.jsonl` | final DPO pairs (ms-swift format) |

The metadata files also contain questions that were not used for training (e.g., a validation split and a set kept for further experiments), so they are larger than the counts reported in the paper (338k original / 316k stage2 questions).

Download: [Lab-MSP/Long-Listening-Thought](https://huggingface.co/datasets/Lab-MSP/Long-Listening-Thought). The data is derived from [AF-Think](https://huggingface.co/datasets/nvidia/AF-Think) and is
released under the NVIDIA OneWay Noncommercial License (research use only).

**Audio is not redistributed.** Download the clips following the
[AF-Think instructions](https://huggingface.co/datasets/nvidia/AF-Think), then arrange them as below (or
rewrite `wav_path` to your own layout). `wav_path` is relative to an audio root that you pass with
`--audio_root`.

### Audio layout

Keys and paths come from the AF-Think `afthink/<Source>.json` files:

- key = `<Source>/<id>`
- `wav_path` = `<Source>/<id>.<ext>`

Here `<Source>` is the json file name and `<id>` is the entry's `id`. Each file holds the audio of the
entry's `sound` field, mixed down to mono at the original sample rate. When `sound` lists several
segments (LibriSpeech, VoxPopuli), they are concatenated in order into one file. For VGGSound the audio
track is extracted from the video.

| `<Source>` (AF-Think json) | Clips | Ext | AF-Think `sound` | Our `wav_path` |
|---|---:|---|---|---|
| YouTube8M | 306,212 | mp3 | `v1yO_FviAZM_74_10.mp3` | `YouTube8M/v1yO_FviAZM_74_10.mp3` |
| AudioSet | 55,872 | flac | `Y5M587BUfo0g.wav` | `AudioSet/Y5M587BUfo0g.flac` |
| VGG | 2,278 | flac | `oK2d2JF27Qk_000000.wav` | `VGG/oK2d2JF27Qk_000000.flac` |
| AudioSet_SL | 946 | flac | `YtF6G0Xryh3c.wav` | `AudioSet_SL/YtF6G0Xryh3c.flac` |
| Freesound | 798 | flac | `644531.wav` | `Freesound/644531.flac` |
| DailyTalk | 752 | flac | `1147.wav` | `DailyTalk/1147.flac` |
| CochlScene | 706 | flac | `Park_user0515_15311135_002.wav` | `CochlScene/Park_user0515_15311135_002.flac` |
| FMA | 396 | mp3 | `047701.wav` | `FMA/047701.mp3` |
| MELD | 393 | flac | `492.wav` | `MELD/492.flac` |
| MagnaTagATune | 238 | flac | `<artist>-<album>-<track>-<n>.mp3` | `MagnaTagATune/<same name>.flac` |
| FSD50k | 98 | flac | `241546.wav` | `FSD50k/241546.flac` |
| MACS | 57 | wav | `park-milan-1063-41819-a.wav` | `MACS/park-milan-1063-41819-a.wav` |
| MusicCaps | 36 | flac | `19Pp9QEw17U.wav` | `MusicCaps/19Pp9QEw17U.flac` |
| Clotho-v2 | 31 | flac | `MetroStation.wav` | `Clotho-v2/MetroStation.flac` |
| LibriSpeech | 29 | flac | `['7505-258958-0000.wav', '7505-258958-0001.wav']` | `LibriSpeech/7505_258958.flac` (concatenated) |
| VoxPopuli | 10 | flac | `['en/2017/20171114-0900-PLENARY-7-en_..._0.wav', ...]` | `VoxPopuli/20171114-0900-PLENARY-7.flac` (concatenated) |
| BBC_Sound_Effects | 6 | flac | `NHU05014209.wav` | `BBC_Sound_Effects/NHU05014209.flac` |
| ESC-50 | 2 | wav | `3-151081-B-20.wav` | `ESC-50/3-151081-B-20.wav` |

Only clips of at most 30 s are kept. The extension does not matter to the scripts (audio is read with
librosa / soundfile), so you can keep the original format and just change the extension in `wav_path`.

### Metadata format

```json
"YouTube8M/4Bftz5jhHFY_23_10_hard": {
  "wav_path": "YouTube8M/4Bftz5jhHFY_23_10.mp3",
  "duration": 10.0,
  "original_key": "YouTube8M/4Bftz5jhHFY_23_10",
  "question": "In what context is this audio most likely occurring, ...",
  "response": "short reasoning + answer",
  "answer": "D",
  "is_mcq": true,
  "choices": ["(A) ...", "(B) ...", "(C) ...", "(D) ..."]
}
```

`examples/afthink_metadata_sample.json` and `examples/llt_hard_metadata_sample.json` hold 20 entries
each, and every script defaults to them. Pass the full metadata (or any subset) with `--input`.

## Environments

```bash
conda env create -f envs/vllm.yml            # vLLM clients, ASR, spectrograms (llt_vllm)
conda env create -f envs/stage1_sed.yml      # PretrainedSED (llt_stage1_sed)
conda env create -f envs/stage1_music.yml    # allin1 + demucs (llt_stage1_music)
conda activate llt_stage1_music && pip install -e data/third_party/demucs_batch && conda deactivate
conda env create -f envs/stage1_chord.yml    # autochord (llt_stage1_chord)
conda env create -f envs/stage1_diarize.yml  # pyannote.audio (llt_stage1_diarize)
```

Run these from the repo root.

## Conventions

- Run each script from its own directory (`cd data/stage1 && python ...`); relative defaults assume this.
- Outputs go to `workdir/` at the repo root (one file per key, so interrupted runs resume).
- LLM steps call OpenAI-compatible vLLM servers. Each model has a fixed default port:

  | Model | Port | Used by |
  |---|---|---|
  | Qwen/Qwen3-Omni-30B-A3B-Captioner | 8001 | stage1 `qwen_omni_caption.py` |
  | Qwen/Qwen3-VL-32B-Instruct | 8002 | stage1 `qwen_vl_caption.py` |
  | Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 | 8003 | stage2, stage3, stage4 `verify_long_reasoning.py` |
  | Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 | 8004 | stage4 `generate_long_reasoning.py` |

  ```bash
  vllm serve Qwen/Qwen3-Omni-30B-A3B-Captioner --port 8001 -tp 2 --allowed-local-media-path /
  vllm serve Qwen/Qwen3-VL-32B-Instruct        --port 8002 -tp 2 --allowed-local-media-path /
  vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 --port 8003 -tp 2
  vllm serve Qwen/Qwen3-Next-80B-A3B-Thinking-FP8 --port 8004 -tp 2 --reasoning-parser deepseek_r1
  ```

  `--reasoning-parser deepseek_r1` separates the thinking model's `<think>` block from its answer; the long
  reasoning traces keep only the answer part.

## Stage 1: dense audio descriptions (`stage1/`)

Input: AF-Think metadata. Output: `workdir/descriptions/<key>.txt`.

| Step | Script | Env | Output |
|---|---|---|---|
| Sound event detection (BEATs) | `sed_runner_full.py` | `llt_stage1_sed` | `workdir/sed_beats/` |
| Music subset | `filter_music_total.py` | `llt_stage1_sed` | `<input>_music_filtered.json` |
| ASR | `fast_whisper.py` | `llt_vllm` | `workdir/whisper_fast/` |
| Speaker diarization | `diarize.py` (needs `HF_TOKEN`) | `llt_stage1_diarize` | `workdir/pyannote/` |
| Chords (music subset) | `extract_chord.py` | `llt_stage1_chord` | `workdir/chord/` |
| Music structure (music subset) | `third_party/demucs_batch/demucs_batch.py` → `extract_music_temporal.py` | `llt_stage1_music` | `workdir/music_temporal/` |
| Acoustic description | `draw_spectrogram.py` → `qwen_vl_caption.py` | `llt_vllm` | `workdir/acoustic_descriptions/` |
| Semantic description | `qwen_omni_caption.py` | `llt_vllm` | `workdir/semantic_descriptions/` |
| Merge | `gather_descriptions.py` | `llt_vllm` | `workdir/descriptions/` |

`gather_descriptions.py` requires the semantic and acoustic descriptions: items missing either are not
saved and are picked up by the next run. Temporal parts (SED, ASR, diarization, chords, music structure)
are optional, since they are legitimately empty for many clips, so an existing complete output is not
rebuilt when a temporal result is added later. Re-run with `--skip_existing 0` to merge such results.

Chords and music structure run on the music subset written by `filter_music_total.py`
(`<input>_music_filtered.json`, e.g. `../examples/afthink_metadata_sample_music_filtered.json`), so run SED
and the filter first. For music structure, precompute the Demucs stems with
`python ../third_party/demucs_batch/demucs_batch.py -i <music_filtered.json> --audio_root <audio_root> -o ../../workdir/demix_cache_full`
(from `data/stage1`); `extract_music_temporal.py --demix_dir` defaults to the same directory.

Scripts that read audio take `--audio_root`. The BEATs checkpoint (`BEATs_strong_1.pt`) downloads
automatically, or get it from the [PretrainedSED release](https://github.com/fschmid56/PretrainedSED/releases/tag/v0.0.1).

## Stage 2: challenging questions (`stage2/`)

```bash
python generate_new_questions.py   # -> workdir/questions_hard/
python verify_questions.py         # -> workdir/stage2_verification/
python gather_verified.py          # -> workdir/llt_hard_metadata.json
```

## Stage 3: incorrect reasoning traces (`stage3/`)

```bash
python wrong_cot_augment.py -i <afthink_metadata.json>           # -> workdir/wrong_cot/
python wrong_cot_augment.py -i ../../workdir/llt_hard_metadata.json
```

## Stage 4: long reasoning traces (`stage4/`)

`--mode oo` continues the original (correct) short reasoning. `--mode xo` continues an incorrect trace
from stage3 into a corrected long one.

```bash
python generate_long_reasoning.py --mode oo    # -> workdir/long_cot/oo/
python generate_long_reasoning.py --mode xo    # -> workdir/long_cot/xo/
python verify_long_reasoning.py --mode oo      # -> workdir/cot_verified/oo/
python verify_long_reasoning.py --mode xo      # -> workdir/cot_verified/xo/
python gather_verified.py                      # -> workdir/llt_full_reasoning_traces.json
```

The generate and verify scripts take `-i`; run them on `../../workdir/llt_hard_metadata.json` as well to
cover the stage2 questions.

Run `gather_verified.py` after stage 4 (generation and verification for both modes) has finished. It
collects the original (`o`), incorrect (`x`), and verified long (`oo`, `xo`) traces into the format of the
released `llt_full_reasoning_traces.json`; pass all metadata files at once, e.g.
`python gather_verified.py -i <afthink_metadata.json> ../../workdir/llt_hard_metadata.json`.
Given the paper's intermediate outputs, it reproduces the released traces exactly, except that it also
includes the questions that were not used for training (see [Released data](#released-data)). Re-running
stages 3–4 yields different traces, since the LLM outputs are sampled.

The released SFT data uses the original short traces (`o`) and verified `oo` long traces. The DPO pairs
take `o` / `oo` / `xo` traces as chosen and stage3 incorrect traces as rejected.

## Building training files

`build_training_data.py` turns `llt_full_reasoning_traces.json` (released, or built with
`stage4/gather_verified.py`) into SFT (ShareGPT, `o` + `oo`) and DPO (ms-swift; `o` / `xo` / `oo` chosen vs `x` rejected of the same question) files:

```bash
python build_training_data.py --traces llt_full_reasoning_traces.json --n_per_type 20000
```

The released `llt_sft_sharegpt.jsonl` / `llt_dpo_60k.jsonl` are the exact paper files; the script is for
building other mixtures and does not reproduce the paper's random 60k selection byte-for-byte.

## GRPO data (`grpo/`)

```bash
python build_avqa_grpo.py --audio_dir /path/to/vggsound_audio
```

This builds `workdir/avqa_8k.jsonl` from the [AVQA](https://mn.cs.tsinghua.edu.cn/avqa/) train set using
the 8k question ids from the paper (`avqa_grpo_8k_ids.json`). The official `train_qa.json` is downloaded
from the [AVQA repository](https://github.com/AlyssaYoung/AVQA) automatically. Audio is not included:
AVQA videos come from VGGSound (YouTube), so obtain them yourself and extract 16 kHz mono FLAC as
`<audio_dir>/<video_name>.flac`.
