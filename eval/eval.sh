#!/bin/bash
# Evaluate a Qwen2.5-Omni checkpoint on MMAR, MMAU (test-mini) and MMSU with vLLM.
#
#   MODEL=/path/to/checkpoint \
#   MMAR_AUDIO=/path/to/mmar   MMAU_AUDIO=/path/to/mmau   MMSU_AUDIO=/path/to/mmsu \
#   bash eval/eval.sh
#
# *_AUDIO is the directory that contains the relative audio paths in the metadata, i.e.
#   MMAR: <MMAR_AUDIO>/audio/*.wav   MMAU: <MMAU_AUDIO>/test-mini-audios/*.wav   MMSU: <MMSU_AUDIO>/audio/*.wav
# LLaMA-Factory SFT checkpoints must be talker-merged first (train/merge_talker.sh).
set -e

MODEL="${MODEL:?set MODEL to a checkpoint path or HF id}"
TAG="${TAG:-$(basename "${MODEL}")}"
OUT_DIR="${OUT_DIR:-$(dirname "$0")/results/${TAG}}"
MMAR_AUDIO="${MMAR_AUDIO:?}"
MMAU_AUDIO="${MMAU_AUDIO:?}"
MMSU_AUDIO="${MMSU_AUDIO:?}"
# Resolve user paths before changing into eval/ (MODEL may also be an HF id)
[ -e "${MODEL}" ] && MODEL="$(realpath "${MODEL}")"
MMAR_AUDIO="$(realpath "${MMAR_AUDIO}")"; MMAU_AUDIO="$(realpath "${MMAU_AUDIO}")"; MMSU_AUDIO="$(realpath "${MMSU_AUDIO}")"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(realpath "${OUT_DIR}")"
cd "$(dirname "$0")"
[ -f MMAR/code/evaluation.py ] && [ -f MMSU/code/mmsu_evaluation.py ] || bash fetch_scorers.sh

python run_inference_vllm.py \
    --model_name "${MODEL}" \
    --data_files MMAR/MMAR-meta.json MMAU/mmau-test-mini.json MMSU/mmsu-test.json \
    --out_files "${OUT_DIR}/mmar_pred.json" "${OUT_DIR}/mmau_pred.json" "${OUT_DIR}/mmsu_pred.jsonl" \
    --response_files "${OUT_DIR}/mmar_response.json" "${OUT_DIR}/mmau_response.json" "${OUT_DIR}/mmsu_response.json" \
    --audio_base_paths "${MMAR_AUDIO}" "${MMAU_AUDIO}" "${MMSU_AUDIO}"

echo "=== MMAR ==="; python MMAR/code/evaluation.py --input "${OUT_DIR}/mmar_pred.json" | tee "${OUT_DIR}/mmar_metrics.txt"
echo "=== MMAU ==="; python MMAU/code/evaluation.py --input "${OUT_DIR}/mmau_pred.json" | tee "${OUT_DIR}/mmau_metrics.txt"
echo "=== MMSU ==="; python MMSU/code/mmsu_evaluation.py "${OUT_DIR}/mmsu_pred.jsonl" | tee "${OUT_DIR}/mmsu_metrics.txt"
