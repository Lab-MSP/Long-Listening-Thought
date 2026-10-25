#!/bin/bash
# LLaMA-Factory saves only the Qwen2.5-Omni thinker. Merge it back with the base talker
# so the checkpoint can be loaded by vLLM / ms-swift (needed before DPO and evaluation).
#   THINKER_PATH=outputs/llt_sft_7b/checkpoint-$CKPT_NUMBER LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory bash train/merge_talker.sh
set -e
: "${THINKER_PATH:?set THINKER_PATH to a LLaMA-Factory checkpoint}"
: "${LLAMA_FACTORY_DIR:?set LLAMA_FACTORY_DIR to your LLaMA-Factory clone}"
SAVE_PATH="${SAVE_PATH:-${THINKER_PATH}_talkermerged}"

python "${LLAMA_FACTORY_DIR}/scripts/qwen_omni_merge.py" save_full \
  --model_path="Qwen/Qwen2.5-Omni-7B" \
  --thinker_path="${THINKER_PATH}" \
  --save_path="${SAVE_PATH}"
