#!/bin/bash
# Stage A: LLT SFT (LLaMA-Factory).
# Requires LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) with Qwen2.5-Omni support.
set -e
CONFIG_FILE="${CONFIG_FILE:-$(dirname "$0")/config/llt_sft_7b.yaml}"

WANDB_PROJECT="${WANDB_PROJECT:-llt}" llamafactory-cli train "${CONFIG_FILE}"
