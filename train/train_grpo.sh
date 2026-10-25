#!/bin/bash
# Stage C (optional): GRPO on AVQA MCQs with ms-swift, full fine-tuning.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to the DPO checkpoint}"
TRAIN_DATA="${TRAIN_DATA:-workdir/avqa_8k.jsonl}"
EXP_NAME="${EXP_NAME:-llt_grpo_7b}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_NAME}}"
REWARD_PLUGIN="${SCRIPT_DIR}/swift/reward_funcs.py"

export VLLM_LOGGING_LEVEL=WARNING

ENABLE_AUDIO_OUTPUT=False \
WANDB_PROJECT="${WANDB_PROJECT:-llt}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
swift rlhf \
    --rlhf_type grpo \
    --model "${MODEL_NAME}" \
    --model_type qwen2_5_omni \
    --train_type full \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --dataset "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --add_version false \
    --save_only_model false \
    --external_plugins "${REWARD_PLUGIN}" \
    --reward_funcs mcq_accuracy format \
    --reward_weights 1.0 0.05 \
    --max_steps 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1.5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --beta 5e-3 \
    --num_generations 8 \
    --generation_batch_size 8 \
    --max_length 4096 \
    --max_completion_length 2048 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --deepspeed zero2 \
    --log_completions true \
    --report_to wandb \
    --run_name "${EXP_NAME}"
