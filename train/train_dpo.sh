#!/bin/bash
# Stage B: DPO (+ RPO NLL term) on LLT preference pairs with ms-swift, full fine-tuning.
set -e
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME to the talker-merged SFT checkpoint}"
TRAIN_DATA="${TRAIN_DATA:-workdir/llt_dpo_60k_abs.jsonl}"
EXP_NAME="${EXP_NAME:-llt_dpo_7b}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_NAME}}"

ENABLE_AUDIO_OUTPUT=False \
WANDB_PROJECT="${WANDB_PROJECT:-llt}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift rlhf \
    --rlhf_type dpo \
    --model "${MODEL_NAME}" \
    --model_type qwen2_5_omni \
    --train_type full \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --loss_type sigmoid \
    --beta 0.2 \
    --rpo_alpha 1.0 \
    --dataset "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --add_version false \
    --save_only_model false \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --num_train_epochs 3 \
    --max_length 32768 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 50 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --attn_impl flash_attn \
    --deepspeed zero2 \
    --report_to wandb \
    --run_name "${EXP_NAME}"
