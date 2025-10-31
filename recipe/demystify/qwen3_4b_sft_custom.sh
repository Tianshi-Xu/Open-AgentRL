#!/bin/bash
set -x
nnodes=1
nproc_per_node=8
project_name=demystify-agentic-rl
experiment_name=qwen3-4b-ra-sft
DATA_ROOT=${DATA_ROOT:-$PWD}
TRAIN_DATA=dataset/Open-AgentRL-SFT-3K/full_sft_3k_shuffled_v4.parquet
EVAL_DATA=dataset/Open-AgentRL-SFT-3K/full_sft_3k_shuffled_v4.parquet
MODEL_PATH=Qwen3-4B-Instruct-2507
SAVE_PATH=checkpoint/$experiment_name

# Set your wandb API key here or export it in your environment
export WANDB_API_KEY="c2ade05262c251418946ecc479a941028eb37bba"

# AMLT_DATA_DIR will be automatically prepended to MODEL_PATH if set
# export AMLT_DATA_DIR="/path/to/your/data"

torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     -m recipe.demystify.custom_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=32768 \
    data.truncation=right \
    data.train_batch_size=32 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp2 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=5 \
    trainer.save_freq=62 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true

