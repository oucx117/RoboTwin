#!/bin/bash

export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "/ssd2/cunxuou/tensorflow_datasets/" \
  --dataset_name "aloha_open_laptop" \
  --run_root_dir "/ssd2/cunxuou/RoboTwin2_project/RoboTwin/policy/openvla-oft/checkpoints" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --grad_accumulation_steps 1 \
  --use_proprio True \
  --batch_size 4 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 15000 \
  --max_steps 30000 \
  --use_val_set True \
  --val_freq 1000 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "ygtreceplain" \
  --wandb_project "openvla_finetune" \
  --run_id_note "open_laptop_8gpu"