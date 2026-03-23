#!/bin/bash
TASK=$1
CONFIG=$2

# 动态拼接路径 (假设你在 policy/openvla-oft 目录下执行)
DATASET_PATH="../../data/${TASK}/${CONFIG}/data"
INSTRUCTION_DIR="../../data/${TASK}/${CONFIG}/instructions"
OUT_BASE_DIR="../../data/${TASK}/${CONFIG}/processed_openvla"

echo "正在处理任务: ${TASK} | 配置: ${CONFIG}"
echo "数据来源: ${DATASET_PATH}"

python preprocess_aloha.py \
    --dataset_path $DATASET_PATH \
    --out_base_dir $OUT_BASE_DIR \
    --instruction_dir $INSTRUCTION_DIR \
    --percent_val 0.05