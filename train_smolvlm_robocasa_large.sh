#!/bin/bash
set -e

BATCH_SIZE=${1:-16}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./runs/simvla_robocasa_large}
DATASET_SOUP=${4:-target_atomic_seen}
RESUME_CKPT=${5:-""}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TF_CPP_MIN_LOG_LEVEL=2

DATASET_BASE_PATH=${DATASET_BASE_PATH:-/home/kslab/zijian/work/robocasa/datasets}
META_DIR=${META_DIR:-./datasets/metas}
NORM_DIR=${NORM_DIR:-./norm_stats}
TRAIN_METAS_PATH=${META_DIR}/robocasa_${DATASET_SOUP}.json
NORM_STATS_PATH=${NORM_DIR}/robocasa_${DATASET_SOUP}_norm.json
SMOLVLM_MODEL=${SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}

LEARNING_RATE=${LEARNING_RATE:-2e-4}
NUM_ACTIONS=${NUM_ACTIONS:-16}
ITERS=${ITERS:-2000}
WARMUP_STEPS=${WARMUP_STEPS:-0}
FREEZE_STEPS=${FREEZE_STEPS:-1000}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

HIDDEN_SIZE=1024
DEPTH=24
NUM_HEADS=16

mkdir -p "${META_DIR}" "${NORM_DIR}" "${OUTPUT_DIR}"

if [ ! -f "${TRAIN_METAS_PATH}" ]; then
  python create_robocasa_meta.py \
    --dataset_soup "${DATASET_SOUP}" \
    --dataset_base_path "${DATASET_BASE_PATH}" \
    --output "${TRAIN_METAS_PATH}" \
    --strict
fi

if [ ! -f "${NORM_STATS_PATH}" ]; then
  python compute_robocasa_norm_stats.py \
    --metas_path "${TRAIN_METAS_PATH}" \
    --num_actions "${NUM_ACTIONS}" \
    --max_samples_per_dataset 2000 \
    --output "${NORM_STATS_PATH}"
fi

ARGS="--output_dir ${OUTPUT_DIR} \
  --train_metas_path ${TRAIN_METAS_PATH} \
  --smolvlm_model_path ${SMOLVLM_MODEL} \
  --action_mode robocasa_12dof \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --learning_coef ${LEARNING_COEF} \
  --num_actions ${NUM_ACTIONS} \
  --iters ${ITERS} \
  --warmup_steps ${WARMUP_STEPS} \
  --freeze_steps ${FREEZE_STEPS} \
  --hidden_size ${HIDDEN_SIZE} \
  --depth ${DEPTH} \
  --num_heads ${NUM_HEADS} \
  --num_workers ${NUM_WORKERS} \
  --save_interval ${SAVE_INTERVAL} \
  --log_interval ${LOG_INTERVAL} \
  --image_size 384 \
  --norm_stats_path ${NORM_STATS_PATH} \
  --max_grad_norm ${MAX_GRAD_NORM}"

if [ -n "${RESUME_CKPT}" ]; then
  ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
  --num_processes=4 \
  --main_process_port 29515 \
  --mixed_precision bf16 \
  train_smolvlm.py ${ARGS}
