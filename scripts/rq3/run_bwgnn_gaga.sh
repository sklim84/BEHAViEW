#!/bin/bash
# BWGNN / GAGA baseline sweep (parallel per dataset)
# 2 models × 4 seeds × 3 datasets = 24 runs
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DS="${DS:-atnet}"
GPU="${GPU:-7}"

python -u models/supervised_baselines.py \
    --gpu "$GPU" --seeds 2024 2025 2026 2027 \
    --dataset "$DS" --models bwgnn gaga \
    --result_file "./results/rq3/bwgnn_gaga_${DS}.csv"
