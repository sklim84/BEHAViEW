#!/bin/bash
# CARE-GNN / PC-GNN baseline sweep
# 2 models × 4 seeds × 3 datasets = 24 runs
set -e
GPU="${GPU:-7}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for DS in atnet amlworld amlnet; do
    echo "[$(date)] === Dataset: $DS ==="
    python -u models/supervised_baselines.py \
        --gpu "$GPU" --seeds 2024 2025 2026 2027 \
        --dataset "$DS" --models caregnn pcgnn \
        --result_file "./results/rq3/caregnn_pcgnn_${DS}.csv"
done
echo "[$(date)] === DONE ==="
