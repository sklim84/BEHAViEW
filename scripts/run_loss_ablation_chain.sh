#!/bin/bash
# =============================================================
# Chain launcher: run Step 1, then Step 2 sequentially on a
# single GPU. Designed to share the GPU with other tenants
# without disruption.
#
# Usage:
#   nice -n 19 GPU=7 bash scripts/run_loss_ablation_chain.sh \
#       > logs/loss_ablation.log 2>&1 &
# =============================================================
set -e

GPU="${GPU:-7}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "[$(date)] === Loss ablation chain started (GPU=$GPU) ==="

GPU="$GPU" bash scripts/run_loss_ablation_step1.sh
echo "[$(date)] --- Step 1 finished ---"

GPU="$GPU" bash scripts/run_loss_ablation_step2.sh
echo "[$(date)] --- Step 2 finished ---"

echo "[$(date)] === Loss ablation chain complete ==="
