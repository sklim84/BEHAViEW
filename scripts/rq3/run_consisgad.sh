#!/bin/bash
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS="${DS:-atnet}"
GPU="${GPU:-7}"
python -u models/supervised_baselines.py \
    --gpu "$GPU" --seeds 2024 2025 2026 2027 \
    --dataset "$DS" --models consisgad \
    --result_file "./results/rq3/consisgad_${DS}.csv"
