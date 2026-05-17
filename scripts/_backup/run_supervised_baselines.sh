#!/bin/bash
# =============================================================
# Supervised baselines (RQ5) on corrected 10/10/80 split via make_split
#
# 6 models (gcn/gat/sage/mlp/lgbm/xgb) × 4 seeds × 2 datasets = 48 runs
# Identical test set as BehaView SSL (paired comparison enabled).
#
# Output:
#   - HOFINET  → results/exp_results_supervised_hofinet.csv
#   - AMLworld → results/exp_results_supervised_amlworld.csv
#
# Parallel:
#   GPU=6 DATASETS=hofinet  bash scripts/run_supervised_baselines.sh &
#   GPU=7 DATASETS=amlworld bash scripts/run_supervised_baselines.sh &
#   wait
# =============================================================
set -e

GPU="${GPU:-0}"
DATASETS="${DATASETS:-hofinet amlworld}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] === Supervised baselines (corrected 10/10/80 split) ==="
echo "  GPU=$GPU"
echo "  DATASETS=$DATASETS"
echo "  SEEDS=$SEEDS"

for DS in $DATASETS; do
    case "$DS" in
        hofinet)
            RESULT="./results/exp_results_supervised_hofinet.csv"
            ;;
        amlworld)
            RESULT="./results/exp_results_supervised_amlworld.csv"
            ;;
        *)
            echo "unknown dataset: $DS"; exit 1
            ;;
    esac

    echo ""
    echo "[$(date)] === Dataset: $DS → $RESULT ==="
    python -u models/supervised_baselines.py \
        --gpu "$GPU" \
        --seeds $SEEDS \
        --dataset "$DS" \
        --result_file "$RESULT" 2>&1 | grep -E "^\[|F1_susp=|Saved:|Suspicious:|Dataset:"
done

echo ""
echo "[$(date)] === Supervised baselines complete ==="
