#!/bin/bash
# =============================================================
# Additional experiments on corrected 10/10/80 split (Gate 1 protocol):
#   (A) Label fraction sensitivity: 3 ratios × 4 seeds = 12 runs
#   (B) Feature ablation: 5 k-NN types × 4 seeds = 20 runs
#
# Output:
#   results/exp_results_label_fraction.csv    (regenerate, ~12 rows)
#   results/exp_results_feature_ablation.csv  (regenerate, ~20 rows)
#
# Computational cost (Table 10) is unchanged — single-run timing,
# protocol-independent, kept from prior measurement.
#
# Usage:
#   GPU=6 bash scripts/run_additional_exp.sh
# =============================================================
set -e

GPU="${GPU:-0}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
HP="--encoder_type gbt --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"
COMMON="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --loss BarlowTwins"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] === Additional experiments (corrected protocol) ==="
echo "  GPU=$GPU"

# ============= (A) Label fraction =============
RESULT_A="./results/exp_results_label_fraction.csv"
echo ""
echo "[$(date)] (A) Label fraction sensitivity → $RESULT_A"
for RATIO in 0.01 0.05 0.10; do
    for SEED in $SEEDS; do
        NAME="labfrac_gbt_d_r${RATIO}_s${SEED}"
        echo "[$(date)] $NAME"
        python -u models/subgraph_cl.py \
            --model_name "$NAME" --gpu "$GPU" --seed "$SEED" \
            $COMMON $HP \
            --knn_graph HOFINET_KNN_BEHAV_k10 --subgraph_pool \
            --train_ratio "$RATIO" \
            --metric_save_path "$RESULT_A" 2>&1 | grep -E "^\(E\)" || true
    done
done

# ============= (B) Feature ablation =============
RESULT_B="./results/exp_results_feature_ablation.csv"
echo ""
echo "[$(date)] (B) Feature ablation → $RESULT_B"
for KNN_TYPE in BEHAV PURE FEAT STRUCT; do
    KNN_FILE="HOFINET_KNN_${KNN_TYPE}_k10"
    if [ ! -f "datasets/${KNN_FILE}.csv" ]; then
        echo "  WARN: ${KNN_FILE}.csv not found, skipping"
        continue
    fi
    for SEED in $SEEDS; do
        NAME="feat_gbt_d_${KNN_TYPE}_s${SEED}"
        echo "[$(date)] $NAME"
        python -u models/subgraph_cl.py \
            --model_name "$NAME" --gpu "$GPU" --seed "$SEED" \
            $COMMON $HP \
            --knn_graph "$KNN_FILE" --subgraph_pool \
            --metric_save_path "$RESULT_B" 2>&1 | grep -E "^\(E\)" || true
    done
done

echo ""
echo "[$(date)] === Additional experiments complete ==="
