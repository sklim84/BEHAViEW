#!/bin/bash
# =============================================================
# Feature ablation: HOFINET k-NN graph variants × 4 seeds.
# 5 k-NN types × 4 seeds = 20 runs.
#
# Output: results/appendix/feature_ablation.csv
# =============================================================
set -e

GPU="${GPU:-0}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
KNN_GRAPHS="${KNN_GRAPHS:-HOFINET_KNN_BEHAV_k10 HOFINET_KNN_PURE_k10 HOFINET_KNN_STRUCT_k10 HOFINET_KNN_HYBRID_k10 HOFINET_KNN_FEAT_k10}"

RESULT="./results/appendix/feature_ablation.csv"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"
COMMON="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --encoder_type gbt"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] Feature ablation -> $RESULT"
for KNN in $KNN_GRAPHS; do
    for SEED in $SEEDS; do
        NAME="featabl_${KNN}_s${SEED}"
        echo "[$(date)] $NAME"
        python -u models/subgraph_cl.py \
            --model_name "$NAME" --gpu "$GPU" --seed "$SEED" \
            $COMMON $HP \
            --knn_graph "$KNN" --subgraph_pool \
            --metric_save_path "$RESULT" 2>&1 | grep -E "^\(E\)" || true
    done
done
