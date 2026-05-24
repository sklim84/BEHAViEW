#!/bin/bash
# =============================================================
# BehaView label-efficiency sweep for {atnet, amlworld, amlnet}.
#
# Env vars: GPU, DATASET (atnet/amlworld/amlnet), ENCODER, KNN
# 1 setting (d) × 3 fractions × 4 seeds = 12 runs/dataset
# =============================================================
set -e

GPU="${GPU:-1}"
DATASET="${DATASET:-amlworld}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
RATIOS="${RATIOS:-0.01 0.05 0.10}"
EPOCHS="${EPOCHS:-200}"

# Per-dataset config (matches paper's best encoder per dataset)
if [[ "$DATASET" == "atnet" ]]; then
    NODE="atnet/ATNET_NODE_FEAT"
    EDGE="atnet/ATNET_EDGES"
    KNN="atnet/ATNET_KNN_BEHAV_k10"
    ENCODER="${ENCODER:-gbt}"
elif [[ "$DATASET" == "amlworld" ]]; then
    NODE="amlworld/AMLWORLD_NODE_FEAT"
    EDGE="amlworld/AMLWORLD_EDGES"
    KNN="amlworld/AMLWORLD_KNN_BEHAV_k10"
    ENCODER="${ENCODER:-bgrl}"
elif [[ "$DATASET" == "amlnet" ]]; then
    NODE="amlnet/AMLNET_NODE_FEAT"
    EDGE="amlnet/AMLNET_EDGES"
    KNN="amlnet/AMLNET_KNN_BEHAV_k10"
    ENCODER="${ENCODER:-gbt}"
else
    echo "Unknown dataset: $DATASET"; exit 1
fi

RESULT="results/rq3/behaview_${DATASET}.csv"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] === BehaView label-efficiency $DATASET ($ENCODER + d) ==="
echo "  GPU=$GPU  Ratios=$RATIOS  Seeds=$SEEDS"

is_done() {
    local NAME="$1"
    [ -f "$RESULT" ] && grep -q ",${NAME}," "$RESULT" 2>/dev/null && return 0
    return 1
}

for RATIO in $RATIOS; do
    for SEED in $SEEDS; do
        NAME="labeff_${ENCODER}_d_${DATASET}_r${RATIO}_s${SEED}"
        if is_done "$NAME"; then
            echo "[$(date)] [skip] $NAME (already in CSV)"
            continue
        fi
        echo "[$(date)] $NAME"
        python3 -u models/subgraph_cl.py \
            --model_name "$NAME" \
            --gpu "$GPU" --seed "$SEED" --epochs "$EPOCHS" \
            --encoder_type "$ENCODER" \
            --node_data_name "$NODE" --edge_data_name "$EDGE" \
            --knn_graph "$KNN" --subgraph_pool \
            --train_ratio "$RATIO" \
            --skip_tsne \
            $HP --loss BootstrapLatent \
            --metric_save_path "$RESULT" 2>&1 | grep -E "^\(E\)" || true
    done
done

echo "[$(date)] === Done: BehaView $DATASET ==="
