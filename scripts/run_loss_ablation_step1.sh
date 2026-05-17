#!/bin/bash
# =============================================================
# Loss ablation Step 1: BEHAViEW (GBT encoder + (d) setting)
#                       across 4 contrastive losses.
#
# Purpose: Identify the contrastive loss best suited to BEHAViEW
# under its dual-view (transaction + behavioral k-NN) framework.
#
# Setup: GBT encoder, setting (d) = behavioral k-NN view + subgraph
#        pooling. HOFINET dataset only.
#
#   Encoders: gbt
#   Losses:   BootstrapLatent, BarlowTwins, InfoNCE, JSD
#   Seeds:    2024 2025 2026 2027
#   Total:    1 * 4 * 4 = 16 runs
#
# Output:  results/exp_results_loss_ablation_step1.csv
#
# Usage (single GPU, sequential, low CPU priority to avoid
# disturbing other GPU tenants):
#
#   nice -n 19 GPU=7 bash scripts/run_loss_ablation_step1.sh
# =============================================================
set -e

GPU="${GPU:-7}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
LOSSES="${LOSSES:-BootstrapLatent BarlowTwins InfoNCE JSD}"
ENCODERS="${ENCODERS:-gbt}"

HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"

NODE="HOFINET_NODE_FEAT"
EDGE="HOFINET_EDGES"
KNN="HOFINET_KNN_BEHAV_k10"
RESULT="./results/exp_results_loss_ablation_step1.csv"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "[$(date)] === Loss ablation Step 1 ==="
echo "  GPU=$GPU"
echo "  ENCODERS=$ENCODERS"
echo "  LOSSES=$LOSSES"
echo "  SEEDS=$SEEDS"
echo "  OUT=$RESULT"

for ENC in $ENCODERS; do
    for LOSS in $LOSSES; do
        for SEED in $SEEDS; do
            NAME="loss_step1_${ENC}_${LOSS}_s${SEED}"
            echo "[$(date)] $NAME"
            python -u models/subgraph_cl.py \
                --model_name "$NAME" \
                --gpu "$GPU" --seed "$SEED" \
                --encoder_type "$ENC" \
                --node_data_name "$NODE" --edge_data_name "$EDGE" \
                --knn_graph "$KNN" --subgraph_pool \
                --skip_tsne \
                --metric_save_path "$RESULT" \
                $HP --loss "$LOSS" 2>&1 | grep -E "^\(E\)" || true
        done
    done
done

echo "[$(date)] === Step 1 complete ==="
