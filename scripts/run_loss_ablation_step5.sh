#!/bin/bash
# Loss ablation Step 5: DGI+BN x 4 losses x setting (b).
# Pairs with Step 3 ((a) setting) to expose (a) vs (b) final-loss
# pattern across all four contrastive losses, supporting the
# claim that the (a) low-loss is structural (view similarity),
# not BYOL-specific.
set -e
GPU="${GPU:-1}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
LOSSES="${LOSSES:-BootstrapLatent BarlowTwins InfoNCE JSD}"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"
NODE="HOFINET_NODE_FEAT"; EDGE="HOFINET_EDGES"; KNN="HOFINET_KNN_BEHAV_k10"
RESULT="./results/exp_results_loss_ablation_step5.csv"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "[$(date)] === Step 5: DGI+BN x 4 losses x (b) ==="
for LOSS in $LOSSES; do
    for SEED in $SEEDS; do
        NAME="loss_step5_dgi_bn_${LOSS}_s${SEED}"
        echo "[$(date)] $NAME"
        # Setting (b): --knn_graph, no --subgraph_pool
        python -u models/subgraph_cl.py \
            --model_name "$NAME" --gpu "$GPU" --seed "$SEED" \
            --encoder_type dgi_bn \
            --node_data_name "$NODE" --edge_data_name "$EDGE" \
            --knn_graph "$KNN" \
            --skip_tsne --metric_save_path "$RESULT" \
            $HP --loss "$LOSS" 2>&1 | grep -E "^\((E|L)\)" || true
    done
done
echo "[$(date)] === Step 5 complete ==="
