#!/bin/bash
# =============================================================
# Loss ablation Step 3: Setting (a), DGI+BN, across 4 losses.
#
# Purpose: Test whether the low loss observed at setting (a)
#   (e.g., DGI+BN seed-2024 L=0.060 in Table~\ref{tab:loss_table})
#   is a loss-formulation-specific quirk of BYOL bootstrap or a
#   structural artifact of view similarity (the two perturbed
#   transaction-graph views being too similar for the encoder to
#   need discriminative encoding).
#
# Expected: All 4 losses produce low F1_susp (~0.27) and converge
#   to their respective minima quickly, confirming that the (a)
#   pathology is structural (low view diversity), not specific to
#   the BYOL bootstrap formulation.
#
# Setup:
#   Encoder: dgi_bn (DGI + per-layer BN)
#   Setting: (a) tx graph + node contrast
#   Losses:  BootstrapLatent BarlowTwins InfoNCE JSD
#   Seeds:   2024 2025 2026 2027
#   Total:   1 * 4 * 4 = 16 runs
#
# Output: results/exp_results_loss_ablation_step3.csv
#
# Usage (low-impact sharing on GPU 1):
#   nice -n 19 GPU=1 bash scripts/run_loss_ablation_step3.sh
# =============================================================
set -e

GPU="${GPU:-1}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
LOSSES="${LOSSES:-BootstrapLatent BarlowTwins InfoNCE JSD}"
ENCODERS="${ENCODERS:-dgi_bn}"

HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"

NODE="HOFINET_NODE_FEAT"
EDGE="HOFINET_EDGES"
RESULT="./results/exp_results_loss_ablation_step3.csv"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "[$(date)] === Loss ablation Step 3 (setting (a) interpretation check) ==="
echo "  GPU=$GPU"
echo "  ENCODERS=$ENCODERS"
echo "  LOSSES=$LOSSES"
echo "  SEEDS=$SEEDS"
echo "  OUT=$RESULT"

for ENC in $ENCODERS; do
    for LOSS in $LOSSES; do
        for SEED in $SEEDS; do
            NAME="loss_step3_${ENC}_${LOSS}_s${SEED}"
            echo "[$(date)] $NAME"
            # Setting (a): no --knn_graph, no --subgraph_pool
            python -u models/subgraph_cl.py \
                --model_name "$NAME" \
                --gpu "$GPU" --seed "$SEED" \
                --encoder_type "$ENC" \
                --node_data_name "$NODE" --edge_data_name "$EDGE" \
                --skip_tsne \
                --metric_save_path "$RESULT" \
                $HP --loss "$LOSS" 2>&1 | grep -E "^\(E\)" || true
        done
    done
done

echo "[$(date)] === Step 3 complete ==="
