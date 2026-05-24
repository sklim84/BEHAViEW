#!/bin/bash
# =============================================================
# Table 8 Native column re-experiment (ATNET only).
# Re-runs each encoder under its paper-native contrastive loss in
# the BehaView setting (d) framework, current env (Step 9 aligned).
#
# Encoder -> native loss mapping:
#   dgi       -> JSD          dgi_bn    -> JSD
#   mvgrl     -> JSD          mvgrl_bn  -> JSD
#   grace     -> InfoNCE      grace_bn  -> InfoNCE
#   gbt       -> BarlowTwins
# Excluded:
#   bgrl  : BootstrapLatent (= main sweep, reuse Table 3 (d) value)
#   gin   : supervised CE (no contrastive native loss)
#
# Env vars:
#   GPU=0           single GPU index
#   PAIRS="..."     space-separated "enc:loss" (override the default)
#   SEEDS="..."     seed list (default: 2024 2025 2026 2027)
#
# Resumability:
#   Each run is named loss_native_<enc>_<loss>_s<seed>. The script
#   greps the output CSV before launching python and skips runs whose
#   model_name is already present. Killing mid-way and restarting
#   resumes from the first missing run.
# =============================================================
set -e

GPU="${GPU:-3}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"

NODE="atnet/ATNET_NODE_FEAT"
EDGE="atnet/ATNET_EDGES"
KNN="atnet/ATNET_KNN_BEHAV_k10"
RESULT="results/appendix/loss_ablation/native_loss.csv"

# Default: full Table 8 Native re-experiment (28 runs = 7 encoders x 4 seeds)
PAIRS="${PAIRS:-dgi:JSD dgi_bn:JSD mvgrl:JSD mvgrl_bn:JSD grace:InfoNCE grace_bn:InfoNCE gbt:BarlowTwins}"

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "[$(date)] === Native loss re-experiment (Table 8 Native, ATNET) ==="
echo "  GPU=$GPU"
echo "  PAIRS=$PAIRS"
echo "  SEEDS=$SEEDS"
echo "  OUT=$RESULT"

mkdir -p "$(dirname "$RESULT")"

is_done() {
    local NAME="$1"
    [ -f "$RESULT" ] && grep -q ",${NAME}," "$RESULT" 2>/dev/null && return 0
    return 1
}

for PAIR in $PAIRS; do
    ENC="${PAIR%:*}"
    LOSS="${PAIR#*:}"
    for SEED in $SEEDS; do
        NAME="loss_native_${ENC}_${LOSS}_s${SEED}"
        if is_done "$NAME"; then
            echo "[$(date)] [skip] $NAME (already in CSV)"
            continue
        fi
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

echo "[$(date)] === Done: native loss (GPU=$GPU) ==="
