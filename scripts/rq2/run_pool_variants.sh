#!/bin/bash
# =============================================================
# Pool variant experiments (HAP / Cycle) for HOFINET
#
# Tests the alternatives to mean pool theoretically motivated by
# Theorem 4 (HAP signal preservation) and the cycle-membership
# domain prior (suspicious accounts in mule-layering cycles).
#
# 4 encoders × {c, d} × {HAP, Cycle} × 4 seeds = 64 runs
#
# Output: results/exp_results_pool_variants.csv
# =============================================================
set -e

GPU="${GPU:-7}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
ENCODERS="${ENCODERS:-gbt bgrl gin dgi}"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NODE="hofinet/HOFINET_NODE_FEAT"
EDGE="hofinet/HOFINET_EDGES"
KNN="hofinet/HOFINET_KNN_BEHAV_k10"
RESULT="./results/rq2/pool_variants.csv"

run_one() {
    local ENC="$1"
    local SETTING="$2"  # c_hap | c_cycle | d_hap | d_cycle
    local SEED="$3"

    local FLAGS=""
    case "$SETTING" in
        c_hap)   FLAGS="--subgraph_pool --pool_variant heterophily" ;;
        c_cycle) FLAGS="--subgraph_pool --pool_variant cycle" ;;
        d_hap)   FLAGS="--knn_graph $KNN --subgraph_pool --pool_variant heterophily" ;;
        d_cycle) FLAGS="--knn_graph $KNN --subgraph_pool --pool_variant cycle" ;;
        *) echo "unknown: $SETTING"; exit 1 ;;
    esac

    local NAME="hof_${ENC}_${SETTING}_s${SEED}"
    echo "[$(date)] $NAME"
    python -u models/subgraph_cl.py \
        --model_name "$NAME" \
        --gpu "$GPU" --seed "$SEED" \
        --encoder_type "$ENC" \
        --node_data_name "$NODE" --edge_data_name "$EDGE" \
        --skip_tsne \
        --metric_save_path "$RESULT" \
        $HP --loss BarlowTwins \
        $FLAGS 2>&1 | grep -E "^\(E\)" || true
}

echo "[$(date)] === Pool variant sweep (HOFINET) ==="
for ENC in $ENCODERS; do
    for SETTING in c_hap c_cycle d_hap d_cycle; do
        for SEED in $SEEDS; do
            run_one "$ENC" "$SETTING" "$SEED"
        done
    done
done
echo "[$(date)] === Sweep complete ==="
