#!/bin/bash
# =============================================================
# DGI (BN-less) TBD-fill sweep for Table 7 (RQ4)
#
# Fills missing cells:
#   B1: AMLworld DGI (d) tuned   (4 seeds)
#   B2: AMLNet   DGI (d) tuned   (4 seeds)
#   B3: AMLNet   DGI (a)(b)(c) untuned (4 seeds × 3)  ← traceability backfill
#
# Setting-to-flags mapping (all 4 ablation settings supported here, unlike
# run_amlworld_safe_improvements.sh which assumes the behavioral view is
# always present):
#   a  → augmentation view, node-level     (no --knn_graph, no --subgraph_pool)
#   b  → behavioral  view, node-level      (--knn_graph)
#   c  → augmentation view, subgraph pool  (no --knn_graph, --subgraph_pool)
#   d  → behavioral  view, subgraph pool   (--knn_graph, --subgraph_pool)
#
# Threshold policy follows the existing Table 7 column convention:
#   - (a)(b)(c) rows: τ=0.5 default only (no --tune_threshold)
#   - (d) rows: --tune_threshold (both default + tuned emitted per refactor,
#     paper uses the tuned row)
# =============================================================
set -e

SEEDS="${SEEDS:-2024 2025 2026 2027}"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"
EPOCHS="${EPOCHS:-200}"

run_one() {
    local DSET="$1"; local NODE="$2"; local EDGE="$3"; local KNN="$4"
    local SETTING="$5"; local SEED="$6"; local GPU="$7"; local RESULT="$8"

    local FLAGS=""
    case "$SETTING" in
        a) ;;                                           # baseline: no knn, no pool
        b) FLAGS="--knn_graph $KNN" ;;
        c) FLAGS="--subgraph_pool" ;;
        d) FLAGS="--knn_graph $KNN --subgraph_pool --tune_threshold" ;;
        *) echo "unknown setting: $SETTING"; exit 1 ;;
    esac

    local NAME="aml_dgi_${SETTING}_s${SEED}"
    echo "[$(date)] [GPU${GPU}] [${DSET}] $NAME"
    python3 -u models/subgraph_cl.py \
        --model_name "$NAME" \
        --gpu "$GPU" --seed "$SEED" --epochs "$EPOCHS" \
        --encoder_type dgi \
        --node_data_name "$NODE" --edge_data_name "$EDGE" \
        --skip_tsne \
        --metric_save_path "$RESULT" \
        $HP --loss BarlowTwins \
        $FLAGS 2>&1 | grep -E "^\(E\)|Tuned threshold" || true
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BLOCK="${BLOCK:-all}"   # all | b1 | b2 | b3

if [[ "$BLOCK" == "all" || "$BLOCK" == "b1" ]]; then
    GPU_B1="${GPU_B1:-6}"
    RES_B1="results/exp_results_amlworld_safe_dgi.csv"
    echo "[$(date)] === B1: AMLworld DGI (d) tuned on GPU${GPU_B1} → ${RES_B1} ==="
    for SEED in $SEEDS; do
        run_one "amlworld" \
            "amlworld/AMLWORLD_NODE_FEAT" "amlworld/AMLWORLD_EDGES" \
            "amlworld/AMLWORLD_KNN_BEHAV_k10" \
            "d" "$SEED" "$GPU_B1" "$RES_B1"
    done
fi

if [[ "$BLOCK" == "all" || "$BLOCK" == "b2" ]]; then
    GPU_B2="${GPU_B2:-0}"
    RES_B2="results/exp_results_amlnet_safe_dgi.csv"
    echo "[$(date)] === B2: AMLNet DGI (d) tuned on GPU${GPU_B2} → ${RES_B2} ==="
    for SEED in $SEEDS; do
        run_one "amlnet" \
            "amlnet/AMLNET_NODE_FEAT" "amlnet/AMLNET_EDGES" \
            "amlnet/AMLNET_KNN_BEHAV_k10" \
            "d" "$SEED" "$GPU_B2" "$RES_B2"
    done
fi

if [[ "$BLOCK" == "all" || "$BLOCK" == "b3" ]]; then
    GPU_B3="${GPU_B3:-0}"
    RES_B3="results/exp_results_amlnet_safe_dgi.csv"
    echo "[$(date)] === B3: AMLNet DGI (a)(b)(c) untuned on GPU${GPU_B3} → ${RES_B3} ==="
    for SETTING in a b c; do
        for SEED in $SEEDS; do
            run_one "amlnet" \
                "amlnet/AMLNET_NODE_FEAT" "amlnet/AMLNET_EDGES" \
                "amlnet/AMLNET_KNN_BEHAV_k10" \
                "$SETTING" "$SEED" "$GPU_B3" "$RES_B3"
        done
    done
fi

echo "[$(date)] === Sweep complete ==="
