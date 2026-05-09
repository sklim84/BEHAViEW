#!/bin/bash
# =============================================================
# (γ) sanity check — does BECON pattern transfer AML → FFD?
#
# AMLNet (AML, AUSTRAC-compliant, 1.09M tx, 0.16% laundering)
# PaySim (FFD only, 6M tx, 0.13% fraud, no AML label)
#
# Run BECON GBT × 4 settings × 4 seeds × 2 datasets = 32 runs.
# Compare (b)→(d) effect, (a)/(c) ordering, absolute F1 floor.
#
# Output:
#   results/exp_results_amlnet.csv
#   results/exp_results_paysim.csv
#
# Usage:
#   GPU=6 bash scripts/run_gamma_sanity.sh
# =============================================================
set -e

GPU="${GPU:-0}"
SEEDS="${SEEDS:-2024 2025 2026 2027}"
SETTINGS="${SETTINGS:-a b c d}"
HP="--encoder_type gbt --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] === (γ) sanity check: AMLNet + PaySim ==="
echo "  GPU=$GPU"

run_one() {
    local DS_TAG="$1" NODE="$2" EDGE="$3" KNN="$4" RESULT="$5" SETTING="$6" SEED="$7"
    local FLAGS=""
    case "$SETTING" in
        a) FLAGS="" ;;
        b) FLAGS="--knn_graph $KNN" ;;
        c) FLAGS="--subgraph_pool" ;;
        d) FLAGS="--knn_graph $KNN --subgraph_pool" ;;
    esac
    local NAME="${DS_TAG}_gbt_${SETTING}_s${SEED}"
    echo "[$(date)] $NAME"
    python -u models/subgraph_cl.py \
        --model_name "$NAME" --gpu "$GPU" --seed "$SEED" \
        --node_data_name "$NODE" --edge_data_name "$EDGE" --skip_tsne \
        --metric_save_path "$RESULT" \
        $HP --loss BarlowTwins \
        $FLAGS 2>&1 | grep -E "^\(E\)" || true
}

for DS in amlnet paysim; do
    case "$DS" in
        amlnet)
            DS_TAG="amlnet"
            NODE="amlnet/AMLNET_NODE_FEAT"
            EDGE="amlnet/AMLNET_EDGES"
            KNN="amlnet/AMLNET_KNN_BEHAV_k10"
            RESULT="./results/exp_results_amlnet.csv"
            ;;
        paysim)
            DS_TAG="paysim"
            NODE="paysim/PAYSIM_NODE_FEAT"
            EDGE="paysim/PAYSIM_EDGES"
            KNN="paysim/PAYSIM_KNN_BEHAV_k10"
            RESULT="./results/exp_results_paysim.csv"
            ;;
    esac
    echo ""
    echo "[$(date)] === Dataset: $DS → $RESULT ==="
    for SETTING in $SETTINGS; do
        for SEED in $SEEDS; do
            run_one "$DS_TAG" "$NODE" "$EDGE" "$KNN" "$RESULT" "$SETTING" "$SEED"
        done
    done
done

echo ""
echo "[$(date)] === (γ) sanity check complete ==="
