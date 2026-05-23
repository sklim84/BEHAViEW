#!/bin/bash
# =============================================================
# Train BehaView (GBT) on HOFINET for the four ablation settings
# (a/b/c/d) and dump the joint node embeddings (z = h^{(1)} || h^{(2)})
# for downstream t-SNE/UMAP visualization. We run a single seed since
# the figure is qualitative.
#
# Env:
#   GPU   single GPU index (default 4 -- the V0 HP sweep uses 0-3)
#   SEED  fixed seed (default 2025)
# Outputs under results/embeddings/tsne_v0/:
#   HOFINET_z_<setting>_s<seed>.npz   each containing arrays z and y
# =============================================================
set -e

GPU="${GPU:-4}"
SEED="${SEED:-2025}"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"
NODE="HOFINET_NODE_FEAT"
EDGE="HOFINET_EDGES"
KNN="HOFINET_KNN_BEHAV_k10"
OUT_DIR="results/embeddings/tsne_v0"

mkdir -p "$OUT_DIR"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"

echo "[$(date)] === Extract BehaView embeddings (GPU=$GPU, SEED=$SEED) ==="

run_one() {
    local SETTING=$1
    local FLAGS=""
    case "$SETTING" in
        a) FLAGS="" ;;
        b) FLAGS="--knn_graph $KNN" ;;
        c) FLAGS="--subgraph_pool" ;;
        d) FLAGS="--knn_graph $KNN --subgraph_pool" ;;
        *) echo "unknown setting"; exit 1 ;;
    esac
    local NAME="tsne_emb_hof_gbt_${SETTING}_s${SEED}"
    local OUT="${OUT_DIR}/HOFINET_z_${SETTING}_s${SEED}.npz"
    if [ -f "$OUT" ]; then
        echo "[$(date)] [skip] $OUT exists"
        return
    fi
    echo "[$(date)] training $NAME -> $OUT"
    python -u models/subgraph_cl.py \
        --model_name "$NAME" \
        --gpu "$GPU" --seed "$SEED" \
        --encoder_type gbt \
        --node_data_name "$NODE" --edge_data_name "$EDGE" \
        --skip_tsne \
        --save_embeddings_to "$OUT" \
        --metric_save_path "results/embeddings/tsne_v0/_metrics.csv" \
        $HP --loss BootstrapLatent \
        $FLAGS 2>&1 | grep -E "^\((E|L)\)" || true
}

for S in a b c d; do
    run_one "$S"
done

echo "[$(date)] === Done. Run plot_tsne.py next. ==="
