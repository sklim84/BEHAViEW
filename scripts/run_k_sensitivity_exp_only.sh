#!/bin/bash
# k-sensitivity experiments only (k-NN graphs already built)
GPU=5
RESULT="./results/exp_results_k_sensitivity.csv"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --metric_save_path $RESULT --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 --loss BarlowTwins --encoder_type gbt"

echo "[$(date)] === k-sensitivity experiments ==="
for K in 5 10 20 50; do
  KNN="HOFINET_KNN_BEHAV_k${K}"
  echo "===== k=$K ====="
  for SEED in 2024 2025 2026 2027; do
    # Skip if already done
    grep -q "ksens_gbt_b_k${K}_s${SEED}" $RESULT 2>/dev/null && echo "  skip b k=$K s=$SEED" && continue
    echo "[$(date)] k=$K s=$SEED (b)"
    python -u models/subgraph_cl.py --model_name ksens_gbt_b_k${K}_s${SEED} --gpu $GPU --seed $SEED $COMMON --knn_graph $KNN 2>&1 | grep "^(E):"

    grep -q "ksens_gbt_d_k${K}_s${SEED}" $RESULT 2>/dev/null && echo "  skip d k=$K s=$SEED" && continue
    echo "[$(date)] k=$K s=$SEED (d)"
    python -u models/subgraph_cl.py --model_name ksens_gbt_d_k${K}_s${SEED} --gpu $GPU --seed $SEED $COMMON --knn_graph $KNN --subgraph_pool 2>&1 | grep "^(E):"
  done
done
echo "[$(date)] === Done ==="
wc -l $RESULT
