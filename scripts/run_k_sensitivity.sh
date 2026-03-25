#!/bin/bash
# k-sensitivity: build k-NN graphs (k=5,20,50) + run GBT experiments
set -e
GPU=5
RESULT="./results/exp_results_k_sensitivity.csv"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COMMON="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --metric_save_path $RESULT --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 --loss BarlowTwins --encoder_type gbt"

echo "[$(date)] === Step 1: Build k-NN graphs ==="
python datasets/build_knn_graph.py --k 5 20 50

echo "[$(date)] === Step 2: Run experiments ==="
for K in 5 10 20 50; do
  KNN="HOFINET_KNN_BEHAV_k${K}"
  echo ""
  echo "===== k=$K ====="
  for SEED in 2024 2025 2026 2027; do
    echo "[$(date)] k=$K seed=$SEED (b)"
    python -u models/subgraph_cl.py --model_name ksens_gbt_b_k${K}_s${SEED} --gpu $GPU --seed $SEED $COMMON --knn_graph $KNN 2>&1 | grep "^(E):"

    echo "[$(date)] k=$K seed=$SEED (d)"
    python -u models/subgraph_cl.py --model_name ksens_gbt_d_k${K}_s${SEED} --gpu $GPU --seed $SEED $COMMON --knn_graph $KNN --subgraph_pool 2>&1 | grep "^(E):"
  done
done

echo ""
echo "[$(date)] === Complete ==="
wc -l $RESULT
