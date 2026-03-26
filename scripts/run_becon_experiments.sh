#!/bin/bash
# BECON paper experiments — run on GPU 1
# Usage: bash scripts/run_becon_experiments.sh

set -e
cd /home/work/kftc_model/KA-003-FraudCenGCL

COMMON="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 --loss BarlowTwins --encoder_type gbt --knn_graph HOFINET_KNN_BEHAV_k10 --subgraph_pool --gpu 1"
SEEDS=(2024 2025 2026 2027)

echo "=========================================="
echo "Experiment A: Label Fraction"
echo "=========================================="
RATIOS=(0.01 0.05 0.10)
for ratio in "${RATIOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        model_name="labfrac_gbt_d_r${ratio}_s${seed}"
        echo "[A] Running $model_name ..."
        python models/subgraph_cl.py $COMMON \
            --model_name "$model_name" \
            --seed "$seed" \
            --train_ratio "$ratio" \
            --metric_save_path results/exp_results_label_fraction.csv \
            2>&1 | tail -3
        echo ""
    done
done

echo "=========================================="
echo "Experiment B: Feature Ablation for k-NN"
echo "=========================================="
# Wait for k-NN graphs if not ready
while [ ! -f datasets/HOFINET_KNN_FEAT_k10.csv ] || [ ! -f datasets/HOFINET_KNN_STRUCT_k10.csv ]; do
    echo "Waiting for k-NN graphs to be built..."
    sleep 30
done

# allfeats (FEAT) and struct runs — BEHAV already in k_sensitivity
declare -A KNN_TYPES=(
    ["allfeats"]="HOFINET_KNN_FEAT_k10"
    ["struct"]="HOFINET_KNN_STRUCT_k10"
)

COMMON_B="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 --loss BarlowTwins --encoder_type gbt --subgraph_pool --gpu 1"

for type_name in allfeats struct; do
    knn="${KNN_TYPES[$type_name]}"
    for seed in "${SEEDS[@]}"; do
        model_name="feat_gbt_d_${type_name}_s${seed}"
        echo "[B] Running $model_name (knn=$knn) ..."
        python models/subgraph_cl.py $COMMON_B \
            --model_name "$model_name" \
            --seed "$seed" \
            --knn_graph "$knn" \
            --metric_save_path results/exp_results_feature_ablation.csv \
            2>&1 | tail -3
        echo ""
    done
done

# Also run BEHAV for completeness in the feature_ablation CSV
for seed in "${SEEDS[@]}"; do
    model_name="feat_gbt_d_behav_s${seed}"
    echo "[B] Running $model_name (knn=HOFINET_KNN_BEHAV_k10) ..."
    python models/subgraph_cl.py $COMMON_B \
        --model_name "$model_name" \
        --seed "$seed" \
        --knn_graph HOFINET_KNN_BEHAV_k10 \
        --metric_save_path results/exp_results_feature_ablation.csv \
        2>&1 | tail -3
    echo ""
done

echo "=========================================="
echo "Experiment D: Computational Cost"
echo "=========================================="
python -c "
import time, os, sys, torch
sys.path.insert(0, '.')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 1. k-NN construction time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('datasets/HOFINET_NODE_FEAT.csv')
agg_cols = [c for c in df.columns if c.startswith(('out_', 'in_', 'md_', 'fnd_', 'entropy'))]
struct_cols_set = {'in_dc', 'out_dc', 'in_count', 'out_count'}
behav_cols = [c for c in agg_cols if c not in struct_cols_set]
X = df[behav_cols].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
norms = np.linalg.norm(X, axis=1, keepdims=True)
norms[norms == 0] = 1
X = X / norms

t0 = time.time()
nn = NearestNeighbors(n_neighbors=11, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
nn.fit(X)
distances, indices = nn.kneighbors(X)
knn_time = time.time() - t0

# 2. Training time (200 epochs)
from config import get_config
from data_loader import load_graph_data, load_knn_graph
from models.subgraph_cl import *

sys.argv = ['', '--node_data_name', 'HOFINET_NODE_FEAT', '--edge_data_name', 'HOFINET_EDGES',
            '--skip_tsne', '--lr', '0.0005', '--hidden_dim', '256', '--gconv_nlayers', '2',
            '--loss', 'BarlowTwins', '--encoder_type', 'gbt', '--knn_graph', 'HOFINET_KNN_BEHAV_k10',
            '--subgraph_pool', '--gpu', '0', '--seed', '2025',
            '--model_name', 'timing_test', '--metric_save_path', '/dev/null']
args = get_config()

device = torch.device('cuda:0')
data, x_struct = load_graph_data(args, device)
knn_edges = load_knn_graph(args, device)

edge_index_v1 = data.edge_index
edge_index_v2 = knn_edges

in_channels = data.x.size(1)
model, optimizer, loss_fn = build_model_and_opt(args, in_channels, device)

# Warm-up
for _ in range(5):
    model.train()
    z1, z2 = model(data.x, edge_index_v1, edge_index_v2)
    loss = loss_fn(z1, z2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()
t0 = time.time()
for _ in range(200):
    model.train()
    z1, z2 = model(data.x, edge_index_v1, edge_index_v2)
    loss = loss_fn(z1, z2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
train_time = time.time() - t0

# 3. Inference time
from GCL.eval import get_split
from utils import evaluate_with_metrics

torch.cuda.synchronize()
t0 = time.time()
z = get_embeddings(model, data.x, edge_index_v1, edge_index_v2)
z_cpu = z.detach().cpu()
torch.cuda.synchronize()
embed_time = time.time() - t0

t0 = time.time()
split = get_split(num_samples=z_cpu.size(0), train_ratio=0.1, test_ratio=0.8)
result = evaluate_with_metrics(z_cpu, data.y, split)
eval_time = time.time() - t0

print()
print('='*50)
print('Computational Cost Summary')
print('='*50)
print(f'k-NN construction (k=10, {len(X)} nodes): {knn_time:.1f}s')
print(f'Training (200 epochs):                     {train_time:.1f}s')
print(f'Embedding extraction:                      {embed_time:.3f}s')
print(f'Evaluation (LogReg):                       {eval_time:.1f}s')
print(f'Total pipeline:                            {knn_time + train_time + embed_time + eval_time:.1f}s')
print('='*50)
" 2>&1

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="

# Print summary tables
python -c "
import pandas as pd
import numpy as np

print()
print('='*60)
print('EXPERIMENT A: Label Fraction Results')
print('='*60)
try:
    df = pd.read_csv('results/exp_results_label_fraction.csv')
    df['ratio'] = df['Model'].str.extract(r'_r([\d.]+)_').astype(float)
    for ratio in sorted(df['ratio'].unique()):
        sub = df[df['ratio'] == ratio]
        print(f'  ratio={ratio:.2f}: F1_1={sub[\"f1_1\"].mean():.4f}+-{sub[\"f1_1\"].std():.4f}, '
              f'F1Ma={sub[\"F1Ma\"].mean():.4f}+-{sub[\"F1Ma\"].std():.4f}, '
              f'AUROC={sub[\"auroc\"].mean():.4f}+-{sub[\"auroc\"].std():.4f}, '
              f'AUPRC={sub[\"auprc\"].mean():.4f}+-{sub[\"auprc\"].std():.4f}')
except Exception as e:
    print(f'  Error: {e}')

print()
print('='*60)
print('EXPERIMENT B: Feature Ablation Results')
print('='*60)
try:
    df = pd.read_csv('results/exp_results_feature_ablation.csv')
    df['type'] = df['Model'].str.extract(r'feat_gbt_d_(\w+)_s')
    for t in ['allfeats', 'behav', 'struct']:
        sub = df[df['type'] == t]
        if len(sub) > 0:
            print(f'  {t:10s}: F1_1={sub[\"f1_1\"].mean():.4f}+-{sub[\"f1_1\"].std():.4f}, '
                  f'F1Ma={sub[\"F1Ma\"].mean():.4f}+-{sub[\"F1Ma\"].std():.4f}, '
                  f'AUROC={sub[\"auroc\"].mean():.4f}+-{sub[\"auroc\"].std():.4f}, '
                  f'AUPRC={sub[\"auprc\"].mean():.4f}+-{sub[\"auprc\"].std():.4f}')
except Exception as e:
    print(f'  Error: {e}')

print()
" 2>&1
