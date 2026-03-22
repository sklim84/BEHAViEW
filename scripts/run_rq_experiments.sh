#!/bin/bash
# =============================================================
# RQ1-RQ5 체계적 실험 (AML 논문용)
# 세션 독립 실행: nohup bash scripts/run_rq_experiments.sh > logs/rq_experiments.log 2>&1 &
# =============================================================
set -e
GPU=5
RESULT="./results/exp_results_rq.csv"
COMMON="--node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --skip_tsne --metric_save_path $RESULT"
HP="--lr 0.0005 --hidden_dim 256 --gconv_nlayers 2"
STRUCT="--struct_feats dc pagerank hits_hub hits_auth kcore triangle betweenness"

echo "============================================================"
echo "[$(date)] RQ Experiments Start (GPU=$GPU)"
echo "============================================================"

# =============================================================
# RQ1-RQ3: BGRL backbone, 4 seeds, 8 settings
# =============================================================
echo ""
echo "===== RQ1-RQ3: BGRL ====="
for SEED in 2024 2025 2026 2027; do
  echo ""
  echo "[$(date)] --- BGRL seed=$SEED ---"

  # RQ1 baseline: _w_org (no k-NN)
  echo "[$(date)] _w_org"
  python -u models/bgrl_w_org.py --model_name rq_bgrl_org_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BarlowTwins 2>&1 | grep "^(E):"

  # RQ1+RQ2: Behavioral k-NN (A, 8 feats)
  echo "[$(date)] behavioral_knn"
  python -u models/bgrl_w_knn.py --model_name rq_bgrl_behav_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BarlowTwins $STRUCT --knn_graph HOFINET_KNN_BEHAV_k10 2>&1 | grep "^(E):"

  # RQ2: Structural k-NN (B, 11 feats)
  echo "[$(date)] structural_knn"
  python -u models/bgrl_w_knn.py --model_name rq_bgrl_struct_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BarlowTwins $STRUCT --knn_graph HOFINET_KNN_STRUCT_k10 2>&1 | grep "^(E):"
done

# =============================================================
# RQ4: Homophily 측정
# =============================================================
echo ""
echo "===== RQ4: Homophily Measurement ====="
echo "[$(date)] Computing homophily ratios..."
python analysis/homophily_knn.py 2>&1

# =============================================================
# RQ5: 다른 GCL backbone (DGI-trs, MVGRL)
# =============================================================
echo ""
echo "===== RQ5: Other Backbones ====="

# DGI-transductive
for SEED in 2024 2025 2026 2027; do
  echo ""
  echo "[$(date)] --- DGI-trs seed=$SEED ---"

  echo "[$(date)] _w_org"
  python -u models/dgi_transductive_w_org.py --model_name rq_dgitrs_org_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss JSD 2>&1 | grep "^(E):"

  echo "[$(date)] behavioral_knn"
  python -u models/dgi_transductive_w_knn.py --model_name rq_dgitrs_behav_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss JSD $STRUCT --knn_graph HOFINET_KNN_BEHAV_k10 2>&1 | grep "^(E):"
done

# MVGRL
for SEED in 2024 2025 2026 2027; do
  echo ""
  echo "[$(date)] --- MVGRL seed=$SEED ---"

  echo "[$(date)] _w_org"
  python -u models/mvgrl_w_org.py --model_name rq_mvgrl_org_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BootstrapLatent 2>&1 | grep "^(E):"

  echo "[$(date)] behavioral_knn"
  python -u models/mvgrl_w_knn.py --model_name rq_mvgrl_behav_s${SEED} --gpu $GPU --seed $SEED $COMMON $HP --loss BootstrapLatent $STRUCT --knn_graph HOFINET_KNN_BEHAV_k10 2>&1 | grep "^(E):"
done

# =============================================================
# 결과 집계
# =============================================================
echo ""
echo "===== Results Summary ====="
echo "[$(date)] Total experiments:"
wc -l $RESULT
echo ""
echo "[$(date)] Per-model averages:"
python3 -c "
import csv
from collections import defaultdict
import statistics

with open('$RESULT') as f:
    rows = list(csv.DictReader(f))

groups = defaultdict(list)
for r in rows:
    m = r['Model']
    if not m.startswith('rq_'): continue
    # Extract setting: rq_bgrl_behav_s2024 → bgrl_behav
    parts = m.split('_')
    key = '_'.join(parts[1:-1])  # remove 'rq_' prefix and '_sXXXX' suffix
    groups[key].append(r)

for key in sorted(groups.keys()):
    vals = groups[key]
    f1s = [float(r['f1_1']) for r in vals]
    aurocs = [float(r['auroc']) for r in vals]
    auprcs = [float(r['auprc']) for r in vals]
    n = len(f1s)
    m_f1 = statistics.mean(f1s)
    s_f1 = statistics.stdev(f1s) if n > 1 else 0
    print(f'{key:25s} n={n}  f1_1={m_f1:.4f}±{s_f1:.4f}  auroc={statistics.mean(aurocs):.4f}  auprc={statistics.mean(auprcs):.4f}')
"

echo ""
echo "============================================================"
echo "[$(date)] RQ Experiments Complete!"
echo "Results: $RESULT"
echo "============================================================"
