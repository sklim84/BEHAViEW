# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FraudCenGCL: Graph Contrastive Learning for financial fraud detection using network centrality features as an auxiliary contrastive view. The goal is to enhance the FraudCenGCL paper for journal submission using HOFINET.csv data.

Core idea: Instead of standard graph augmentation (edge removal, feature masking) for both contrastive views, use **node aggregate features** as one view and **graph centrality features** (degree, closeness, betweenness) as the other view.

## Running Experiments

All commands run from the project root directory.

```bash
# Run a single model (with centrality features)
python models/grace_w_cen.py --node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --gpu 0

# Run a single model (baseline without centrality)
python models/grace_w_org.py --node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --gpu 0

# Run all models comparison experiment
bash scripts/ALL_MODEL_rq5.sh
```

Key arguments (defined in `config.py`):
- `--model_name`: model identifier for result tracking
- `--node_data_name` / `--edge_data_name`: CSV filenames (without .csv) in `datasets/`
- `--cen_feats`: centrality features to use, e.g. `"dc cc bc"`
- `--loss`: contrastive loss function (`InfoNCE`, `JSD`, `BarlowTwins`, `BootstrapLatent`)
- `--input_dim`, `--hidden_dim`, `--proj_dim`, `--gconv_nlayers`, `--lr`: hyperparameters

## Architecture

### Dual-View Contrastive Learning Pattern

Every `_w_cen` model follows the same pattern:
1. **Data loading** (`data_loader.py`): CSV → `x_agg` (node features: out_*, in_*, md_*, fnd_*, entropy*) + `x_cen` (centrality: dc, cc, bc) + edge_index
2. **Projection**: `proj_agg` and `proj_cen` linear layers align feature dimensions to `input_dim`
3. **Augmentation**: Separate augmentors applied to each view
4. **Encoding**: Shared GCN encoder processes both projected views
5. **Contrastive loss**: Computed between the two view embeddings
6. **Evaluation**: Frozen embeddings → LogisticRegression → F1, AUROC, AUPRC + t-SNE visualization with ARI/Silhouette

The `_w_org` models are baselines using only node features with standard dual-augmentation.

### Model Variants (6 architectures x 2 variants = 12 models in `models/`)

| Model | Contrast Type | GNN | Epochs | Key Difference |
|-------|--------------|-----|--------|----------------|
| GRACE | DualBranch L2L | GCNConv | 1000 | Mini-batch NeighborLoader, projection head |
| GBT | WithinEmbed | GCNConv (2-layer BN+PReLU) | 4000 | BarlowTwins default, no projection head |
| BGRL | Bootstrap L2L | GCNConv + dropout/BN | 100 | Momentum target encoder, predictor head |
| DGI-TRS | SingleBranch G2L | GCNConv + PReLU per-layer | 300 | Corruption-based, averages dual projections |
| DGI-IND | SingleBranch G2L | SAGEConv | 30 | Inductive with NeighborSampler |
| MVGRL | DualBranch G2L | GCNConv | 200 | Two separate encoders, PPRDiffusion augmentor |

### Common Utilities

- `config.py`: Centralized argparse for all models
- `data_loader.py`: `load_graph_data(args, device)` → returns `(Data, x_cen)`
- `utils.py`: `set_seed()`, `create_loss()`, `build_result_dict()`, `save_results_to_csv()`, `evaluate_with_metrics()`, `visualize_tsne()`

### Data Pipeline (`datasets/`)

**Main pipeline**: `pp_hofinet.py` — HOFINET.csv (한글 컬럼) → 컬럼 영문 변환 → source/target 생성 → 노드 피처 집계 → 중심성 계산 → `HOFINET_NODE_FEAT.csv` + `HOFINET_EDGES.csv`

3가지 실행 모드 지원:

```bash
# CPU only (NetworkX)
python datasets/pp_hofinet.py

# GPU 가속 (cuGraph)
python datasets/pp_hofinet.py --use_gpu

# Memgraph 사용
python datasets/pp_hofinet.py --use_memgraph
python datasets/pp_hofinet.py --use_memgraph --mg_host 192.168.1.10 --mg_port 7687
```

그래프 피처 목록: dc, cc, pagerank, hits_hub, hits_auth, katz, eigenvector, kcore, triangle

HOFINET.csv 컬럼 매핑: 거래일자→tran_dt, 출금금융회사일련번호→wd_fc_sn, 출금계좌일련번호→wd_ac_sn, 입금금융회사일련번호→dps_fc_sn, 입금계좌일련번호→dps_ac_sn, 자금구분→fnd_type, 매체구분→md_type, 거래금액→tran_amt, 이상거래여부→label(0/1)

Legacy scripts (이전 HF_TRNS_TRAN 데이터용): `pp_00_fnd_type.py`, `pp_01_data_divider.py`, `pp_02_sp_sampling.py`, `pp_centrality.py`, `pp_03_*.py`

### Other Directories

- `benchmarks/`: Original GCL library reference implementations (Planetoid, WikiCS, TUDataset) — not used in experiments
- `scripts/`: Shell scripts for hyperparameter sweeps and model comparisons
- `analysis/`: Network statistics, centrality distribution analysis, homophily analysis
- `results/`: Experiment result CSVs with metrics per run
- `visualize/`: t-SNE plots and hyperparameter sensitivity heatmaps

## Memgraph 설정

Memgraph v3.8.1이 로컬에 설치/기동됨 (sudo 없이 .deb 추출 방식)

- **바이너리**: `/home/work/kftc_sklim/memgraph/extracted/usr/lib/memgraph/memgraph`
- **데이터**: `/home/work/kftc_sklim/memgraph/data`
- **로그**: `/home/work/kftc_sklim/memgraph/log/`
- **쿼리 모듈**: `/home/work/kftc_sklim/memgraph/extracted/usr/lib/memgraph/query_modules/`
- **커스텀 모듈**: `graph_features.py` — degree_centrality, hits, katz_centrality, eigenvector_centrality, triangle_count, closeness_centrality
- **내장 모듈**: `nxalg.py` — pagerank, core_number

기동 명령:
```bash
export LD_LIBRARY_PATH=/home/work/kftc_sklim/memgraph/extracted/usr/lib/memgraph:$LD_LIBRARY_PATH
nohup /home/work/kftc_sklim/memgraph/extracted/usr/lib/memgraph/memgraph \
  --bolt-port 7687 \
  --data-directory /home/work/kftc_sklim/memgraph/data \
  --log-file /home/work/kftc_sklim/memgraph/log/memgraph.log \
  --query-modules-directory /home/work/kftc_sklim/memgraph/extracted/usr/lib/memgraph/query_modules \
  --storage-properties-on-edges=true \
  --log-level WARNING \
  --telemetry-enabled=false \
  > /home/work/kftc_sklim/memgraph/log/stdout.log 2>&1 &
```

다른 서버에서 Memgraph 설정 재현:
```bash
mkdir -p ~/memgraph && cd ~/memgraph
curl -L -o memgraph.deb https://download.memgraph.com/memgraph/v3.8.1/ubuntu-24.04/memgraph_3.8.1-1_amd64.deb
dpkg-deb -x memgraph.deb ./extracted
pip install neo4j
# graph_features.py를 query_modules/에 복사 후 기동
```

## Git

- Repository: github.com/sklim84/KA-003-FraudCenGCL
- Branch: main
- User: sklim84 / captiong84@gmail.com
