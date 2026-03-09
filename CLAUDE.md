# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FraudCenGCL: Graph Contrastive Learning for financial fraud detection using network centrality features as an auxiliary contrastive view.

Core idea: Instead of standard graph augmentation for both contrastive views, use **node aggregate features** (x_agg) as one view and **graph centrality features** (x_cen) as the other view. This replaces traditional augmentations like edge removal/feature masking.

Dataset: HOFINET — 452K nodes, 2.56M directed edges, 2.13% fraud rate.

## Running Experiments

All commands run from the project root.

```bash
# Single model (with centrality / baseline)
python models/grace_w_cen.py --node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --gpu 0
python models/grace_w_org.py --node_data_name HOFINET_NODE_FEAT --edge_data_name HOFINET_EDGES --gpu 0

# All 12 models comparison (fixed HP)
bash scripts/ALL_MODEL_rq5.sh

# Hyperparameter search (3 presets)
python scripts/hp_search.py --preset baseline           # 7 features, 6 GPU, 432 jobs
python scripts/hp_search.py --preset 17feat             # 17 features, 1 GPU, 216 jobs
python scripts/hp_search.py --preset v2 --wait_gpu_idle # 3 feature sets, 6 GPU, 1296 jobs

# Custom HP search
python scripts/hp_search.py --cen_feats dc cc pagerank --num_gpus 2 --lr 1e-4 5e-4

# Data preprocessing (from raw HOFINET.csv)
python datasets/pp_hofinet.py                   # hybrid: cuGraph(GPU) + Memgraph
python datasets/pp_hofinet.py --gpu_only        # cuGraph only (9 features)
python datasets/pp_hofinet.py --cpu_only        # NetworkX only (no GPU/Memgraph)

# Standalone feature computation
python datasets/compute_features_hybrid.py --gpu 0              # all phases
python datasets/compute_features_hybrid.py --memgraph_only      # Memgraph only
python datasets/compute_features_hybrid.py --skip_memgraph      # cuGraph + NetworkX

# Orchestration: feature discovery → GPU idle → HP search
nohup python -u scripts/run_features_and_hp.py --gpu 5 > logs/run_all.log 2>&1 &
```

Key arguments (defined in `config.py`):
- `--model_name`: identifier for result tracking
- `--node_data_name` / `--edge_data_name`: CSV filenames (without .csv) in `datasets/`
- `--cen_feats`: centrality features list (default: dc cc pagerank hits_hub hits_auth kcore triangle)
- `--loss`: contrastive loss (`InfoNCE`, `JSD`, `BarlowTwins`, `BootstrapLatent`)
- `--input_dim` (16), `--hidden_dim` (256), `--proj_dim` (32), `--gconv_nlayers` (3), `--lr` (0.001)
- `--skip_tsne`: skip t-SNE visualization (for HP search)

## Architecture

### Dual-View Contrastive Learning Pipeline

Every `_w_cen` model:
1. **Data loading** (`data_loader.py`): CSV → `x_agg` (out_\*, in_\*, md_\*, fnd_\*, entropy\*) + `x_cen` (centrality features from `--cen_feats`) + edge_index
2. **Projection**: `proj_agg` / `proj_cen` linear layers → `input_dim`
3. **Augmentation**: GCL augmentors (EdgeRemoving, FeatureMasking) per view
4. **Encoding**: Shared GNN encoder on both projected views
5. **Contrastive loss**: Between the two view embeddings
6. **Evaluation**: Frozen embeddings → LogisticRegression (10% train, 80% test) → F1, AUROC, AUPRC + t-SNE with ARI/Silhouette

The `_w_org` models are baselines using only node features with standard dual-augmentation.

### Model Variants (6 architectures × 2 variants = 12 models)

| Model | Contrast Type | GNN | Epochs | Loss | Key Difference |
|-------|--------------|-----|--------|------|----------------|
| GRACE | DualBranch L2L | GCNConv | 1000 | InfoNCE | NeighborLoader, projection head |
| GBT | WithinEmbed | GCNConv (2-layer BN+PReLU) | 4000 | BarlowTwins | No projection head |
| BGRL | Bootstrap L2L | GCNConv + dropout/BN | 100 | BarlowTwins | Momentum target encoder |
| DGI-TRS | SingleBranch G2L | GCNConv + PReLU | 300 | JSD | Corruption-based, averages dual projections |
| DGI-IND | SingleBranch G2L | SAGEConv | 30 | JSD | Inductive NeighborSampler |
| MVGRL | DualBranch G2L | GCNConv | 200 | BootstrapLatent | Two separate encoders, EdgeRemoving(0.3) |

### Shared Modules

| File | Purpose |
|------|---------|
| `config.py` | Centralized argparse for all models |
| `data_loader.py` | `load_graph_data(args, device)` → `(Data, x_cen)` |
| `utils.py` | `set_seed()`, `create_loss()`, `evaluate_with_metrics()`, `save_results_to_csv()`, `visualize_tsne()` |
| `scripts/hp_common.py` | `MODELS` config, `run_parallel()`, `build_command()`, `load_completed_jobs()`, GPU idle monitoring |
| `datasets/feature_utils.py` | `run_feature()`, `compute_feature_stats()`, `save_feature_progress()` |

### Data Pipeline

**`datasets/pp_hofinet.py`**: HOFINET.csv (Korean columns) → English mapping → node feature aggregation → graph centrality computation → `HOFINET_NODE_FEAT.csv` + `HOFINET_EDGES.csv`

3 execution modes: hybrid (cuGraph GPU + Memgraph), `--gpu_only` (cuGraph), `--cpu_only` (NetworkX)

**`datasets/compute_features_hybrid.py`**: Standalone 3-phase feature computation:
- Phase 1 (cuGraph GPU, seconds): dc, in_dc, out_dc, pagerank, hits, katz, eigenvector, kcore, triangle, betweenness, louvain
- Phase 2 (Memgraph CPU, minutes): clustering, sq_clustering, avg_neigh_deg, greedy_color
- Phase 3 (NetworkX CPU, slow): cc, harmonic, load_cen, voterank, constraint, eff_size

Flags: `--skip_cugraph`, `--skip_memgraph`, `--skip_networkx`, `--memgraph_only`

HOFINET.csv column mapping: 거래일자→tran_dt, 출금금융회사일련번호→wd_fc_sn, 출금계좌일련번호→wd_ac_sn, 입금금융회사일련번호→dps_fc_sn, 입금계좌일련번호→dps_ac_sn, 자금구분→fnd_type, 매체구분→md_type, 거래금액→tran_amt, 이상거래여부→label(0/1)

### HP Search Architecture

`scripts/hp_search.py` uses 3 presets defined in `PRESETS` dict. Each preset specifies `cen_feats_sets`, `search_space`, `result_file`, `num_gpus`. All settings can be overridden via CLI.

Job execution via `hp_common.run_parallel()`: deque-based GPU scheduling, subprocess per job, incremental CSV results, `load_completed_jobs()` for resume support. Model names are normalized to lowercase for resume matching.

Results CSV schema: timestamp, Model, Data, Seed, cen_feats, lr, input_dim, hidden_dim, proj_dim, gconv_nlayers, loss, pre_0/1, rec_0/1, f1_0/1, F1Mi, F1Ma, auroc, auprc, ari_score, sil_score

### Other Directories

- `benchmarks/`: GCL library reference implementations (Planetoid, WikiCS, TUDataset) — not used
- `analysis/`: Network statistics, centrality distributions, homophily analysis
- `results/`: Experiment result CSVs
- `visualize/`: t-SNE plots per model

## Known Issues

- **cuGraph eigenvector_centrality**: Fails to converge on HOFINET graph. Use Memgraph's `eigenvector_centrality_numpy` instead.
- **CUDA_VISIBLE_DEVICES**: Must be restored after cuGraph usage (handled in code via `os.environ.pop`), otherwise GPU 1-5 become invisible.
- **Memgraph query.timeout**: Default 600s is insufficient. Set to 7200s: `SET DATABASE SETTING "query.timeout" TO "7200"`
- **Memgraph module reload**: Fails if active transactions exist. Run `TERMINATE TRANSACTIONS` first.
- **PyGCL patch**: `GCL/utils.py` has `import dgl` changed to lazy import to avoid dependency conflicts.

## Memgraph Setup

Memgraph v3.8.1 installed via .deb extraction (no sudo).

- **Binary**: `/home/work/kftc_sklim/memgraph/extracted/usr/lib/memgraph/memgraph`
- **Data**: `/home/work/kftc_sklim/memgraph/data`
- **Query modules**: `.../query_modules/` — custom `graph_features.py` (15 procedures) + built-in `nxalg.py`

```bash
# Start Memgraph
export LD_LIBRARY_PATH=/home/work/kftc_sklim/memgraph/extracted/usr/lib/memgraph:$LD_LIBRARY_PATH
nohup /home/work/kftc_sklim/memgraph/extracted/usr/lib/memgraph/memgraph \
  --bolt-port 7687 \
  --data-directory /home/work/kftc_sklim/memgraph/data \
  --log-file /home/work/kftc_sklim/memgraph/log/memgraph.log \
  --query-modules-directory /home/work/kftc_sklim/memgraph/extracted/usr/lib/memgraph/query_modules \
  --storage-properties-on-edges=true --log-level WARNING --telemetry-enabled=false \
  > /home/work/kftc_sklim/memgraph/log/stdout.log 2>&1 &

# Reproduce on another server
mkdir -p ~/memgraph && cd ~/memgraph
curl -L -o memgraph.deb https://download.memgraph.com/memgraph/v3.8.1/ubuntu-24.04/memgraph_3.8.1-1_amd64.deb
dpkg-deb -x memgraph.deb ./extracted && pip install neo4j
```

## Git

- Repository: github.com/sklim84/KA-003-FraudCenGCL
- Branch: main
- User: sklim84 / captiong84@gmail.com
