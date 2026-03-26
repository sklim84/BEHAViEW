# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BECON: Behavioral Subgraph Contrast for Anti-Money Laundering in Low-Homophily Transaction Networks.

Core idea: Construct a **behavioral k-NN graph** as a contrastive view (instead of standard graph augmentation) and apply **subgraph-level pooling** to amplify suspicious signals. Two independent axes — view construction and contrastive level — are systematically analyzed across 10 GCL encoders.

Datasets: HOFINET (452K nodes, 4.7M edges, 2.13% suspicious) and AMLworld HI-Small (515K nodes, 5.1M edges, 1.23% suspicious).

## Running Experiments

All commands run from the project root.

```bash
# (d) Proposed: behavioral k-NN view + subgraph pooling
python models/subgraph_cl.py \
  --encoder_type gbt --knn_graph HOFINET_KNN_BEHAV_k10 --subgraph_pool \
  --gpu 0 --seed 2025 --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2

# 4 settings via flags:
#   (a) baseline:        --encoder_type gbt
#   (b) +view:           --encoder_type gbt --knn_graph HOFINET_KNN_BEHAV_k10
#   (c) +level:          --encoder_type gbt --subgraph_pool
#   (d) +both (proposed): --encoder_type gbt --knn_graph ... --subgraph_pool

# Supervised baselines
python models/supervised_baselines.py --gpu 0 --dataset hofinet

# Experiment scripts
bash scripts/run_ablation_abcd.sh              # HOFINET 4-setting ablation
bash scripts/run_amlworld.sh                    # AMLworld experiments
bash scripts/run_hofinet_ab_multiseed.sh        # HOFINET (a)/(b) multi-seed
bash scripts/run_k_sensitivity_exp_only.sh      # k sensitivity (k=5,10,20,50)
bash scripts/run_all_additional_exp.sh          # Label fraction + feature ablation + cost

# k-NN graph construction
python datasets/build_knn_graph.py --k 10       # builds BEHAV/STRUCT/FEAT k-NN graphs

# Data preprocessing
python datasets/pp_hofinet.py                   # HOFINET raw → NODE_FEAT + EDGES
python datasets/pp_amlworld.py                  # AMLworld raw → NODE_FEAT + EDGES
```

Key arguments (defined in `config.py`):
- `--encoder_type`: gbt, bgrl, dgi, mvgrl, grace, gca, dgi_bn, mvgrl_bn, grace_bn, gin
- `--knn_graph`: k-NN graph name (e.g., HOFINET_KNN_BEHAV_k10)
- `--subgraph_pool`: enable subgraph pooling
- `--train_ratio`: train split ratio (default 0.1; val=same, test=1-2*train)
- `--hidden_dim` (256), `--gconv_nlayers` (2), `--lr` (0.0005)
- `--loss`: contrastive loss (BarlowTwins default, also InfoNCE, JSD, BootstrapLatent)
- `--skip_tsne`: skip t-SNE visualization

## Architecture

### BECON Pipeline

1. **Data loading** (`data_loader.py`): CSV → behavioral features (x_behav, 22 vars) + edge_index + optional k-NN graph
2. **k-NN view**: Behavioral features → StandardScaler → L2-norm → ball_tree k-NN → G_knn
3. **Dual-view encoding**: Shared GNN encoder (GCNConv + BN + PReLU + Dropout) processes both transaction graph and k-NN graph
4. **Subgraph pooling**: Mean pooling over ego-neighborhood on each view's graph
5. **Contrastive loss**: BYOL-style bootstrap loss with momentum target encoder
6. **Evaluation**: Frozen [h_trans ∥ h_knn] → LogisticRegression → F1_susp, AUROC, AUPRC

### 10 Encoder Variants

| Encoder | BN Type | Conv | Origin |
|---------|---------|------|--------|
| GBT | Per-layer | GCNConv | BarlowTwins |
| DGI+BN | Per-layer | GCNConv | DGI |
| MVGRL+BN | Per-layer | GCNConv | MVGRL |
| GRACE+BN | Per-layer | GCNConv | GRACE |
| BGRL | Final-only | GCNConv | BYOL |
| GIN | Per-layer | GINConv | GIN |
| DGI | None | GCNConv | DGI |
| MVGRL | None | GCNConv | MVGRL |
| GRACE | None | GCNConv | GRACE |
| GCA | None | GCNConv | GCA |

All encoders use the same BYOL-style bootstrap loss for fair comparison.

### Project Structure

```
_paper/                          # Paper source (LaTeX)
  figures/                       # Paper figures (PDF/SVG/PNG)
  reviews/                       # Reviewer feedback
models/
  subgraph_cl.py                 # Unified framework (10 encoders × 4 settings)
  supervised_baselines.py        # Supervised comparison (6 models)
datasets/
  build_knn_graph.py             # k-NN graph construction
  pp_hofinet.py                  # HOFINET preprocessing
  pp_amlworld.py                 # AMLworld preprocessing
analysis/
  homophily_knn.py               # S-S/S-B ratio measurement
scripts/
  run_ablation_abcd.sh           # HOFINET ablation
  run_amlworld.sh                # AMLworld experiments
  run_hofinet_ab_multiseed.sh    # Multi-seed (a)/(b)
  run_k_sensitivity_exp_only.sh  # k sensitivity
  run_all_additional_exp.sh      # Additional experiments
visualize/
  gen_paper_figures.py           # Paper figures (matplotlib)
  gen_intro_variants.py          # Intro figure variants
  gen_framework_svg.py           # Framework SVG figure
config.py                       # Shared argparse
data_loader.py                   # Data loading
utils.py                         # Shared utilities
results/                         # Experiment result CSVs
```

### Feature Taxonomy

| Category | Count | Features | GNN Input | k-NN |
|----------|-------|----------|:---------:|:----:|
| Amount stats | 6 | out/in_mean, _max, _std | ✓ | ✓ |
| Temporal | 12 | out/in_{3,6,12}m_mean, _count | ✓ | ✓ |
| Entropy | 2 | md_type_entropy, fnd_type_entropy | ✓ | ✓ |
| Count | 2 | out_count, in_count | ✓ | |
| Degree | 2 | in_dc, out_dc | ✓ | |
| Structural | 9 | dc, pagerank, betweenness, hits_*, kcore, triangle | | |

## Known Issues

- **PyGCL patches**: `GCL/utils.py` has lazy `import dgl`; `GCL/augmentors/functional.py` has try/except for torch_sparse/torch_scatter
- **Git LFS**: Large CSV files (HOFINET_*.csv) use Git LFS. Run `git lfs install && git lfs pull` after cloning
- **PYTORCH_CUDA_ALLOC_CONF**: Use `expandable_segments:True` for large graphs

## Paper

- Title: BECON: Behavioral Subgraph Contrast for Anti-Money Laundering in Low-Homophily Transaction Networks
- Framework name: `\method` (BECON\xspace) in LaTeX
- Style: Korean draft body, English section names and captions
- No `\textbf` in body text; `\texttt` only for framework name
- Terminology: suspicious (not fraud), F1_susp (not F1_fraud), S-S/S-B (not F-F/F-B)

## Git

- Repository: github.com/sklim84/KA-003-FraudCenGCL
- Branch: main
- User: sklim84 / captiong84@gmail.com
