# BEHAViEW: Self-Supervised Money Laundering Detection via Behavioral Topology Repair

In anti-money laundering (AML) transaction graphs, suspicious accounts have predominantly benign neighbors, so graph neural network (GNN) aggregation dilutes a rare suspicious signal. We trace this to a *topology-behavior mismatch*: the evasion-shaped transaction network fails to reflect homophily within the suspicious class.

**BEHAViEW** is a self-supervised graph contrastive learning framework that performs **topology repair through behavioral homophily recovery**. It builds a topology-independent recovered neighborhood graph from behavioral similarity (e.g., transaction amount and timing patterns) and aligns it with the transaction graph through contrastive learning. Across three AML datasets, the repair compresses the suspicious-edge S-S:S-B ratio (suspicious-to-suspicious vs suspicious-to-benign edges) from 1:5.7–15.8 to 1:1.3–9.4. At 1% labels, BEHAViEW attains the highest suspicious-class F1 on all three datasets, with a 0.010–0.016 gap over the strongest supervised baseline.

---

## Two design axes

| Axis | Standard GCL | BEHAViEW |
|---|---|---|
| **View construction** | augmentations of the same graph | **behavior-only k-NN graph** (links behaviorally similar accounts) |
| **Contrastive level** | node-level only | **subgraph pooling** (ego + neighbors) |

The 4-setting ablation:

| | Transaction view | Behavior-recovered view |
|---|:---:|:---:|
| **Node-level** | (a) baseline | (b) +view |
| **Subgraph pool** | (c) +level | **(d) proposed ★** |

---

## Research questions (current paper)

| RQ | Question | Key finding (ATNet) |
|---|---|---|
| **RQ1** Homophily Recovery | Does the recovered graph carry class-relevant evidence? | edge homophily 0.690 → 0.981; S-S:S-B 1:5.7 → 1:1.4 |
| **RQ2** Signal Preservation | Does pooling on the repaired topology preserve or dilute the signal? | sign-flip threshold separates amplification from dilution; (c) hurts by 17–30%, (d) wins |
| **RQ3** Label-Efficiency | Is the self-supervised representation competitive under label scarcity? | best F1_susp at 1% labels on all three datasets; +0.010–0.016 over the strongest baseline |
| **RQ4** Cross-dataset Robustness | Do the patterns transfer across AML prevalences? | topology repair dominates on AMLworld (ρ=1.23%) and AMLNet (ρ=13.52%): (b),(d) ≫ (a),(c); the (d)-over-(b) pooling gain is conditional |

---

## Datasets

| | ATNet | AMLworld HI-Small | AMLNet |
|---|---|---|---|
| Source | Real-derived synthetic interbank transfers (40 months) | [NeurIPS 2023 benchmark](https://arxiv.org/abs/2306.16424) | Huda et al., Expert Systems with Applications 2025 |
| Nodes | 452,816 | 515,088 | 11,000 |
| Edges | 4,732,130 (directed multi-edges) | 5,078,345 | 1,090,172 |
| Suspicious ratio ρ | 2.13% | 1.23% | 13.52% |

**Availability.** ATNet is a real-derived synthetic dataset and cannot be shared under its data-sharing agreement. The other two are public: AMLworld HI-Small from the [NeurIPS 2023 benchmark](https://arxiv.org/abs/2306.16424) (IBM Transactions for Anti-Money Laundering, on Kaggle), and AMLNet from Huda et al., *Expert Systems with Applications* 2025. AMLworld and AMLNet experiments are fully reproducible from these public sources.

---

## Selected results

### Topology repair (Table 4 in the paper)

| Dataset | Graph | Homophily | S-S:S-B |
|---|---|---|---|
| ATNet | Transaction | 0.690 | 1:5.7 |
| ATNet | Behavioral repair | **0.981** | **1:1.4** |
| AMLworld | Transaction | 0.886 | 1:15.8 |
| AMLworld | Behavioral repair | **0.980** | **1:9.4** |
| AMLNet | Transaction | 0.791 | 1:7.2 |
| AMLNet | Behavioral repair | **0.900** | **1:1.3** |

### 4-setting ablation on ATNet (F1_susp, 4-seed mean±std)

Selected encoders (full table in `results/rq1/main_sweep.csv`):

| Encoder | (a) Tx + node | (b) Recovered + node | (c) Tx + pool | (d) Repaired + pool |
|---|---|---|---|---|
| GBT | 0.265±.011 | 0.664±.002 | 0.197±.002 | **0.667±.002** |
| DGI+BN | 0.266±.010 | 0.664±.002 | 0.198±.002 | **0.667±.001** |
| BGRL | 0.306±.013 | 0.542±.012 | 0.252±.015 | **0.629±.005** |
| GIN | 0.181±.022 | 0.591±.017 | 0.127±.001 | **0.590±.012** |
| DGI (no BN) | 0.044±.002 | 0.054±.007 | 0.048±.004 | **0.068±.007** |

### Label-efficiency at 1% labels (BEHAViEW vs. supervised baselines)

| Dataset | BEHAViEW (d) | Best supervised baseline | Gap |
|---|---|---|---|
| ATNet | **0.647±.007** | XGBoost 0.633±.015 | +0.014 |
| AMLworld | **0.062±.014** | GAT 0.046±.002 | +0.016 |
| AMLNet | **0.632±.081** | MLP 0.622±.085 | +0.010 |

Full tables for label fractions {1%, 5%, 10%}: `results/rq3/`.

---

## Setup

Python 3.10+ and a CUDA GPU are recommended.

```bash
pip install -r requirements.txt
```

Under PyTorch 2.10+, PyGCL 0.1.2 needs four small patches (lazy `dgl` / `RWSampling` imports and optional `torch_sparse` / `torch_scatter`); the exact files are listed in `requirements.txt`. For the larger graphs (AMLworld has 515K nodes), set `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation; the scripts under `scripts/` already export this. Public datasets are not bundled: download AMLworld and AMLNet (see the Datasets table) into `datasets/amlworld/` and `datasets/amlnet/`, then run the matching `datasets/pp_*.py` preprocessing and `python datasets/build_knn_graph.py --k 10`.

---

## Quick start

```bash
# (d) proposed: behavior-recovered view + subgraph pooling (AMLworld, public)
python models/subgraph_cl.py \
    --encoder_type gbt \
    --node_data_name amlworld/AMLWORLD_NODE_FEAT \
    --edge_data_name amlworld/AMLWORLD_EDGES \
    --knn_graph amlworld/AMLWORLD_KNN_BEHAV_k10 \
    --subgraph_pool \
    --gpu 0 --seed 2025 \
    --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2

# 4-setting ablation (toggle topology and level on the same command):
#   (a) baseline:  omit --knn_graph and --subgraph_pool
#   (b) +view:     --knn_graph amlworld/AMLWORLD_KNN_BEHAV_k10
#   (c) +level:    --subgraph_pool
#   (d) proposed:  --knn_graph amlworld/AMLWORLD_KNN_BEHAV_k10 --subgraph_pool

# Supervised baselines used in the paper: XGBoost, LightGBM, MLP, GCN, GAT, CARE-GNN.
# Heterophily-aware models (MixHop, FAGCN, H2GCN, ACM-GCN) and fraud-specific GNNs
# (BWGNN, GAGA, PCGNN) are also implemented in models/supervised_baselines.py.
python models/supervised_baselines.py --gpu 0 --dataset amlworld

# Main table sweep (9 encoders × 4 settings × 4 seeds × dataset)
GPU=0 DATASETS=amlworld bash scripts/run_main_table.sh &
GPU=1 DATASETS=amlnet   bash scripts/run_main_table.sh &
wait

# Build the behavioral k-NN graph (one-time preprocessing)
python datasets/build_knn_graph.py --k 10
```

Each run appends metrics (`F1_susp`, AUROC, AUPRC) to `--metric_save_path` (default `results/exp_results.csv`); the sweep scripts write per-RQ CSVs under `results/rq*/`.

Key arguments (see `models/config.py`):
- `--encoder_type`: gbt, bgrl, dgi, mvgrl, grace, gca, dgi_bn, mvgrl_bn, grace_bn, gin
- `--knn_graph`: k-NN graph name (e.g., `amlworld/AMLWORLD_KNN_BEHAV_k10`)
- `--subgraph_pool`: enable subgraph pooling
- `--loss`: contrastive loss (`BootstrapLatent` default, also `InfoNCE`, `JSD`, `BarlowTwins`)
- `--train_ratio`: train split ratio (default 0.1; val = same; test = 1 − 2 · train)

---

## Project layout

```
models/
  subgraph_cl.py                 # Main BEHAViEW training (9 encoders × 4 settings)
  supervised_baselines.py        # Supervised comparison (10+ baselines)
  config.py, data_loader.py, utils.py
datasets/
  atnet/                       # ATNet: real-derived synthetic interbank dataset (not shared)
  amlworld/                      # AMLworld HI-Small (NeurIPS 2023)
  amlnet/                        # AMLNet
  build_knn_graph.py             # Builds the behavioral / structural / feature k-NN graphs
  pp_{atnet,amlworld,amlnet}.py  # Per-dataset preprocessing
scripts/
  run_main_table.sh              # Multi-GPU main table dispatcher
  rq{1,2,3,4}/                   # RQ-specific scripts (HP sweeps, t-SNE, etc.)
  appendix/                      # Loss-substitution and feature-ablation scripts
analysis/
  homophily_knn.py               # S-S vs S-B ratio measurement utilities
results/
  rq{1,2,3,4}/                   # Per-RQ result CSVs (see results/README.md)
  embeddings/                    # Cached embeddings for downstream visualization
  appendix/                      # Appendix-table CSVs
visualize/
  gen_fig1_embedding.py          # Embedding / t-SNE figure generation
  method_visualizer/             # Streamlit app for interactive 3D method comparison
```

---

## Related work

| Paper | Venue | Relation |
|---|---|---|
| [MLGCL](https://arxiv.org/abs/2107.02639) | Neurocomputing 2023 | k-NN view for general GCL |
| [GCPAL](https://doi.org/10.1007/s44196-024-00720-4) | IJCIS 2024 | k-NN view for AML |
| [SUBG-CON](https://arxiv.org/abs/2009.10564) | ICDM 2020 | Subgraph contrastive learning |
| [AMLworld](https://arxiv.org/abs/2306.16424) | NeurIPS 2023 | AML synthetic benchmark |

