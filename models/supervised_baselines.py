"""
Supervised baseline models for AML comparison.
- GCN, GAT, GraphSAGE (supervised GNN)
- LightGBM, XGBoost (tabular)
- MLP (neural network)

동일 10%/80% train/test split, F1_susp metric.

사용법:
    python models/supervised_baselines.py --gpu 0 --seed 2025
    python models/supervised_baselines.py --gpu 0 --seed 2025 --dataset amlworld
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import FAConv, GCNConv, GATConv, MixHopConv, SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from config import get_config
from data_loader import load_graph_data
from utils import set_seed, make_split, load_split


# ============================================================
# Supervised GNN Models
# ============================================================
class SupervisedGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class SupervisedGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(GATConv(input_dim, hidden_dim // heads, heads=heads))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class SupervisedSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class SupervisedMixHop(nn.Module):
    """MixHop baseline for heterophily-aware propagation.

    MixHop mixes multiple adjacency powers in each layer, allowing the model to
    combine ego, 1-hop, and 2-hop signals instead of relying on one smoothing
    channel.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2, powers=(0, 1, 2)):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.proj = nn.ModuleList()
        in_dim = input_dim
        out_dim = hidden_dim * len(powers)
        for _ in range(num_layers):
            self.layers.append(MixHopConv(in_dim, hidden_dim, powers=list(powers)))
            self.bns.append(nn.BatchNorm1d(out_dim))
            self.proj.append(nn.Linear(out_dim, hidden_dim))
            in_dim = hidden_dim
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn, proj in zip(self.layers, self.bns, self.proj):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(proj(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class SupervisedFAGCN(nn.Module):
    """FAGCN baseline with adaptive low/high-frequency aggregation."""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([FAConv(hidden_dim, eps=0.1, dropout=dropout) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x0 = F.dropout(F.relu(self.input_proj(x)), p=self.dropout, training=self.training)
        h = x0
        for conv, bn in zip(self.layers, self.bns):
            h = conv(h, x0, edge_index)
            h = F.relu(bn(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


class SupervisedH2GCN(nn.Module):
    """H2GCN-style baseline with separated ego, 1-hop, and 2-hop channels."""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.one_hop = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=False) for _ in range(num_layers)
        ])
        self.two_hop = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=False) for _ in range(num_layers)
        ])
        self.mix = nn.ModuleList([nn.Linear(hidden_dim * 3, hidden_dim) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x))
        for conv1, conv2, mix, bn in zip(self.one_hop, self.two_hop, self.mix, self.bns):
            h1 = conv1(h, edge_index)
            h2 = conv2(h1, edge_index)
            h = mix(torch.cat([h, h1, h2], dim=1))
            h = F.relu(bn(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


class SupervisedACMGCN(nn.Module):
    """ACM-GCN-style baseline with adaptive low/high/MLP channel mixing."""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.low_convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.high_convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.self_lins = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.mlp_lins = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_dim, 3) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x))
        for low_conv, high_conv, self_lin, mlp_lin, gate, bn in zip(
            self.low_convs, self.high_convs, self.self_lins, self.mlp_lins, self.gates, self.bns
        ):
            low = low_conv(h, edge_index)
            high = self_lin(h) - high_conv(h, edge_index)
            mlp = mlp_lin(h)
            alpha = torch.softmax(gate(h), dim=1)
            h = alpha[:, 0:1] * low + alpha[:, 1:2] * high + alpha[:, 2:3] * mlp
            h = F.relu(bn(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


class SupervisedMLP(nn.Module):
    """Standard (bare) MLP baseline: stacked Linear + ReLU. No BatchNorm, no
    Dropout. Matches the convention used in AML/fraud-GNN literature
    (CARE-GNN, PC-GNN, BWGNN, GAGA) where "MLP" denotes a plain feedforward
    network without modern regularization tricks."""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.net(x)


class CAREGNN(nn.Module):
    """CARE-GNN (Dou et al. CIKM 2020) simplified single-relation variant.

    Core idea: similarity-aware neighbor filtering. Edges are weighted by
    cosine similarity of endpoint representations, so dissimilar (likely
    camouflage) neighbors contribute less to aggregation. Equivalent to
    soft version of CARE-GNN's RL-learned top-theta neighbor selection.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, add_self_loops=False, normalize=False))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv, bn in zip(self.layers, self.bns):
            # Compute per-edge cosine similarity (clamped to [0,1])
            src, tgt = edge_index[0], edge_index[1]
            sim = F.cosine_similarity(x[src], x[tgt], dim=1).clamp(min=0.0)
            x = conv(x, edge_index, edge_weight=sim)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class PCGNN(nn.Module):
    """PC-GNN (Liu et al. WWW 2021) simplified single-relation variant.

    Core idea: label-distribution-aware neighborhood sampling. During
    training, edges between two majority-class nodes are randomly dropped
    so that the GNN sees a relatively balanced ego-neighborhood. Minority
    nodes always keep all their neighbors (high-recall) ; majority-majority
    edges keep with prob p_keep (default 0.3, roughly matching prevalence
    ratio for HOFINET 2%-prevalence).
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2, p_keep_mm=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout
        self.p_keep_mm = p_keep_mm
        self._y_train = None  # set externally via set_train_labels()

    def set_train_labels(self, y_train, train_mask):
        """Provide training labels (full vector, -1 for non-train) for sampling."""
        self._y_train = y_train.clone()
        self._y_train[~train_mask] = -1  # mark non-train as -1 to keep their edges by default

    def _sample_edges(self, edge_index):
        if self._y_train is None or not self.training:
            return edge_index
        src, tgt = edge_index[0], edge_index[1]
        y_src = self._y_train[src]
        y_tgt = self._y_train[tgt]
        # Edge is "majority-majority" iff both endpoints labeled 0 in train
        mm_mask = (y_src == 0) & (y_tgt == 0)
        # Random keep mask for mm edges; non-mm edges always kept
        rnd = torch.rand(edge_index.size(1), device=edge_index.device)
        keep = (~mm_mask) | (rnd < self.p_keep_mm)
        return edge_index[:, keep]

    def forward(self, x, edge_index):
        ei = self._sample_edges(edge_index)
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, ei)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class BWGNN(nn.Module):
    """BWGNN (Tang et al. ICML 2022) simplified single-relation variant.

    Core idea: multi-band spectral filtering via polynomial graph filters
    of different orders. The original uses Beta-wavelet basis to cover
    distinct frequency bands; here we use a fixed multi-hop polynomial
    approximation: stack representations from K propagation orders, then
    concatenate. This captures the multi-frequency spectral coverage
    that distinguishes BWGNN from single-band GCN.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2, num_filters=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)
        self.filters = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_filters):
            self.filters.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Linear(hidden_dim * num_filters, 2)
        self.dropout = dropout
        self.num_filters = num_filters

    def forward(self, x, edge_index):
        h0 = F.relu(self.bn_in(self.input_proj(x)))
        outs = []
        h = h0
        # Each filter = one more hop of propagation; concatenate multi-hop reprs
        for conv, bn in zip(self.filters, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            outs.append(h)
        h_cat = torch.cat(outs, dim=1)
        return self.classifier(h_cat)


class GAGA(nn.Module):
    """GAGA (Wang et al. WWW 2023) simplified single-relation variant.

    Core idea: group neighbors by their (training) class label and apply
    separate aggregation per group. The original uses transformer-based
    aggregation over group representations; here we use two parallel
    GCN channels (minority-class neighbors, majority-class neighbors)
    with concatenation, faithfully capturing the label-group separation.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        # Two relational paths: minority-class neighbors, majority-class neighbors
        self.conv_min = nn.ModuleList([GCNConv(input_dim if l == 0 else hidden_dim, hidden_dim) for l in range(num_layers)])
        self.conv_maj = nn.ModuleList([GCNConv(input_dim if l == 0 else hidden_dim, hidden_dim) for l in range(num_layers)])
        self.bn_min = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.bn_maj = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim * 2, 2)
        self.dropout = dropout
        self._y_train = None
        self._train_mask = None

    def set_train_labels(self, y_train, train_mask):
        self._y_train = y_train.clone()
        self._train_mask = train_mask.clone()

    def _group_edges(self, edge_index):
        if self._y_train is None:
            return edge_index, edge_index
        src, tgt = edge_index[0], edge_index[1]
        # Group by source (the neighbor sending message): minority (label=1) vs majority (label=0)
        # For non-train nodes (-1), treat as majority by default
        y_src = self._y_train[src].clone()
        if self._train_mask is not None:
            y_src[~self._train_mask[src]] = 0
        ei_min = edge_index[:, y_src == 1]
        ei_maj = edge_index[:, y_src == 0]
        return ei_min, ei_maj

    def forward(self, x, edge_index):
        ei_min, ei_maj = self._group_edges(edge_index)
        h_min, h_maj = x, x
        for c_min, c_maj, bn_min, bn_maj in zip(self.conv_min, self.conv_maj, self.bn_min, self.bn_maj):
            h_min = F.dropout(F.relu(bn_min(c_min(h_min, ei_min))), p=self.dropout, training=self.training)
            h_maj = F.dropout(F.relu(bn_maj(c_maj(h_maj, ei_maj))), p=self.dropout, training=self.training)
        return self.classifier(torch.cat([h_min, h_maj], dim=1))


class ConsisGAD(nn.Module):
    """ConsisGAD (Chen et al. ICLR 2024) simplified single-relation variant.

    Core idea: consistency training between original graph view and a
    learnably-augmented graph view. The original learns a soft edge mask
    via attention; here we use a simpler shared-parameter dual-view setup
    where the same GCN encodes (1) the original graph and (2) a stochastic
    edge-perturbed view, with consistency loss between the two outputs.
    Final classification combines both views. Captures the multi-view
    consistency principle that distinguishes ConsisGAD.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2, p_edge_drop=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.classifier = nn.Linear(hidden_dim * 2, 2)
        self.dropout = dropout
        self.p_edge_drop = p_edge_drop
        self._consistency_loss = 0.0  # tracked externally if needed

    def _encode(self, x, edge_index):
        h = x
        for conv, bn in zip(self.layers, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index):
        # View 1: original graph
        h1 = self._encode(x, edge_index)
        # View 2: edge-perturbed graph
        if self.training:
            mask = torch.rand(edge_index.size(1), device=edge_index.device) >= self.p_edge_drop
            ei2 = edge_index[:, mask]
        else:
            ei2 = edge_index
        h2 = self._encode(x, ei2)
        # Combine both views
        h = torch.cat([h1, h2], dim=1)
        # Track consistency loss (L2 between h1 and h2) for external use
        if self.training:
            self._consistency_loss = ((h1 - h2) ** 2).mean()
        return self.classifier(h)


MODELS = {
    'gcn': SupervisedGCN,
    'gat': SupervisedGAT,
    'sage': SupervisedSAGE,
    'h2gcn': SupervisedH2GCN,
    'mixhop': SupervisedMixHop,
    'fagcn': SupervisedFAGCN,
    'acmgcn': SupervisedACMGCN,
    'mlp': SupervisedMLP,
    'caregnn': CAREGNN,
    'pcgnn': PCGNN,
    'bwgnn': BWGNN,
    'gaga': GAGA,
    'consisgad': ConsisGAD,
}


def train_supervised_gnn(model, data, train_mask, optimizer, class_weight):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=class_weight)
    # ConsisGAD: add multi-view consistency regularization
    if hasattr(model, '_consistency_loss') and isinstance(model._consistency_loss, torch.Tensor):
        loss = loss + 0.1 * model._consistency_loss
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_supervised(model, data, test_mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[test_mask].argmax(dim=1).cpu().numpy()
    prob = F.softmax(out[test_mask], dim=1)[:, 1].cpu().numpy()
    y = data.y[test_mask].cpu().numpy()

    f1_1 = f1_score(y, pred, pos_label=1, zero_division=0)
    pre_1 = precision_score(y, pred, pos_label=1, zero_division=0)
    rec_1 = recall_score(y, pred, pos_label=1, zero_division=0)
    auroc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else 0
    auprc = average_precision_score(y, prob) if len(np.unique(y)) > 1 else 0
    return f1_1, pre_1, rec_1, auroc, auprc


def run_gnn_model(model_name, data, device, seed, hidden_dim=256, num_layers=2, lr=0.001, epochs=200, train_ratio=0.1, split=None):
    set_seed(seed)
    N = data.num_nodes
    y = data.y.cpu().numpy()

    # train/val/test split via make_split, or externally supplied temporal split.
    if split is None:
        split = make_split(N, train_ratio=train_ratio, val_ratio=0.1, seed=seed)
    train_idx = split['train'].numpy()
    test_idx = split['test'].numpy()
    train_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    train_mask = train_mask.to(device)
    test_mask = test_mask.to(device)

    # Class weight for imbalance
    n_pos = y[train_idx].sum()
    n_neg = len(train_idx) - n_pos
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    model_cls = MODELS[model_name]
    model = model_cls(data.x.size(1), hidden_dim, num_layers).to(device)
    # PC-GNN, GAGA need train labels for label-aware neighbor sampling/grouping
    if model_name in ('pcgnn', 'gaga'):
        model.set_train_labels(data.y, train_mask)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_f1 = 0
    best_result = None
    for epoch in range(1, epochs + 1):
        train_supervised_gnn(model, data, train_mask, optimizer, weight)
        if epoch % 10 == 0 or epoch == epochs:
            f1_1, pre_1, rec_1, auroc, auprc = eval_supervised(model, data, test_mask)
            if f1_1 > best_f1:
                best_f1 = f1_1
                best_result = (f1_1, pre_1, rec_1, auroc, auprc)

    return best_result


# ============================================================
# Tabular Baselines (LightGBM, XGBoost)
# ============================================================
def run_tabular_model(model_name, X, y, seed, train_ratio=0.1, split=None):
    # split via make_split, or externally supplied temporal split.
    if split is None:
        split = make_split(len(y), train_ratio=train_ratio, val_ratio=0.1, seed=seed)
    train_idx = split['train'].numpy()
    test_idx = split['test'].numpy()
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if model_name == 'lgbm':
        import lightgbm as lgb
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=n_neg / max(n_pos, 1),
            verbose=-1, random_state=seed, n_jobs=-1)
    elif model_name == 'xgb':
        import xgboost as xgb
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric='logloss', verbosity=0, random_state=seed, n_jobs=-1)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    f1_1 = f1_score(y_test, pred, pos_label=1, zero_division=0)
    pre_1 = precision_score(y_test, pred, pos_label=1, zero_division=0)
    rec_1 = recall_score(y_test, pred, pos_label=1, zero_division=0)
    auroc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0
    auprc = average_precision_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0
    return f1_1, pre_1, rec_1, auroc, auprc


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2024, 2025, 2026, 2027])
    parser.add_argument('--dataset', type=str, default='hofinet', choices=['hofinet', 'amlworld', 'amlnet'])
    parser.add_argument('--result_file', type=str, default='./results/exp_results_supervised.csv')
    parser.add_argument('--train_ratio', type=float, default=0.1,
                        help='Train split ratio (val_ratio fixed at 0.1; test = 1 - train - val)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--exclude_struct', action='store_true',
                        help='If set, tabular models (lgbm, xgb) use only behavioral features (data.x), matching BehaView encoder input')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='Specific models to run (default: all)')
    parser.add_argument('--split_path', type=str, default=None,
                        help='Optional .npz file with train/valid/test node indices, e.g. temporal split')
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu}')
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    if args.dataset == 'hofinet':
        node_data = 'HOFINET_NODE_FEAT'
        edge_data = 'HOFINET_EDGES'
    elif args.dataset == 'amlworld':
        node_data = 'amlworld/AMLWORLD_NODE_FEAT'
        edge_data = 'amlworld/AMLWORLD_EDGES'
    else:
        node_data = 'amlnet/AMLNET_NODE_FEAT'
        edge_data = 'amlnet/AMLNET_EDGES'

    # Load data using config-compatible args
    class Args:
        pass
    fake_args = Args()
    fake_args.node_data_name = node_data
    fake_args.edge_data_name = edge_data
    fake_args.struct_feats = ['dc', 'pagerank', 'hits_hub', 'hits_auth', 'kcore', 'triangle', 'betweenness']

    data, x_struct = load_graph_data(fake_args, device=device)
    print(f'Dataset: {args.dataset}, Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}')
    print(f'Suspicious: {(data.y==1).sum().item()} ({(data.y==1).float().mean().item()*100:.2f}%)')

    # Feature matrix for tabular models. By default behavioral + structural;
    # with --exclude_struct only behavioral (matches BehaView encoder input).
    if args.exclude_struct:
        X_all = data.x.cpu().numpy()
        print(f'[exclude_struct] tabular models use behavioral-only features ({X_all.shape[1]} cols)')
    else:
        X_all = torch.cat([data.x.cpu(), x_struct], dim=1).numpy()
    y_all = data.y.cpu().numpy()
    external_split = load_split(args.split_path) if args.split_path else None
    if external_split is not None:
        print(f'Loaded split: {args.split_path}')

    results = []
    all_models = args.models if args.models else ['gcn', 'gat', 'sage', 'mlp', 'lgbm', 'xgb', 'caregnn', 'pcgnn']

    for model_name in all_models:
        for seed in args.seeds:
            set_seed(seed)
            print(f'[{model_name}] seed={seed}...', end=' ')

            if model_name in ('lgbm', 'xgb'):
                f1_1, pre_1, rec_1, auroc, auprc = run_tabular_model(
                    model_name, X_all, y_all, seed, train_ratio=args.train_ratio, split=external_split)
            else:
                f1_1, pre_1, rec_1, auroc, auprc = run_gnn_model(
                    model_name, data, device, seed, hidden_dim=args.hidden_dim, num_layers=2,
                    epochs=args.epochs, train_ratio=args.train_ratio, split=external_split)

            print(f'F1_susp={f1_1:.4f}, AUROC={auroc:.4f}')
            results.append({
                'dataset': args.dataset, 'model': model_name, 'seed': seed,
                'f1_1': f1_1, 'pre_1': pre_1, 'rec_1': rec_1,
                'auroc': auroc, 'auprc': auprc,
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.result_file, index=False)
    print(f'\nSaved: {args.result_file}')

    # Summary
    print(f'\n{"Model":6s} | {"F1_susp":>10s} | {"AUROC":>8s} | {"AUPRC":>8s}')
    print('-' * 40)
    for m in all_models:
        sub = df[df['model'] == m]
        print(f'{m:6s} | {sub["f1_1"].mean():.4f}±{sub["f1_1"].std():.4f} | {sub["auroc"].mean():.4f} | {sub["auprc"].mean():.4f}')


if __name__ == '__main__':
    main()
