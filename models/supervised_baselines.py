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
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from config import get_config
from data_loader import load_graph_data
from utils import set_seed, make_split


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


class SupervisedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
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


MODELS = {
    'gcn': SupervisedGCN,
    'gat': SupervisedGAT,
    'sage': SupervisedSAGE,
    'mlp': SupervisedMLP,
    'caregnn': CAREGNN,
    'pcgnn': PCGNN,
}


def train_supervised_gnn(model, data, train_mask, optimizer, class_weight):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=class_weight)
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


def run_gnn_model(model_name, data, device, seed, hidden_dim=256, num_layers=2, lr=0.001, epochs=200):
    set_seed(seed)
    N = data.num_nodes
    y = data.y.cpu().numpy()

    # 10/10/80 train/val/test split via make_split (paired with BECON SSL evaluation)
    split = make_split(N, train_ratio=0.1, val_ratio=0.1, seed=seed)
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
    # PC-GNN needs train labels for neighbor sampling
    if model_name == 'pcgnn':
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
def run_tabular_model(model_name, X, y, seed):
    # 10/10/80 split via make_split — same accounts as GNN supervised and BECON SSL
    split = make_split(len(y), train_ratio=0.1, val_ratio=0.1, seed=seed)
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
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help='Specific models to run (default: all)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    if args.dataset == 'hofinet':
        node_data = 'hofinet/HOFINET_NODE_FEAT'
        edge_data = 'hofinet/HOFINET_EDGES'
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

    # Feature matrix for tabular models (behavioral + structural)
    X_all = torch.cat([data.x.cpu(), x_struct], dim=1).numpy()
    y_all = data.y.cpu().numpy()

    results = []
    all_models = args.models if args.models else ['gcn', 'gat', 'sage', 'mlp', 'lgbm', 'xgb', 'caregnn', 'pcgnn']

    for model_name in all_models:
        for seed in args.seeds:
            set_seed(seed)
            print(f'[{model_name}] seed={seed}...', end=' ')

            if model_name in ('lgbm', 'xgb'):
                f1_1, pre_1, rec_1, auroc, auprc = run_tabular_model(model_name, X_all, y_all, seed)
            else:
                f1_1, pre_1, rec_1, auroc, auprc = run_gnn_model(
                    model_name, data, device, seed, hidden_dim=256, num_layers=2)

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
