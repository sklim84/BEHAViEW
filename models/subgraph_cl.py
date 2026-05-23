"""
Subgraph-Level Contrastive Learning for AML Detection.

두 이종 그래프(Transaction + Behavioral k-NN)에서 각 노드의 ego-subgraph 표현을
대조학습으로 일치시킴. G2L의 class imbalance 문제와 L2L의 경로 정보 부재를 동시 해결.

사용법:
    python models/subgraph_cl.py \
        --model_name subgraph_cl \
        --gpu 5 --seed 2025 \
        --node_data_name hofinet/HOFINET_NODE_FEAT \
        --edge_data_name hofinet/HOFINET_EDGES \
        --knn_graph hofinet/HOFINET_KNN_BEHAV_k10 \
        --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 \
        --loss BarlowTwins --skip_tsne
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import re
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GINConv as torch_geometric_GINConv
from tqdm import tqdm

from config import get_config
from data_loader import load_graph_data, load_knn_graph
from utils import set_seed, build_result_dict, save_results_to_csv, evaluate_with_metrics, visualize_tsne, make_split, load_split


class GNNEncoder_BGRL(nn.Module):
    """BGRL-style: GCN + BN + PReLU + Dropout + projection head."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.activation = nn.PReLU()
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.layers:
            x = conv(x, edge_index, edge_weight)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.bn(x)


class GNNEncoder_DGI(nn.Module):
    """DGI-style: GCN + per-layer PReLU, no dropout/BN."""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.activations.append(nn.PReLU(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for conv, act in zip(self.layers, self.activations):
            x = conv(x, edge_index, edge_weight)
            x = act(x)
        return x


class GNNEncoder_MVGRL(nn.Module):
    """MVGRL-style: dual GCN encoders (하나로 두 view 처리).
    forward는 single encoder로 동작, SubgraphCL에서 두 번 호출."""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.activation = nn.PReLU(hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.layers:
            x = conv(x, edge_index, edge_weight)
            x = self.activation(x)
        return x


class GNNEncoder_GBT(nn.Module):
    """GBT-style: GCN + BN + PReLU + Dropout (similar to BGRL but 2-layer BN)."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.activation = nn.PReLU()
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GNNEncoder_GRACE(nn.Module):
    """GRACE-style: GCN + activation, with projection head built into SubgraphCL."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GNNEncoder_DGI_BN(nn.Module):
    """DGI + BatchNorm: BN 효과 검증용."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.activations.append(nn.PReLU(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv, bn, act in zip(self.layers, self.bns, self.activations):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GNNEncoder_MVGRL_BN(nn.Module):
    """MVGRL + BatchNorm: BN 효과 검증용."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.activation = nn.PReLU(hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GNNEncoder_GRACE_BN(nn.Module):
    """GRACE + BatchNorm: BN 효과 검증용."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for i, (conv, bn) in enumerate(zip(self.layers, self.bns)):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GNNEncoder_GCA(nn.Module):
    """GCA-style: GCNConv + ReLU, no BN (same as GRACE). GCA's novelty is in augmentation, not encoder."""
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index, edge_weight)
            x = self.activation(x)
        return x


class GNNEncoder_GIN(nn.Module):
    """GIN-style (PGCL/BGRL_G2L): GINConv + BN + ReLU. Used by PGCL and BGRL_G2L."""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(torch_geometric_GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        for conv, bn in zip(self.layers, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


ENCODERS = {
    'bgrl': GNNEncoder_BGRL,
    'dgi': GNNEncoder_DGI,
    'mvgrl': GNNEncoder_MVGRL,
    'gbt': GNNEncoder_GBT,
    'grace': GNNEncoder_GRACE,
    'dgi_bn': GNNEncoder_DGI_BN,
    'mvgrl_bn': GNNEncoder_MVGRL_BN,
    'grace_bn': GNNEncoder_GRACE_BN,
    'gca': GNNEncoder_GCA,
    'gin': GNNEncoder_GIN,
}


class SubgraphPooling(nn.Module):
    """각 노드의 ego-subgraph 표현을 mean pooling으로 계산.

    GNN 출력(모든 노드 임베딩) + 그래프의 edge_index를 받아,
    각 노드의 이웃 임베딩을 mean pooling하여 subgraph representation 생성.
    self-loop 포함 (자기 자신 + 이웃).
    """
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_index):
        """
        z: [N, D] node embeddings
        edge_index: [2, E] adjacency
        Returns: [N, D] subgraph representations (mean of self + neighbors)
        """
        N, D = z.size()
        # Scatter mean: 각 target 노드에 대해 source 노드 임베딩 합산
        src, tgt = edge_index[0], edge_index[1]

        # 이웃 합산
        neigh_sum = torch.zeros(N, D, device=z.device)
        neigh_count = torch.zeros(N, 1, device=z.device)
        neigh_sum.scatter_add_(0, tgt.unsqueeze(1).expand(-1, D), z[src])
        neigh_count.scatter_add_(0, tgt.unsqueeze(1), torch.ones(src.size(0), 1, device=z.device))

        # self + neighbors mean
        total_sum = neigh_sum + z  # self-loop
        total_count = neigh_count + 1
        subgraph_repr = total_sum / total_count.clamp(min=1)

        return subgraph_repr


class HeterophilyAwarePool(nn.Module):
    """Cosine-similarity-weighted ego-subgraph pooling.

    Mean pool의 한계 (heterophilous 이웃에서 noise amplification) 회피.
    각 edge (u, v)의 가중치 w_uv = max(0, cos(z_u, z_v)). self-loop 가중치 1.0 고정.
    이질적 이웃 (cos < 0) 은 가중치 0으로 사실상 제외 → 의심 신호 희석 완화.

        s_v = (z_v + sum_u w_uv * z_u) / (1 + sum_u w_uv)
    """
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_index):
        N, D = z.size()
        src, tgt = edge_index[0], edge_index[1]

        z_norm = F.normalize(z, p=2, dim=1)
        cos_sim = (z_norm[src] * z_norm[tgt]).sum(dim=1, keepdim=True).clamp(min=0.0)

        weighted_z  = cos_sim * z[src]
        neigh_sum   = torch.zeros(N, D, device=z.device)
        weight_sum  = torch.zeros(N, 1, device=z.device)
        neigh_sum.scatter_add_(0, tgt.unsqueeze(1).expand(-1, D), weighted_z)
        weight_sum.scatter_add_(0, tgt.unsqueeze(1), cos_sim)

        total_sum    = neigh_sum + z              # self-loop weight = 1
        total_weight = weight_sum + 1.0
        return total_sum / total_weight.clamp(min=1e-6)


class CycleAwarePool(nn.Module):
    """HAP + cycle-membership boost.

    HeterophilyAwarePool 의 cosine 가중치에 triangle (cycle) 멤버십 가중치 추가.
    각 edge (u, v) 의 weight = max(0, cos(z_u, z_v)) * (1 + alpha * 1[tri_u > 0]).
    self-loop weight = 1 + alpha * 1[tri_v > 0].

    triangle membership 은 외부에서 set_tri() 로 주입한다 (HOFINET 의 'triangle' feature).
    의심 계좌가 거래 cycle (mule layering 패턴) 에 더 자주 참여한다는 도메인 가정 활용.
    """
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
        self.tri_indicator = None

    def set_tri(self, tri_count):
        """tri_count: [N] tensor of triangle counts; 0/1 indicator 로 변환."""
        self.tri_indicator = (tri_count > 0).float().unsqueeze(1)

    def forward(self, z, edge_index):
        if self.tri_indicator is None:
            raise RuntimeError("CycleAwarePool: tri_indicator not set; call set_tri() first")
        N, D = z.size()
        src, tgt = edge_index[0], edge_index[1]
        tri = self.tri_indicator.to(z.device)

        z_norm = F.normalize(z, p=2, dim=1)
        cos_sim = (z_norm[src] * z_norm[tgt]).sum(dim=1, keepdim=True).clamp(min=0.0)

        edge_boost = 1.0 + self.alpha * tri[src]
        weighted_w = cos_sim * edge_boost

        weighted_z = weighted_w * z[src]
        neigh_sum = torch.zeros(N, D, device=z.device)
        weight_sum = torch.zeros(N, 1, device=z.device)
        neigh_sum.scatter_add_(0, tgt.unsqueeze(1).expand(-1, D), weighted_z)
        weight_sum.scatter_add_(0, tgt.unsqueeze(1), weighted_w)

        self_boost = 1.0 + self.alpha * tri
        total_sum = neigh_sum + self_boost * z
        total_weight = weight_sum + self_boost
        return total_sum / total_weight.clamp(min=1e-6)


class SubgraphCL(nn.Module):
    """Unified Contrastive Learning framework.

    4가지 설정 지원:
    (a) aug view + node-level: use_knn=False, use_subgraph=False
    (b) behav view + node-level: use_knn=True, use_subgraph=False
    (c) aug view + subgraph: use_knn=False, use_subgraph=True
    (d) behav view + subgraph: use_knn=True, use_subgraph=True

    pool_variant ('mean' | 'heterophily' | 'cycle'): subgraph pool 방식 선택.
      - 'mean': 표준 mean pool (BehaView 기본)
      - 'heterophily': cosine-sim weighted pool (heterophilous 이웃 noise 완화)
      - 'cycle': HAP + triangle membership boost (cycle 패턴 amplification, P5)
    """
    def __init__(self, encoder, hidden_dim, proj_dim=64, use_subgraph=True,
                 pool_variant='mean', cycle_alpha=2.0):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.use_subgraph = use_subgraph
        self.pool_variant = pool_variant
        if pool_variant == 'heterophily':
            self._subgraph_pool = HeterophilyAwarePool()
        elif pool_variant == 'cycle':
            self._subgraph_pool = CycleAwarePool(alpha=cycle_alpha)
        else:
            self._subgraph_pool = SubgraphPooling()

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Predictor (BYOL-style asymmetry)
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum=0.99):
        for p, new_p in zip(self.get_target_encoder().parameters(),
                           self.online_encoder.parameters()):
            p.data = momentum * p.data + (1 - momentum) * new_p.data

    def _encode(self, encoder, x, edge_index):
        """GNN encoding (gradient checkpointing 지원)."""
        if self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(encoder, x, edge_index, use_reentrant=False)
        return encoder(x, edge_index)

    def _get_repr(self, encoder, x, edge_index):
        """Encode → optionally subgraph pool → return representation."""
        z = self._encode(encoder, x, edge_index)
        if self.use_subgraph:
            return self._subgraph_pool(z, edge_index)
        return z

    def forward(self, x, edge_index_v1, edge_index_v2):
        # View 1
        r1 = self._get_repr(self.online_encoder, x, edge_index_v1)
        p1 = self.predictor(self.projector(r1))

        # View 2
        r2 = self._get_repr(self.online_encoder, x, edge_index_v2)
        p2 = self.predictor(self.projector(r2))

        # Target
        with torch.no_grad():
            t1 = self.projector(self._get_repr(self.get_target_encoder(), x, edge_index_v1))
            t2 = self.projector(self._get_repr(self.get_target_encoder(), x, edge_index_v2))

        return r1, r2, p1, p2, t1, t2


def bootstrap_loss(p1, p2, t1, t2):
    """BYOL-style bootstrap loss; no negatives required."""
    loss = (2 - 2 * F.cosine_similarity(p1, t2.detach(), dim=-1).mean()
          + 2 - 2 * F.cosine_similarity(p2, t1.detach(), dim=-1).mean())
    return loss


def barlow_twins_loss(p1, p2, lambda_=None, eps=1e-5):
    """Cross-correlation matrix loss (Zbontar et al., 2021)."""
    N, D = p1.shape
    if lambda_ is None:
        lambda_ = 1.0 / D
    z1 = (p1 - p1.mean(0)) / (p1.std(0) + eps)
    z2 = (p2 - p2.mean(0)) / (p2.std(0) + eps)
    c = (z1.T @ z2) / N
    eye = torch.eye(D, dtype=torch.bool, device=c.device)
    on_diag = (1 - c.diagonal()).pow(2).sum()
    off_diag = c[~eye].pow(2).sum()
    return on_diag + lambda_ * off_diag


def infonce_loss(p1, p2, tau=0.5, k=2048):
    """GRACE-style symmetric InfoNCE with anchor subsampling for memory."""
    N = p1.shape[0]
    if N > k:
        idx = torch.randperm(N, device=p1.device)[:k]
        p1 = p1[idx]
        p2 = p2[idx]
    z1 = F.normalize(p1, dim=-1)
    z2 = F.normalize(p2, dim=-1)

    def _half(za, zb):
        sim_ab = za @ zb.t() / tau
        sim_aa = za @ za.t() / tau
        exp_ab = torch.exp(sim_ab)
        exp_aa = torch.exp(sim_aa)
        pos = exp_ab.diag()
        denom = exp_ab.sum(1) + exp_aa.sum(1) - exp_aa.diag()
        return -torch.log(pos / denom).mean()

    return (_half(z1, z2) + _half(z2, z1)) * 0.5


def jsd_loss(p1, p2, k=2048):
    """JSD discriminative loss (DGI-style) with anchor subsampling."""
    import math
    N = p1.shape[0]
    if N > k:
        idx = torch.randperm(N, device=p1.device)[:k]
        p1 = p1[idx]
        p2 = p2[idx]
    Nk = p1.shape[0]
    sim = p1 @ p2.t()
    eye = torch.eye(Nk, dtype=torch.bool, device=sim.device)
    pos_sim = sim[eye]
    neg_sim = sim[~eye]
    E_pos = (math.log(2) - F.softplus(-pos_sim)).mean()
    E_neg = (F.softplus(-neg_sim) + neg_sim - math.log(2)).mean()
    return E_neg - E_pos


_LOSS_REGISTRY = {
    'BootstrapLatent': lambda p1, p2, t1, t2: bootstrap_loss(p1, p2, t1, t2),
    'BarlowTwins':     lambda p1, p2, t1, t2: barlow_twins_loss(p1, p2),
    'InfoNCE':         lambda p1, p2, t1, t2: infonce_loss(p1, p2),
    'JSD':             lambda p1, p2, t1, t2: jsd_loss(p1, p2),
}


def compute_loss(loss_name, p1, p2, t1, t2):
    if loss_name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown --loss '{loss_name}'. "
                         f"Choose from {list(_LOSS_REGISTRY)}.")
    return _LOSS_REGISTRY[loss_name](p1, p2, t1, t2)


def train(model, x, edge_index_v1, edge_index_v2, optimizer, loss_name='BootstrapLatent'):
    model.train()
    optimizer.zero_grad()
    _, _, p1, p2, t1, t2 = model(x, edge_index_v1, edge_index_v2)
    loss = compute_loss(loss_name, p1, p2, t1, t2)
    loss.backward()
    optimizer.step()
    model.update_target_encoder()
    return loss.item()


def get_embeddings(model, x, edge_index_v1, edge_index_v2):
    """추론 시 두 view의 표현을 결합."""
    model.eval()
    with torch.no_grad():
        r1 = model._get_repr(model.online_encoder, x, edge_index_v1)
        r2 = model._get_repr(model.online_encoder, x, edge_index_v2)
    return torch.cat([r1, r2], dim=1)


def main(args):
    set_seed(args.seed)
    device_arg = getattr(args, 'device', 'auto')
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu}')
        elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    elif device_arg == 'cuda':
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device(device_arg)
    print(f'##### device: {device}')

    data, _ = load_graph_data(args, device=device)
    print(f'Transaction graph: {data}')

    # View 설정: k-NN graph 또는 augmented transaction graph
    use_knn = args.knn_graph is not None
    use_subgraph = getattr(args, 'subgraph_pool', False)

    if use_knn:
        edge_index_knn = load_knn_graph(args.knn_graph, device=device)
        edge_index_v1 = data.edge_index  # transaction graph
        edge_index_v2 = edge_index_knn   # behavioral k-NN graph
        print(f'k-NN graph: {args.knn_graph} ({edge_index_knn.size(1)} edges)')
    else:
        edge_index_v1 = data.edge_index  # both views use transaction graph
        edge_index_v2 = data.edge_index  # (augmentation creates diversity)

    setting = {(False, False): '(a)', (True, False): '(b)',
               (False, True): '(c)', (True, True): '(d)'}[(use_knn, use_subgraph)]
    print(f'Setting {setting}: knn={use_knn}, subgraph={use_subgraph}')

    encoder_type = getattr(args, 'encoder_type', 'bgrl')
    encoder_cls = ENCODERS[encoder_type]
    encoder_kwargs = dict(input_dim=data.x.size(1), hidden_dim=args.hidden_dim, num_layers=args.gconv_nlayers)
    if encoder_type in ('bgrl', 'gbt', 'grace', 'dgi_bn', 'mvgrl_bn', 'grace_bn', 'gin'):
        encoder_kwargs['dropout'] = 0.2
    encoder = encoder_cls(**encoder_kwargs)
    print(f'Encoder: {encoder_type} ({encoder_cls.__name__})')

    pool_variant = getattr(args, 'pool_variant', 'mean')
    cycle_alpha = getattr(args, 'cycle_alpha', 2.0)
    model = SubgraphCL(
        encoder=encoder,
        hidden_dim=args.hidden_dim,
        use_subgraph=use_subgraph,
        pool_variant=pool_variant,
        cycle_alpha=cycle_alpha,
    ).to(device)
    if pool_variant == 'cycle' and use_subgraph:
        # Inject triangle feature for CycleAwarePool
        df_full = pd.read_csv(f'./datasets/{args.node_data_name}.csv')
        if 'triangle' not in df_full.columns:
            raise RuntimeError(f"pool_variant=cycle requires 'triangle' feature column in {args.node_data_name}")
        tri_count = torch.tensor(df_full['triangle'].values, dtype=torch.float, device=device)
        model._subgraph_pool.set_tri(tri_count)
    if use_subgraph:
        print(f'Subgraph pool variant: {pool_variant}'
              + (f' (cycle_alpha={cycle_alpha})' if pool_variant == 'cycle' else ''))

    optimizer = Adam(model.parameters(), lr=args.lr)

    epochs = getattr(args, 'epochs', 200)
    final_loss = float('nan')
    with tqdm(total=epochs, desc='(T)') as pbar:
        for epoch in range(1, epochs + 1):
            loss = train(model, data.x, edge_index_v1, edge_index_v2, optimizer, args.loss)
            final_loss = loss
            pbar.set_postfix({'loss': f'{loss:.4f}'})
            pbar.update()
    print(f'(L) {args.model_name}: final_loss={final_loss:.6f} ({args.loss})')

    # Evaluation
    z = get_embeddings(model, data.x, edge_index_v1, edge_index_v2)
    z_cpu = z.detach().cpu()

    # Optional: dump the joint embeddings for downstream visualization
    # (e.g., t-SNE / UMAP) by isolated scripts under _exp/tsne_viz/.
    embedding_path = getattr(args, 'save_embeddings_to', None)
    if embedding_path:
        os.makedirs(os.path.dirname(embedding_path) or '.', exist_ok=True)
        import numpy as _np
        _np.savez_compressed(embedding_path,
                             z=z_cpu.numpy(),
                             y=data.y.cpu().numpy(),
                             model_name=args.model_name)
        print(f'(E) saved embeddings -> {embedding_path}')

    os.makedirs('./visualize/SubgraphCL', exist_ok=True)
    vis_save_path = f'./visualize/SubgraphCL/tsne_{args.model_name}.png'
    ari_score, sil_score = visualize_tsne(
        args.seed, z_cpu.numpy(), data.y, save_path=vis_save_path, skip=args.skip_tsne)

    split_path = getattr(args, 'split_path', None)
    if split_path:
        split = load_split(split_path)
        print(f'Loaded split: {split_path}')
    else:
        train_ratio = getattr(args, 'train_ratio', 0.1)
        val_ratio = train_ratio  # paper convention: train=val=10%, test=80%
        split = make_split(z_cpu.size(0), train_ratio=train_ratio, val_ratio=val_ratio, seed=args.seed)
    tune_threshold = getattr(args, 'tune_threshold', False)
    test_results = evaluate_with_metrics(z_cpu, data.y, split, tune_threshold=tune_threshold)

    rows = []
    for r in test_results:
        variant = r.pop('variant', 'default')
        if variant == 'tuned':
            m = re.match(r'(.*)_s(\d+)$', args.model_name)
            name = f'{m.group(1)}_tuned_s{m.group(2)}' if m else f'{args.model_name}_tuned'
        else:
            name = args.model_name
        print(f'(E)[{variant}] {name}: F1Mi={r["micro_f1"]:.4f}, '
              f'F1Ma={r["macro_f1"]:.4f}, F1_susp={r["f1_1"]:.4f}, τ={r["threshold"]:.3f}')
        rows.append(build_result_dict(name, args, r, ari_score, sil_score, use_cen=False, final_loss=final_loss))
    save_results_to_csv(rows, args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    print(args)
    main(args)
