"""
Subgraph-Level Contrastive Learning for AML Detection.

두 이종 그래프(Transaction + Behavioral k-NN)에서 각 노드의 ego-subgraph 표현을
대조학습으로 일치시킴. G2L의 class imbalance 문제와 L2L의 경로 정보 부재를 동시 해결.

사용법:
    python models/subgraph_cl.py \
        --model_name subgraph_cl \
        --gpu 5 --seed 2025 \
        --node_data_name HOFINET_NODE_FEAT \
        --edge_data_name HOFINET_EDGES \
        --knn_graph HOFINET_KNN_BEHAV_k10 \
        --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 \
        --loss BarlowTwins --skip_tsne
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GINConv as torch_geometric_GINConv
from tqdm import tqdm

from config import get_config
from data_loader import load_graph_data, load_knn_graph
from utils import set_seed, build_result_dict, save_results_to_csv, evaluate_with_metrics, visualize_tsne, make_split


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
    """각 노드의 ego-subgraph 표현을 계산.

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


class SubgraphCL(nn.Module):
    """Unified Contrastive Learning framework.

    4가지 설정 지원:
    (a) aug view + node-level: use_knn=False, use_subgraph=False
    (b) behav view + node-level: use_knn=True, use_subgraph=False
    (c) aug view + subgraph: use_knn=False, use_subgraph=True
    (d) behav view + subgraph: use_knn=True, use_subgraph=True
    """
    def __init__(self, encoder, hidden_dim, proj_dim=64, use_subgraph=True):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.use_subgraph = use_subgraph
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
    """BYOL-style bootstrap loss — negative sample 불필요."""
    loss = (2 - 2 * F.cosine_similarity(p1, t2.detach(), dim=-1).mean()
          + 2 - 2 * F.cosine_similarity(p2, t1.detach(), dim=-1).mean())
    return loss


def train(model, x, edge_index_v1, edge_index_v2, optimizer):
    model.train()
    optimizer.zero_grad()
    _, _, p1, p2, t1, t2 = model(x, edge_index_v1, edge_index_v2)
    loss = bootstrap_loss(p1, p2, t1, t2)
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
    device = torch.device(f'cuda:{args.gpu}')
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

    model = SubgraphCL(
        encoder=encoder,
        hidden_dim=args.hidden_dim,
        use_subgraph=use_subgraph,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    with tqdm(total=200, desc='(T)') as pbar:
        for epoch in range(1, 201):
            loss = train(model, data.x, edge_index_v1, edge_index_v2, optimizer)
            pbar.set_postfix({'loss': f'{loss:.4f}'})
            pbar.update()

    # Evaluation
    z = get_embeddings(model, data.x, edge_index_v1, edge_index_v2)
    z_cpu = z.detach().cpu()

    os.makedirs('./visualize/SubgraphCL', exist_ok=True)
    vis_save_path = f'./visualize/SubgraphCL/tsne_{args.model_name}.png'
    ari_score, sil_score = visualize_tsne(
        args.seed, z_cpu.numpy(), data.y, save_path=vis_save_path, skip=args.skip_tsne)

    train_ratio = getattr(args, 'train_ratio', 0.1)
    val_ratio = train_ratio  # paper convention: train=val=10%, test=80%
    split = make_split(z_cpu.size(0), train_ratio=train_ratio, val_ratio=val_ratio, seed=args.seed)
    test_result = evaluate_with_metrics(z_cpu, data.y, split)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, '
          f'F1Ma={test_result["macro_f1"]:.4f}, '
          f'F1_susp={test_result["f1_1"]:.4f}')

    result = build_result_dict(args.model_name, args, test_result, ari_score, sil_score, use_cen=False)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    print(args)
    main(args)
