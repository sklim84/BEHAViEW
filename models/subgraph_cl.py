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
from torch_geometric.nn import GCNConv
from GCL.eval import get_split
from tqdm import tqdm

from config import get_config
from data_loader import load_graph_data, load_knn_graph
from utils import set_seed, build_result_dict, save_results_to_csv, evaluate_with_metrics, visualize_tsne


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


ENCODERS = {
    'bgrl': GNNEncoder_BGRL,
    'dgi': GNNEncoder_DGI,
    'mvgrl': GNNEncoder_MVGRL,
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
    """Subgraph-level Contrastive Learning.

    두 그래프에서 각각 GNN → subgraph pooling → projection → contrastive loss.
    """
    def __init__(self, encoder, hidden_dim, proj_dim=64):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.subgraph_pool = SubgraphPooling()

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

    def _encode_and_pool(self, encoder, x, edge_index):
        """GNN encoding + subgraph pooling (gradient checkpointing 지원)."""
        if self.training:
            from torch.utils.checkpoint import checkpoint
            z = checkpoint(encoder, x, edge_index, use_reentrant=False)
        else:
            z = encoder(x, edge_index)
        s = self.subgraph_pool(z, edge_index)
        return z, s

    def forward(self, x, edge_index_trans, edge_index_knn):
        # View 1: Transaction graph
        z1, s1 = self._encode_and_pool(self.online_encoder, x, edge_index_trans)
        p1 = self.predictor(self.projector(s1))

        # View 2: k-NN graph (detach intermediate to save memory)
        z2, s2 = self._encode_and_pool(self.online_encoder, x, edge_index_knn)
        p2 = self.predictor(self.projector(s2))

        # Target representations (no gradient)
        with torch.no_grad():
            _, s1_t = self._encode_and_pool(self.get_target_encoder(), x, edge_index_trans)
            t1 = self.projector(s1_t)
            _, s2_t = self._encode_and_pool(self.get_target_encoder(), x, edge_index_knn)
            t2 = self.projector(s2_t)

        return z1, z2, s1, s2, p1, p2, t1, t2


def bootstrap_loss(p1, p2, t1, t2):
    """BYOL-style bootstrap loss — negative sample 불필요."""
    loss = (2 - 2 * F.cosine_similarity(p1, t2.detach(), dim=-1).mean()
          + 2 - 2 * F.cosine_similarity(p2, t1.detach(), dim=-1).mean())
    return loss


def train(model, x, edge_index_trans, edge_index_knn, optimizer):
    model.train()
    optimizer.zero_grad()
    _, _, _, _, p1, p2, t1, t2 = model(x, edge_index_trans, edge_index_knn)
    loss = bootstrap_loss(p1, p2, t1, t2)
    loss.backward()
    optimizer.step()
    model.update_target_encoder()
    return loss.item()


def get_embeddings(model, x, edge_index_trans, edge_index_knn):
    """추론 시 두 view의 subgraph 표현을 결합."""
    model.eval()
    with torch.no_grad():
        z1 = model.online_encoder(x, edge_index_trans)
        s1 = model.subgraph_pool(z1, edge_index_trans)

        z2 = model.online_encoder(x, edge_index_knn)
        s2 = model.subgraph_pool(z2, edge_index_knn)

    # 두 subgraph 표현 결합
    return torch.cat([s1, s2], dim=1)


def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    print(f'##### device: {device}')

    data, _ = load_graph_data(args, device=device)
    print(f'Transaction graph: {data}')

    edge_index_knn = load_knn_graph(args.knn_graph, device=device)
    print(f'k-NN graph: {args.knn_graph} ({edge_index_knn.size(1)} edges)')

    encoder_type = getattr(args, 'encoder_type', 'bgrl')
    encoder_cls = ENCODERS[encoder_type]
    encoder_kwargs = dict(input_dim=data.x.size(1), hidden_dim=args.hidden_dim, num_layers=args.gconv_nlayers)
    if encoder_type == 'bgrl':
        encoder_kwargs['dropout'] = 0.2
    encoder = encoder_cls(**encoder_kwargs)
    print(f'Encoder: {encoder_type} ({encoder_cls.__name__})')

    model = SubgraphCL(
        encoder=encoder,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    with tqdm(total=200, desc='(T)') as pbar:
        for epoch in range(1, 201):
            loss = train(model, data.x, data.edge_index, edge_index_knn, optimizer)
            pbar.set_postfix({'loss': f'{loss:.4f}'})
            pbar.update()

    # Evaluation
    z = get_embeddings(model, data.x, data.edge_index, edge_index_knn)
    z_cpu = z.detach().cpu()

    os.makedirs('./visualize/SubgraphCL', exist_ok=True)
    vis_save_path = f'./visualize/SubgraphCL/tsne_{args.model_name}.png'
    ari_score, sil_score = visualize_tsne(
        args.seed, z_cpu.numpy(), data.y, save_path=vis_save_path, skip=args.skip_tsne)

    split = get_split(num_samples=z_cpu.size(0), train_ratio=0.1, test_ratio=0.8)
    test_result = evaluate_with_metrics(z_cpu, data.y, split)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, '
          f'F1Ma={test_result["macro_f1"]:.4f}, '
          f'F1_fraud={test_result["f1_1"]:.4f}')

    result = build_result_dict(args.model_name, args, test_result, ari_score, sil_score, use_cen=False)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    assert args.knn_graph is not None, '--knn_graph required'
    print(args)
    main(args)
