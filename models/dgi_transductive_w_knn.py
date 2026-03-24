"""
DGI (Transductive) with k-NN Graph View — contrastive learning between
original transaction graph and k-NN similarity graph.

Settings:
  (D) --knn_graph HOFINET_KNN_FEAT_k10  → feature similarity k-NN
  (F) --knn_graph HOFINET_KNN_CEN_k10   → centrality similarity k-NN
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn
from torch.optim import Adam
from GCL.eval import get_split
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from tqdm import tqdm

from config import get_config
from data_loader import load_graph_data, load_knn_graph
from utils import set_seed, create_loss, build_result_dict, save_results_to_csv, evaluate_with_metrics, visualize_tsne


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv, act in zip(self.layers, self.activations):
            z = conv(z, edge_index, edge_weight)
            z = act(z)
        return z


def subgraph_pool(z, edge_index):
    """각 노드의 k-NN 이웃 임베딩 평균 (self 포함)."""
    N, D = z.size()
    src, tgt = edge_index[0], edge_index[1]
    neigh_sum = torch.zeros(N, D, device=z.device)
    neigh_count = torch.zeros(N, 1, device=z.device)
    neigh_sum.scatter_add_(0, tgt.unsqueeze(1).expand(-1, D), z[src])
    neigh_count.scatter_add_(0, tgt.unsqueeze(1), torch.ones(src.size(0), 1, device=z.device))
    return (neigh_sum + z) / (neigh_count + 1).clamp(min=1)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim, use_subgraph_pool=False):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)
        self.use_subgraph_pool = use_subgraph_pool

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def _compute_summary(self, z, edge_index_knn):
        """Global pooling 또는 subgraph pooling으로 graph summary 계산."""
        if self.use_subgraph_pool:
            # Subgraph pooling: 각 노드의 k-NN 이웃 평균 → per-node summary
            s = subgraph_pool(z, edge_index_knn)
            g = self.project(torch.sigmoid(s))
        else:
            # Global pooling: 전체 노드 평균 → single summary
            g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        return g

    def forward(self, x, edge_index_trans, edge_index_knn):
        z1 = self.encoder(x, edge_index_trans)
        z2 = self.encoder(x, edge_index_knn)
        z = (z1 + z2) / 2

        g = self._compute_summary(z, edge_index_knn)

        x_corr1, ei_corr1 = self.corruption(x, edge_index_trans)
        zn1 = self.encoder(x_corr1, ei_corr1)
        x_corr2, ei_corr2 = self.corruption(x, edge_index_knn)
        zn2 = self.encoder(x_corr2, ei_corr2)
        zn = (zn1 + zn2) / 2

        return z, g, zn


def subgraph_contrastive_loss(z, g, zn):
    """Per-node subgraph contrastive: node vs its local summary (positive), corrupted vs local summary (negative)."""
    # g: [N, D] per-node subgraph summary
    # z: [N, D] node embeddings (positive)
    # zn: [N, D] corrupted node embeddings (negative)
    pos = torch.sigmoid((z * g).sum(dim=1))    # positive score per node
    neg = torch.sigmoid((zn * g).sum(dim=1))   # negative score per node
    loss = -torch.log(pos + 1e-8).mean() - torch.log(1 - neg + 1e-8).mean()
    return loss


def train(encoder_model, contrast_model, x, edge_index_trans, edge_index_knn, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, g, zn = encoder_model(x, edge_index_trans, edge_index_knn)
    if encoder_model.use_subgraph_pool:
        loss = subgraph_contrastive_loss(z, g, zn)
    else:
        loss = contrast_model(h=z, g=g, hn=zn)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(seed, encoder_model, x, edge_index_trans, edge_index_knn, y, vis_save_path, skip_tsne=False):
    encoder_model.eval()
    with torch.no_grad():
        z, _, _ = encoder_model(x, edge_index_trans, edge_index_knn)
    ari_score, sil_score = visualize_tsne(seed, z.detach().cpu().numpy(), y, save_path=vis_save_path, skip=skip_tsne)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = evaluate_with_metrics(z, y, split)
    return result, ari_score, sil_score



def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    print(f'##### device: {device}')

    data, x_struct = load_graph_data(args, device=device)
    print(f'Transaction graph: {data}')

    # Load k-NN graph
    edge_index_knn = load_knn_graph(args.knn_graph, device=device)
    print(f'k-NN graph: {args.knn_graph} ({edge_index_knn.size(1)} edges)')

    # Single feature set — no projection needed
    input_dim = data.x.size(1)
    use_subgraph = getattr(args, 'subgraph_pool', False)
    gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.gconv_nlayers).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=args.hidden_dim, use_subgraph_pool=use_subgraph).to(device)
    print(f'Subgraph pooling: {use_subgraph}')

    loss_fn = create_loss(args.loss)
    contrast_model = SingleBranchContrast(loss=loss_fn, mode='G2L').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    with tqdm(total=300, desc='(T)') as pbar:
        for epoch in range(1, 301):
            loss = train(encoder_model, contrast_model, data.x, data.edge_index, edge_index_knn, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs('./visualize/DGI_TRN_KNN', exist_ok=True)
    vis_save_path = f'./visualize/DGI_TRN_KNN/tsne_{args.model_name}.png'
    test_result, ari_score, sil_score = test(
        args.seed, encoder_model, data.x, data.edge_index, edge_index_knn, data.y, vis_save_path, args.skip_tsne)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}, F1_fraud={test_result["f1_1"]:.4f}')

    result = build_result_dict(args.model_name, args, test_result, ari_score, sil_score, use_cen=False)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    assert args.knn_graph is not None, '--knn_graph required (e.g., HOFINET_KNN_CEN_k10)'
    print(args)
    main(args)
