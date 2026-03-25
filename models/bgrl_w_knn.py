"""
BGRL with k-NN Graph View — contrastive learning between
original transaction graph and k-NN similarity graph.

Settings:
  (D) --knn_graph HOFINET_KNN_FEAT_k10  → feature similarity k-NN
  (F) --knn_graph HOFINET_KNN_CEN_k10   → centrality similarity k-NN
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import copy
import GCL.augmentors as A
import torch
import torch.nn.functional as F
from GCL.eval import get_split
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from config import get_config
from data_loader import load_graph_data, load_knn_graph
from utils import set_seed, create_loss, build_result_dict, save_results_to_csv, evaluate_with_metrics, visualize_tsne


class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)


def subgraph_pool(z, edge_index):
    N, D = z.size()
    src, tgt = edge_index[0], edge_index[1]
    neigh_sum = torch.zeros(N, D, device=z.device)
    neigh_count = torch.zeros(N, 1, device=z.device)
    neigh_sum.scatter_add_(0, tgt.unsqueeze(1).expand(-1, D), z[src])
    neigh_count.scatter_add_(0, tgt.unsqueeze(1), torch.ones(src.size(0), 1, device=z.device))
    return (neigh_sum + z) / (neigh_count + 1).clamp(min=1)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch',
                 use_subgraph_pool=False):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.use_subgraph_pool = use_subgraph_pool
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = copy.deepcopy(self.online_encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    def update_target_encoder(self, momentum):
        for p, new_p in zip(self.get_target_encoder().parameters(), self.online_encoder.parameters()):
            next_p = momentum * p.data + (1 - momentum) * new_p.data
            p.data = next_p

    def forward(self, x, edge_index_trans, edge_index_knn, edge_weight=None):
        aug1, aug2 = self.augmentor

        x1, ei1, ew1 = aug1(x, edge_index_trans, edge_weight)
        h1, h1_online = self.online_encoder(x1, ei1, ew1)

        x2, ei2, ew2 = aug2(x, edge_index_knn, edge_weight)
        h2, h2_online = self.online_encoder(x2, ei2, ew2)

        if self.use_subgraph_pool:
            h1_online = subgraph_pool(h1_online, ei1)
            h2_online = subgraph_pool(h2_online, ei2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            x1_t, ei1_t, ew1_t = aug1(x, edge_index_trans, edge_weight)
            _, h1_target = self.get_target_encoder()(x1_t, ei1_t, ew1_t)
            x2_t, ei2_t, ew2_t = aug2(x, edge_index_knn, edge_weight)
            _, h2_target = self.get_target_encoder()(x2_t, ei2_t, ew2_t)
            if self.use_subgraph_pool:
                h1_target = subgraph_pool(h1_target, ei1_t)
                h2_target = subgraph_pool(h2_target, ei2_t)

        if self.use_subgraph_pool:
            h1 = subgraph_pool(h1, ei1)
            h2 = subgraph_pool(h2, ei2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target


def bootstrap_latent_loss(h1_pred, h2_pred, h1_target, h2_target):
    loss = (2 - 2 * F.cosine_similarity(h1_pred, h2_target.detach(), dim=-1).mean() +
            2 - 2 * F.cosine_similarity(h2_pred, h1_target.detach(), dim=-1).mean())
    return loss


def train(encoder_model, x, edge_index_trans, edge_index_knn, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(
        x, edge_index_trans, edge_index_knn)
    loss = bootstrap_latent_loss(h1_pred, h2_pred, h1_target, h2_target)
    loss.backward()
    optimizer.step()
    encoder_model.update_target_encoder(0.99)
    return loss.item()


def test(seed, encoder_model, x, edge_index_trans, edge_index_knn, y, vis_save_path, skip_tsne=False):
    encoder_model.eval()
    with torch.no_grad():
        h1, h2, _, _, _, _ = encoder_model(x, edge_index_trans, edge_index_knn)
    z = torch.cat([h1, h2], dim=1)
    ari_score, sil_score = visualize_tsne(seed, z.detach().cpu().numpy(), y, save_path=vis_save_path, skip=skip_tsne)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = evaluate_with_metrics(z, y, split)
    return result, ari_score, sil_score


def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    print(f'##### device: {device}')

    data, _ = load_graph_data(args, device=device)
    print(f'Transaction graph: {data}')

    edge_index_knn = load_knn_graph(args.knn_graph, device=device)
    print(f'k-NN graph: {args.knn_graph} ({edge_index_knn.size(1)} edges)')

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    use_subgraph = getattr(args, 'subgraph_pool', False)
    gconv = GConv(input_dim=data.x.size(1), hidden_dim=args.hidden_dim,
                  num_layers=args.gconv_nlayers).to(device)
    encoder_model = Encoder(
        encoder=gconv, augmentor=(aug1, aug2), hidden_dim=args.hidden_dim,
        use_subgraph_pool=use_subgraph,
    ).to(device)
    print(f'Subgraph pooling: {use_subgraph}')

    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train(encoder_model, data.x,
                         data.edge_index, edge_index_knn, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs('./visualize/BGRL_KNN', exist_ok=True)
    vis_save_path = f'./visualize/BGRL_KNN/tsne_{args.model_name}.png'
    test_result, ari_score, sil_score = test(
        args.seed, encoder_model, data.x,
        data.edge_index, edge_index_knn, data.y, vis_save_path, args.skip_tsne)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}, F1_susp={test_result["f1_1"]:.4f}')

    result = build_result_dict(args.model_name, args, test_result, ari_score, sil_score, use_cen=False)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    assert args.knn_graph is not None, '--knn_graph required (e.g., HOFINET_KNN_CEN_k10)'
    print(args)
    main(args)
