"""
MVGRL with k-NN Graph View — contrastive learning between
original transaction graph and k-NN similarity graph using dual encoders.

Settings:
  (D) --knn_graph HOFINET_KNN_FEAT_k10  → feature similarity k-NN
  (F) --knn_graph HOFINET_KNN_CEN_k10   → centrality similarity k-NN
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.checkpoint import checkpoint
from GCL.eval import get_split
from GCL.models import DualBranchContrast
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
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x, edge_index_trans, edge_index_knn, edge_weight=None):
        # View 1: encoder1 on transaction graph
        if self.training:
            z1 = checkpoint(self.encoder1, x, edge_index_trans, edge_weight, use_reentrant=False)
        else:
            z1 = self.encoder1(x, edge_index_trans, edge_weight)

        # View 2: encoder2 on k-NN graph
        if self.training:
            z2 = checkpoint(self.encoder2, x, edge_index_knn, edge_weight, use_reentrant=False)
        else:
            z2 = self.encoder2(x, edge_index_knn, edge_weight)

        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))

        # Corrupted views
        x1n, ei1n, ew1n = self.corruption(x, edge_index_trans, edge_weight)
        x2n, ei2n, ew2n = self.corruption(x, edge_index_knn, edge_weight)

        if self.training:
            z1n = checkpoint(self.encoder1, x1n, ei1n, ew1n, use_reentrant=False)
            z2n = checkpoint(self.encoder2, x2n, ei2n, ew2n, use_reentrant=False)
        else:
            z1n = self.encoder1(x1n, ei1n, ew1n)
            z2n = self.encoder2(x2n, ei2n, ew2n)

        return z1, z2, g1, g2, z1n, z2n


def train(encoder_model, contrast_model, x, edge_index_trans, edge_index_knn, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(x, edge_index_trans, edge_index_knn)
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(seed, encoder_model, x, edge_index_trans, edge_index_knn, y, vis_save_path, skip_tsne=False):
    encoder_model.eval()
    with torch.no_grad():
        z1, z2, _, _, _, _ = encoder_model(x, edge_index_trans, edge_index_knn)
    z = z1 + z2
    ari_score, sil_score = visualize_tsne(seed, z.detach().cpu().numpy(), y, save_path=vis_save_path, skip=skip_tsne)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = evaluate_with_metrics(z, y, split)
    return result, ari_score, sil_score



def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    print(f'##### device: {device}')

    data, x_cen = load_graph_data(args, device=device)
    print(f'Transaction graph: {data}')

    # Load k-NN graph
    edge_index_knn = load_knn_graph(args.knn_graph, device=device)
    print(f'k-NN graph: {args.knn_graph} ({edge_index_knn.size(1)} edges)')

    # Single feature set — no projection needed, dual encoders
    input_dim = data.x.size(1)
    gconv1 = GConv(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.gconv_nlayers).to(device)
    gconv2 = GConv(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.gconv_nlayers).to(device)
    encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, hidden_dim=args.hidden_dim).to(device)

    loss_fn = create_loss(args.loss)
    contrast_model = DualBranchContrast(loss=loss_fn, mode='G2L').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    with tqdm(total=200, desc='(T)') as pbar:
        for epoch in range(1, 201):
            loss = train(encoder_model, contrast_model, data.x, data.edge_index, edge_index_knn, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs('./visualize/MVGRL_KNN', exist_ok=True)
    vis_save_path = f'./visualize/MVGRL_KNN/tsne_{args.model_name}.png'
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
