import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import GCL.augmentors as A
import GCL.losses as L
import torch
import torch.nn.functional as F
from GCL.eval import get_split
from GCL.models import DualBranchContrast
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from config import get_config
from data_loader import load_graph_data
from utils import set_seed, build_result_dict, save_results_to_csv, evaluate_with_metrics, visualize_tsne


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z = self.encoder(x, edge_index, edge_weight)
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2

    def project(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(train_loader, encoder_model, contrast_model, optimizer, device):
    encoder_model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z, z1, z2 = encoder_model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None))
        h1, h2 = [encoder_model.project(x) for x in (z1, z2)]
        loss = contrast_model(h1, h2)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(train_loader))


def test(seed, encoder_model, data, vis_save_path, num_layers=2, batch_size=4096, device='cuda'):
    encoder_model.eval()
    encoder_model.augmentor = (A.Identity(), A.Identity())

    out_dim = encoder_model.fc2.out_features
    Z = torch.empty((data.num_nodes, out_dim), device=device)

    test_loader = NeighborLoader(data, num_neighbors=[-1] * num_layers, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            z, _, _ = encoder_model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None))
            h = encoder_model.project(z)
            Z[batch.n_id] = h

    split = get_split(num_samples=Z.size()[0], train_ratio=0.1, test_ratio=0.8)
    ari_score, sil_score = visualize_tsne(seed, Z.detach().cpu().numpy(), data.y, save_path=vis_save_path)
    result = evaluate_with_metrics(Z, data.y, split)
    return result, ari_score, sil_score


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'##### device: {device}')

    data, _ = load_graph_data(args, device=device)
    print(data)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    gconv = GConv(input_dim=data.x.shape[1], hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    batch_size = 4096
    train_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=batch_size, shuffle=True)

    with tqdm(total=1000, desc='(T)') as pbar:
        for epoch in range(1, 1001):
            loss = train(train_loader, encoder_model, contrast_model, optimizer, device)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs('./visualize/GRACE', exist_ok=True)
    vis_save_path = f'./visualize/GRACE/tsne_GRACE_w_org_{args.node_data_name}.png'
    test_result, ari_score, sil_score = test(args.seed, encoder_model, data, vis_save_path, 2, batch_size, device)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    result = build_result_dict('GRACE_w_org', args, test_result, ari_score, sil_score, use_cen=False)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    main(args)
