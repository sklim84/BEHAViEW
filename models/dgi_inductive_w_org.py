import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import GCL.losses as L
import torch
from torch import nn
from torch.optim import Adam
from GCL.eval import get_split
from GCL.models import SingleBranchContrast
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.inits import uniform
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from config import get_config
from data_loader import load_graph_data
from utils import set_seed, build_result_dict, save_results_to_csv, evaluate_with_metrics, visualize_tsne


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activations = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(SAGEConv(input_dim, hidden_dim))
            else:
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.activations.append(nn.PReLU(hidden_dim))

    def forward(self, x, edge_index):
        z = x
        for conv, act in zip(self.layers, self.activations):
            z = conv(z, edge_index)
            z = act(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index):
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        g = self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))
        zn = self.encoder(*self.corruption(x, edge_index))
        return z, g, zn


def train(encoder_model, contrast_model, data, dataloader, optimizer, device):
    encoder_model.train()
    total_loss = total_examples = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        z, g, zn = encoder_model(batch.x, batch.edge_index)
        loss = contrast_model(h=z, g=g, hn=zn)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * z.shape[0]
        total_examples += z.shape[0]
    return total_loss / total_examples


def test(seed, encoder_model, data, dataloader, device, vis_save_path, skip_tsne=False):
    encoder_model.eval()
    zs, n_ids = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            z, _, _ = encoder_model(batch.x, batch.edge_index)
            zs.append(z.cpu())
            n_ids.append(batch.n_id.cpu())

    Z = torch.zeros(data.num_nodes, zs[0].shape[1])
    for z, nid in zip(zs, n_ids):
        Z[nid] = z

    ari_score, sil_score = visualize_tsne(seed, Z.numpy(), data.y, save_path=vis_save_path, skip=skip_tsne)
    split = get_split(num_samples=Z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = evaluate_with_metrics(Z, data.y, split)
    return result, ari_score, sil_score


def main(args):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    print(f'##### device: {device}')

    data, _ = load_graph_data(args, device=None)
    print(data)

    train_loader = NeighborLoader(data, num_neighbors=[10, 10, 25], batch_size=4096, shuffle=True)
    test_loader = NeighborLoader(data, num_neighbors=[10, 10, 25], batch_size=4096, shuffle=False)

    gconv = GConv(input_dim=data.x.shape[1], hidden_dim=512, num_layers=3).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=512).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=0.0001)

    with tqdm(total=30, desc='(T)') as pbar:
        for epoch in range(1, 31):
            loss = train(encoder_model, contrast_model, data, train_loader, optimizer, device)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs('./visualize/DGI_IND', exist_ok=True)
    vis_save_path = f'./visualize/DGI_IND/tsne_DGI_inductive_w_org_{args.node_data_name}.png'
    test_result, ari_score, sil_score = test(args.seed, encoder_model, data, test_loader, device, vis_save_path, args.skip_tsne)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    result = build_result_dict('DGI_inductive_w_org', args, test_result, ari_score, sil_score, use_cen=False)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    main(args)
