import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import GCL.augmentors as A
import GCL.losses as L
import torch
from torch import nn
from torch.optim import Adam
from GCL.eval import get_split
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from tqdm import tqdm

from config import get_config
from data_loader import load_graph_data
from utils import set_seed, build_result_dict, save_results_to_csv, evaluate_with_metrics, visualize_tsne


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


def subgraph_pool(z, edge_index):
    N, D = z.size()
    src, tgt = edge_index[0], edge_index[1]
    neigh_sum = torch.zeros(N, D, device=z.device)
    neigh_count = torch.zeros(N, 1, device=z.device)
    neigh_sum.scatter_add_(0, tgt.unsqueeze(1).expand(-1, D), z[src])
    neigh_count.scatter_add_(0, tgt.unsqueeze(1), torch.ones(src.size(0), 1, device=z.device))
    return (neigh_sum + z) / (neigh_count + 1).clamp(min=1)


class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim, use_subgraph_pool=False):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)
        self.use_subgraph_pool = use_subgraph_pool

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def _compute_summary(self, z, edge_index):
        if self.use_subgraph_pool:
            return self.project(torch.sigmoid(subgraph_pool(z, edge_index)))
        else:
            return self.project(torch.sigmoid(z.mean(dim=0, keepdim=True)))

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        z1 = self.encoder1(x1, edge_index1, edge_weight1)
        z2 = self.encoder2(x2, edge_index2, edge_weight2)
        g1 = self._compute_summary(z1, edge_index1)
        g2 = self._compute_summary(z2, edge_index2)
        z1n = self.encoder1(*self.corruption(x1, edge_index1, edge_weight1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2, edge_weight2))
        return z1, z2, g1, g2, z1n, z2n


def subgraph_contrastive_loss_dual(z1, z2, g1, g2, z1n, z2n):
    """Per-node dual-branch subgraph contrastive loss."""
    pos1 = torch.sigmoid((z1 * g2).sum(dim=1))
    pos2 = torch.sigmoid((z2 * g1).sum(dim=1))
    neg1 = torch.sigmoid((z1n * g2).sum(dim=1))
    neg2 = torch.sigmoid((z2n * g1).sum(dim=1))
    loss = (-torch.log(pos1 + 1e-8).mean() - torch.log(1 - neg1 + 1e-8).mean()
            - torch.log(pos2 + 1e-8).mean() - torch.log(1 - neg2 + 1e-8).mean())
    return loss


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, data.edge_index)
    if encoder_model.use_subgraph_pool:
        loss = subgraph_contrastive_loss_dual(z1, z2, g1, g2, z1n, z2n)
    else:
        loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(seed, encoder_model, data, vis_save_path, skip_tsne=False):
    encoder_model.eval()
    with torch.no_grad():
        z1, z2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = z1 + z2
    ari_score, sil_score = visualize_tsne(seed, z.detach().cpu().numpy(), data.y, save_path=vis_save_path, skip=skip_tsne)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = evaluate_with_metrics(z, data.y, split)
    return result, ari_score, sil_score


def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    print(f'##### device: {device}')

    data, _ = load_graph_data(args, device=device)
    print(data)

    aug1 = A.Identity()
    aug2 = A.EdgeRemoving(pe=0.3)
    use_subgraph = getattr(args, 'subgraph_pool', False)
    gconv1 = GConv(input_dim=data.x.shape[1], hidden_dim=args.hidden_dim, num_layers=args.gconv_nlayers).to(device)
    gconv2 = GConv(input_dim=data.x.shape[1], hidden_dim=args.hidden_dim, num_layers=args.gconv_nlayers).to(device)
    encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2),
                           hidden_dim=args.hidden_dim, use_subgraph_pool=use_subgraph).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)
    print(f'Subgraph pooling: {use_subgraph}')

    with tqdm(total=200, desc='(T)') as pbar:
        for epoch in range(1, 201):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs('./visualize/MVGRL', exist_ok=True)
    vis_save_path = f'./visualize/MVGRL/tsne_MVGRL_w_org_{args.node_data_name}.png'
    test_result, ari_score, sil_score = test(args.seed, encoder_model, data, vis_save_path, args.skip_tsne)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}, F1_susp={test_result["f1_1"]:.4f}')

    result = build_result_dict('MVGRL_node_w_org', args, test_result, ari_score, sil_score, use_cen=False)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    main(args)
