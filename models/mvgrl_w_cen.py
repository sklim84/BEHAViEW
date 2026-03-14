import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import GCL.augmentors as A
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
from data_loader import load_graph_data
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
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, x_agg, x_cen, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x_agg, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x_cen, edge_index, edge_weight)

        # Pre-compute corrupted inputs (random permutation) before checkpoint
        x1n, edge_index1n, edge_weight1n = self.corruption(x1, edge_index1, edge_weight1)
        x2n, edge_index2n, edge_weight2n = self.corruption(x2, edge_index2, edge_weight2)

        # Gradient checkpointing: recompute intermediates during backward to save memory
        z1 = checkpoint(self.encoder1, x1, edge_index1, edge_weight1, use_reentrant=False)
        z2 = checkpoint(self.encoder2, x2, edge_index2, edge_weight2, use_reentrant=False)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = checkpoint(self.encoder1, x1n, edge_index1n, edge_weight1n, use_reentrant=False)
        z2n = checkpoint(self.encoder2, x2n, edge_index2n, edge_weight2n, use_reentrant=False)
        return z1, z2, g1, g2, z1n, z2n


def train(encoder_model, contrast_model, data, x_cen, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, x_cen, data.edge_index)
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(seed, encoder_model, data, x_cen, vis_save_path, skip_tsne):
    encoder_model.eval()
    with torch.no_grad():
        z1, z2, _, _, _, _ = encoder_model(data.x, x_cen, data.edge_index)
    z = z1 + z2
    ari_score, sil_score = visualize_tsne(seed, z.detach().cpu().numpy(), data.y, save_path=vis_save_path, skip=skip_tsne)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = evaluate_with_metrics(z, data.y, split)
    return result, ari_score, sil_score


def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    print(f'##### device: {device}')

    data, x_cen = load_graph_data(args, device=device)
    print(data)

    aug1 = A.Identity()
    aug2 = A.EdgeRemoving(pe=0.3)
    gconv1 = GConv(input_dim=data.x.shape[1], hidden_dim=args.hidden_dim, num_layers=args.gconv_nlayers).to(device)
    gconv2 = GConv(input_dim=x_cen.shape[1], hidden_dim=args.hidden_dim, num_layers=args.gconv_nlayers).to(device)
    encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=args.hidden_dim).to(device)

    loss_fn = create_loss(args.loss)
    contrast_model = DualBranchContrast(loss=loss_fn, mode='G2L').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    x_cen_gpu = x_cen.to(device)

    with tqdm(total=200, desc='(T)') as pbar:
        for epoch in range(1, 201):
            loss = train(encoder_model, contrast_model, data, x_cen_gpu, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs('./visualize/MVGRL', exist_ok=True)
    cen_feats = "_".join(str(item) for item in args.cen_feats)
    vis_save_path = f'./visualize/MVGRL/tsne_{args.model_name}_{args.node_data_name}_{cen_feats}_{args.lr}_{args.input_dim}_{args.hidden_dim}_{args.gconv_nlayers}_{args.loss}.png'
    test_result, ari_score, sil_score = test(args.seed, encoder_model, data, x_cen_gpu, vis_save_path, args.skip_tsne)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    result = build_result_dict('MVGRL_node_w_cen', args, test_result, ari_score, sil_score, use_cen=True)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    print(args)
    main(args)
