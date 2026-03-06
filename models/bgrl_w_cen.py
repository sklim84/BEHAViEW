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
from data_loader import load_graph_data
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


class Encoder(torch.nn.Module):
    def __init__(self, encoder, proj_agg, proj_cen, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.proj_agg = proj_agg
        self.proj_cen = proj_cen
        self.augmentor = augmentor
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

    def forward(self, x_agg, x_cen, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x_agg, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x_cen, edge_index, edge_weight)

        z1 = self.proj_agg(x1)
        h1, h1_online = self.online_encoder(z1, edge_index1, edge_weight1)

        z2 = self.proj_cen(x2)
        h2, h2_online = self.online_encoder(z2, edge_index2, edge_weight2)

        h1_pred = self.predictor(h1_online)
        h2_pred = self.predictor(h2_online)

        with torch.no_grad():
            z1 = self.proj_agg(x1)
            _, h1_target = self.get_target_encoder()(z1, edge_index1, edge_weight1)
            z2 = self.proj_cen(x2)
            _, h2_target = self.get_target_encoder()(z2, edge_index2, edge_weight2)

        return h1, h2, h1_pred, h2_pred, h1_target, h2_target


def bootstrap_latent_loss(h1_pred, h2_pred, h1_target, h2_target):
    """BootstrapLatent L2L loss without N×N matrix (memory-efficient)"""
    loss = (2 - 2 * F.cosine_similarity(h1_pred, h2_target.detach(), dim=-1).mean() +
            2 - 2 * F.cosine_similarity(h2_pred, h1_target.detach(), dim=-1).mean())
    return loss


def train(encoder_model, data, x_cen, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, _, h1_pred, h2_pred, h1_target, h2_target = encoder_model(data.x, x_cen, data.edge_index, data.edge_attr)
    loss = bootstrap_latent_loss(h1_pred, h2_pred, h1_target, h2_target)
    loss.backward()
    optimizer.step()
    encoder_model.update_target_encoder(0.99)
    return loss.item()


def test(seed, encoder_model, data, x_cen, vis_save_path):
    encoder_model.eval()
    h1, h2, _, _, _, _ = encoder_model(data.x, x_cen, data.edge_index)
    z = torch.cat([h1, h2], dim=1)
    ari_score, sil_score = visualize_tsne(seed, z.detach().cpu().numpy(), data.y, save_path=vis_save_path)
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = evaluate_with_metrics(z, data.y, split)
    return result, ari_score, sil_score


def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    print(f'##### device: {device}')

    data, x_cen = load_graph_data(args, device=device)
    print(data)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    gconv = GConv(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.gconv_nlayers).to(device)
    proj_agg = torch.nn.Linear(data.x.size(1), args.input_dim)
    proj_cen = torch.nn.Linear(x_cen.size(1), args.input_dim)

    encoder_model = Encoder(
        encoder=gconv, proj_agg=proj_agg, proj_cen=proj_cen,
        augmentor=(aug1, aug2), hidden_dim=args.hidden_dim
    ).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    total = 100
    with tqdm(total=total, desc='(T)') as pbar:
        for epoch in range(1, total + 1):
            loss = train(encoder_model, data, x_cen.to(device), optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs('./visualize/BGRL_L2L', exist_ok=True)
    cen_feats = "_".join(str(item) for item in args.cen_feats)
    vis_save_path = f'./visualize/BGRL_L2L/tsne_{args.model_name}_{args.node_data_name}_{cen_feats}_{args.lr}_{args.input_dim}_{args.hidden_dim}_{args.gconv_nlayers}_{args.loss}.png'
    test_result, ari_score, sil_score = test(args.seed, encoder_model, data, x_cen.to(device), vis_save_path)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    result = build_result_dict(args.model_name, args, test_result, ari_score, sil_score, use_cen=True)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    print(args)
    main(args)
