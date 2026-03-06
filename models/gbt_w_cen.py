import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import GCL.augmentors as A
import torch
from GCL.eval import get_split
from GCL.models.contrast_model import WithinEmbedContrast
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from config import get_config
from data_loader import load_graph_data
from utils import set_seed, create_loss, build_result_dict, save_results_to_csv, evaluate_with_metrics, visualize_tsne


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GConv, self).__init__()
        self.act = torch.nn.PReLU()
        self.bn = torch.nn.BatchNorm1d(2 * hidden_dim, momentum=0.01)
        self.conv1 = GCNConv(input_dim, 2 * hidden_dim, cached=False)
        self.conv2 = GCNConv(2 * hidden_dim, hidden_dim, cached=False)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.conv1(x, edge_index, edge_weight)
        z = self.bn(z)
        z = self.act(z)
        z = self.conv2(z, edge_index, edge_weight)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder, proj_agg, proj_cen, augmentor):
        super(Encoder, self).__init__()
        self.proj_agg = proj_agg
        self.proj_cen = proj_cen
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x_agg, x_cen, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x_agg, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x_cen, edge_index, edge_weight)

        z = self.proj_agg(x_agg)
        z = self.encoder(z, edge_index, edge_weight)

        z1 = self.proj_agg(x1)
        z1 = self.encoder(z1, edge_index1, edge_weight1)

        z2 = self.proj_cen(x2)
        z2 = self.encoder(z2, edge_index2, edge_weight2)

        return z, z1, z2


def train(encoder_model, contrast_model, data, x_cen, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    _, z1, z2 = encoder_model(data.x, x_cen, data.edge_index, data.edge_attr)
    loss = contrast_model(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(seed, encoder_model, data, x_cen, vis_save_path):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, x_cen, data.edge_index, data.edge_attr)
    ari_score, sil_score = visualize_tsne(seed, z.detach().cpu().numpy(), data.y, save_path=vis_save_path, skip=args.skip_tsne)
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

    gconv = GConv(input_dim=args.input_dim, hidden_dim=args.hidden_dim).to(device)
    proj_agg = torch.nn.Linear(data.x.shape[1], args.input_dim)
    proj_cen = torch.nn.Linear(x_cen.shape[1], args.input_dim)

    encoder_model = Encoder(encoder=gconv, proj_agg=proj_agg, proj_cen=proj_cen, augmentor=(aug1, aug2)).to(device)
    loss_fn = create_loss(args.loss)
    contrast_model = WithinEmbedContrast(loss=loss_fn).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    with tqdm(total=4000, desc='(T)') as pbar:
        for epoch in range(1, 4001):
            loss = train(encoder_model, contrast_model, data, x_cen.to(device), optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs('./visualize/GBT', exist_ok=True)
    cen_feats = "_".join(str(item) for item in args.cen_feats)
    vis_save_path = f'./visualize/GBT/tsne_{args.model_name}_{args.node_data_name}_{cen_feats}_{args.lr}_{args.input_dim}_{args.hidden_dim}_{args.gconv_nlayers}_{args.loss}.png'
    test_result, ari_score, sil_score = test(args.seed, encoder_model, data, x_cen.to(device), vis_save_path)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    result = build_result_dict('GBT_w_cen', args, test_result, ari_score, sil_score, use_cen=True)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    print(args)
    main(args)
