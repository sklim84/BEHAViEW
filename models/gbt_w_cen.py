import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import GCL.augmentors as A
import torch
from GCL.eval import get_split
from GCL.models.contrast_model import WithinEmbedContrast
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.loader import NeighborLoader
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
    def __init__(self, encoder, proj_behav, proj_struct, augmentor):
        super(Encoder, self).__init__()
        self.proj_behav = proj_behav
        self.proj_struct = proj_struct
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x_behav, x_struct, edge_index, edge_weight=None, return_base=False):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x_behav, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x_struct, edge_index, edge_weight)

        z1 = self.proj_behav(x1)
        z1 = self.encoder(z1, edge_index1, edge_weight1)

        z2 = self.proj_struct(x2)
        z2 = self.encoder(z2, edge_index2, edge_weight2)

        if return_base:
            z = self.proj_behav(x_behav)
            z = self.encoder(z, edge_index, edge_weight)
            return z, z1, z2

        return z1, z2


def train(loader, encoder_model, contrast_model, optimizer, x_struct, device):
    encoder_model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch_x_struct = x_struct[batch.n_id]

        optimizer.zero_grad()
        z1, z2 = encoder_model(batch.x, batch_x_struct, batch.edge_index, batch.edge_attr)
        z1, z2 = z1[:batch.batch_size], z2[:batch.batch_size]
        loss = contrast_model(z1, z2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(encoder_model, data, x_struct, device, batch_size=4096):
    encoder_model.eval()
    old_aug = encoder_model.augmentor
    encoder_model.augmentor = (A.Identity(), A.Identity())

    out_dim = encoder_model.encoder.conv2.out_channels
    Z = torch.empty((data.num_nodes, out_dim), device=device)

    test_loader = NeighborLoader(data, num_neighbors=[-1, -1], batch_size=batch_size, shuffle=False)

    try:
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                z, _, _ = encoder_model(
                    batch.x, x_struct[batch.n_id],
                    batch.edge_index, getattr(batch, 'edge_attr', None),
                    return_base=True)
                Z[batch.n_id[:batch.batch_size]] = z[:batch.batch_size]
    finally:
        encoder_model.augmentor = old_aug
    return Z


def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}')
    print(f'##### device: {device}')

    data, x_struct = load_graph_data(args, device=None)
    print(data)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    gconv = GConv(input_dim=args.input_dim, hidden_dim=args.hidden_dim).to(device)
    proj_behav = torch.nn.Linear(data.x.shape[1], args.input_dim)
    proj_struct = torch.nn.Linear(x_struct.shape[1], args.input_dim)

    encoder_model = Encoder(encoder=gconv, proj_behav=proj_behav, proj_struct=proj_struct, augmentor=(aug1, aug2)).to(device)
    loss_fn = create_loss(args.loss)
    contrast_model = WithinEmbedContrast(loss=loss_fn).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    batch_size = 4096
    train_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=batch_size, shuffle=True)
    x_struct_gpu = x_struct.to(device)

    with tqdm(total=4000, desc='(T)') as pbar:
        for epoch in range(1, 4001):
            loss = train(train_loader, encoder_model, contrast_model, optimizer, x_struct_gpu, device)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    Z = test(encoder_model, data, x_struct_gpu, device, batch_size)
    z = Z.detach().cpu()

    os.makedirs('./visualize/GBT', exist_ok=True)
    struct_feats = "_".join(str(item) for item in args.struct_feats)
    vis_save_path = f'./visualize/GBT/tsne_{args.model_name}_{args.node_data_name}_{struct_feats}_{args.lr}_{args.input_dim}_{args.hidden_dim}_{args.gconv_nlayers}_{args.loss}.png'

    ari_score, sil_score = visualize_tsne(args.seed, z.numpy(), data.y, save_path=vis_save_path, skip=args.skip_tsne)
    split = get_split(num_samples=z.size(0), train_ratio=0.1, test_ratio=0.8)
    test_result = evaluate_with_metrics(z, data.y, split)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}, F1_susp={test_result["f1_1"]:.4f}')

    result = build_result_dict('GBT_w_cen', args, test_result, ari_score, sil_score, use_cen=True)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    print(args)
    main(args)
