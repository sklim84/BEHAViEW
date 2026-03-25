import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import GCL.augmentors as A
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
from utils import set_seed, create_loss, build_result_dict, save_results_to_csv, evaluate_with_metrics, visualize_tsne


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation
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
    def __init__(self, encoder, proj_behav, proj_struct, augmentor1, augmentor2, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.proj_behav = proj_behav
        self.proj_struct = proj_struct
        self.encoder = encoder
        self.augmentor1 = augmentor1
        self.augmentor2 = augmentor2
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x_behav, x_struct, edge_index, edge_weight=None):
        x1, edge_index1, edge_weight1 = self.augmentor1(x_behav, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = self.augmentor2(x_struct, edge_index, edge_weight)

        z1 = self.proj_behav(x1)
        z1 = self.encoder(z1, edge_index1, edge_weight1)

        z2 = self.proj_struct(x2)
        z2 = self.encoder(z2, edge_index2, edge_weight2)

        return z1, z2

    def project(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(loader, encoder_model, contrast_model, optimizer, x_struct, device):
    encoder_model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch_x_struct = x_struct[batch.n_id]

        optimizer.zero_grad()
        z1, z2 = encoder_model(batch.x, batch_x_struct, batch.edge_index, batch.edge_attr)
        h1 = encoder_model.project(z1)
        h2 = encoder_model.project(z2)
        h1, h2 = h1[:batch.batch_size], h2[:batch.batch_size]

        loss = contrast_model(h1, h2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(encoder_model, data, x_struct, num_layers=2, batch_size=4096, device=None):
    encoder_model.eval()
    old_aug1 = encoder_model.augmentor1
    old_aug2 = encoder_model.augmentor2
    encoder_model.augmentor1 = A.Identity()
    encoder_model.augmentor2 = A.Identity()

    out_dim = encoder_model.fc2.out_features
    Z = torch.empty((data.num_nodes, out_dim), device=device)

    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1] * num_layers,
        batch_size=batch_size,
        shuffle=False
    )

    try:
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                z1, _ = encoder_model(
                    batch.x,
                    x_struct[batch.n_id],
                    batch.edge_index,
                    getattr(batch, 'edge_attr', None)
                )
                h1 = encoder_model.project(z1)
                Z[batch.n_id[:batch.batch_size]] = h1[:batch.batch_size]
    finally:
        encoder_model.augmentor1 = old_aug1
        encoder_model.augmentor2 = old_aug2

    return Z


def run_experiment(data, x_struct, args, device, vis_save_path):
    gconv = GConv(args.input_dim, args.hidden_dim, torch.nn.ReLU(), args.gconv_nlayers).to(device)
    proj_behav = torch.nn.Linear(data.x.shape[1], args.input_dim).to(device)
    proj_struct = torch.nn.Linear(x_struct.shape[1], args.input_dim).to(device)

    encoder_model = Encoder(
        encoder=gconv,
        proj_behav=proj_behav,
        proj_struct=proj_struct,
        augmentor1=A.Compose([A.EdgeRemoving(pe=0.0), A.FeatureMasking(pf=0.3)]),
        augmentor2=A.Compose([A.EdgeRemoving(pe=0.0), A.FeatureMasking(pf=0.3)]),
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim
    ).to(device)

    loss_fn = create_loss(args.loss)
    contrast_model = DualBranchContrast(loss=loss_fn, mode='L2L', intraview_negs=True).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    batch_size = 4096
    train_loader = NeighborLoader(data, num_neighbors=[10, 10], batch_size=batch_size, shuffle=True)
    x_struct_gpu = x_struct.to(device)

    with tqdm(total=1000, desc='(T) With Centrality') as pbar:
        for epoch in range(1, 1001):
            loss = train(train_loader, encoder_model, contrast_model, optimizer, x_struct_gpu, device)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    Z = test(encoder_model, data, x_struct_gpu, args.gconv_nlayers, batch_size, device)
    z1 = Z.detach().cpu()

    ari_score, sil_score = visualize_tsne(args.seed, z1.numpy(), data.y, save_path=vis_save_path, skip=args.skip_tsne)
    split = get_split(num_samples=z1.size(0), train_ratio=0.1, test_ratio=0.8)
    eval_result = evaluate_with_metrics(z1, data.y, split)

    return eval_result, ari_score, sil_score


def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"##### device: {device}")

    data, x_struct = load_graph_data(args, device=None)
    print(data)

    os.makedirs('./visualize/GRACE', exist_ok=True)
    struct_feats = "_".join(str(item) for item in args.struct_feats)
    vis_save_path = f'./visualize/GRACE/tsne_{args.model_name}_{args.node_data_name}_{struct_feats}_{args.lr}_{args.input_dim}_{args.hidden_dim}_{args.proj_dim}_{args.gconv_nlayers}_{args.loss}.png'

    test_result, ari_score, sil_score = run_experiment(data, x_struct, args, device, vis_save_path)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}, F1_susp={test_result["f1_1"]:.4f}')

    result = build_result_dict('GRACE_w_cen', args, test_result, ari_score, sil_score, use_cen=True)
    save_results_to_csv([result], args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    print(args)
    main(args)
