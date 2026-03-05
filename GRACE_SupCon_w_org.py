import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import from_predefined_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

from utils import set_seed, save_results_to_csv
from torch_geometric.data import Data
import pandas as pd


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
        for i, conv in enumerate(self.layers):
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

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]

    # compute extra pos and neg masks for semi-supervised learning
    extra_pos_mask = torch.eq(data.y, data.y.unsqueeze(dim=1)).to('cuda')
    # construct extra supervision signals for only training samples
    extra_pos_mask[~data.train_mask][:, ~data.train_mask] = False
    extra_pos_mask.fill_diagonal_(False)
    # pos_mask: [N, 2N] for both inter-view and intra-view samples
    extra_pos_mask = torch.cat([extra_pos_mask, extra_pos_mask], dim=1).to('cuda')
    # fill interview positives only; pos_mask for intraview samples should have zeros in diagonal
    extra_pos_mask.fill_diagonal_(True)

    extra_neg_mask = torch.ne(data.y, data.y.unsqueeze(dim=1)).to('cuda')
    extra_neg_mask[~data.train_mask][:, ~data.train_mask] = True
    extra_neg_mask.fill_diagonal_(False)
    extra_neg_mask = torch.cat([extra_neg_mask, extra_neg_mask], dim=1).to('cuda')

    loss = contrast_model(h1=h1, h2=h2, extra_pos_mask=extra_pos_mask, extra_neg_mask=extra_neg_mask)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(encoder_model, data):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)
    split = from_predefined_split(data=data)
    result = LREvaluator()(z, data.y, split)
    return result


def main():
    seed = 2024
    set_seed(seed)  # 시드 고정

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'##### device: {device}')

    # Load your CSV data
    df = pd.read_csv('./_datasets/hf_ctgan_base_10000_3_CEN_FEAT_ALL.csv')[:100]
    df['Source'] = df['wd_fc_sn'].astype(str) + '_' + df['wd_ac_sn'].astype(str)
    df['Target'] = df['dps_fc_sn'].astype(str) + '_' + df['dps_ac_sn'].astype(str)
    df.drop(columns=['wd_fc_sn', 'wd_ac_sn', 'dps_fc_sn', 'dps_ac_sn'], inplace=True)

    df['ff_sp_ai'] = df['ff_sp_ai'].replace(pd.NA, 0)
    df['ff_sp_ai'] = df['ff_sp_ai'].replace('01', 1)
    df['ff_sp_ai'] = df['ff_sp_ai'].replace('02', 1)
    df['ff_sp_ai'] = df['ff_sp_ai'].replace('SP', 1)

    # Define node features (e.g., `tran_amt`, `md_type`, `fnd_type`)
    node_features = df[['tran_amt', 'md_type', 'fnd_type']].fillna(0).to_numpy()
    y = df['ff_sp_ai'].fillna(0).to_numpy()

    # Convert `WD_NODE` and `DPS_NODE` into categorical codes for graph construction
    source_nodes = df['Source'].astype('category').cat.codes
    target_nodes = df['Target'].astype('category').cat.codes
    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    # Create the PyTorch Geometric data object
    data = Data(x=torch.tensor(node_features, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.long))
    data = data.to(device)
    print(data)
    print(type(data))
    print(data.y)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    gconv = GConv(input_dim=node_features.shape[1], hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=1000, desc='(T)') as pbar:
        for epoch in range(1, 1001):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, data)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    results = []
    # 저장할 CSV 파일 경로
    csv_path = '_results/exp_results.csv'
    # 결과 저장
    results.append({
        'Model': 'GRACE_SupCon_w_org',
        'Seed': seed,
        'F1Mi': test_result["micro_f1"],
        'F1Ma': test_result["macro_f1"],
    })

    # 실험 결과를 CSV 파일로 저장
    save_results_to_csv(results, csv_path)

if __name__ == '__main__':
    main()
