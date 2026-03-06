import torch
import os.path as osp
import GCL.losses as L

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from utils import set_seed, save_results_to_csv
from torch_geometric.data import Data
import pandas as pd


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        z, g = encoder_model(data.x, data.edge_index, data.batch)
        z, g = encoder_model.project(z, g)
        loss = contrast_model(h=z, g=g, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        z, g = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=True)(x, y, split)
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

    # # # Create the PyTorch Geometric data object
    # data = Data(x=torch.tensor(node_features, dtype=torch.float),
    #             edge_index=edge_index,
    #             y=torch.tensor(y, dtype=torch.long))
    # data = data.to(device)
    # print(data)
    # print(type(data))
    # print(data.y)

    df['Source'] = df['Source'].astype('category').cat.codes
    df['Target'] = df['Target'].astype('category').cat.codes

    data_list = []
    for i in range(len(df)):
        node_features = df.iloc[i][['tran_amt', 'md_type', 'fnd_type']].fillna(0).to_numpy()
        y = df.iloc[i]['ff_sp_ai']
        source_node = df.iloc[i]['Source']
        target_node = df.iloc[i]['Target']

        edge_index = torch.tensor([[source_node], [target_node]], dtype=torch.long)

        data = Data(x=torch.tensor(node_features, dtype=torch.float).unsqueeze(0),
                    edge_index=edge_index,
                    y=torch.tensor([y], dtype=torch.long))
        data_list.append(data)


    # 2. DataLoader 생성
    dataloader = DataLoader(data_list, batch_size=1)

    gconv = GConv(input_dim=data_list[0].x.shape[1], hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    fc1 = FC(hidden_dim=32 * 2)
    fc2 = FC(hidden_dim=32 * 2)
    encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=100, desc='(T)') as pbar:
        for epoch in range(1, 101):
            loss = train(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(encoder_model, dataloader)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
