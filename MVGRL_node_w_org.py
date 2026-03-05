import GCL.augmentors as A
import GCL.losses as L
import pandas as pd
import torch
from GCL.eval import get_split
from GCL.models import DualBranchContrast
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from tqdm import tqdm

from _config import get_config
from utils import set_seed, save_results_to_csv, evaluate_with_metrics, visualize_tsne
import os


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

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        # x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        x2, ei2, ew2 = aug2(x.cpu(), edge_index.cpu(), edge_weight)
        x2, edge_index2, edge_weight2 = x2.to(x.device), ei2.to(x.device), ew2.to(x.device)

        z1 = self.encoder1(x1, edge_index1, edge_weight1)
        z2 = self.encoder2(x2, edge_index2, edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1, edge_weight1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2, edge_weight2))
        return z1, z2, g1, g2, z1n, z2n


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, data.edge_index)

    # loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h1n=z1n, h2n=z2n)
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(seed, encoder_model, data, vis_save_path):
    encoder_model.eval()
    z1, z2, _, _, _, _ = encoder_model(data.x, data.edge_index)
    z = z1 + z2

    # 시각화
    ari_score, sil_score = visualize_tsne(seed, z.detach().cpu().numpy(), data.y, save_path=vis_save_path)

    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    # result = LREvaluator()(z, data.y, split)

    # 새로운 평가
    result = evaluate_with_metrics(z, data.y, split)

    return result, ari_score, sil_score


def main(args):
    seed = args.seed
    set_seed(seed)  # 시드 고정

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'##### device: {device}')

    # Load your CSV data
    df = pd.read_csv(f'./_datasets/{args.node_data_name}.csv')

    edge_df = pd.read_csv(f'./_datasets/{args.edge_data_name}.csv')
    node_index = {acc: i for i, acc in enumerate(df['account'])}
    src = edge_df['source'].map(node_index)
    tgt = edge_df['target'].map(node_index)
    edge_index = torch.tensor([src.values, tgt.values], dtype=torch.long)

    label = torch.tensor(df['label'].values, dtype=torch.long)

    x = torch.tensor(df[[c for c in df.columns if c.startswith(('out_', 'in_', 'md_', 'fnd_', 'entropy'))]].values,
                     dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=label).to(device)
    print(data)
    print(type(data))
    print(data.y)

    aug1 = A.Identity()
    aug2 = A.PPRDiffusion(alpha=0.2)
    gconv1 = GConv(input_dim=x.shape[1], hidden_dim=512, num_layers=2).to(device)
    gconv2 = GConv(input_dim=x.shape[1], hidden_dim=512, num_layers=2).to(device)
    encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=512).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.001)

    with tqdm(total=200, desc='(T)') as pbar:
        for epoch in range(1, 201):
            loss = train(encoder_model, contrast_model, data, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs(f'./_visualize/MVGRL', exist_ok=True)
    vis_save_path = f'./_visualize/MVGRL/tsne_MVGRL_w_org_{args.node_data_name}_0.0001_xshape_512_2_BootstrapLatent.png'
    test_result, ari_score, sil_score = test(seed, encoder_model, data, vis_save_path)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    results = []

    # 결과 저장
    results.append({
        'Model': 'MVGRL_node_w_org',
        'Data': args.node_data_name,
        'Seed': seed,
        'cen_feats': "None",
        'lr': -1,
        'input_dim': -1,
        'hidden_dim': -1,
        'gconv_nlayers': -1,
        'loss': args.loss,
        'pre_1': test_result['pre_1'],
        'rec_1': test_result['rec_1'],
        'f1_1': test_result['f1_1'],
        'pre_0': test_result['pre_0'],
        'rec_0': test_result['rec_0'],
        'f1_0': test_result['f1_0'],
        'F1Mi': test_result["micro_f1"],
        'F1Ma': test_result["macro_f1"],
        'auroc': test_result["auroc"],
        'auprc': test_result["auprc"],
        'ari_score': ari_score,
        'sil_score': sil_score
    })

    # 실험 결과를 CSV 파일로 저장
    save_results_to_csv(results, args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    main(args)
