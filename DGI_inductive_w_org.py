import GCL.losses as L
import pandas as pd
import torch
from GCL.eval import get_split
from GCL.models import SingleBranchContrast
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.inits import uniform
from tqdm import tqdm

from _config import get_config
from utils import set_seed, save_results_to_csv, evaluate_with_metrics, visualize_tsne
import os


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

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.layers[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x


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


def train(encoder_model, contrast_model, data, dataloader, optimizer):
    encoder_model.train()
    total_loss = total_examples = 0
    for batch_size, node_id, adjs in dataloader:
        adjs = [adj.to('cuda') for adj in adjs]
        optimizer.zero_grad()
        z, g, zn = encoder_model(data.x[node_id], adjs)
        loss = contrast_model(h=z, g=g, hn=zn)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * z.shape[0]
        total_examples += z.shape[0]
    return total_loss / total_examples


def test(seed, encoder_model, data, dataloader, vis_save_path):
    encoder_model.eval()
    zs = []
    for i, (batch_size, node_id, adjs) in enumerate(dataloader):
        adjs = [adj.to('cuda') for adj in adjs]
        z, _, _ = encoder_model(data.x[node_id], adjs)
        zs.append(z)
    x = torch.cat(zs, dim=0)

    # 시각화
    ari_score, sil_score = visualize_tsne(seed, x.detach().cpu().numpy(), data.y, save_path=vis_save_path)

    split = get_split(num_samples=x.size()[0], train_ratio=0.1, test_ratio=0.8)
    # result = LREvaluator()(x, data.y, split)

    # 새로운 평가
    result = evaluate_with_metrics(x, data.y, split)

    return result, ari_score, sil_score


def main(args):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

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

    train_loader = NeighborSampler(data.edge_index, node_idx=None,
                                   sizes=[10, 10, 25], batch_size=128,
                                   shuffle=True, num_workers=32)
    test_loader = NeighborSampler(data.edge_index, node_idx=None,
                                  sizes=[10, 10, 25], batch_size=128,
                                  shuffle=False, num_workers=32)

    gconv = GConv(input_dim=x.shape[1], hidden_dim=512, num_layers=3).to(device)
    encoder_model = Encoder(encoder=gconv, hidden_dim=512).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.0001)

    with tqdm(total=30, desc='(T)') as pbar:
        for epoch in range(1, 31):
            loss = train(encoder_model, contrast_model, data, train_loader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs(f'./_visualize/DGI_IND', exist_ok=True)
    vis_save_path = f'./_visualize/DGI_IND/tsne_DGI_inductive_w_org_{args.node_data_name}_0.0001_xshape_512_3_JSD.png'
    test_result, ari_score, sil_score= test(seed, encoder_model, data, test_loader, vis_save_path)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    results = []

    # 결과 저장
    results.append({
        'Model': 'DGI_inductive_w_org',
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
