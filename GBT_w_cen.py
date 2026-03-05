import GCL.augmentors as A
import GCL.losses as L
import pandas as pd
import torch
from GCL.eval import get_split, LREvaluator
from GCL.models.contrast_model import WithinEmbedContrast
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from _config import get_config
from utils import set_seed, save_results_to_csv, evaluate_with_metrics, visualize_tsne
import os


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

        # 노드 피처와 중심성 피처 shape을 맞추기 위한 projection layer
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

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f'cuda:{args.gpu}')
    print(f'##### device: {device}')

    # Load your CSV data
    df = pd.read_csv(f'./_datasets/{args.node_data_name}.csv')

    edge_df = pd.read_csv(f'./_datasets/{args.edge_data_name}.csv')
    node_index = {acc: i for i, acc in enumerate(df['account'])}
    src = edge_df['source'].map(node_index)
    tgt = edge_df['target'].map(node_index)
    edge_index = torch.tensor([src.values, tgt.values], dtype=torch.long)

    label = torch.tensor(df['label'].values, dtype=torch.long)

    x_agg = torch.tensor(df[[c for c in df.columns if c.startswith(('out_', 'in_', 'md_', 'fnd_', 'entropy'))]].values,
                         dtype=torch.float)
    x_cen = torch.tensor(df[[c for c in df.columns if c in args.cen_feats]].values, dtype=torch.float)

    data = Data(x=x_agg, edge_index=edge_index, y=label).to(device)
    print(data)
    print(type(data))
    print(data.y)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.5), A.FeatureMasking(pf=0.1)])

    input_dim = args.input_dim
    # hidden_dim: 256
    gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    # 노드 피처와 중심성 피처 shape을 맞추기 위한 projection layer
    proj_agg = torch.nn.Linear(data.x.shape[1], input_dim)
    proj_cen = torch.nn.Linear(x_cen.shape[1], input_dim)

    encoder_model = Encoder(encoder=gconv,
                            proj_agg=proj_agg,
                            proj_cen=proj_cen,
                            augmentor=(aug1, aug2)).to(device)

    if args.loss == 'BootstrapLatent':
        loss = L.BootstrapLatent()
    elif args.loss == 'JSD':
        loss = L.JSD()
    elif args.loss == 'BarlowTwins':
        loss = L.BarlowTwins()
    elif args.loss == 'InfoNCE':
        loss = L.InfoNCE(tau=0.2)

    # contrast_model = WithinEmbedContrast(loss=L.BarlowTwins()).to(device)
    contrast_model = WithinEmbedContrast(loss=loss).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr)  # 5e-4

    with tqdm(total=4000, desc='(T)') as pbar:
        for epoch in range(1, 4001):
            loss = train(encoder_model, contrast_model, data, x_cen.to(device), optimizer)
            # scheduler.step()
            pbar.set_postfix({'loss': loss})
            pbar.update()

    os.makedirs(f'./_visualize/GBT', exist_ok=True)
    cen_feats = "_".join(str(item) for item in args.cen_feats)
    vis_save_path = f'./_visualize/GBT/tsne_{args.model_name}_{args.node_data_name}_{cen_feats}_{args.lr}_{args.input_dim}_{args.hidden_dim}_{args.gconv_nlayers}_{args.loss}.png'
    test_result, ari_score, sil_score  = test(seed, encoder_model, data, x_cen.to(device), vis_save_path)
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    results = []
    # 결과 저장
    results.append({
        'Model': 'GBT_w_cen',
        'Data': args.node_data_name,
        'Seed': seed,
        'cen_feats': args.cen_feats,
        'lr': args.lr,
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'gconv_nlayers': args.gconv_nlayers,
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
    print(args)

    main(args)
