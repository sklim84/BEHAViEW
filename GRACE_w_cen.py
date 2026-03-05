import GCL.augmentors as A
import GCL.losses as L
import pandas as pd
import torch
import torch.nn.functional as F
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from _config import get_config
from utils import set_seed, save_results_to_csv, evaluate_with_metrics, visualize_tsne
import os


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
    def __init__(self, encoder, proj_agg, proj_cen, augmentor1, augmentor2, hidden_dim, proj_dim):
        super(Encoder, self).__init__()

        # 노드 피처와 중심성 피처 shape을 맞추기 위한 projection layer
        self.proj_agg = proj_agg
        self.proj_cen = proj_cen

        self.encoder = encoder  # 노드 피처에 대한 GCN

        self.augmentor1 = augmentor1  # 노드 피처에 대한 augmentation
        self.augmentor2 = augmentor2  # 중심성 피처에 대한 augmentation

        # 임베딩 투영 레이어
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x_agg, x_cen, edge_index, edge_weight=None):
        # Augmentors 적용
        x1, edge_index1, edge_weight1 = self.augmentor1(x_agg, edge_index, edge_weight)  # 노드 피처에 대해 적용
        x2, edge_index2, edge_weight2 = self.augmentor2(x_cen, edge_index, edge_weight)  # 중앙성 피처에 대해 적용

        # 두 augmentation의 차원 일치 확인을 위해 각 augmentation에 대해 별도의 레이어 적용

        # 증강된 그래프의 임베딩
        z1 = self.proj_agg(x1)
        z1 = self.encoder(z1, edge_index1, edge_weight1)

        z2 = self.proj_cen(x2)
        z2 = self.encoder(z2, edge_index2, edge_weight2)

        return z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


def train(encoder_model, contrast_model, data, x_cen, optimizer):
    encoder_model.train()
    optimizer.zero_grad()

    # 노드 피처와 중앙성 피처를 각각 모델에 전달
    z1, z2 = encoder_model(data.x, x_cen, data.edge_index, data.edge_attr)

    # 프로젝트된 임베딩을 통해 대조 학습
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(seed, encoder_model, data, x_cen, vis_save_path):
    encoder_model.eval()

    # 노드 피처와 중앙성 피처를 각각 모델에 전달
    z1, z2 = encoder_model(data.x, x_cen, data.edge_index, data.edge_attr)

    # 시각화
    ari_score, sil_score = visualize_tsne(seed, z1.detach().cpu().numpy(), data.y, save_path=vis_save_path)

    # 임베딩 결과에 대해 평가
    split = get_split(num_samples=z1.size()[0], train_ratio=0.1, test_ratio=0.8)
    # result = LREvaluator()(z1, data.y, split)

    # 새로운 평가
    result = evaluate_with_metrics(z1, data.y, split)

    return result, z1.detach().cpu().numpy(), ari_score, sil_score  # 임베딩도 반환


def run_experiment(data, x_cen, args, vis_save_path):
    device = torch.device(f'cuda:{args.gpu}')

    input_dim = args.input_dim
    hidden_dim = args.input_dim  # 32
    proj_dim = args.proj_dim  # 32
    num_layers = args.gconv_nlayers  # 2

    # Augmentation 설정: aug1은 노드 피처에, aug2는 중앙성 피처에 적용
    aug1 = A.Compose([A.EdgeRemoving(pe=0.0), A.FeatureMasking(pf=0.3)])  # 노드 피처에 적용
    aug2 = A.Compose([A.EdgeRemoving(pe=0.0), A.FeatureMasking(pf=0.3)])  # 중앙성 피처에 적용

    # GCN 모델 설정
    gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=num_layers).to(
        device)

    # 노드 피처와 중심성 피처 shape을 맞추기 위한 projection layer
    proj_agg = torch.nn.Linear(data.x.shape[1], input_dim)
    proj_cen = torch.nn.Linear(x_cen.shape[1], input_dim)

    # Encoder 설정
    encoder_model = Encoder(
        encoder=gconv,
        proj_agg=proj_agg,
        proj_cen=proj_cen,
        augmentor1=aug1,
        augmentor2=aug2,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim
    ).to(device)

    if args.loss == 'BootstrapLatent':
        loss = L.BootstrapLatent()
    elif args.loss == 'JSD':
        loss = L.JSD()
    elif args.loss == 'BarlowTwins':
        loss = L.BarlowTwins()
    elif args.loss == 'InfoNCE':
        loss = L.InfoNCE(tau=0.2)

    # contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)
    contrast_model = DualBranchContrast(loss=loss, mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=args.lr)  # 0.01

    # edge_index의 dtype을 torch.long으로 변환 (정수형이어야 함)
    if data.edge_index.dtype != torch.long:
        data.edge_index = data.edge_index.long()

    # 중앙성 피처는 별도로 처리하므로 노드 피처와 결합하지 않음
    with tqdm(total=1000, desc=f'(T) With Centrality') as pbar:
        for epoch in range(1, 1001):
            loss = train(encoder_model, contrast_model, data, x_cen.to(device), optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result, embeddings, ari_score, sil_score = test(args.seed, encoder_model, data, x_cen.to(device),
                                                         vis_save_path)
    return test_result, embeddings, ari_score, sil_score


def main(args):
    seed = args.seed
    set_seed(seed)  # 시드 고정

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

    os.makedirs(f'./_visualize/GRACE', exist_ok=True)
    cen_feats = "_".join(str(item) for item in args.cen_feats)
    vis_save_path = f'./_visualize/GRACE/tsne_{args.model_name}_{args.node_data_name}_{cen_feats}_{args.lr}_{args.input_dim}_{args.hidden_dim}_{args.proj_dim}_{args.gconv_nlayers}_{args.loss}.png'

    # 실험: 중앙성 지표 포함
    test_result, embeddings_with_centrality, ari_score, sil_score = run_experiment(data, x_cen, args, vis_save_path)
    print(test_result)
    print(
        f'(E) With Centrality: Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    results = []
    # 결과 저장
    results.append({
        'Model': 'GRACE_w_cen',
        'Data': args.node_data_name,
        'Seed': seed,
        'cen_feats': args.cen_feats,
        'lr': args.lr,
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'proj_dim': args.proj_dim,
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
