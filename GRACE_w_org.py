import GCL.augmentors as A
import GCL.losses as L
import pandas as pd
import torch
import torch.nn.functional as F
from GCL.eval import get_split
from GCL.models import DualBranchContrast
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm import tqdm

from _config import get_config
from utils import set_seed, save_results_to_csv, evaluate_with_metrics, visualize_tsne
import os
from torch_geometric.loader import NeighborLoader

class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()  # 활성화 함수 설정 (예: ReLU)
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))  # 첫 번째 GCN 레이어
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))  # 추가 GCN 레이어

    def forward(self, x, edge_index, edge_weight=None):
        z = x  # 입력 feature를 임베딩으로 초기화
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)  # 각 GCN 레이어를 통과하며 임베딩 업데이트
            z = self.activation(z)  # 활성화 함수 적용
        return z  # 최종 임베딩 반환


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.encoder = encoder  # GCN 인코더 모델
        self.augmentor = augmentor  # 두 개의 증강기 (augmentor) 설정

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)  # 임베딩을 프로젝션하는 첫 번째 FC 레이어
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)  # 프로젝션 후 임베딩 복구하는 두 번째 FC 레이어

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor  # 두 개의 증강 방법 가져오기
        # 첫 번째 증강을 통해 그래프 변형
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        # 두 번째 증강을 통해 그래프 변형
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        # 원본 그래프 임베딩
        z = self.encoder(x, edge_index, edge_weight)
        # 증강된 그래프의 임베딩
        z1 = self.encoder(x1, edge_index1, edge_weight1)
        z2 = self.encoder(x2, edge_index2, edge_weight2)
        return z, z1, z2  # 원본 및 증강된 그래프의 임베딩 반환

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))  # 첫 번째 프로젝션 레이어에 ELU 활성화 함수 적용
        return self.fc2(z)  # 두 번째 프로젝션 레이어를 통과한 결과 반환


# def train(encoder_model, contrast_model, data, optimizer):
#     encoder_model.train()  # 모델을 학습 모드로 전환
#     optimizer.zero_grad()  # 옵티마이저의 그래디언트 초기화
#     z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)  # 원본 및 증강된 임베딩 생성
#     h1, h2 = [encoder_model.project(x) for x in [z1, z2]]  # 증강된 임베딩을 프로젝션하여 hidden representation 생성
#     loss = contrast_model(h1, h2)  # 두 representation 간의 contrastive loss 계산
#     loss.backward()  # 그래디언트 역전파
#     optimizer.step()  # 모델 파라미터 업데이트
#     return loss.item()  # 손실 값 반환

def train(train_loader, encoder_model, contrast_model, optimizer, device):
    encoder_model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Encode mini-batch subgraph
        z, z1, z2 = encoder_model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None))
        h1, h2 = [encoder_model.project(x) for x in (z1, z2)]

        loss = contrast_model(h1, h2)  # intra-view negatives OFF → O(N) 수준
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(train_loader))

# def test(seed, encoder_model, data, vis_save_path):
#     encoder_model.eval()  # 모델을 평가 모드로 전환
#     z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)  # 원본 임베딩 계산
#     split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)  # 데이터셋 분할 (학습/테스트)
#
#     # 시각화
#     ari_score, sil_score = visualize_tsne(seed, z.detach().cpu().numpy(), data.y, save_path=vis_save_path)
#
#     # result = LREvaluator()(z, data.y, split)  # Logistic Regression 평가 수행
#
#     # 새로운 평가
#     result = evaluate_with_metrics(z, data.y, split)
#
#     return result, ari_score, sil_score  # 평가 결과 반환

def test(seed, encoder_model, data, vis_save_path, num_layers=2, batch_size=4096, device='cuda'):
    """
    NeighborLoader(num_neighbors=[-1]*num_layers)로 풀그래프와 동일한 임베딩을 배치 추론.
    반환: Z (torch.Tensor on GPU), shape [num_nodes, proj_hidden_dim]
    """
    import GCL.augmentors as A
    encoder_model.eval()
    # 평가 시엔 augmentation 비활성화 (원본 기준)
    encoder_model.augmentor = (A.Identity(), A.Identity())

    out_dim = encoder_model.fc2.out_features
    Z = torch.empty((data.num_nodes, out_dim), device=device)

    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1] * num_layers,   # all neighbors → exact results
        batch_size=batch_size,
        shuffle=False
    )

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            z, _, _ = encoder_model(batch.x, batch.edge_index, getattr(batch, 'edge_attr', None))
            h = encoder_model.project(z)
            Z[batch.n_id] = h


    split = get_split(num_samples=Z.size()[0], train_ratio=0.1, test_ratio=0.8)  # 데이터셋 분할 (학습/테스트)

    # 시각화
    ari_score, sil_score = visualize_tsne(seed, Z.detach().cpu().numpy(), data.y, save_path=vis_save_path)

    # result = LREvaluator()(z, data.y, split)  # Logistic Regression 평가 수행

    # 새로운 평가
    result = evaluate_with_metrics(Z, data.y, split)

    return result, ari_score, sil_score  # 평가 결과 반환


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

    # 증강 방법 설정: 엣지 제거 및 feature 마스킹
    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    # GCN 모델 설정
    gconv = GConv(input_dim=x.shape[1], hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    # 인코더 및 증강기 설정
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    # 대조 학습 모델 설정
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)  # 옵티마이저 설정

    batch_size=4096
    # ==== Mini-batch training loader ====
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],   # k-hop 샘플링
        batch_size=batch_size,          # 필요시 512/256으로 조정
        shuffle=True
    )

    # 1000
    with tqdm(total=1000, desc='(T)') as pbar:  # 1000번 에폭 동안 학습
        for epoch in range(1, 1000 + 1):
            loss = train(train_loader, encoder_model, contrast_model, optimizer, device)  # 학습 수행
            pbar.set_postfix({'loss': loss})  # 진행 상태바에 손실 값 표시
            pbar.update()  # 진행 상태 업데이트

    os.makedirs(f'./_visualize/GRACE', exist_ok=True)
    vis_save_path = f'./_visualize/GRACE/tsne_GRACE_w_org_{args.node_data_name}_0.01_xshape_256_32_2_InfoNCE.png'
    test_result, ari_score, sil_score = test(seed, encoder_model, data, vis_save_path, args.gconv_nlayers, batch_size, device)  # 테스트 수행
    print(test_result)
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')  # 평가 결과 출력

    results = []

    # 결과 저장
    results.append({
        'Model': 'GRACE_w_org',
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
