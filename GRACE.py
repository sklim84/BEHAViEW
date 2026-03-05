import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F
import torch_geometric.transforms as T

from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid


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


def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()  # 모델을 학습 모드로 전환
    optimizer.zero_grad()  # 옵티마이저의 그래디언트 초기화
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr)  # 원본 및 증강된 임베딩 생성
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]  # 증강된 임베딩을 프로젝션하여 hidden representation 생성
    loss = contrast_model(h1, h2)  # 두 representation 간의 contrastive loss 계산
    loss.backward()  # 그래디언트 역전파
    optimizer.step()  # 모델 파라미터 업데이트
    return loss.item()  # 손실 값 반환


def test(encoder_model, data):
    encoder_model.eval()  # 모델을 평가 모드로 전환
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)  # 원본 임베딩 계산
    split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)  # 데이터셋 분할 (학습/테스트)
    result = LREvaluator()(z, data.y, split)  # Logistic Regression 평가 수행
    return result  # 평가 결과 반환


def main():
    device = torch.device('cuda')  # CUDA 장치를 사용하여 GPU에서 연산 수행
    path = osp.join(osp.expanduser('~'), 'datasets')  # 데이터셋 저장 경로 설정
    dataset = Planetoid(path, name='Cora', transform=T.NormalizeFeatures())  # Planetoid 데이터셋 로드 및 feature 정규화
    data = dataset[0].to(device)  # 데이터를 GPU로 이동

    # 증강 방법 설정: 엣지 제거 및 feature 마스킹
    aug1 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    # GCN 모델 설정
    gconv = GConv(input_dim=dataset.num_features, hidden_dim=32, activation=torch.nn.ReLU, num_layers=2).to(device)
    # 인코더 및 증강기 설정
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=32, proj_dim=32).to(device)
    # 대조 학습 모델 설정
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)  # 옵티마이저 설정

    with tqdm(total=1000, desc='(T)') as pbar:  # 1000번 에폭 동안 학습
        for epoch in range(1, 1001):
            loss = train(encoder_model, contrast_model, data, optimizer)  # 학습 수행
            pbar.set_postfix({'loss': loss})  # 진행 상태바에 손실 값 표시
            pbar.update()  # 진행 상태 업데이트

    test_result = test(encoder_model, data)  # 테스트 수행
    print(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')  # 평가 결과 출력


if __name__ == '__main__':
    main()  # 메인 함수 실행
