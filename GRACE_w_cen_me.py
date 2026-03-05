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
from torch_geometric.loader import NeighborLoader

# ===========================
# GConv 정의
# ===========================
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


# ===========================
# Encoder 정의
# ===========================
class Encoder(torch.nn.Module):
    def __init__(self, encoder, proj_agg, proj_cen, augmentor1, augmentor2, hidden_dim, proj_dim):
        super(Encoder, self).__init__()
        self.proj_agg = proj_agg
        self.proj_cen = proj_cen
        self.encoder = encoder
        self.augmentor1 = augmentor1
        self.augmentor2 = augmentor2

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x_agg, x_cen, edge_index, edge_weight=None):
        x1, edge_index1, edge_weight1 = self.augmentor1(x_agg, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = self.augmentor2(x_cen, edge_index, edge_weight)

        z1 = self.proj_agg(x1)
        z1 = self.encoder(z1, edge_index1, edge_weight1)

        z2 = self.proj_cen(x2)
        z2 = self.encoder(z2, edge_index2, edge_weight2)

        return z1, z2

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


# ===========================
# Train (Mini-batch)
# ===========================
def train(loader, encoder_model, contrast_model, optimizer, x_cen, device):
    encoder_model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch_x_cen = x_cen[batch.n_id].to(device)

        optimizer.zero_grad()
        z1, z2 = encoder_model(batch.x, batch_x_cen, batch.edge_index, batch.edge_attr)
        h1, h2 = [encoder_model.project(z) for z in [z1, z2]]

        loss = contrast_model(h1, h2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ===========================
# Test (Full graph inference)
# ===========================
# def test(encoder_model, data, x_cen, contrast_model, device):
#     encoder_model.eval()
#     data = data.to(device)
#     x_cen = x_cen.to(device)
#
#     with torch.no_grad():
#         z1, z2 = encoder_model(data.x, x_cen, data.edge_index, data.edge_attr)
#         h1, h2 = [encoder_model.project(z) for z in [z1, z2]]
#         loss = contrast_model(h1, h2).item()
#
#     return z1.cpu(), z2.cpu(), loss

# FIXME 메모리 제한으로 아래와 같이 Loader를 사용하는 방식으로 변경
def test(encoder_model, data, x_cen, num_layers=2, batch_size=4096, device='cuda'):
    """
    NeighborLoader로 배치 추론하여 풀그래프와 동일한 임베딩(z1)을 반환한다.
    - augmentation은 테스트에서 비활성화한다.
    - 반환값: torch.Tensor [num_nodes, proj_hidden_dim]  (Encoder.project 출력 차원)
    """
    encoder_model.eval()

    # 테스트에서는 augmentation 비활성화
    encoder_model.augmentor1 = A.Identity()
    encoder_model.augmentor2 = A.Identity()

    out_dim = encoder_model.fc2.out_features  # Encoder.project의 출력 차원
    Z = torch.empty((data.num_nodes, out_dim), device=device)

    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1] * num_layers,  # 모든 이웃 포함 → 풀그래프와 동일 결과
        batch_size=batch_size,
        shuffle=False
    )

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            z1, _ = encoder_model(
                batch.x,
                x_cen[batch.n_id].to(device),
                batch.edge_index,
                getattr(batch, 'edge_attr', None)
            )
            # project까지 적용해서 평가와 시각화에 바로 쓰기
            h1 = encoder_model.project(z1)
            Z[batch.n_id] = h1

    return Z

# ===========================
# run_experiment
# ===========================
def run_experiment(data, x_cen, args, device, vis_save_path):
    gconv = GConv(args.input_dim, args.hidden_dim, torch.nn.ReLU(), args.gconv_nlayers).to(device)
    proj_agg = torch.nn.Linear(data.x.shape[1], args.input_dim).to(device)
    proj_cen = torch.nn.Linear(x_cen.shape[1], args.input_dim).to(device)

    encoder_model = Encoder(
        encoder=gconv,
        proj_agg=proj_agg,
        proj_cen=proj_cen,
        augmentor1=A.Compose([A.EdgeRemoving(pe=0.0), A.FeatureMasking(pf=0.3)]),
        augmentor2=A.Compose([A.EdgeRemoving(pe=0.0), A.FeatureMasking(pf=0.3)]),
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim
    ).to(device)

    # Loss
    if args.loss == 'BootstrapLatent':
        loss_fn = L.BootstrapLatent()
    elif args.loss == 'JSD':
        loss_fn = L.JSD()
    elif args.loss == 'BarlowTwins':
        loss_fn = L.BarlowTwins()
    elif args.loss == 'InfoNCE':
        loss_fn = L.InfoNCE(tau=0.2)
    else:
        raise ValueError("Unknown loss")

    contrast_model = DualBranchContrast(loss=loss_fn, mode='L2L', intraview_negs=True).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=args.lr)

    # FIXME 메모리 제한으로 아래와 같이 Loader를 사용하는 방식으로 변경
    batch_size = 4096
    # Mini-batch loader
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=batch_size,
        shuffle=True
    )

    # 1000
    with tqdm(total=1000, desc=f'(T) With Centrality') as pbar:
        for epoch in range(1, 1000 + 1):
            loss = train(train_loader, encoder_model, contrast_model, optimizer, x_cen, device)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    # Full-graph inference
    # 테스트: 임베딩만 받음
    Z = test(
        encoder_model=encoder_model,
        data=data,
        x_cen=x_cen,
        num_layers=args.gconv_nlayers,
        batch_size=batch_size,
        device=device
    )
    z1 = Z.detach().cpu()

    # print(f"Final Contrastive Loss: {final_loss:.4f}")

    # t-SNE + 평가
    ari_score, sil_score = visualize_tsne(args.seed, z1.numpy(), data.y, save_path=vis_save_path)
    split = get_split(num_samples=z1.size(0), train_ratio=0.1, test_ratio=0.8)
    eval_result = evaluate_with_metrics(z1, data.y, split)

    return eval_result, ari_score, sil_score


# ===========================
# Main
# ===========================
def main(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"##### device: {device}")

    # Load dataset
    df = pd.read_csv(f'./_datasets/{args.node_data_name}.csv')
    edge_df = pd.read_csv(f'./_datasets/{args.edge_data_name}.csv')

    node_index = {acc: i for i, acc in enumerate(df['account'])}
    src = edge_df['source'].map(node_index)
    tgt = edge_df['target'].map(node_index)
    edge_index = torch.tensor([src.values, tgt.values], dtype=torch.long)

    label = torch.tensor(df['label'].values, dtype=torch.long)
    x_agg = torch.tensor(
        df[[c for c in df.columns if c.startswith(('out_', 'in_', 'md_', 'fnd_', 'entropy'))]].values,
        dtype=torch.float
    )
    x_cen = torch.tensor(
        df[[c for c in df.columns if c in args.cen_feats]].values,
        dtype=torch.float
    )

    data = Data(x=x_agg, edge_index=edge_index, y=label)
    print(data)
    print(type(data))
    print(data.y)

    os.makedirs(f'./_visualize/GRACE', exist_ok=True)
    cen_feats = "_".join(str(item) for item in args.cen_feats)
    vis_save_path = f'./_visualize/GRACE/tsne_{args.model_name}_{args.node_data_name}_{cen_feats}_{args.lr}_{args.input_dim}_{args.hidden_dim}_{args.proj_dim}_{args.gconv_nlayers}_{args.loss}.png'

    test_result, ari_score, sil_score = run_experiment(data, x_cen, args, device, vis_save_path)
    print(test_result)
    print(
        f'(E) With Centrality: Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')

    results = []
    # 결과 저장
    results.append({
        'Model': 'GRACE_w_cen',
        'Data': args.node_data_name,
        'Seed': args.seed,
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

    save_results_to_csv(results, args.metric_save_path)


if __name__ == '__main__':
    args = get_config()
    print(args)

    main(args)