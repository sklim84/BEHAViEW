import pandas as pd
import torch
from torch_geometric.data import Data


def load_graph_data(args, device=None):
    """CSV 파일에서 노드 피처, 엣지, 중심성 피처를 로드하여 PyG Data 객체를 반환한다."""
    df = pd.read_csv(f'./datasets/{args.node_data_name}.csv')
    edge_df = pd.read_csv(f'./datasets/{args.edge_data_name}.csv')

    node_index = {acc: i for i, acc in enumerate(df['account'])}
    src = edge_df['source'].map(node_index)
    tgt = edge_df['target'].map(node_index)
    edge_index = torch.tensor([src.values, tgt.values], dtype=torch.long)

    label = torch.tensor(df['label'].values, dtype=torch.long)

    structural = {'dc', 'in_dc', 'out_dc', 'pagerank', 'hits_hub', 'hits_auth',
                   'kcore', 'triangle', 'betweenness'}
    x_behav = torch.tensor(
        df[[c for c in df.columns
            if c.startswith(('out_', 'in_', 'md_', 'fnd_', 'entropy')) and c not in structural]].values,
        dtype=torch.float
    )
    x_struct = torch.tensor(
        df[[c for c in df.columns if c in args.struct_feats]].values,
        dtype=torch.float
    )

    data = Data(x=x_behav, edge_index=edge_index, y=label)
    if device:
        data = data.to(device)

    return data, x_struct


def load_knn_graph(knn_graph_name, device=None):
    """k-NN 그래프 edge index 로드."""
    path = f'./datasets/{knn_graph_name}.csv'
    df = pd.read_csv(path)
    edge_index = torch.tensor([df['source'].values, df['target'].values], dtype=torch.long)
    if device:
        edge_index = edge_index.to(device)
    return edge_index
