"""
HOFINET.csv 전처리 파이프라인
원시 거래 데이터 → 노드 피처 CSV + 엣지 CSV 생성

사용법:
    python datasets/pp_hofinet.py              # CPU only
    python datasets/pp_hofinet.py --use_gpu    # GPU 가속 (cuGraph)
"""

import argparse
import os
import time
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy

# HOFINET 한글 → 영문 컬럼 매핑
COLUMN_MAP = {
    '거래일자': 'tran_dt',
    '거래시간대': 'tran_tmrg',
    '출금금융회사일련번호': 'wd_fc_sn',
    '출금계좌일련번호': 'wd_ac_sn',
    '입금금융회사일련번호': 'dps_fc_sn',
    '입금계좌일련번호': 'dps_ac_sn',
    '자금구분': 'fnd_type',
    '매체구분': 'md_type',
    '거래금액': 'tran_amt',
    '이상거래여부': 'label',
    '이상거래유형': 'fraud_type',
    '이상거래설명': 'fraud_desc',
}


def load_and_rename(input_path):
    print(f'[INFO] Loading {input_path}...')
    df = pd.read_csv(input_path)
    df.rename(columns=COLUMN_MAP, inplace=True)
    df['tran_dt'] = df['tran_dt'].astype(str)
    print(f'[INFO] Loaded {len(df):,} transactions')
    return df


def add_source_target(df):
    df['source'] = df['wd_fc_sn'].astype(str) + ':' + df['wd_ac_sn'].astype(str)
    df['target'] = df['dps_fc_sn'].astype(str) + ':' + df['dps_ac_sn'].astype(str)
    return df


def compute_entropy_feat(group):
    counts = group.value_counts(normalize=True)
    return entropy(counts)


def compute_graph_features_gpu(edge_df, node_index):
    """cuGraph GPU 기반 그래프 피처 계산
    Returns: dict of {feature_name: {node_id: value}}
    피처: dc, pagerank, hits_hub, hits_auth, katz, eigenvector, kcore, triangle
    """
    import cudf
    import cugraph
    import gc

    src_ids = edge_df['source'].map(node_index).values
    tgt_ids = edge_df['target'].map(node_index).values
    gdf = cudf.DataFrame({'src': src_ids, 'dst': tgt_ids})
    n_nodes = len(node_index)

    results = {}

    # --- Directed graph features ---
    G_di = cugraph.Graph(directed=True)
    G_di.from_cudf_edgelist(gdf, source='src', destination='dst')
    print(f'[INFO] cuGraph directed: {n_nodes:,} nodes, {len(gdf):,} edges')

    # Degree Centrality
    t0 = time.time()
    dc_df = cugraph.degree_centrality(G_di)
    results['dc'] = dict(zip(dc_df['vertex'].to_pandas(), dc_df['degree_centrality'].to_pandas()))
    print(f'[INFO] Degree centrality done (GPU, {time.time()-t0:.1f}s)')

    # PageRank
    t0 = time.time()
    pr_df = cugraph.pagerank(G_di)
    results['pagerank'] = dict(zip(pr_df['vertex'].to_pandas(), pr_df['pagerank'].to_pandas()))
    print(f'[INFO] PageRank done (GPU, {time.time()-t0:.1f}s)')

    # HITS
    t0 = time.time()
    hits_df = cugraph.hits(G_di)
    results['hits_hub'] = dict(zip(hits_df['vertex'].to_pandas(), hits_df['hubs'].to_pandas()))
    results['hits_auth'] = dict(zip(hits_df['vertex'].to_pandas(), hits_df['authorities'].to_pandas()))
    print(f'[INFO] HITS done (GPU, {time.time()-t0:.1f}s)')

    # Katz Centrality
    t0 = time.time()
    try:
        katz_df = cugraph.katz_centrality(G_di)
        results['katz'] = dict(zip(katz_df['vertex'].to_pandas(), katz_df['katz_centrality'].to_pandas()))
        print(f'[INFO] Katz centrality done (GPU, {time.time()-t0:.1f}s)')
    except Exception as e:
        print(f'[WARN] Katz centrality failed: {e}')
        results['katz'] = {}

    del G_di
    gc.collect()

    # --- Undirected graph features ---
    G_un = cugraph.Graph(directed=False)
    G_un.from_cudf_edgelist(gdf, source='src', destination='dst')
    print(f'[INFO] cuGraph undirected: {G_un.number_of_vertices():,} nodes, {G_un.number_of_edges():,} edges')

    # Eigenvector Centrality
    t0 = time.time()
    try:
        ev_df = cugraph.eigenvector_centrality(G_un)
        results['eigenvector'] = dict(zip(ev_df['vertex'].to_pandas(), ev_df['eigenvector_centrality'].to_pandas()))
        print(f'[INFO] Eigenvector centrality done (GPU, {time.time()-t0:.1f}s)')
    except Exception as e:
        print(f'[WARN] Eigenvector centrality failed: {e}')
        results['eigenvector'] = {}

    # K-core (core number)
    t0 = time.time()
    kcore_df = cugraph.core_number(G_un)
    results['kcore'] = dict(zip(kcore_df['vertex'].to_pandas(), kcore_df['core_number'].to_pandas()))
    print(f'[INFO] K-core done (GPU, {time.time()-t0:.1f}s)')

    # Triangle Count
    t0 = time.time()
    tri_df = cugraph.triangle_count(G_un)
    results['triangle'] = dict(zip(tri_df['vertex'].to_pandas(), tri_df['counts'].to_pandas()))
    print(f'[INFO] Triangle count done (GPU, {time.time()-t0:.1f}s)')

    del G_un, gdf
    gc.collect()

    return results


def compute_graph_features_cpu(G):
    """NetworkX CPU 기반 그래프 피처 계산"""
    results = {}

    t0 = time.time()
    results['dc'] = nx.degree_centrality(G)
    print(f'[INFO] Degree centrality done (CPU, {time.time()-t0:.1f}s)')

    t0 = time.time()
    results['pagerank'] = nx.pagerank(G)
    print(f'[INFO] PageRank done (CPU, {time.time()-t0:.1f}s)')

    t0 = time.time()
    hubs, auths = nx.hits(G)
    results['hits_hub'] = hubs
    results['hits_auth'] = auths
    print(f'[INFO] HITS done (CPU, {time.time()-t0:.1f}s)')

    t0 = time.time()
    try:
        results['katz'] = nx.katz_centrality(G)
        print(f'[INFO] Katz centrality done (CPU, {time.time()-t0:.1f}s)')
    except Exception as e:
        print(f'[WARN] Katz centrality failed: {e}')
        results['katz'] = {}

    G_un = G.to_undirected()

    t0 = time.time()
    try:
        results['eigenvector'] = nx.eigenvector_centrality(G_un)
        print(f'[INFO] Eigenvector centrality done (CPU, {time.time()-t0:.1f}s)')
    except Exception as e:
        print(f'[WARN] Eigenvector centrality failed: {e}')
        results['eigenvector'] = {}

    t0 = time.time()
    results['kcore'] = nx.core_number(G_un)
    print(f'[INFO] K-core done (CPU, {time.time()-t0:.1f}s)')

    t0 = time.time()
    results['triangle'] = nx.triangles(G_un)
    print(f'[INFO] Triangle count done (CPU, {time.time()-t0:.1f}s)')

    return results


GRAPH_FEATURE_NAMES = ['dc', 'cc', 'pagerank', 'hits_hub', 'hits_auth', 'katz', 'eigenvector', 'kcore', 'triangle']


def build_node_features(df, seed=2025, use_gpu=False):
    """거래 데이터로부터 노드 피처 + 그래프 피처 + 엣지 리스트 생성"""
    print('[INFO] Building node features...')

    # 계좌별 거래 집계
    features_out = df.groupby('source')['tran_amt'].agg(['mean', 'max', 'std', 'count']).add_prefix('out_')
    features_in = df.groupby('target')['tran_amt'].agg(['mean', 'max', 'std', 'count']).add_prefix('in_')
    node_features = pd.concat([features_out, features_in], axis=1).fillna(0)

    # 사기 레이블 (출금/입금 중 하나라도 사기 거래가 있으면 1)
    src_label = df.groupby('source')['label'].max()
    tgt_label = df.groupby('target')['label'].max()
    labels = src_label.combine(tgt_label, func=lambda s, t: max(s, t)).fillna(
        src_label).fillna(tgt_label).fillna(0).astype(int)
    node_features['label'] = labels

    # md_type, fnd_type entropy
    md_entropy = df.groupby('source')['md_type'].apply(compute_entropy_feat).rename('md_type_entropy')
    fnd_entropy = df.groupby('source')['fnd_type'].apply(compute_entropy_feat).rename('fnd_type_entropy')
    node_features = node_features.join(md_entropy, how='left').join(fnd_entropy, how='left')
    node_features.fillna(0, inplace=True)

    # 엣지 리스트 (중복 제거)
    edge_df = df[['source', 'target']].drop_duplicates()

    # 그래프 피처 계산
    print('[INFO] Computing graph features...')
    if use_gpu:
        node_index = {acc: i for i, acc in enumerate(node_features.index)}
        feat_dicts = compute_graph_features_gpu(edge_df, node_index)
        for feat_name, feat_dict in feat_dicts.items():
            node_features[feat_name] = node_features.index.map(
                lambda x, fd=feat_dict, ni=node_index: fd.get(ni.get(x, -1), 0))
    else:
        G = nx.from_pandas_edgelist(df, source='source', target='target', create_using=nx.DiGraph())
        print(f'[INFO] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges')
        feat_dicts = compute_graph_features_cpu(G)
        for feat_name, feat_dict in feat_dicts.items():
            node_features[feat_name] = pd.Series(feat_dict)

    # Closeness centrality (CPU only — cuGraph 미지원)
    print('[INFO] Computing closeness centrality (CPU)...')
    if use_gpu:
        node_index = {acc: i for i, acc in enumerate(node_features.index)}
        src_ids = edge_df['source'].map(node_index).values
        tgt_ids = edge_df['target'].map(node_index).values
        G_nx = nx.DiGraph()
        G_nx.add_edges_from(zip(src_ids, tgt_ids))
    else:
        G_nx = G
    t0 = time.time()
    cc_dict = nx.closeness_centrality(G_nx)
    if use_gpu:
        cc_dict = {int(k): v for k, v in cc_dict.items()}
        node_features['cc'] = node_features.index.map(
            lambda x: cc_dict.get(node_index.get(x, -1), 0))
    else:
        node_features['cc'] = pd.Series(cc_dict)
    print(f'[INFO] Closeness centrality done (CPU, {time.time()-t0:.1f}s)')

    node_features.fillna(0, inplace=True)

    # 피처별 통계 출력
    print(f'\n[INFO] === Graph Feature Statistics ===')
    for feat in GRAPH_FEATURE_NAMES:
        if feat in node_features.columns:
            col = node_features[feat]
            n_nz = (col > 0).sum()
            fraud_mean = col[node_features['label'] == 1].mean()
            benign_mean = col[node_features['label'] == 0].mean()
            print(f'  {feat:12s}: non-zero={n_nz:>7,}/{len(col):,} ({n_nz*100/len(col):5.1f}%), '
                  f'fraud_mean={fraud_mean:.6f}, benign_mean={benign_mean:.6f}')

    node_features['account'] = node_features.index

    return node_features, edge_df


def save_outputs(node_features, edge_df, output_name, output_dir='./datasets'):
    os.makedirs(output_dir, exist_ok=True)

    node_path = os.path.join(output_dir, f'{output_name}_NODE_FEAT.csv')
    edge_path = os.path.join(output_dir, f'{output_name}_EDGES.csv')

    node_features.reset_index(drop=True).to_csv(node_path, index=False)
    edge_df.to_csv(edge_path, index=False)

    n_fraud = (node_features['label'] == 1).sum()
    n_total = len(node_features)
    print(f'[INFO] Saved {node_path}: {n_total:,} nodes (fraud: {n_fraud:,}, {n_fraud*100/n_total:.2f}%)')
    print(f'[INFO] Saved {edge_path}: {len(edge_df):,} edges')


def main():
    parser = argparse.ArgumentParser(description='HOFINET preprocessing pipeline')
    parser.add_argument('--input', type=str, default='./datasets/HOFINET.csv')
    parser.add_argument('--output_name', type=str, default='HOFINET')
    parser.add_argument('--output_dir', type=str, default='./datasets')
    parser.add_argument('--use_gpu', action='store_true',
                        help='GPU 가속 사용 (cuGraph, cc는 CPU)')
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()

    df = load_and_rename(args.input)
    df = add_source_target(df)

    node_features, edge_df = build_node_features(
        df, seed=args.seed, use_gpu=args.use_gpu)
    save_outputs(node_features, edge_df, args.output_name, args.output_dir)

    print(f'\n[INFO] Preprocessing complete!')
    print(f'[INFO] Graph features: {[f for f in GRAPH_FEATURE_NAMES if f in node_features.columns]}')
    print(f'[INFO] To run experiments:')
    print(f'  python models/grace_w_cen.py --node_data_name {args.output_name}_NODE_FEAT --edge_data_name {args.output_name}_EDGES')


if __name__ == '__main__':
    main()
