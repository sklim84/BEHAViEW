"""
HOFINET.csv 전처리 파이프라인
원시 거래 데이터 → 노드 피처 CSV + 엣지 CSV 생성

사용법:
    python datasets/pp_hofinet.py                          # 하이브리드: cuGraph(GPU) + Memgraph (권장)
    python datasets/pp_hofinet.py --mg_host 192.168.1.10   # 원격 Memgraph
    python datasets/pp_hofinet.py --gpu_only               # cuGraph만 (Memgraph 미사용, 피처 8개)
    python datasets/pp_hofinet.py --cpu_only               # NetworkX만 (GPU/Memgraph 없는 환경)

그래프 피처 (19개):
    cuGraph (GPU, 수초): dc, in_dc, out_dc, pagerank, hits_hub, hits_auth, kcore, triangle, betweenness
    Memgraph (CPU): cc, harmonic, clustering, sq_clustering, avg_neighbor_deg,
                    load_cen, voterank, constraint, effective_size, eigenvector
"""

import argparse
import os
import time
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import entropy

try:
    from datasets.feature_utils import run_feature
except ImportError:
    from feature_utils import run_feature

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

GRAPH_FEATURE_NAMES = [
    'dc', 'in_dc', 'out_dc', 'cc', 'harmonic',
    'pagerank', 'hits_hub', 'hits_auth', 'eigenvector',
    'kcore', 'triangle', 'clustering', 'sq_clustering',
    'avg_neighbor_deg', 'betweenness', 'load_cen',
    'voterank', 'constraint', 'effective_size',
]


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


# ============================================================
# cuGraph (GPU) — 빠른 피처 9개
# ============================================================
def compute_graph_features_gpu(edge_df, node_index, gpu_id=0):
    """cuGraph GPU 기반 그래프 피처 계산 (수초~수십초)
    피처: dc, in_dc, out_dc, pagerank, hits_hub, hits_auth, kcore, triangle, betweenness
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import cudf
    import cugraph
    import gc

    src_ids = edge_df['source'].map(node_index).values
    tgt_ids = edge_df['target'].map(node_index).values
    gdf = cudf.DataFrame({'src': src_ids, 'dst': tgt_ids})
    n_nodes = len(node_index)

    results = {}

    # --- Directed graph ---
    G_di = cugraph.Graph(directed=True)
    G_di.from_cudf_edgelist(gdf, source='src', destination='dst')
    print(f'[INFO] cuGraph directed: {n_nodes:,} nodes, {len(gdf):,} edges')

    r = run_feature('dc', lambda: cugraph.degree_centrality(G_di))
    if r is not None:
        results['dc'] = dict(zip(r['vertex'].to_pandas(),
                                 r['degree_centrality'].to_pandas()))

    def _compute_in_out_dc():
        in_deg = G_di.in_degree()
        out_deg = G_di.out_degree()
        n1 = max(n_nodes - 1, 1)
        return in_deg, out_deg, n1
    r = run_feature('in_dc, out_dc', _compute_in_out_dc)
    if r is not None:
        in_deg, out_deg, n1 = r
        results['in_dc'] = dict(zip(in_deg['vertex'].to_pandas(),
                                    (in_deg['degree'] / n1).to_pandas()))
        results['out_dc'] = dict(zip(out_deg['vertex'].to_pandas(),
                                     (out_deg['degree'] / n1).to_pandas()))

    r = run_feature('pagerank', lambda: cugraph.pagerank(G_di))
    if r is not None:
        results['pagerank'] = dict(zip(r['vertex'].to_pandas(),
                                       r['pagerank'].to_pandas()))

    r = run_feature('hits', lambda: cugraph.hits(G_di))
    if r is not None:
        results['hits_hub'] = dict(zip(r['vertex'].to_pandas(),
                                       r['hubs'].to_pandas()))
        results['hits_auth'] = dict(zip(r['vertex'].to_pandas(),
                                        r['authorities'].to_pandas()))

    del G_di
    gc.collect()

    # --- Undirected graph ---
    G_un = cugraph.Graph(directed=False)
    G_un.from_cudf_edgelist(gdf, source='src', destination='dst')
    print(f'[INFO] cuGraph undirected: {G_un.number_of_vertices():,} nodes, '
          f'{G_un.number_of_edges():,} edges')

    r = run_feature('kcore', lambda: cugraph.core_number(G_un))
    if r is not None:
        results['kcore'] = dict(zip(r['vertex'].to_pandas(),
                                    r['core_number'].to_pandas()))

    r = run_feature('triangle', lambda: cugraph.triangle_count(G_un))
    if r is not None:
        results['triangle'] = dict(zip(r['vertex'].to_pandas(),
                                       r['counts'].to_pandas()))

    r = run_feature('betweenness(k=500)',
                    lambda: cugraph.betweenness_centrality(G_un, k=500,
                                                           seed=2025))
    if r is not None:
        results['betweenness'] = dict(zip(
            r['vertex'].to_pandas(),
            r['betweenness_centrality'].to_pandas()))

    del G_un, gdf
    gc.collect()

    # CUDA_VISIBLE_DEVICES 복원
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)

    return results


# ============================================================
# Memgraph — 느린 피처 (cuGraph 미지원)
# ============================================================
def _load_graph_to_memgraph(session, edge_df, node_index, batch_size=10000):
    """Memgraph에 그래프 데이터 적재"""
    session.run('MATCH (n) DETACH DELETE n')
    session.run('CREATE INDEX ON :Account(nid)')

    node_ids = list(node_index.values())
    for i in range(0, len(node_ids), batch_size):
        batch = node_ids[i:i + batch_size]
        session.run('UNWIND $nodes AS nid CREATE (:Account {nid: nid})', nodes=batch)

    src_ids = edge_df['source'].map(node_index).values
    tgt_ids = edge_df['target'].map(node_index).values
    edges = [{'s': int(s), 't': int(t)} for s, t in zip(src_ids, tgt_ids)]
    for i in range(0, len(edges), batch_size):
        batch = edges[i:i + batch_size]
        session.run(
            'UNWIND $edges AS e '
            'MATCH (a:Account {nid: e.s}), (b:Account {nid: e.t}) '
            'CREATE (a)-[:TX]->(b)',
            edges=batch
        )
    return len(node_ids), len(edges)


def _run_memgraph_proc(session, query, key_field='nid', value_fields=None):
    """Memgraph 프로시저 실행 후 {nid: value} dict 반환"""
    result = session.run(query)
    if len(value_fields) == 1:
        return {r[key_field]: r[value_fields[0]] for r in result}
    else:
        records = list(result)
        return {vf: {r[key_field]: r[vf] for r in records} for vf in value_fields}


def compute_graph_features_memgraph(edge_df, node_index, host='127.0.0.1', port=7687):
    """Memgraph 기반 그래프 피처 계산 — cuGraph 미지원 피처만
    cc, harmonic, clustering, sq_clustering, avg_neighbor_deg,
    load_cen, voterank, constraint, effective_size, eigenvector
    """
    from neo4j import GraphDatabase

    results = {}

    print(f'[INFO] Connecting to Memgraph at {host}:{port}...')
    driver = GraphDatabase.driver(f'bolt://{host}:{port}', auth=('', ''))
    driver.verify_connectivity()
    print('[INFO] Connected to Memgraph')

    try:
        # --- 그래프 적재 ---
        t0 = time.time()
        with driver.session(database='memgraph') as session:
            n_n, n_e = _load_graph_to_memgraph(session, edge_df, node_index)
        print(f'[INFO] Graph loaded into Memgraph: {n_n:,} nodes, {n_e:,} edges ({time.time()-t0:.1f}s)')

        # cuGraph 미지원 피처만 Memgraph로 계산
        FEATURES = [
            ('cc', 'CALL graph_features.closeness_centrality() YIELD node, rank RETURN node.nid AS nid, rank', ['rank']),
            ('harmonic', 'CALL graph_features.harmonic_centrality() YIELD node, rank RETURN node.nid AS nid, rank', ['rank']),
            ('clustering', 'CALL nxalg.clustering() YIELD node, clustering RETURN node.nid AS nid, clustering', ['clustering']),
            ('sq_clustering', 'CALL graph_features.square_clustering() YIELD node, coeff RETURN node.nid AS nid, coeff', ['coeff']),
            ('avg_neighbor_deg', 'CALL graph_features.average_neighbor_degree() YIELD node, avg_deg RETURN node.nid AS nid, avg_deg', ['avg_deg']),
            ('load_cen', 'CALL graph_features.load_centrality() YIELD node, load RETURN node.nid AS nid, load', ['load']),
            ('voterank', 'CALL graph_features.voterank_score() YIELD node, rank RETURN node.nid AS nid, rank', ['rank']),
            ('constraint', 'CALL graph_features.constraint() YIELD node, value RETURN node.nid AS nid, value', ['value']),
            ('effective_size', 'CALL graph_features.effective_size() YIELD node, value RETURN node.nid AS nid, value', ['value']),
            ('eigenvector', 'CALL graph_features.eigenvector_centrality() YIELD node, rank RETURN node.nid AS nid, rank', ['rank']),
        ]

        for feat_name, query, value_fields in FEATURES:
            t0 = time.time()
            try:
                with driver.session(database='memgraph') as session:
                    data = _run_memgraph_proc(session, query, value_fields=value_fields)
                results[feat_name] = data
                print(f'[INFO] {feat_name} done (Memgraph, {time.time()-t0:.1f}s)')
            except Exception as e:
                print(f'[WARN] {feat_name} failed: {e}')
                results[feat_name] = {}

    finally:
        driver.close()
        print('[INFO] Memgraph connection closed')

    return results


# ============================================================
# NetworkX CPU — GPU/Memgraph 없는 환경용 폴백
# ============================================================
def compute_graph_features_cpu(G):
    """NetworkX CPU 기반 그래프 피처 계산"""
    results = {}
    G_un = G.to_undirected()

    # Directed features
    cpu_features_directed = [
        ('dc',       lambda: nx.degree_centrality(G)),
        ('in_dc',    lambda: nx.in_degree_centrality(G)),
        ('out_dc',   lambda: nx.out_degree_centrality(G)),
        ('pagerank', lambda: nx.pagerank(G)),
        ('cc',       lambda: nx.closeness_centrality(G)),
        ('harmonic', lambda: nx.harmonic_centrality(G)),
    ]
    for name, func in cpu_features_directed:
        r = run_feature(name, func)
        if r is not None:
            results[name] = r

    # HITS (returns tuple)
    r = run_feature('hits', lambda: nx.hits(G))
    if r is not None:
        results['hits_hub'], results['hits_auth'] = r

    # Undirected features
    cpu_features_undirected = [
        ('eigenvector',  lambda: nx.eigenvector_centrality(G_un, max_iter=1000)),
        ('kcore',        lambda: nx.core_number(G_un)),
        ('triangle',     lambda: nx.triangles(G_un)),
        ('betweenness',  lambda: nx.betweenness_centrality(G_un, k=500, seed=2025)),
        ('clustering',   lambda: nx.clustering(G_un)),
    ]
    for name, func in cpu_features_undirected:
        r = run_feature(name, func)
        if r is not None:
            results[name] = r

    return results


# ============================================================
# build_node_features — 메인 파이프라인
# ============================================================
def build_node_features(df, seed=2025, cpu_only=False, gpu_only=False,
                        mg_host='127.0.0.1', mg_port=7687, gpu_id=0):
    """거래 데이터로부터 노드 피처 + 그래프 피처 + 엣지 리스트 생성"""
    print('[INFO] Building node features...')

    # 계좌별 거래 집계 (전체 기간)
    features_out = df.groupby('source')['tran_amt'].agg(['mean', 'max', 'std', 'count']).add_prefix('out_')
    features_in = df.groupby('target')['tran_amt'].agg(['mean', 'max', 'std', 'count']).add_prefix('in_')
    node_features = pd.concat([features_out, features_in], axis=1).fillna(0)

    # 시간 윈도우별 거래 집계 (최근 3/6/12개월)
    df['tran_date'] = pd.to_datetime(df['tran_dt'], format='%Y%m%d')
    max_date = df['tran_date'].max()
    for months in [3, 6, 12]:
        cutoff = max_date - pd.DateOffset(months=months)
        df_win = df[df['tran_date'] >= cutoff]
        win_out = df_win.groupby('source')['tran_amt'].agg(['mean', 'count']).add_prefix(f'out_{months}m_')
        win_in = df_win.groupby('target')['tran_amt'].agg(['mean', 'count']).add_prefix(f'in_{months}m_')
        node_features = node_features.join(win_out, how='left').join(win_in, how='left')
    node_features.fillna(0, inplace=True)
    df.drop(columns=['tran_date'], inplace=True)

    # 사기 레이블
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

    # 엣지 리스트 (멀티엣지 보존 — 동일 계좌 간 반복 거래를 개별 엣지로 유지)
    edge_df = df[['source', 'target']]
    node_index = {acc: i for i, acc in enumerate(node_features.index)}

    # ---- 그래프 피처 계산 ----
    print('[INFO] Computing graph features...')

    def _apply_feats(feat_dicts, use_node_index=True):
        for feat_name, feat_dict in feat_dicts.items():
            if not feat_dict:
                node_features[feat_name] = 0
            elif use_node_index:
                node_features[feat_name] = node_features.index.map(
                    lambda x, fd=feat_dict, ni=node_index: fd.get(ni.get(x, -1), 0))
            else:
                node_features[feat_name] = pd.Series(feat_dict)

    if cpu_only:
        # NetworkX CPU only
        G = nx.from_pandas_edgelist(df, source='source', target='target', create_using=nx.DiGraph())
        print(f'[INFO] Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges')
        cpu_feats = compute_graph_features_cpu(G)
        _apply_feats(cpu_feats, use_node_index=False)
    else:
        # Step 1: cuGraph (GPU) — 빠른 피처 9개
        print('[INFO] === Step 1: cuGraph (GPU) ===')
        gpu_feats = compute_graph_features_gpu(edge_df, node_index, gpu_id=gpu_id)
        _apply_feats(gpu_feats)

        if not gpu_only:
            # Step 2: Memgraph — 느린 피처 10개
            print('[INFO] === Step 2: Memgraph ===')
            mg_feats = compute_graph_features_memgraph(edge_df, node_index,
                                                       host=mg_host, port=mg_port)
            _apply_feats(mg_feats)

    node_features.fillna(0, inplace=True)

    # 피처별 통계 출력
    print(f'\n[INFO] === Graph Feature Statistics ===')
    for feat in GRAPH_FEATURE_NAMES:
        if feat in node_features.columns:
            col = node_features[feat]
            n_nz = (col > 0).sum()
            fraud_mean = col[node_features['label'] == 1].mean()
            benign_mean = col[node_features['label'] == 0].mean()
            ratio = fraud_mean / benign_mean if benign_mean > 0 else float('inf')
            print(f'  {feat:18s}: non-zero={n_nz:>7,}/{len(col):,} ({n_nz*100/len(col):5.1f}%), '
                  f'fraud={fraud_mean:.6f}, benign={benign_mean:.6f}, ratio={ratio:.2f}x')

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
    parser.add_argument('--cpu_only', action='store_true',
                        help='NetworkX CPU만 사용 (GPU/Memgraph 없는 환경)')
    parser.add_argument('--gpu_only', action='store_true',
                        help='cuGraph GPU만 사용 (Memgraph 미사용, 피처 9개)')
    parser.add_argument('--mg_host', type=str, default='127.0.0.1',
                        help='Memgraph 호스트 (기본: 127.0.0.1)')
    parser.add_argument('--mg_port', type=int, default=7687,
                        help='Memgraph 포트 (기본: 7687)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='cuGraph에 사용할 GPU ID (기본: 0)')
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()

    if args.cpu_only and args.gpu_only:
        parser.error('--cpu_only와 --gpu_only는 동시에 사용할 수 없습니다')

    df = load_and_rename(args.input)
    df = add_source_target(df)

    node_features, edge_df = build_node_features(
        df, seed=args.seed, cpu_only=args.cpu_only, gpu_only=args.gpu_only,
        mg_host=args.mg_host, mg_port=args.mg_port, gpu_id=args.gpu)
    save_outputs(node_features, edge_df, args.output_name, args.output_dir)

    print(f'\n[INFO] Preprocessing complete!')
    print(f'[INFO] Graph features: {[f for f in GRAPH_FEATURE_NAMES if f in node_features.columns]}')
    print(f'[INFO] To run experiments:')
    print(f'  python models/grace_w_cen.py --node_data_name {args.output_name}_NODE_FEAT --edge_data_name {args.output_name}_EDGES')


if __name__ == '__main__':
    main()
