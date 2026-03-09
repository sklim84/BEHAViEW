"""
그래프 피처 계산 — cuGraph(GPU) + Memgraph + NetworkX 하이브리드 방식.

cuGraph: 빠른 GPU 연산 (pagerank, hits, betweenness, eigenvector 등)
Memgraph: cuGraph 미지원 피처 (closeness, harmonic, greedy_color 등)
NetworkX: Memgraph에서도 느린 피처 (load_centrality, voterank, constraint 등)

사용법:
    nohup python datasets/compute_features_hybrid.py --gpu 0 > logs/features_hybrid.log 2>&1 &

결과: logs/memgraph_features.csv
"""
import argparse
import gc
import os
import time

import networkx as nx
import pandas as pd

from datasets.feature_utils import (
    run_feature, print_feature_summary, save_feature_progress,
)

LOG_DIR = './logs'
NODE_CSV = './datasets/HOFINET_NODE_FEAT.csv'
EDGE_CSV = './datasets/HOFINET_EDGES.csv'
OUTPUT_CSV = os.path.join(LOG_DIR, 'memgraph_features.csv')
PROGRESS_JSON = os.path.join(LOG_DIR, 'memgraph_features_progress.json')


def compute_cugraph_features(src_ids, tgt_ids, n_nodes, gpu_id=0):
    """cuGraph GPU 기반 그래프 피처 계산"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    import cudf
    import cugraph

    gdf = cudf.DataFrame({'src': src_ids, 'dst': tgt_ids})
    results = {}

    # --- Directed graph ---
    G_di = cugraph.Graph(directed=True)
    G_di.from_cudf_edgelist(gdf, source='src', destination='dst')
    print(f'[cuGraph] Directed: {n_nodes:,} nodes, {len(gdf):,} edges')

    r = run_feature('degree_centrality',
                    lambda: cugraph.degree_centrality(G_di))
    if r is not None:
        results['dc'] = dict(zip(r['vertex'].to_pandas(),
                                 r['degree_centrality'].to_pandas()))

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

    r = run_feature('katz_centrality',
                    lambda: cugraph.katz_centrality(G_di))
    if r is not None:
        results['katz'] = dict(zip(r['vertex'].to_pandas(),
                                   r['katz_centrality'].to_pandas()))

    del G_di
    gc.collect()

    # --- Undirected graph ---
    G_un = cugraph.Graph(directed=False)
    G_un.from_cudf_edgelist(gdf, source='src', destination='dst')
    print(f'[cuGraph] Undirected: {G_un.number_of_vertices():,} nodes, '
          f'{G_un.number_of_edges():,} edges')

    r = run_feature('eigenvector_centrality',
                    lambda: cugraph.eigenvector_centrality(G_un))
    if r is not None:
        results['eigenvector'] = dict(zip(
            r['vertex'].to_pandas(),
            r['eigenvector_centrality'].to_pandas()))

    r = run_feature('core_number', lambda: cugraph.core_number(G_un))
    if r is not None:
        results['kcore'] = dict(zip(r['vertex'].to_pandas(),
                                    r['core_number'].to_pandas()))

    r = run_feature('triangle_count', lambda: cugraph.triangle_count(G_un))
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

    r = run_feature('louvain', lambda: cugraph.louvain(G_un))
    if r is not None:
        parts, modularity = r
        results['louvain'] = dict(zip(parts['vertex'].to_pandas(),
                                      parts['partition'].to_pandas()))
        print(f'    modularity: {modularity:.4f}')

    del G_un, gdf
    gc.collect()

    return results


def compute_memgraph_features(node_index, host='127.0.0.1', port=7687):
    """Memgraph 기반 피처 계산 (cuGraph 미지원 피처)"""
    from neo4j import GraphDatabase

    results = {}
    driver = GraphDatabase.driver(f'bolt://{host}:{port}', auth=('', ''))
    driver.verify_connectivity()
    print(f'[Memgraph] Connected to {host}:{port}')

    FEATURES = [
        ('in_dc',         'CALL graph_features.in_degree_centrality() YIELD node, degree RETURN node.nid AS nid, degree',   'degree'),
        ('out_dc',        'CALL graph_features.out_degree_centrality() YIELD node, degree RETURN node.nid AS nid, degree',  'degree'),
        ('clustering',    'CALL nxalg.clustering() YIELD node, clustering RETURN node.nid AS nid, clustering',              'clustering'),
        ('sq_clustering', 'CALL graph_features.square_clustering() YIELD node, coeff RETURN node.nid AS nid, coeff',        'coeff'),
        ('avg_neigh_deg', 'CALL graph_features.average_neighbor_degree() YIELD node, avg_deg RETURN node.nid AS nid, avg_deg', 'avg_deg'),
        ('greedy_color',  'CALL nxalg.greedy_color() YIELD node, color RETURN node.nid AS nid, color',                      'color'),
    ]

    for feat_name, query, val_field in FEATURES:
        t0 = time.time()
        try:
            with driver.session(database='memgraph') as s:
                records = list(s.run(query))
            results[feat_name] = {r['nid']: r[val_field] for r in records}
            print(f'  {feat_name:25s}: {time.time()-t0:7.1f}s  OK '
                  f'({len(records):,} records)')
        except Exception as e:
            print(f'  {feat_name:25s}: {time.time()-t0:7.1f}s  FAILED - {e}')
            results[feat_name] = {}

    driver.close()
    return results


def compute_networkx_features(src_ids, tgt_ids):
    """NetworkX CPU 기반 피처 계산 (cuGraph, Memgraph 미지원 피처)"""
    print('[NetworkX] Building graph...')
    t0 = time.time()

    # Build both graph types from same edge arrays
    edges = list(zip(src_ids.astype(int), tgt_ids.astype(int)))
    G_un = nx.Graph()
    G_un.add_edges_from(edges)
    G_di = nx.DiGraph()
    G_di.add_edges_from(edges)
    print(f'  Graph built: {G_un.number_of_nodes():,} nodes, '
          f'{G_un.number_of_edges():,} edges ({time.time()-t0:.1f}s)')

    results = {}

    # Directed features
    r = run_feature('closeness_centrality',
                    lambda: nx.closeness_centrality(G_di))
    if r is not None:
        results['cc'] = {int(k): v for k, v in r.items()}

    r = run_feature('harmonic_centrality',
                    lambda: nx.harmonic_centrality(G_di))
    if r is not None:
        results['harmonic'] = {int(k): v for k, v in r.items()}

    del G_di
    gc.collect()

    # Undirected features
    r = run_feature('load_centrality', lambda: nx.load_centrality(G_un))
    if r is not None:
        results['load_cen'] = {int(k): v for k, v in r.items()}

    r = run_feature('voterank', lambda: nx.voterank(G_un))
    if r is not None:
        results['voterank'] = {n: float(i + 1) for i, n in enumerate(r)}

    r = run_feature('constraint', lambda: nx.constraint(G_un))
    if r is not None:
        results['constraint'] = {
            int(k): (v if v is not None else 0.0) for k, v in r.items()}

    r = run_feature('effective_size', lambda: nx.effective_size(G_un))
    if r is not None:
        results['eff_size'] = {int(k): v for k, v in r.items()}

    del G_un
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mg_host', type=str, default='127.0.0.1')
    parser.add_argument('--mg_port', type=int, default=7687)
    parser.add_argument('--skip_cugraph', action='store_true',
                        help='cuGraph(GPU) 피처 생략')
    parser.add_argument('--skip_memgraph', action='store_true',
                        help='Memgraph 피처 생략')
    parser.add_argument('--skip_networkx', action='store_true',
                        help='NetworkX 피처 생략')
    parser.add_argument('--memgraph_only', action='store_true',
                        help='Memgraph만 실행 (cuGraph, NetworkX 생략)')
    args = parser.parse_args()

    if args.memgraph_only:
        args.skip_cugraph = True
        args.skip_networkx = True

    os.makedirs(LOG_DIR, exist_ok=True)

    print('[INFO] Loading data...')
    node_df = pd.read_csv(NODE_CSV)
    edge_df = pd.read_csv(EDGE_CSV)
    node_index = {acc: i for i, acc in enumerate(node_df['account'])}
    labels = node_df['label'].values
    n_nodes = len(node_df)
    print(f'  Nodes: {n_nodes:,}, Edges: {len(edge_df):,}')

    # Pre-compute edge index arrays (reused by cuGraph and NetworkX)
    src_ids = edge_df['source'].map(node_index).values
    tgt_ids = edge_df['target'].map(node_index).values

    all_results = {}
    total_start = time.time()

    # === Phase 1: cuGraph (GPU) ===
    if not args.skip_cugraph:
        print(f'\n{"="*60}')
        print(f'Phase 1: cuGraph (GPU {args.gpu})')
        print(f'{"="*60}')
        cugraph_results = compute_cugraph_features(
            src_ids, tgt_ids, n_nodes, gpu_id=args.gpu)
        all_results.update(cugraph_results)
        save_feature_progress(all_results, n_nodes, labels, PROGRESS_JSON)
        print(f'  cuGraph: {len(cugraph_results)} features computed')
    else:
        print('\n[SKIP] cuGraph features')

    # === Phase 2: Memgraph ===
    if not args.skip_memgraph:
        print(f'\n{"="*60}')
        print(f'Phase 2: Memgraph ({args.mg_host}:{args.mg_port})')
        print(f'{"="*60}')
        try:
            mg_results = compute_memgraph_features(
                node_index, host=args.mg_host, port=args.mg_port)
            all_results.update(mg_results)
            save_feature_progress(all_results, n_nodes, labels, PROGRESS_JSON)
            print(f'  Memgraph: {len(mg_results)} features computed')
        except Exception as e:
            print(f'  Memgraph: FAILED - {e}')
    else:
        print('\n[SKIP] Memgraph features')

    # === Phase 3: NetworkX (CPU) ===
    if not args.skip_networkx:
        print(f'\n{"="*60}')
        print(f'Phase 3: NetworkX (CPU)')
        print(f'{"="*60}')
        nx_results = compute_networkx_features(src_ids, tgt_ids)
        all_results.update(nx_results)
        save_feature_progress(all_results, n_nodes, labels, PROGRESS_JSON)
        print(f'  NetworkX: {len(nx_results)} features computed')
    else:
        print('\n[SKIP] NetworkX features')

    # === 결과 저장 ===
    print(f'\n{"="*60}')
    print('Saving results')
    print(f'{"="*60}')

    # Build DataFrame at once instead of column-by-column
    feat_data = {feat_name: [feat_dict.get(i, 0) for i in range(n_nodes)]
                 for feat_name, feat_dict in all_results.items()}
    feat_df = pd.DataFrame(feat_data)
    feat_df.fillna(0, inplace=True)
    feat_df.to_csv(OUTPUT_CSV, index=False)

    total_elapsed = time.time() - total_start
    print(f'Saved {len(feat_df.columns)} features to {OUTPUT_CSV}')
    print(f'Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)')

    # 최종 통계
    print(f'\n{"="*60}')
    print('Feature Statistics (fraud vs benign)')
    print(f'{"="*60}')
    print_feature_summary(all_results, n_nodes, labels)

    print('\nDone!')


if __name__ == '__main__':
    main()
