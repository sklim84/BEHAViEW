"""
통합 오케스트레이션: 피처 발굴 → GPU idle 모니터링 → HP search 자동 시작

수행절차:
1. 그래프 피처 최대한 발굴 (cuGraph → Memgraph → NetworkX, 각 피처 타임아웃 적용)
2. 발굴된 피처를 NODE_FEAT.csv에 병합
3. GPU idle 모니터링 (1분 간격)
4. GPU idle 시 해당 시점까지 확보된 피처로 HP search 시작

사용법:
    nohup python -u scripts/run_features_and_hp.py --gpu 5 > logs/run_all.log 2>&1 &
"""
import argparse
import gc
import itertools
import json
import os
import signal
import time

import numpy as np
import pandas as pd

from scripts.hp_common import (
    MODELS, load_completed_jobs, run_parallel, wait_for_gpu_idle,
)

# ============================================================
# 설정
# ============================================================
NUM_GPUS = 6
FEATURE_TIMEOUT = 600  # 개별 피처 최대 10분

NODE_CSV = './datasets/HOFINET_NODE_FEAT.csv'
EDGE_CSV = './datasets/HOFINET_EDGES.csv'
PROGRESS_JSON = './logs/feature_discovery_progress.json'
HP_RESULT_FILE = './results/exp_results_hp_search_v2.csv'
HP_LOG_DIR = './logs/hp_search_v2'

# 기존 NODE_FEAT.csv에 없는 새 피처 후보
NEW_FEATURE_CANDIDATES = ['betweenness', 'louvain', 'constraint']

SEARCH_SPACE = {
    'input_dim': [8, 16, 32],
    'hidden_dim': [128, 256, 512],
    'lr': [1e-4, 5e-4, 1e-3, 1e-2],
    'gconv_nlayers': [2, 3],
}

COMMON_ARGS = {
    'seed': 2025,
    'node_data_name': 'HOFINET_NODE_FEAT',
    'edge_data_name': 'HOFINET_EDGES',
}


# ============================================================
# 피처 통계 유틸리티
# ============================================================
def compute_feature_stats(feat_dict, n_nodes, labels):
    """피처 dict에서 nz, susp_mean, benign_mean, ratio 계산"""
    vals = np.array([feat_dict.get(i, 0) for i in range(n_nodes)], dtype=float)
    nz = int((vals != 0).sum())
    susp_mean = float(vals[labels == 1].mean())
    benign_mean = float(vals[labels == 0].mean())
    ratio = susp_mean / benign_mean if benign_mean > 0 else float('inf')
    return nz, susp_mean, benign_mean, ratio


def save_feature_progress(features, n_nodes, labels):
    """진행 상황 JSON 저장"""
    progress = {}
    for k, v in features.items():
        nz, fm, bm, _ = compute_feature_stats(v, n_nodes, labels)
        progress[k] = {'count': len(v), 'nz': nz,
                        'susp_mean': fm, 'benign_mean': bm}
    with open(PROGRESS_JSON, 'w') as f:
        json.dump(progress, f, indent=2)


# ============================================================
# Step 1: 피처 발굴
# ============================================================
def run_with_timeout(name, func, timeout=FEATURE_TIMEOUT):
    """개별 피처 계산을 타임아웃과 함께 실행"""
    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    t0 = time.time()
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        result = func()
        print(f'  {name:25s}: {time.time()-t0:7.1f}s  OK')
        return result
    except TimeoutError:
        print(f'  {name:25s}: {time.time()-t0:7.1f}s  TIMEOUT ({timeout}s)')
        return None
    except Exception as e:
        print(f'  {name:25s}: {time.time()-t0:7.1f}s  FAILED - {e}')
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def discover_features(gpu_id=0):
    """피처 발굴: cuGraph → NetworkX"""
    node_df = pd.read_csv(NODE_CSV)
    edge_df = pd.read_csv(EDGE_CSV)
    node_index = {acc: i for i, acc in enumerate(node_df['account'])}
    labels = node_df['label'].values
    n_nodes = len(node_df)

    print(f'[Step 1] 피처 발굴 시작')
    print(f'  Nodes: {n_nodes:,}, Edges: {len(edge_df):,}')
    print(f'  발굴 대상: {NEW_FEATURE_CANDIDATES}')

    new_features = {}
    src_ids = edge_df['source'].map(node_index).values
    tgt_ids = edge_df['target'].map(node_index).values

    # --- Phase 1: cuGraph ---
    print(f'\n{"="*60}')
    print(f'Phase 1: cuGraph (GPU {gpu_id})')
    print(f'{"="*60}')
    orig_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        import cudf
        import cugraph

        gdf = cudf.DataFrame({'src': src_ids, 'dst': tgt_ids})
        G_un = cugraph.Graph(directed=False)
        G_un.from_cudf_edgelist(gdf, source='src', destination='dst')

        r = run_with_timeout('betweenness(k=500)',
            lambda: cugraph.betweenness_centrality(G_un, k=500, seed=2025))
        if r is not None:
            new_features['betweenness'] = dict(zip(
                r['vertex'].to_pandas(), r['betweenness_centrality'].to_pandas()))

        r = run_with_timeout('louvain', lambda: cugraph.louvain(G_un))
        if r is not None:
            parts, mod = r
            new_features['louvain'] = dict(zip(
                parts['vertex'].to_pandas(), parts['partition'].to_pandas()))
            print(f'    modularity: {mod:.4f}')

        del G_un, gdf
        gc.collect()
    except Exception as e:
        print(f'  cuGraph phase failed: {e}')
    finally:
        if orig_cuda is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = orig_cuda
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

    save_feature_progress(new_features, n_nodes, labels)

    # --- Phase 2: NetworkX ---
    print(f'\n{"="*60}')
    print(f'Phase 2: NetworkX (CPU)')
    print(f'{"="*60}')

    import networkx as nx
    G_un = nx.Graph()
    G_un.add_edges_from(zip(src_ids.astype(int), tgt_ids.astype(int)))
    print(f'  Undirected graph: {G_un.number_of_nodes():,} nodes, '
          f'{G_un.number_of_edges():,} edges')

    r = run_with_timeout('constraint', lambda: nx.constraint(G_un))
    if r is not None:
        new_features['constraint'] = {
            int(k): (v if v is not None else 0.0) for k, v in r.items()}
        save_feature_progress(new_features, n_nodes, labels)

    del G_un
    gc.collect()

    # --- 결과 요약 ---
    print(f'\n{"="*60}')
    print(f'피처 발굴 완료: {len(new_features)}개 새 피처')
    print(f'{"="*60}')
    for feat_name, feat_dict in new_features.items():
        nz, fm, bm, ratio = compute_feature_stats(feat_dict, n_nodes, labels)
        print(f'  {feat_name:16s}: nz={nz:>7,}/{n_nodes:,}  '
              f'susp/benign={ratio:.2f}x')

    return new_features, n_nodes, node_df


def merge_features_to_csv(new_features, n_nodes, node_df):
    """새 피처를 NODE_FEAT.csv에 병합"""
    if not new_features:
        print('[Merge] 새 피처 없음, 병합 스킵')
        return []

    added = []
    for feat_name, feat_dict in new_features.items():
        if feat_name in node_df.columns:
            print(f'  {feat_name}: 이미 존재, 스킵')
            continue
        col_vals = [feat_dict.get(i, 0) for i in range(n_nodes)]
        acc_idx = list(node_df.columns).index('account')
        node_df.insert(acc_idx, feat_name, col_vals)
        added.append(feat_name)
        print(f'  {feat_name}: 추가됨')

    if added:
        node_df.to_csv(NODE_CSV, index=False)
        print(f'[Merge] {len(added)}개 피처 추가 → {NODE_CSV} 저장 완료')
    return added


# ============================================================
# Step 2 & 3: GPU 모니터링 → HP search
# ============================================================
def build_struct_feats_sets(node_df):
    """NODE_FEAT.csv의 centrality 피처 기반으로 struct_feats 조합 생성"""
    agg_prefixes = ('out_', 'in_mean', 'in_max', 'in_std', 'in_count',
                    'md_', 'fnd_')
    cen_cols = [c for c in node_df.columns
                if c not in ('account', 'label')
                and not any(c.startswith(p) for p in agg_prefixes)]

    baseline = [c for c in
                ['dc', 'cc', 'pagerank', 'hits_hub', 'hits_auth',
                 'kcore', 'triangle']
                if c in cen_cols]

    top = [c for c in
           ['triangle', 'dc', 'hits_auth', 'hits_hub', 'kcore',
            'in_dc', 'out_dc', 'betweenness', 'load_cen',
            'constraint', 'eff_size']
           if c in cen_cols]

    sets = {
        'baseline': ' '.join(baseline),
        'all_cen': ' '.join(cen_cols),
        'top_discrim': ' '.join(top),
    }

    print(f'\n[struct_feats 조합]')
    for name, feats in sets.items():
        print(f'  {name}: {feats}')

    return sets


def generate_hp_jobs(struct_feats_sets):
    """struct_feats 조합별 HP search 작업 생성"""
    key_fields = [
        ('Model', str.lower),
        ('struct_feats', lambda x: x),
        ('lr', float),
        ('input_dim', int),
        ('hidden_dim', int),
        ('gconv_nlayers', int),
    ]
    completed = load_completed_jobs(HP_RESULT_FILE, key_fields)
    jobs = []
    skipped = 0
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())

    for model_name, model_cfg in MODELS.items():
        for cen_name, struct_feats in struct_feats_sets.items():
            for combo in itertools.product(*values):
                hp = dict(zip(keys, combo))
                key = (model_name, struct_feats, hp['lr'], hp['input_dim'],
                       hp['hidden_dim'], hp['gconv_nlayers'])
                if key in completed:
                    skipped += 1
                    continue
                jobs.append((model_name, model_cfg, hp, struct_feats, cen_name))

    if skipped:
        print(f'[HP Search] {skipped} 이미 완료된 작업 스킵')
    return jobs


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5, help='cuGraph용 GPU')
    parser.add_argument('--skip_features', action='store_true',
                        help='피처 발굴 생략')
    args = parser.parse_args()

    os.makedirs('./logs', exist_ok=True)
    os.makedirs(HP_LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(HP_RESULT_FILE), exist_ok=True)

    total_start = time.time()

    # Step 1: 피처 발굴
    if not args.skip_features:
        new_features, n_nodes, node_df = discover_features(gpu_id=args.gpu)
        added = merge_features_to_csv(new_features, n_nodes, node_df)
        print(f'[Step 1 완료] 새 피처 {len(added)}개 추가: {added}')
    else:
        print('[Step 1 스킵] 기존 CSV 피처만 사용')

    # Step 2: GPU idle 대기
    wait_for_gpu_idle()

    # Step 3: HP search
    node_df = pd.read_csv(NODE_CSV)
    struct_feats_sets = build_struct_feats_sets(node_df)
    jobs = generate_hp_jobs(struct_feats_sets)

    if not jobs:
        print('[HP Search] 모든 작업 완료됨')
        return

    run_parallel(jobs, NUM_GPUS, HP_RESULT_FILE, COMMON_ARGS,
                 log_dir=HP_LOG_DIR, log_prefix='gpu', label='HP Search')

    total_elapsed = time.time() - total_start
    print(f'\n전체 소요시간: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)')


if __name__ == '__main__':
    main()
