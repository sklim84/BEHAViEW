"""
하이퍼파라미터 서치 통합 스크립트 — 프리셋 또는 커스텀 설정 지원.

사용법:
    # 프리셋 실행
    python scripts/hp_search.py --preset baseline          # 7개 피처, 432 jobs
    python scripts/hp_search.py --preset 17feat            # 17개 피처, 216 jobs
    python scripts/hp_search.py --preset v2                # 3개 피처 세트 비교, 1296 jobs

    # 커스텀 실행
    python scripts/hp_search.py --cen_feats dc cc pagerank --num_gpus 2
    python scripts/hp_search.py --preset 17feat --num_gpus 6 --lr 1e-4 5e-4 1e-3

    # GPU idle 대기 후 시작
    python scripts/hp_search.py --preset v2 --wait_gpu_idle

    # nohup 백그라운드
    nohup python -u scripts/hp_search.py --preset 17feat --num_gpus 1 > logs/hp_search.log 2>&1 &
"""
import argparse
import itertools
import os

from scripts.hp_common import (
    MODELS, MODELS_CEN, MODELS_ORG, MODELS_ALL,
    load_completed_jobs, run_parallel, wait_for_gpu_idle,
)

# ============================================================
# 프리셋 정의
# ============================================================
PRESETS = {
    'baseline': {
        'cen_feats_sets': {
            'baseline': 'dc cc pagerank hits_hub hits_auth kcore triangle',
        },
        'search_space': {
            'input_dim': [8, 16, 32],
            'hidden_dim': [128, 256, 512],
            'lr': [1e-4, 5e-4, 1e-3, 1e-2],
            'gconv_nlayers': [2, 3],
        },
        'result_file': './results/exp_results_hp_search.csv',
        'num_gpus': 6,
    },
    '17feat': {
        'cen_feats_sets': {
            '17feat': ('dc in_dc out_dc cc harmonic pagerank hits_hub '
                       'hits_auth eigenvector kcore triangle clustering '
                       'sq_clustering avg_neigh_deg betweenness louvain '
                       'constraint'),
        },
        'search_space': {
            'input_dim': [16, 32, 64],
            'hidden_dim': [256, 512],
            'lr': [1e-4, 5e-4, 1e-3],
            'gconv_nlayers': [2, 3],
        },
        'result_file': './results/exp_results_hp_search_17feat.csv',
        'num_gpus': 1,
    },
    'v2': {
        'cen_feats_sets': {
            'baseline': 'dc cc pagerank hits_hub hits_auth kcore triangle',
            'extended': ('dc cc pagerank hits_hub hits_auth kcore triangle '
                         'in_dc out_dc clustering avg_neigh_deg harmonic '
                         'sq_clustering'),
            'top_discrim': 'dc hits_auth hits_hub kcore triangle in_dc out_dc',
        },
        'search_space': {
            'input_dim': [8, 16, 32],
            'hidden_dim': [128, 256, 512],
            'lr': [1e-4, 5e-4, 1e-3, 1e-2],
            'gconv_nlayers': [2, 3],
        },
        'result_file': './results/exp_results_hp_search_v2.csv',
        'num_gpus': 6,
    },
    'compare': {
        'cen_feats_sets': {
            'baseline': 'dc cc pagerank hits_hub hits_auth kcore triangle',
        },
        'search_space': {
            'input_dim': [8, 16, 32],
            'hidden_dim': [128, 256, 512],
            'lr': [1e-4, 5e-4, 1e-3, 1e-2],
            'gconv_nlayers': [2, 3],
        },
        'result_file': './results/exp_results_hp_compare.csv',
        'num_gpus': 6,
        'models': 'all',  # _w_cen + _w_org 모두
    },
}

COMMON_ARGS = {
    'seed': 2025,
    'node_data_name': 'HOFINET_NODE_FEAT',
    'edge_data_name': 'HOFINET_EDGES',
}


# ============================================================
# 작업 생성
# ============================================================
def generate_jobs(search_space, result_file, cen_feats_sets, models=None):
    """모델 × cen_feats × HP 조합 생성 (완료 제외)"""
    if models is None:
        models = MODELS
    multi_set = len(cen_feats_sets) > 1

    if multi_set:
        key_fields = [
            ('Model', str.lower),
            ('cen_feats', lambda x: x),
            ('lr', float),
            ('input_dim', int),
            ('hidden_dim', int),
            ('gconv_nlayers', int),
        ]
    else:
        key_fields = [
            ('Model', str.lower),
            ('lr', float),
            ('input_dim', int),
            ('hidden_dim', int),
            ('gconv_nlayers', int),
        ]

    completed = load_completed_jobs(result_file, key_fields)
    jobs = []
    skipped = 0
    keys = list(search_space.keys())
    values = list(search_space.values())

    for model_name, model_cfg in models.items():
        is_org = '_w_org' in model_name
        feats_iter = {'none': ''} if is_org else cen_feats_sets
        for cen_name, cen_feats in feats_iter.items():
            for combo in itertools.product(*values):
                hp = dict(zip(keys, combo))
                if multi_set and not is_org:
                    key = (model_name, cen_feats, hp['lr'], hp['input_dim'],
                           hp['hidden_dim'], hp['gconv_nlayers'])
                else:
                    key = (model_name, hp['lr'], hp['input_dim'],
                           hp['hidden_dim'], hp['gconv_nlayers'])
                if key in completed:
                    skipped += 1
                    continue
                jobs.append((model_name, model_cfg, hp, cen_feats, cen_name))

    return jobs, skipped


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='HP Search (unified)')
    parser.add_argument('--preset', choices=list(PRESETS.keys()),
                        help='프리셋 선택 (baseline/17feat/v2/compare)')
    parser.add_argument('--cen_feats', nargs='+',
                        help='커스텀 cen_feats (프리셋 대신 사용)')
    parser.add_argument('--num_gpus', type=int, help='사용할 GPU 수')
    parser.add_argument('--result_file', type=str, help='결과 CSV 경로')
    parser.add_argument('--wait_gpu_idle', action='store_true',
                        help='GPU idle 대기 후 시작')

    # HP 범위 오버라이드
    parser.add_argument('--input_dim', nargs='+', type=int)
    parser.add_argument('--hidden_dim', nargs='+', type=int)
    parser.add_argument('--lr', nargs='+', type=float)
    parser.add_argument('--gconv_nlayers', nargs='+', type=int)

    args = parser.parse_args()

    # 프리셋 로드 또는 기본값
    if args.preset:
        cfg = PRESETS[args.preset].copy()
        label = f'HP Search [{args.preset}]'
    else:
        cfg = PRESETS['baseline'].copy()
        label = 'HP Search [custom]'

    search_space = cfg['search_space'].copy()
    cen_feats_sets = cfg['cen_feats_sets'].copy()
    result_file = cfg['result_file']
    num_gpus = cfg['num_gpus']

    # 모델 세트 결정
    models_key = cfg.get('models', 'cen')
    if models_key == 'all':
        selected_models = MODELS_ALL
    elif models_key == 'org':
        selected_models = MODELS_ORG
    else:
        selected_models = MODELS_CEN

    # CLI 오버라이드 적용
    if args.cen_feats:
        cen_feats_sets = {'custom': ' '.join(args.cen_feats)}
    if args.num_gpus is not None:
        num_gpus = args.num_gpus
    if args.result_file:
        result_file = args.result_file
    if args.input_dim:
        search_space['input_dim'] = args.input_dim
    if args.hidden_dim:
        search_space['hidden_dim'] = args.hidden_dim
    if args.lr:
        search_space['lr'] = args.lr
    if args.gconv_nlayers:
        search_space['gconv_nlayers'] = args.gconv_nlayers

    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    if args.wait_gpu_idle:
        wait_for_gpu_idle()

    # 작업 생성
    jobs, skipped = generate_jobs(search_space, result_file, cen_feats_sets,
                                  models=selected_models)
    n_hp = len(list(itertools.product(*search_space.values())))

    print(f'[{label}] Skipping {skipped} already completed jobs')
    print(f'[{label}] {len(selected_models)} models × '
          f'{len(cen_feats_sets)} cen_feats sets × {n_hp} HP = {len(jobs)} jobs')
    print(f'[{label}] Search space: {search_space}')
    for name, feats in cen_feats_sets.items():
        print(f'[{label}] cen_feats[{name}] ({len(feats.split())}): {feats}')

    if not jobs:
        print(f'[{label}] No jobs to run — all completed!')
        return

    run_parallel(jobs, num_gpus, result_file, COMMON_ARGS,
                 log_prefix='hp_search', label=label)


if __name__ == '__main__':
    main()
