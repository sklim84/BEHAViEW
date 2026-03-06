"""
하이퍼파라미터 서치 스크립트 — 6 GPU 병렬 실행
사용법: python scripts/hp_search.py
"""
import subprocess
import itertools
import time
import os
import csv
from collections import deque

NUM_GPUS = 6

# 모델별 설정: (script, model_name, epochs_approx, extra_fixed_args)
MODELS = {
    'bgrl_w_cen': {
        'script': 'models/bgrl_w_cen.py',
        'fixed': {'loss': 'BarlowTwins'},
    },
    'dgi_inductive_w_cen': {
        'script': 'models/dgi_inductive_w_cen.py',
        'fixed': {'loss': 'JSD'},
    },
    'dgi_transductive_w_cen': {
        'script': 'models/dgi_transductive_w_cen.py',
        'fixed': {'loss': 'JSD'},
    },
    'gbt_w_cen': {
        'script': 'models/gbt_w_cen.py',
        'fixed': {'loss': 'BarlowTwins'},
    },
    'grace_w_cen': {
        'script': 'models/grace_w_cen.py',
        'fixed': {'loss': 'InfoNCE', 'proj_dim': 32},
    },
    'mvgrl_w_cen': {
        'script': 'models/mvgrl_w_cen.py',
        'fixed': {'loss': 'BootstrapLatent'},
    },
}

# 하이퍼파라미터 서치 공간
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
    'cen_feats': 'dc cc pagerank hits_hub hits_auth kcore triangle',
}

RESULT_FILE = './results/exp_results_hp_search.csv'


def load_completed_jobs():
    """이미 완료된 (model, lr, input_dim, hidden_dim, gconv_nlayers) 조합 로드"""
    completed = set()
    if not os.path.exists(RESULT_FILE):
        return completed
    with open(RESULT_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row['Model'],
                float(row['lr']),
                int(row['input_dim']),
                int(row['hidden_dim']),
                int(row['gconv_nlayers']),
            )
            completed.add(key)
    return completed


def generate_jobs():
    """모든 모델 × 하이퍼파라미터 조합 생성 (완료된 작업 제외)"""
    completed = load_completed_jobs()
    jobs = []
    skipped = 0
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())

    for model_name, model_cfg in MODELS.items():
        for combo in itertools.product(*values):
            hp = dict(zip(keys, combo))
            key = (model_name, hp['lr'], hp['input_dim'], hp['hidden_dim'], hp['gconv_nlayers'])
            if key in completed:
                skipped += 1
                continue
            jobs.append((model_name, model_cfg, hp))

    print(f'[HP Search] Skipping {skipped} already completed jobs')
    return jobs


def build_command(model_name, model_cfg, hp, gpu_id):
    """실행 커맨드 생성"""
    cmd = ['python', '-u', model_cfg['script']]
    cmd += ['--model_name', model_name]
    cmd += ['--gpu', str(gpu_id)]
    cmd += ['--metric_save_path', RESULT_FILE]
    cmd += ['--skip_tsne']

    for k, v in COMMON_ARGS.items():
        if k == 'cen_feats':
            cmd += ['--cen_feats'] + v.split()
        else:
            cmd += [f'--{k}', str(v)]

    for k, v in hp.items():
        cmd += [f'--{k}', str(v)]

    for k, v in model_cfg['fixed'].items():
        cmd += [f'--{k}', str(v)]

    return cmd


def run_parallel(jobs):
    """6 GPU에 분배하여 병렬 실행"""
    total = len(jobs)
    print(f'[HP Search] Total jobs: {total} ({total // NUM_GPUS} batches + {total % NUM_GPUS} remaining)')
    print(f'[HP Search] Search space: {SEARCH_SPACE}')
    print(f'[HP Search] Models: {list(MODELS.keys())}')
    print(f'[HP Search] Results: {RESULT_FILE}')
    print()

    queue = deque(jobs)
    running = {}  # gpu_id -> (process, model_name, hp, start_time)
    completed = 0
    failed = 0

    while queue or running:
        # 빈 GPU에 작업 할당
        for gpu_id in range(NUM_GPUS):
            if gpu_id not in running and queue:
                model_name, model_cfg, hp = queue.popleft()
                cmd = build_command(model_name, model_cfg, hp, gpu_id)
                log_file = open(f'/tmp/hp_search_gpu{gpu_id}.log', 'w')
                proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                running[gpu_id] = (proc, model_name, hp, time.time(), log_file)
                hp_str = ', '.join(f'{k}={v}' for k, v in hp.items())
                print(f'  [GPU {gpu_id}] START {model_name} | {hp_str}')

        # 완료된 작업 확인
        done_gpus = []
        for gpu_id, (proc, model_name, hp, start_time, log_file) in running.items():
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - start_time
                log_file.close()
                if ret == 0:
                    completed += 1
                    print(f'  [GPU {gpu_id}] DONE  {model_name} ({elapsed:.0f}s) [{completed + failed}/{total}]')
                else:
                    failed += 1
                    print(f'  [GPU {gpu_id}] FAIL  {model_name} (exit={ret}, {elapsed:.0f}s) [{completed + failed}/{total}]')
                done_gpus.append(gpu_id)

        for gpu_id in done_gpus:
            del running[gpu_id]

        if running:
            time.sleep(5)

    print(f'\n[HP Search] Complete! Success: {completed}, Failed: {failed}')
    print(f'[HP Search] Results saved to {RESULT_FILE}')


def main():
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    jobs = generate_jobs()
    run_parallel(jobs)


if __name__ == '__main__':
    main()
