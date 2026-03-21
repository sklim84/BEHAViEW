"""
HP Search 공통 모듈 — 모델 설정, 작업 실행, 결과 로드 등 공통 로직.
hp_search.py, run_features_and_hp.py에서 사용.
"""
import csv
import itertools
import os
import subprocess
import time
from collections import deque

# 6 아키텍처 × 2 변형 = 12 모델
MODELS_CEN = {
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

MODELS_ORG = {
    'bgrl_w_org': {
        'script': 'models/bgrl_w_org.py',
        'fixed': {'loss': 'BarlowTwins'},
    },
    'dgi_inductive_w_org': {
        'script': 'models/dgi_inductive_w_org.py',
        'fixed': {'loss': 'JSD'},
    },
    'dgi_transductive_w_org': {
        'script': 'models/dgi_transductive_w_org.py',
        'fixed': {'loss': 'JSD'},
    },
    'gbt_w_org': {
        'script': 'models/gbt_w_org.py',
        'fixed': {'loss': 'BarlowTwins'},
    },
    'grace_w_org': {
        'script': 'models/grace_w_org.py',
        'fixed': {'loss': 'InfoNCE', 'proj_dim': 32},
    },
    'mvgrl_w_org': {
        'script': 'models/mvgrl_w_org.py',
        'fixed': {'loss': 'BootstrapLatent'},
    },
}

MODELS = MODELS_CEN  # 기존 호환성 유지
MODELS_ALL = {**MODELS_CEN, **MODELS_ORG}


def build_command(model_cfg, model_name, hp, gpu_id, result_file,
                  common_args, cen_feats=None):
    """모델 실행 커맨드 생성"""
    is_org = '_w_org' in model_name
    cmd = ['python', '-u', model_cfg['script']]
    cmd += ['--model_name', model_name]
    cmd += ['--gpu', str(gpu_id)]
    cmd += ['--metric_save_path', result_file]
    cmd += ['--skip_tsne']

    for k, v in common_args.items():
        if k == 'cen_feats':
            if not is_org:
                cmd += ['--cen_feats'] + str(v).split()
        else:
            cmd += [f'--{k}', str(v)]

    if cen_feats and not is_org:
        cmd += ['--cen_feats'] + cen_feats.split()

    for k, v in hp.items():
        cmd += [f'--{k}', str(v)]

    for k, v in model_cfg['fixed'].items():
        cmd += [f'--{k}', str(v)]

    return cmd


def load_completed_jobs(result_file, key_fields=None):
    """결과 CSV에서 완료된 작업 키 세트를 로드.

    key_fields: CSV 컬럼 → 변환 함수 튜플 리스트.
        기본: [('Model', str.lower), ('lr', float), ('input_dim', int),
               ('hidden_dim', int), ('gconv_nlayers', int)]
    """
    if key_fields is None:
        key_fields = [
            ('Model', str.lower),
            ('lr', float),
            ('input_dim', int),
            ('hidden_dim', int),
            ('gconv_nlayers', int),
        ]

    completed = set()
    if not os.path.exists(result_file):
        return completed

    with open(result_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = tuple(conv(row[col]) for col, conv in key_fields)
                completed.add(key)
            except (KeyError, ValueError):
                continue
    return completed


def generate_jobs(search_space, result_file, models=None,
                  key_fields=None, extra_job_data=None):
    """모델 × HP 조합 생성 (완료 작업 제외).

    extra_job_data: 각 job에 추가할 데이터를 생성하는 함수
        (model_name, model_cfg, hp) -> tuple of extra values
    Returns: (jobs list, skipped count)
    """
    if models is None:
        models = MODELS

    completed = load_completed_jobs(result_file, key_fields)
    jobs = []
    skipped = 0
    keys = list(search_space.keys())
    values = list(search_space.values())

    for model_name, model_cfg in models.items():
        for combo in itertools.product(*values):
            hp = dict(zip(keys, combo))
            key = (model_name, hp['lr'], hp['input_dim'],
                   hp['hidden_dim'], hp['gconv_nlayers'])
            if key in completed:
                skipped += 1
                continue
            job = (model_name, model_cfg, hp)
            if extra_job_data:
                job += extra_job_data(model_name, model_cfg, hp)
            jobs.append(job)

    return jobs, skipped


def run_parallel(jobs, num_gpus, result_file, common_args, log_dir='logs',
                 log_prefix='hp_search', cen_feats=None, label='HP Search'):
    """GPU에 분배하여 병렬 실행.

    jobs: [(model_name, model_cfg, hp, ...)] 리스트
    cen_feats: 문자열 또는 None. None이면 common_args에서 가져옴.
        job이 5-tuple이면 (model_name, model_cfg, hp, cen_feats, cen_name)으로 간주.
    """
    total = len(jobs)
    print(f'[{label}] Total jobs: {total}, GPUs: {num_gpus}')
    print(f'[{label}] Results: {result_file}')
    print()

    queue = deque(jobs)
    running = {}  # gpu_id -> (process, model_name, hp, start_time, log_file, extra_label)
    completed = 0
    failed = 0

    os.makedirs(log_dir, exist_ok=True)

    while queue or running:
        for gpu_id in range(num_gpus):
            if gpu_id not in running and queue:
                job = queue.popleft()
                model_name, model_cfg, hp = job[0], job[1], job[2]

                # cen_feats / cen_name 결정
                job_cen_feats = cen_feats
                extra_label = ''
                if len(job) >= 5:
                    job_cen_feats = job[3]
                    extra_label = f' [{job[4]}]'

                cmd = build_command(
                    model_cfg, model_name, hp, gpu_id, result_file,
                    common_args, cen_feats=job_cen_feats)

                log_file = open(
                    os.path.join(log_dir, f'{log_prefix}_gpu{gpu_id}.log'), 'w')
                proc = subprocess.Popen(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT)
                running[gpu_id] = (
                    proc, model_name, hp, time.time(), log_file, extra_label)

                hp_str = ', '.join(f'{k}={v}' for k, v in hp.items())
                print(f'  [GPU {gpu_id}] START {model_name}{extra_label} | {hp_str}')

        done_gpus = []
        for gpu_id, (proc, mn, hp, t0, lf, el) in running.items():
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - t0
                lf.close()
                if ret == 0:
                    completed += 1
                    print(f'  [GPU {gpu_id}] DONE  {mn}{el} ({elapsed:.0f}s) '
                          f'[{completed + failed}/{total}]')
                else:
                    failed += 1
                    print(f'  [GPU {gpu_id}] FAIL  {mn}{el} '
                          f'(exit={ret}, {elapsed:.0f}s) '
                          f'[{completed + failed}/{total}]')
                done_gpus.append(gpu_id)

        for gpu_id in done_gpus:
            del running[gpu_id]

        if running:
            time.sleep(5)

    print(f'\n[{label}] Complete! Success: {completed}, Failed: {failed}')
    print(f'[{label}] Results saved to {result_file}')


def check_gpu_idle(threshold=10):
    """모든 GPU utilization이 threshold 이하인지 확인"""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True)
        for line in out.strip().split('\n'):
            parts = line.split(',')
            util = int(parts[1].strip())
            if util > threshold:
                return False, out.strip()
        return True, out.strip()
    except Exception as e:
        return False, str(e)


def wait_for_gpu_idle(threshold=10, interval=60):
    """GPU가 idle해질 때까지 대기"""
    print(f'[Monitor] Waiting for GPUs idle (threshold={threshold}%, '
          f'interval={interval}s)')
    while True:
        idle, info = check_gpu_idle(threshold)
        now = time.strftime('%H:%M:%S')
        if idle:
            print(f'[Monitor] [{now}] All GPUs idle!')
            return
        print(f'[Monitor] [{now}] GPUs busy — '
              f'{info.replace(chr(10), " | ")}')
        time.sleep(interval)
