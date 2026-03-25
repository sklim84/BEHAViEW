"""
그래프 피처 계산 공통 유틸리티.
compute_features_hybrid.py, pp_hofinet.py에서 사용.
"""
import json
import time

import numpy as np


def run_feature(name, func):
    """피처 계산 함수를 실행하고 소요시간/결과를 출력한다.

    Returns: 계산 결과 또는 None (실패 시)
    """
    t0 = time.time()
    try:
        result = func()
        print(f'  {name:25s}: {time.time()-t0:7.1f}s  OK')
        return result
    except Exception as e:
        print(f'  {name:25s}: {time.time()-t0:7.1f}s  FAILED - {e}')
        return None


def compute_feature_stats(feat_dict, n_nodes, labels):
    """피처 dict → (nz, susp_mean, benign_mean, ratio)"""
    vals = np.array([feat_dict.get(i, 0) for i in range(n_nodes)], dtype=float)
    nz = int((vals != 0).sum())
    susp_mean = float(vals[labels == 1].mean())
    benign_mean = float(vals[labels == 0].mean())
    ratio = susp_mean / benign_mean if benign_mean > 0 else float('inf')
    return nz, susp_mean, benign_mean, ratio


def print_feature_stats(feat_name, feat_dict, n_nodes, labels, elapsed=None):
    """피처 통계를 포맷팅하여 출력"""
    nz, sm, bm, ratio = compute_feature_stats(feat_dict, n_nodes, labels)
    time_str = f'{elapsed:7.1f}s' if elapsed and elapsed > 0 else '       '
    print(f'  {feat_name:16s}: {time_str}  nz={nz:>7,}/{n_nodes:,} '
          f'({nz*100/n_nodes:5.1f}%)  suspicious={sm:.6f}  benign={bm:.6f}  '
          f'ratio={ratio:.2f}x')


def print_feature_summary(features, n_nodes, labels):
    """여러 피처의 통계를 일괄 출력"""
    for feat_name, feat_dict in features.items():
        if isinstance(feat_dict, dict):
            nz, sm, bm, ratio = compute_feature_stats(feat_dict, n_nodes, labels)
            print(f'  {feat_name:16s}: nz={nz:>7,}/{n_nodes:,} '
                  f'({nz*100/n_nodes:5.1f}%)  suspicious={sm:.6f}  '
                  f'benign={bm:.6f}  ratio={ratio:.2f}x')


def save_feature_progress(features, n_nodes, labels, output_path):
    """피처 발굴 진행 상황을 JSON으로 저장"""
    progress = {}
    for k, v in features.items():
        nz, sm, bm, _ = compute_feature_stats(v, n_nodes, labels)
        progress[k] = {'count': len(v), 'nz': nz,
                        'susp_mean': sm, 'benign_mean': bm}
    with open(output_path, 'w') as f:
        json.dump(progress, f, indent=2)
