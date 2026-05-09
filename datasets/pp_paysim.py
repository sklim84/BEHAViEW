"""PaySim 전처리 파이프라인.

Transaction CSV → Node features + Edge list (HOFINET/AMLworld와 동일 schema)

PaySim은 FFD task (isMoneyLaundering 라벨 없음, isFraud만 존재).
BECON 의 (γ) AML vs FFD transfer test 용으로 추출.

- step: 1 step = 1 hour, 743 steps = ~30.96 days
- Amount stats: out/in × {mean, max, std, count}
- Entropy: type (5 payment types: PAYMENT/TRANSFER/CASH_OUT/DEBIT/CASH_IN). 카테고리 컬럼 없음 → 1 entropy
- Temporal: 1w/2w/3w windows (step 단위)
- Label: isFraud (FFD!) — account-level max 집계

Usage:
    python datasets/pp_paysim.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy


def compute_entropy_feat(group):
    counts = group.value_counts(normalize=True)
    return entropy(counts)


def main():
    input_path = 'datasets/paysim/PS_20174392719_1491204439457_log.csv'
    output_dir = 'datasets/paysim'

    print(f'[INFO] Loading {input_path}...')
    df = pd.read_csv(input_path)
    print(f'[INFO] Loaded {len(df):,} transactions')

    df['source']   = df['nameOrig'].astype(str)
    df['target']   = df['nameDest'].astype(str)
    df['tran_amt'] = df['amount'].fillna(0)
    df['Is_Fraud'] = df['isFraud'].astype(int)

    print(f'[INFO] Fraud tx: {df["Is_Fraud"].sum():,} ({df["Is_Fraud"].mean()*100:.2f}%)')

    # === Node Features ===
    print('[INFO] Building amount statistics...')
    features_out = df.groupby('source')['tran_amt'].agg(['mean','max','std','count']).add_prefix('out_')
    features_in  = df.groupby('target')['tran_amt'].agg(['mean','max','std','count']).add_prefix('in_')
    node_features = pd.concat([features_out, features_in], axis=1).fillna(0)

    print('[INFO] Building entropy feature (type only — no category column)...')
    type_entropy = df.groupby('source')['type'].apply(compute_entropy_feat).rename('payment_entropy')
    node_features = node_features.join(type_entropy, how='left')
    # Add zero column to preserve schema parity (BECON expects 2 entropy features)
    node_features['currency_entropy'] = 0.0
    node_features.fillna(0, inplace=True)

    print('[INFO] Building temporal features (1w/2w/3w windows on step)...')
    max_step = df['step'].max()
    for weeks, hours in [(1, 168), (2, 336), (3, 504)]:
        cutoff = max_step - hours
        df_win = df[df['step'] >= cutoff]
        win_out = df_win.groupby('source')['tran_amt'].agg(['mean','count']).add_prefix(f'out_{weeks}m_')
        win_in  = df_win.groupby('target')['tran_amt'].agg(['mean','count']).add_prefix(f'in_{weeks}m_')
        node_features = node_features.join(win_out, how='left').join(win_in, how='left')
    node_features.fillna(0, inplace=True)

    # === Labels (account is fraud if any of its tx is fraud) ===
    src_label = df.groupby('source')['Is_Fraud'].max()
    tgt_label = df.groupby('target')['Is_Fraud'].max()
    labels = src_label.combine(
        tgt_label, func=lambda s, t: max(s, t)
    ).fillna(src_label).fillna(tgt_label).fillna(0).astype(int)
    node_features['label']   = labels
    node_features['account'] = node_features.index

    edge_df = df[['source', 'target']]

    os.makedirs(output_dir, exist_ok=True)
    node_path = os.path.join(output_dir, 'PAYSIM_NODE_FEAT.csv')
    edge_path = os.path.join(output_dir, 'PAYSIM_EDGES.csv')
    node_features.reset_index(drop=True).to_csv(node_path, index=False)
    edge_df.to_csv(edge_path, index=False)

    n_fraud, n_total = (node_features['label']==1).sum(), len(node_features)
    print(f'[INFO] Saved {node_path}: {n_total:,} nodes '
          f'(fraud: {n_fraud:,}, {n_fraud*100/n_total:.2f}%)')
    print(f'[INFO] Saved {edge_path}: {len(edge_df):,} edges')


if __name__ == '__main__':
    main()
