"""AMLNet preprocessing pipeline.

Transaction CSV -> node features + edge list (same schema as the AMLworld pipeline)

- Source/Target: nameOrig/nameDest (account ID)
- Amount stats: out/in × {mean, max, std, count}
- Entropy: type (6 payment methods), category (11 transaction categories)
- Temporal: 1/2/3-month windows (195-day simulation, span ~6 months)
- Label: isMoneyLaundering at tx level → max-aggregated to account level

Usage:
    python datasets/pp_amlnet.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy


def compute_entropy_feat(group):
    counts = group.value_counts(normalize=True)
    return entropy(counts)


def main():
    input_path = 'datasets/amlnet/AMLNet_August 2025.csv'
    output_dir = 'datasets/amlnet'

    print(f'[INFO] Loading {input_path}...')
    usecols = ['step', 'type', 'amount', 'category', 'nameOrig', 'nameDest',
               'isMoneyLaundering', 'hour', 'day_of_week', 'day_of_month', 'month']
    df = pd.read_csv(input_path, usecols=usecols)
    print(f'[INFO] Loaded {len(df):,} transactions')

    df['source'] = df['nameOrig'].astype(str)
    df['target'] = df['nameDest'].astype(str)
    df['tran_amt'] = df['amount'].fillna(0)
    df['Is_Laundering'] = df['isMoneyLaundering'].astype(int)

    print(f'[INFO] Laundering tx: {df["Is_Laundering"].sum():,} '
          f'({df["Is_Laundering"].mean()*100:.2f}%)')

    # === Node Features ===
    print('[INFO] Building amount statistics...')
    features_out = df.groupby('source')['tran_amt'].agg(['mean','max','std','count']).add_prefix('out_')
    features_in  = df.groupby('target')['tran_amt'].agg(['mean','max','std','count']).add_prefix('in_')
    node_features = pd.concat([features_out, features_in], axis=1).fillna(0)

    print('[INFO] Building entropy features...')
    payment_entropy  = df.groupby('source')['type'].apply(compute_entropy_feat).rename('payment_entropy')
    currency_entropy = df.groupby('source')['category'].apply(compute_entropy_feat).rename('currency_entropy')
    node_features = node_features.join(payment_entropy, how='left').join(currency_entropy, how='left')
    node_features.fillna(0, inplace=True)

    print('[INFO] Building temporal features (1/2/3-month windows)...')
    df['tran_date'] = pd.to_datetime(dict(year=2025, month=df['month'], day=df['day_of_month']))
    max_date = df['tran_date'].max()
    for months in [1, 2, 3]:
        cutoff = max_date - pd.DateOffset(months=months)
        df_win = df[df['tran_date'] >= cutoff]
        win_out = df_win.groupby('source')['tran_amt'].agg(['mean','count']).add_prefix(f'out_{months}m_')
        win_in  = df_win.groupby('target')['tran_amt'].agg(['mean','count']).add_prefix(f'in_{months}m_')
        node_features = node_features.join(win_out, how='left').join(win_in, how='left')
    node_features.fillna(0, inplace=True)
    df.drop(columns=['tran_date'], inplace=True)

    # === Labels (max-aggregated across roles) ===
    src_label = df.groupby('source')['Is_Laundering'].max()
    tgt_label = df.groupby('target')['Is_Laundering'].max()
    labels = src_label.combine(
        tgt_label, func=lambda s, t: max(s, t)
    ).fillna(src_label).fillna(tgt_label).fillna(0).astype(int)
    node_features['label']   = labels
    node_features['account'] = node_features.index

    edge_df = df[['source', 'target']]

    os.makedirs(output_dir, exist_ok=True)
    node_path = os.path.join(output_dir, 'AMLNET_NODE_FEAT.csv')
    edge_path = os.path.join(output_dir, 'AMLNET_EDGES.csv')
    node_features.reset_index(drop=True).to_csv(node_path, index=False)
    edge_df.to_csv(edge_path, index=False)

    n_susp, n_total = (node_features['label']==1).sum(), len(node_features)
    print(f'[INFO] Saved {node_path}: {n_total:,} nodes '
          f'(laundering: {n_susp:,}, {n_susp*100/n_total:.2f}%)')
    print(f'[INFO] Saved {edge_path}: {len(edge_df):,} edges')


if __name__ == '__main__':
    main()
