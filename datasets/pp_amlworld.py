"""
AMLworld HI-Small 전처리 파이프라인.
Transaction CSV → Node features + Edge list (HOFINET과 동일 형식)

사용법:
    python datasets/pp_amlworld.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import entropy

def compute_entropy_feat(group):
    counts = group.value_counts(normalize=True)
    return entropy(counts)

def main():
    input_path = 'datasets/amlworld/HI-Small_Trans.csv'
    output_dir = 'datasets/amlworld'

    print(f'[INFO] Loading {input_path}...')
    df = pd.read_csv(input_path)
    print(f'[INFO] Loaded {len(df):,} transactions')

    # Column mapping — AMLworld has duplicate 'Account' columns
    df.columns = ['Timestamp', 'From_Bank', 'From_Account', 'To_Bank', 'To_Account',
                   'Amount_Received', 'Receiving_Currency', 'Amount_Paid',
                   'Payment_Currency', 'Payment_Format', 'Is_Laundering']

    # Node ID = Bank:Account
    df['source'] = df['From_Bank'].astype(str) + ':' + df['From_Account'].astype(str)
    df['target'] = df['To_Bank'].astype(str) + ':' + df['To_Account'].astype(str)
    df['tran_amt'] = df['Amount_Paid'].fillna(0)

    print(f'[INFO] Transactions: {len(df):,}')
    print(f'[INFO] Laundering: {df["Is_Laundering"].sum():,} ({df["Is_Laundering"].mean()*100:.2f}%)')

    # === Node Features ===
    print('[INFO] Building node features...')

    # A1. Transaction amount statistics
    features_out = df.groupby('source')['tran_amt'].agg(['mean', 'max', 'std', 'count']).add_prefix('out_')
    features_in = df.groupby('target')['tran_amt'].agg(['mean', 'max', 'std', 'count']).add_prefix('in_')
    node_features = pd.concat([features_out, features_in], axis=1).fillna(0)

    # A2. Diversity entropy
    payment_entropy = df.groupby('source')['Payment_Format'].apply(compute_entropy_feat).rename('payment_entropy')
    currency_entropy = df.groupby('source')['Payment_Currency'].apply(compute_entropy_feat).rename('currency_entropy')
    node_features = node_features.join(payment_entropy, how='left').join(currency_entropy, how='left')
    node_features.fillna(0, inplace=True)

    # A3. Temporal features (3/6/12 month windows based on timestamp)
    df['tran_date'] = pd.to_datetime(df['Timestamp'])
    max_date = df['tran_date'].max()
    # AMLworld is ~3 months, use 1/2/3 month windows instead
    for months in [1, 2, 3]:
        cutoff = max_date - pd.DateOffset(months=months)
        df_win = df[df['tran_date'] >= cutoff]
        win_out = df_win.groupby('source')['tran_amt'].agg(['mean', 'count']).add_prefix(f'out_{months}m_')
        win_in = df_win.groupby('target')['tran_amt'].agg(['mean', 'count']).add_prefix(f'in_{months}m_')
        node_features = node_features.join(win_out, how='left').join(win_in, how='left')
    node_features.fillna(0, inplace=True)
    df.drop(columns=['tran_date'], inplace=True)

    # Label: account is laundering if involved in any laundering transaction
    src_label = df.groupby('source')['Is_Laundering'].max()
    tgt_label = df.groupby('target')['Is_Laundering'].max()
    labels = src_label.combine(tgt_label, func=lambda s, t: max(s, t)).fillna(
        src_label).fillna(tgt_label).fillna(0).astype(int)
    node_features['label'] = labels

    node_features['account'] = node_features.index

    # === Edge List (multi-edge preserved) ===
    edge_df = df[['source', 'target']]

    # === Save ===
    os.makedirs(output_dir, exist_ok=True)
    node_path = os.path.join(output_dir, 'AMLWORLD_NODE_FEAT.csv')
    edge_path = os.path.join(output_dir, 'AMLWORLD_EDGES.csv')

    node_features.reset_index(drop=True).to_csv(node_path, index=False)
    edge_df.to_csv(edge_path, index=False)

    n_fraud = (node_features['label'] == 1).sum()
    n_total = len(node_features)
    print(f'[INFO] Saved {node_path}: {n_total:,} nodes (laundering: {n_fraud:,}, {n_fraud*100/n_total:.2f}%)')
    print(f'[INFO] Saved {edge_path}: {len(edge_df):,} edges')

    # Feature summary
    behav_cols = [c for c in node_features.columns if c not in ('account', 'label')]
    fraud = node_features[node_features['label'] == 1]
    benign = node_features[node_features['label'] == 0]
    print(f'\n[INFO] Feature statistics (fraud/benign ratio):')
    for c in behav_cols:
        fm, bm = fraud[c].mean(), benign[c].mean()
        ratio = fm / bm if bm > 0 else float('inf')
        print(f'  {c:20s}: fraud={fm:.4f}, benign={bm:.4f}, ratio={ratio:.2f}x')

if __name__ == '__main__':
    main()
