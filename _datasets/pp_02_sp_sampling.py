import pandas as pd
import numpy as np
import os

def extract_anomaly_hop(df, acc_set, labels, suffix):
    df_hop = df[(df['source'].isin(acc_set)) | (df['target'].isin(acc_set))]
    df_hop.to_csv(f'{output_base}/{data_name}_{"-".join(labels)}_{suffix}.csv', index=False)
    count = len(df_hop)

    src = df_hop['source'].to_numpy()
    tar = df_hop['target'].to_numpy()
    acc = np.unique(np.concatenate([src, tar]))

    anomaly_count = df_hop['ff_sp_ai'].value_counts()
    count_01 = anomaly_count.get('01', 0)
    count_02 = anomaly_count.get('02', 0)
    count_SP = anomaly_count.get('SP', 0)

    print(f'\n###### {suffix}-hop Data Information')
    print(f'# 총 계좌수: {len(acc):,}')
    print(f'# 총 거래수: {count:,}')
    print(f'# 전체 사기거래 수: {count_01 + count_02 + count_SP:,}  ({(count_01 + count_02 + count_SP) * 100 / count:.2f} %)')
    print(f'# 01사기거래 수: {count_01:,}  ({count_01 * 100 / count:.2f} %)')
    print(f'# 02사기거래 수: {count_02:,}  ({count_02 * 100 / count:.2f} %)')
    print(f'# SP사기거래 수: {count_SP:,}  ({count_SP * 100 / count:.2f} %)')

    return acc

if __name__ == '__main__':
    data_name = 'HF_FND_2024_03'
    input_path = f'./{data_name}.csv'
    output_base = './'

    df = pd.read_csv(input_path)
    df['tran_dt'] = pd.to_datetime(df['tran_dt'], format='ISO8601')
    df = df.astype({
        'tran_amt': 'float',
        'wd_fc_sn': 'int',
        'wd_ac_sn': 'str',
        'dps_fc_sn': 'int',
        'dps_ac_sn': 'str'
    })

    df['source'] = df[['wd_fc_sn', 'wd_ac_sn']].agg(':'.join, axis=1)
    df['target'] = df[['dps_fc_sn', 'dps_ac_sn']].agg(':'.join, axis=1)
    df.to_csv(f'{output_base}/{data_name}_ST.csv', index=False)

    # 통계용 기본 변수
    count_tran = len(df)
    anomaly_count = df['ff_sp_ai'].value_counts()
    count_01 = anomaly_count.get('01', 0)
    count_02 = anomaly_count.get('02', 0)
    count_SP = anomaly_count.get('SP', 0)

    # 전체 계좌
    src_acc = df['source'].to_numpy()
    tar_acc = df['target'].to_numpy()
    total_acc = np.unique(np.concatenate([src_acc, tar_acc]))

    # Original Data 통계 출력
    print('##### Original Data Information')
    print(f'# 총 계좌수: {len(total_acc):,}')
    print(f'# 총 거래수: {count_tran:,}')
    print(f'# 전체 사기거래수: {count_01 + count_02 + count_SP:,}  ({(count_01 + count_02 + count_SP) * 100 / count_tran:.2f} %)')
    print(f'# 01사기거래수: {count_01:,}  ({count_01 * 100 / count_tran:.2f} %)')
    print(f'# 02사기거래수: {count_02:,}  ({count_02 * 100 / count_tran:.2f} %)')
    print(f'# SP사기거래수: {count_SP:,}  ({count_SP * 100 / count_tran:.2f} %)')

    # 이상 거래 레이블
    anomaly_labels = ['SP']

    # 0-hop: SP 사기거래
    df_h0 = df[df['ff_sp_ai'].isin(anomaly_labels)]
    acc_h0 = np.unique(np.concatenate([df_h0['source'].to_numpy(), df_h0['target'].to_numpy()]))
    acc_h0 = extract_anomaly_hop(df, acc_h0, anomaly_labels, 'H0')

    # 1-hop: H0에 연결된 거래
    acc_h1 = extract_anomaly_hop(df, acc_h0, anomaly_labels, 'H1')

    # 2-hop: H1에 연결된 거래
    acc_h2 = extract_anomaly_hop(df, acc_h1, anomaly_labels, 'H2')
