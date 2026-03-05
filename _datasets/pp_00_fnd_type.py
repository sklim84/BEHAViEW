import os
import pandas as pd

# HF_TRNS_TRAN CD_TRNS_TRAN
data_name = 'HF_TRNS_TRAN'

# 설정
input_path = f'./{data_name}.csv'
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

# 데이터 로딩
df = pd.read_csv(input_path)
print(df.columns)
print(f"[INFO] {data_name} 전체 거래 건수: {len(df):,}")

# 데이터 타입 정리
df = df.astype({
    'tran_tmrg': 'int',
    'wd_fc_sn': 'int',
    'wd_ac_sn': 'str',
    'dps_fc_sn': 'int',
    'dps_ac_sn': 'str',
    'tran_amt': 'float',
    'md_type': 'int',
    'fnd_type': 'int'
}, errors='ignore')

# 제외할 fnd_type
exclude_types = [2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30]

# fnd_type이 제외 리스트에 포함되지 않은 행만 선택
filtered_df = df[~df['fnd_type'].isin(exclude_types)]
filtered_df.to_csv('HF_FND.csv', index=False)
print(f"[INFO] FND_TYPE 필터링 후 HF_FND 전체 거래 건수: {len(filtered_df):,}")
