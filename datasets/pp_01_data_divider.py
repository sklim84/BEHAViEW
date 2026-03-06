import pandas as pd
import numpy as np
import os

# HF_TRNS_TRAN HF_FND
data_name = 'HF_FND'

# 설정
input_path = f'./{data_name}.csv'
output_dir = './'
os.makedirs(output_dir, exist_ok=True)

# 데이터 로딩
df = pd.read_csv(input_path)
print(df.columns)
print(f"[INFO] {data_name} 전체 거래 건수: {len(df):,}")

# 날짜 변환 및 정렬
df['tran_dt'] = pd.to_datetime(df['tran_dt'], format='%Y%m%d', errors='coerce')
df.dropna(subset=['tran_dt'], inplace=True)

df.sort_values(by=['tran_dt', 'tran_tmrg'], ascending=True, inplace=True)

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

# 연도-월 컬럼 생성
df['year_month'] = df['tran_dt'].dt.to_period('M')
print(df['year_month'].unique())

# 월별 저장
for ym in sorted(df['year_month'].unique()):
    ym_str = str(ym).replace('-', '_')  # 예: '2024_01'
    df_month = df[df['year_month'] == ym]
    output_path = os.path.join(output_dir, f"{data_name}_{ym_str}.csv")

    df_month.drop(columns=['year_month'], inplace=True)
    df_month.to_csv(output_path, index=False)
    print(f"[INFO] {str(ym)} 저장 완료: {len(df_month):,}건 → {output_path}")
