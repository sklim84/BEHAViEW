import pandas as pd

# HF_TRNS_TRAN
# HF_TRNS_TRAN_2024_03
# HF_TRNS_TRAN_2024_03_SP_H1
data_name = 'HF_TRNS_TRAN_2024_03_SP_H1'
input_path = f'../_datasets/{data_name}.csv'

# 데이터 로딩
df = pd.read_csv(input_path)

# NaN 포함한 상태로 ff_sp_ai 집계 (NaN은 문자열로 대체해도 됨)
df['ff_sp_ai'] = df['ff_sp_ai'].fillna('NaN')

# fnd_type별 ff_sp_ai의 빈도수 계산
# fnd_type_stats = df.groupby('fnd_type')['ff_sp_ai'].value_counts().unstack(fill_value=0)
# print("fnd_type별 ff_sp_ai 통계:")
# print(fnd_type_stats)

# md_type별 ff_sp_ai의 빈도수 계산
md_type_stats = df.groupby('md_type')['ff_sp_ai'].value_counts().unstack(fill_value=0)
print("md_type별 ff_sp_ai 통계:")
print(md_type_stats)
