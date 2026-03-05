import pandas as pd

df = pd.read_csv("HF_TRNS_TRAN.csv")

# 거래일자 컬럼을 날짜 형식으로 변환 및 '월' 추출
df['tran_dt'] = pd.to_datetime(df['tran_dt'], format='%Y%m%d')
df['month'] = df['tran_dt'].dt.to_period('M')  # 월 단위

# 1. 전체 거래 건수
total_count = len(df)

# 2. ff_sp_ai 컬럼의 각 값별 건수
ff_sp_ai_counts = df['ff_sp_ai'].fillna('없음').value_counts()

# 3. 월별 거래 건수
monthly_counts = df.groupby('month').size()

# 4. 월별 ff_sp_ai 값별 건수
monthly_ff_sp_ai = df.copy()
monthly_ff_sp_ai['ff_sp_ai'] = monthly_ff_sp_ai['ff_sp_ai'].fillna('없음')
monthly_ff_sp_ai_counts = monthly_ff_sp_ai.groupby(['month', 'ff_sp_ai']).size().unstack(fill_value=0)

# 결과 출력
print(f"1. 전체 거래 건수: {total_count}\n")
print("2. ff_sp_ai 값별 건수:")
print(ff_sp_ai_counts)

print("\n3. 월별 거래 건수:")
print(monthly_counts)

print("\n4. 월별 ff_sp_ai 값별 건수:")
print(monthly_ff_sp_ai_counts)
