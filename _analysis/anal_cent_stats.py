import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

# CSV 로딩
# HF_FND_2024_03_SAMPLE_12K_NODE_FEAT HF_FND_2024_03_SAMPLE_100K_NODE_FEAT HF_FND_2024_03_SP_H1_NODE_FEAT
data_name = 'HF_FND_2024_03_SP_H1_NODE_FEAT'
df = pd.read_csv(f'../_datasets/{data_name}.csv')
save_dir = './'
os.makedirs(save_dir, exist_ok=True)

centrality_cols = ['dc', 'cc', 'bc']

# 📈 1. 통계 요약
summary = df[centrality_cols].describe().T
summary.to_csv(os.path.join(save_dir, 'centrality_summary.csv'))
print(summary)

# Seaborn 스타일 설정
sns.set(style="whitegrid", context="notebook")

# 반복문: 각 중심성 컬럼에 대해 시각화 수행
for col in centrality_cols:
    print(f'##### label 0 {col} 통계정보')
    print(df[df['label'] == 0][col].describe())  # 값 분포 확인
    print(f'##### label 1 {col} 통계정보')
    print(df[df['label'] == 1][col].describe())  # 값 분포 확인

    plt.figure(figsize=(5, 3))
    flierprops = dict(markeredgewidth=0.3)
    # palette = sns.color_palette("pastel")
    palette = {'0': '#A1C9F4', '1': '#F4A1A1'}
    sns.boxplot(x='label', y=col,
                data=df,
                flierprops=flierprops,
                width=0.5,
                linewidth=0.5,
                fliersize=5,
                palette=palette,
                medianprops=dict(linewidth=1.2),
                showmeans=True)

    plt.yscale('log')
    plt.ylabel('')
    plt.xlabel('')
    plt.xticks([0, 1], ['Benign', 'Fraud'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{data_name}_{col}_boxplot_log.png'), dpi=300)
    plt.show()

    # sns.kdeplot(df[df['label'] == 0][col], color='blue', label='Benign', fill=True, alpha=0.5, log_scale=True,
    #             warn_singular=False)
    # sns.kdeplot(df[df['label'] == 1][col], color='red', label='Fraud', fill=True, alpha=0.5, log_scale=True,
    #             warn_singular=False)

    # sns.histplot(data=df[df['label'] == 0], x=col, label='Benign', color='blue', alpha=0.5, stat='density', element='step',
    #              fill=True, bins=1000, log_scale=True)
    # sns.histplot(data=df[df['label'] == 1], x=col, label='Fraud', color='red', alpha=0.5, stat='density', element='step',
    #              fill=True, bins=1000, log_scale=True)

    # plt.xlabel(col)
    # plt.legend(title='Label')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, f'{data_name}_{col}_kde_log.png'))
    # plt.show()

# # 📊 2. 전체 분포 (log scale) 저장 및 출력
# for col in centrality_cols:
#     plt.figure(figsize=(8, 4))
#     sns.histplot(df[col], kde=True, bins=100)
#     plt.title(f'Distribution of {col} (log-scale)')
#     plt.xlabel(col)
#     plt.xscale('log')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f'{data_name}_{col}_hist_log.png'))
#     plt.show()
#
#
# # 📊 3. Fraud vs Normal boxplot 저장 및 출력
# for col in centrality_cols:
#     plt.figure(figsize=(8, 4))
#     sns.boxplot(x='label', y=col, data=df)
#     plt.title(f'{col} by Fraud Label (0:Normal, 1:Fraud)')
#     plt.yscale('log')
#     plt.xlabel('Label')
#     plt.ylabel(col)
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, f'{data_name}_{col}_boxplot_log.png'))
#     plt.show()
