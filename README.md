# BECON: Behavioral Subgraph Contrast for Anti-Money Laundering in Low-Homophily Transaction Networks

자금세탁(AML) 탐지를 위한 그래프 대조학습(GCL) 프레임워크. GCL의 성능을 결정하는 **두 독립 축**(View Construction × Contrastive Level)을 체계적으로 분석하고, 최적 조합을 도출합니다.

---

## 프레임워크

![Framework](_paper/figures/fig_framework.png)

**두 축:**

| 축 | 기존 GCL | BECON |
|----|---------|-------|
| **View 구성** | 동일 그래프에 확률적 증강 | **Behavioral k-NN graph** (행동적 유사 계좌끼리 연결) |
| **Contrastive Level** | Node-level 대조 | **Subgraph pooling** (이웃 포함 집계) |

**4가지 설정 (ablation):**

```
                        View 구성
                   Augmentation    Behavioral k-NN
Contrastive  Node │ (a) baseline  │ (b) view 변경    │
Level        Sub  │ (c) level 변경│ (d) 최종 제안  ★ │
```

---

## 연구 질문

| RQ | 질문 | 핵심 발견 |
|----|------|----------|
| **RQ1** | 행동적 k-NN view가 왜 효과적인가? | S-S/S-B가 1:5.7→1:1.4로 개선, +80~151% F1_susp 향상 |
| **RQ2** | 서브그래프 풀링이 언제 효과적인가? | 행동적 view와 결합 시에만 효과. 단독(c)은 오히려 악화 |
| **RQ3** | BatchNorm이 성능에 미치는 영향은? | BN만으로 0.07→0.68 (~10배). 인코더 아키텍처는 부차적 |
| **RQ4** | 다른 AML 데이터셋에서도 재현되는가? | HOFINET + AMLworld 모두 (d)>(b)>(a)>(c) 일관 |
| **RQ5** | 지도학습 대비 경쟁력이 있는가? | SSL 0.682 ≈ Supervised 0.678, 라벨 없이 동등 |

---

## 데이터셋

| | HOFINET | AMLworld HI-Small |
|---|---------|------------------|
| **출처** | 전자금융공동망 (KFTC) | [NeurIPS 2023 벤치마크](https://arxiv.org/abs/2306.16424) |
| **노드** | 452,816 계좌 | 515,088 계좌 |
| **엣지** | 4,732,130 (유향, 멀티엣지) | 5,078,345 (유향, 멀티엣지) |
| **의심 비율** | 2.13% | 1.23% (계좌), 0.10% (거래) |
| **AML 유형** | 6가지 (structuring, layering 등) | 8가지 (fan-out, cycle 등) |

---

## 실험 결과

### Suspicious 연결성 분석

| 그래프 | Homophily | S-S | S-B | S-S/S-B |
|--------|-----------|-----|-----|---------|
| Transaction | 0.690 | 257K | 1,469K | 1:5.7 |
| Structural k-NN | 0.965 | 16K | 159K | 1:9.7 |
| **Behavioral k-NN** | **0.981** | **62K** | **88K** | **1:1.4** |

### 두 축 Ablation (HOFINET, F1_susp, 4-seed 평균)

| Encoder | (a) org | (b) behav view | (c) subgraph | (d) both ★ | BN |
|---------|---------|---------------|-------------|------------|-----|
| GBT | 0.270 | **0.678** | 0.205 | **0.682** | Per-layer |
| DGI+BN | 0.271 | **0.678** | 0.204 | **0.682** | Per-layer |
| MVGRL+BN | 0.270 | 0.677 | 0.204 | **0.682** | Per-layer |
| GRACE+BN | 0.286 | 0.676 | 0.218 | **0.681** | Per-layer |
| BGRL | 0.315 | 0.566 | 0.254 | **0.647** | Final |
| GIN | 0.149 | 0.545 | 0.121 | **0.570** | GINConv |
| GCA | 0.048 | 0.064 | 0.049 | 0.085 | None |
| DGI | 0.045 | 0.058 | 0.048 | 0.074 | None |
| MVGRL | 0.045 | 0.056 | 0.048 | 0.071 | None |
| GRACE | 0.045 | 0.056 | 0.046 | 0.069 | None |

### Supervised 모델 비교 (HOFINET)

| 모델 | F1_susp | AUROC | 타입 |
|------|---------|-------|------|
| **BECON (d)** | **0.682** | 0.985 | Self-supervised |
| MLP | 0.678 | 0.991 | Supervised |
| GraphSAGE | 0.677 | 0.991 | Supervised GNN |
| XGBoost | 0.675 | 0.992 | Tabular |
| LightGBM | 0.674 | 0.992 | Tabular |
| GAT | 0.469 | 0.984 | Supervised GNN |
| GCN | 0.250 | 0.954 | Supervised GNN |

### 핵심 발견

1. **Behavioral view가 핵심**: 모든 encoder에서 일관된 개선, S-S/S-B 비율 4배 개선
2. **Subgraph pooling은 view와 결합 시에만 효과적**: 단독 적용 시 noise 증폭으로 악화
3. **BatchNorm이 결정적**: ~10배 성능 격차, per-layer BN encoder 4종 모두 0.681~0.682 수렴
4. **일반성 확인**: HOFINET + AMLworld 두 데이터셋에서 동일 (d)>(b)>(a)>(c) 패턴
5. **라벨 없이 Supervised와 동등**: Self-supervised 0.682 ≈ MLP 0.678

---

## 빠른 시작

```bash
# (d) 최고 성능: GCN+BN encoder + behavioral view + subgraph pooling
python models/subgraph_cl.py \
  --encoder_type gbt \
  --knn_graph HOFINET_KNN_BEHAV_k10 \
  --subgraph_pool \
  --gpu 0 --seed 2025 \
  --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2

# 4가지 설정 선택:
#   (a) --encoder_type gbt                                          # baseline
#   (b) --encoder_type gbt --knn_graph HOFINET_KNN_BEHAV_k10       # +view
#   (c) --encoder_type gbt --subgraph_pool                          # +level
#   (d) --encoder_type gbt --knn_graph ... --subgraph_pool          # +both ★

# Supervised baseline 비교
python models/supervised_baselines.py --gpu 0 --dataset hofinet

# 전체 ablation (HOFINET / AMLworld)
bash scripts/run_ablation_abcd.sh
bash scripts/run_amlworld.sh
```

---

## 프로젝트 구조

```
_paper/                          # 논문 소스 (LaTeX)
models/
  subgraph_cl.py                 # 통합 프레임워크 (10종 encoder, 4 settings)
  supervised_baselines.py        # Supervised 비교 (6종)
datasets/
  build_knn_graph.py             # k-NN 그래프 구축
  pp_hofinet.py                  # HOFINET 전처리
  pp_amlworld.py                 # AMLworld 전처리
analysis/
  homophily_knn.py               # S-S/S-B 비율 측정
scripts/
  run_ablation_abcd.sh           # HOFINET ablation 실험
  run_amlworld.sh                # AMLworld 실험
visualize/
  gen_paper_figures.py           # 논문 figure 생성
  gen_intro_variants.py          # Intro figure 변형
  gen_framework_svg.py           # Framework SVG figure
config.py                       # 공통 argparse
data_loader.py                   # 데이터 로딩
utils.py                         # 공통 유틸리티
```

---

## 관련 연구

| 논문 | 학회 | 관계 |
|------|------|------|
| [MLGCL](https://arxiv.org/abs/2107.02639) | Neurocomputing 2023 | k-NN view for GCL |
| [GCPAL](https://doi.org/10.1007/s44196-024-00720-4) | IJCIS 2024 | k-NN view for AML |
| [SUBG-CON](https://arxiv.org/abs/2009.10564) | ICDM 2020 | Subgraph contrastive learning |
| [AMLworld](https://arxiv.org/abs/2306.16424) | NeurIPS 2023 | AML 합성 벤치마크 |

---

## 라이선스

MIT
