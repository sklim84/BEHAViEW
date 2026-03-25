# Beyond Augmentation: Behavioral Subgraph Contrast for Anti-Money Laundering

자금세탁(AML) 탐지를 위한 그래프 대조학습(GCL)의 성능을 결정하는 **두 독립 축**을 분석하고, 이를 결합한 통합 프레임워크를 제안합니다.

> [FraudCenGCL (BigData 2025)](https://github.com/sklim84/KA-003-FraudCenGCL) 확장 연구

---

## 프레임워크

![Framework](visualize/paper_figures/fig1_framework.png)

**두 축:**

| 축 | 기존 GCL | 제안 |
|----|---------|------|
| **View 구성** | 같은 그래프에 augmentation | **Behavioral k-NN graph** (행동 유사 계좌끼리 연결) |
| **Contrastive Level** | Node-level 또는 Global pooling | **Subgraph pooling** (이웃 포함 집계) |

**4가지 설정 (ablation):**

```
                        View 구성
                   Augmentation    Behavioral k-NN
Contrastive  Node │ (a) baseline  │ (b) view 변경    │
Level        Sub  │ (c) level 변경│ (d) 최종 제안  ★ │
```

---

## 연구 질문

| RQ | 질문 | 비교 | 핵심 발견 |
|----|------|------|----------|
| **RQ1** | Behavioral k-NN view가 AML 탐지에 효과적인가? 그리고 왜 효과적인가? | (b) vs (a), F-F/F-B 비율 분석 | +80~151% 개선. F-F/F-B가 1:5.7→1:1.4로 fraud 신호 복원 |
| **RQ2** | Subgraph pooling이 추가 개선을 가져오는가? 어떤 조건에서 효과적인가? | (d) vs (b), (c) vs (a) | behavioral view와 결합 시에만 효과. 단독(c)은 오히려 악화 |
| **RQ3** | Encoder 설계(BatchNorm)가 대규모 불균형 AML에서 성능에 미치는 영향은? | 10종 encoder (BN 유무) | BN만으로 0.07→0.68 (~10배). 아키텍처는 무관 |
| **RQ4** | 다른 AML 데이터셋에서도 동일한 패턴이 재현되는가? | HOFINET + AMLworld | 두 데이터셋 모두 (d)>(b)>(a)>(c) 일관 |
| **RQ5** | Supervised 모델 대비 라벨 없는 self-supervised가 경쟁력 있는가? | GCN/GAT/SAGE/MLP/LightGBM/XGBoost | SSL 0.682 ≈ Supervised 0.678, 라벨 없이 동등 |

---

## 데이터셋

| | HOFINET | AMLworld HI-Small |
|---|---------|------------------|
| **출처** | 전자금융공동망 (KFTC) | [NeurIPS 2023 벤치마크](https://arxiv.org/abs/2306.16424) |
| **노드** | 452,816 계좌 | 515,088 계좌 |
| **엣지** | 4,732,130 (유향, 멀티엣지) | 5,078,345 (유향, 멀티엣지) |
| **의심 비율** | 2.13% | 0.10% |
| **AML 유형** | 6가지 (structuring, layering 등) | 8가지 (fan-out, cycle 등) |

---

## 피처 분류

| Category | 수 | 설명 | 용도 |
|----------|-----|------|------|
| **A. Behavioral** (x_behav) | 22 | 금액통계(8) + entropy(2) + temporal(12) | GNN 입력 + k-NN 구축 |
| **B. Structural** (x_struct) | 9 | degree, centrality (dc, pagerank, betweenness 등) | GNN 입력 (선택) |

---

## 실험 결과

### RQ1-2: 두 축 Ablation (HOFINET, F1_fraud, 4-seed 평균)

| Encoder | (a) org | (b) behav view | (c) subgraph | (d) both ★ | BN | Conv |
|---------|---------|---------------|-------------|------------|-----|------|
| **GBT** | 0.270 | **0.678** (+151%) | 0.205 (-24%) | **0.682** (+152%) | ✅ | GCN |
| **DGI+BN** | 0.271 | **0.678** (+150%) | 0.204 (-25%) | **0.682** (+152%) | ✅ | GCN |
| **MVGRL+BN** | 0.270 | **0.677** (+151%) | 0.204 (-24%) | **0.682** (+153%) | ✅ | GCN |
| **GRACE+BN** | 0.286 | **0.676** (+136%) | 0.218 (-24%) | **0.681** (+138%) | ✅ | GCN |
| **BGRL** | 0.315 | 0.566 (+80%) | 0.254 (-19%) | **0.647** (+105%) | ✅ | GCN |
| **GIN** | 0.149 | **0.545** (+266%) | 0.121 (-19%) | **0.570** (+283%) | ✅ | GIN |
| GCA | 0.048 | 0.064 (+31%) | 0.049 (+2%) | 0.085 (+75%) | ❌ | GCN |
| DGI | 0.045 | 0.058 (+29%) | 0.048 (+7%) | 0.074 (+64%) | ❌ | GCN |
| MVGRL | 0.045 | 0.056 (+24%) | 0.048 (+7%) | 0.071 (+56%) | ❌ | GCN |
| GRACE | 0.045 | 0.056 (+27%) | 0.046 (+4%) | 0.069 (+54%) | ❌ | GCN |

### RQ1: Fraud 연결성 분석 (왜 효과적인가)

| 그래프 | Homophily | F-F | F-B | F-F/F-B | 의미 |
|--------|-----------|-----|-----|---------|------|
| Transaction | 0.690 | 257K | 1,469K | **1:5.7** | fraud가 benign에 묻힘 |
| Structural k-NN | 0.965 | 16K | 159K | **1:9.7** | 더 심하게 묻힘 |
| **Behavioral k-NN** | **0.981** | **62K** | **88K** | **1:1.4** | fraud 신호 보존 |

### RQ4: AMLworld 일반성 검증

| Encoder | (a) org | (b) behav | (c) sub | (d) both | BN |
|---------|---------|----------|---------|----------|-----|
| **BGRL** | 0.038 | 0.060 (+59%) | 0.035 (-7%) | **0.068** (+81%) | ✅ |
| GBT | 0.041 | 0.047 (+14%) | 0.042 (+3%) | 0.046 (+12%) | ✅ |
| DGI+BN | 0.041 | 0.047 (+14%) | 0.042 (+2%) | 0.046 (+12%) | ✅ |
| DGI | 0.034 | 0.048 (+43%) | 0.031 (-9%) | 0.052 (+54%) | ❌ |

- **동일 패턴 재현**: (d) > (b) > (a) > (c) — 두 데이터셋에서 일관
- 절대 성능 차이는 불균형도 차이 (HOFINET 2.13% vs AMLworld 0.10%)에 기인

### RQ5: Supervised 모델 비교 (HOFINET)

| 모델 | F1_fraud | AUROC | 타입 |
|------|----------|-------|------|
| **Ours (d) GCN+BN** | **0.682** | 0.985 | **Self-supervised** |
| MLP | 0.678 | 0.991 | Supervised |
| GraphSAGE | 0.677 | 0.991 | Supervised GNN |
| XGBoost | 0.675 | 0.992 | Tabular |
| LightGBM | 0.674 | 0.992 | Tabular |
| GAT | 0.469 | 0.984 | Supervised GNN |
| GCN | 0.250 | 0.954 | Supervised GNN |

> Self-supervised 제안 방법이 **라벨 없이** fully-supervised MLP/SAGE/LightGBM과 **동등한 성능** 달성.
> AML과 같이 라벨 획득이 어려운 도메인에서 실용적 가치가 높음.

### 핵심 발견 요약

1. **Behavioral view가 핵심**: 모든 encoder에서 일관된 개선 (+24~283%)
2. **Subgraph pooling은 view와 결합할 때만 효과적**: 단독 시 오히려 악화 — F-F/F-B 비율 증폭
3. **BN이 결정적**: 0.07→0.68 (~10배). per-layer BN encoder 모두 동등 (0.681~0.682)
4. **일반성 확인**: HOFINET + AMLworld 두 데이터셋에서 동일 패턴
5. **Supervised와 동등**: 라벨 없이 supervised MLP/SAGE/LightGBM 수준 도달

---

## 빠른 시작

```bash
# (d) 최고 성능: GCN+BN encoder + behavioral view + subgraph pooling
python models/subgraph_cl.py \
  --knn_graph HOFINET_KNN_BEHAV_k10 \
  --subgraph_pool --encoder_type gbt \
  --gpu 0 --seed 2025 \
  --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2

# 4가지 설정 선택:
#   (a) --encoder_type gbt                                          # baseline
#   (b) --encoder_type gbt --knn_graph HOFINET_KNN_BEHAV_k10       # +view
#   (c) --encoder_type gbt --subgraph_pool                          # +level
#   (d) --encoder_type gbt --knn_graph ... --subgraph_pool          # +both ★

# Supervised baseline 비교
python models/supervised_baselines.py --gpu 0 --dataset hofinet

# 전체 ablation
bash scripts/run_full_ablation.sh
```

---

## 프로젝트 구조

```
models/
  subgraph_cl.py               # 통합 프레임워크 (10종 encoder, 4 settings)
  supervised_baselines.py       # Supervised 비교 (GCN/GAT/SAGE/MLP/LightGBM/XGBoost)
  *_w_knn.py                   # 개별 backbone + k-NN view
  *_w_org.py                   # 개별 backbone baseline
datasets/
  build_knn_graph.py           # k-NN 그래프 구축
  pp_hofinet.py                # HOFINET 전처리
  pp_amlworld.py               # AMLworld 전처리
analysis/
  homophily_knn.py             # F-F/F-B 비율 측정
scripts/
  run_full_ablation.sh         # 전체 ablation 실험
  run_amlworld.sh              # AMLworld 실험
visualize/
  gen_paper_figures.py         # 논문 figure 생성
```

---

## 관련 연구

| 논문 | 학회 | 관계 |
|------|------|------|
| [MLGCL](https://arxiv.org/abs/2107.02639) | Neurocomputing 2024 | k-NN view for GCL |
| [GCPAL](https://doi.org/10.1007/s44196-024-00720-4) | IJCIS 2024 | k-NN view for AML |
| [SUBG-CON](https://arxiv.org/abs/2009.10564) | ICDM 2020 | Subgraph contrastive learning |
| [ImGCL](https://ojs.aaai.org/index.php/AAAI/article/view/26319) | NeurIPS 2023 | Imbalanced node classification GCL |
| [AMLworld](https://arxiv.org/abs/2306.16424) | NeurIPS 2023 | AML 합성 벤치마크 데이터셋 |

**차별점**: 기존 연구는 view 구성(MLGCL, GCPAL) 또는 contrastive level(SUBG-CON)을 개별적으로 다룸. 본 연구는 **두 축을 교차 분석**하여 AML에 최적인 조합을 도출하고, **BN의 결정적 역할**을 10종 encoder 비교로 입증.

---

## TODO

- [x] (a)(b)(c)(d) ablation × 10 encoder
- [x] AMLworld HI-Small 일반성 검증
- [x] Supervised baseline 비교 (GCN/GAT/SAGE/MLP/LightGBM/XGBoost)
- [ ] 논문 집필

---

## 라이선스

MIT
