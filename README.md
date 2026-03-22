# 피처 인식 그래프 뷰 구성을 통한 행동 동질성 복원 기반 자금세탁 탐지

> [FraudCenGCL (BigData 2025)](https://github.com/sklim84/KA-003-FraudCenGCL) 논문을 확장한 AML 탑티어 연구 프로젝트

## 개요

자금세탁(AML) 탐지를 위한 그래프 대조학습(GCL)에서 **효과적인 contrastive view를 어떻게 구성할 것인가**를 연구합니다. 노드 피처를 **행동(Behavioral)** 피처와 **구조(Structural)** 피처로 체계적으로 분류하고, 각 피처 유형으로 구축한 k-NN 유사도 그래프의 contrastive view 효과를 비교합니다.

### 핵심 발견

1. **Behavioral k-NN view로 AML 탐지 성능 +82% 향상** (f1_1: 0.323 → 0.586)
2. **Behavioral >> Structural** — 중심성/차수 피처로는 효과적인 k-NN 구축 불가
3. **Structural 피처의 보완적 가치 없음** — edge weight, augmentation 등 다양한 결합 시도 모두 gain 0
4. **동질성(homophily) 복원이 핵심 메커니즘** — Behavioral k-NN이 homophily를 0.814에서 0.981로 복원
5. **k=5~10이 최적** — 가장 유사한 소수 이웃만 연결할수록 효과적

## 연구 질문

| RQ | 질문 | 상태 |
|----|------|------|
| RQ1 | k-NN view가 augmentation 기반 view보다 AML 탐지에 효과적인가? | 완료 |
| RQ2 | Behavioral vs Structural — 어떤 피처로 k-NN을 구축해야 하는가? | 완료 |
| RQ3 | Structural 피처가 Behavioral k-NN에 보완적 가치가 있는가? | 완료 |
| RQ4 | Behavioral k-NN이 왜 효과적인가? (동질성 복원 검증) | 완료 |
| RQ5 | 다른 GCL backbone과 AML 데이터셋에서 일반화되는가? | 일부 |

---

## 피처 분류

### Category A: Behavioral 피처 (8개)

거래 행동 패턴 — 계좌가 "무엇을 하는가"

| # | 피처 | 설명 | Fraud/Benign 비율 |
|---|------|------|------------------|
| 1 | out_mean | 출금 평균 금액 | 76.5x |
| 2 | out_max | 출금 최대 금액 | 117.5x |
| 3 | out_std | 출금 금액 표준편차 | 104.1x |
| 4 | in_mean | 입금 평균 금액 | 39.2x |
| 5 | in_max | 입금 최대 금액 | 51.1x |
| 6 | in_std | 입금 금액 표준편차 | 51.3x |
| 7 | md_type_entropy | 거래 매체 다양성 (PC/인터넷/모바일 등) | 7.1x |
| 8 | fnd_type_entropy | 자금 구분 다양성 (급여/일반 등) | 28.4x |

### Category B: Structural 피처 (20개)

네트워크 위상 — 계좌가 "네트워크에서 어디에 위치하는가"

- **B1. Degree/Count (4)**: out_count, in_count, in_dc, out_dc
- **B2. Graph Centrality (16)**: dc, pagerank, hits_hub, hits_auth, katz, eigenvector, kcore, triangle, cc, clustering, avg_neigh_deg, harmonic, sq_clustering, betweenness, louvain, constraint

---

## 데이터셋

### HOFINET (전자금융공동망)

| 항목 | 값 |
|------|-----|
| 계좌 (노드) | 452,816 |
| 이체 (엣지) | 2,558,743 (유향) |
| 의심 계좌 | 9,644 (2.13%) |
| 기간 | 2021년 3분기 ~ 2024년 4분기 (40개월) |
| AML 유형 | 6가지 |

AML 유형별 분포:
- 신규 수신처 거래 (layering) — 63.9%
- 분할 거래 (structuring) — 14.3%
- 갑작스러운 거래패턴 변화 — 13.5%
- 다중거래 동시 요청 (rapid movement) — 6.4%
- 거액 입금 후 당일 인출 (placement-extraction) — 1.7%
- 심야/새벽 대량 거래 — 0.2%

---

## 실험 결과 (HOFINET, 4-seed 평균 ± 표준편차)

### RQ1: k-NN view vs augmentation 기반 view (BGRL)

| 설정 | f1_1 (의심 계좌) | auroc | auprc |
|------|-----------------|-------|-------|
| _w_org (augmentation only) | 0.323 ± 0.008 | 0.934 | 0.298 |
| **Behavioral k-NN view** | **0.586 ± 0.014** | **0.985** | **0.565** |

### RQ2: Behavioral vs Structural k-NN (BGRL)

| k-NN 그래프 | 피처 분류 | f1_1 | auroc | auprc |
|-------------|----------|------|-------|-------|
| **Behavioral** | **A (8개)** | **0.586 ± 0.014** | **0.985** | **0.565** |
| Structural | B (11개) | 0.375 ± 0.012 | 0.958 | 0.368 |

### RQ3: Structural 보완 효과 (BGRL)

| 방법 | f1_1 | Δ |
|------|------|----|
| Behavioral k-NN | 0.586 | baseline |
| + Structural 결합 (A+B) | 0.582 | -0.004 |
| + Structural edge weight | 0.582 | -0.004 |
| + Structural guided aug | 0.571 | -0.015 |

### RQ4: 동질성 복원

| 그래프 | Homophily | Fraud-Fraud 엣지 수 |
|--------|-----------|-------------------|
| Transaction (원본) | 0.814 | 41,081 |
| **Behavioral k-NN** | **0.981** | **61,987** |
| Structural k-NN | 0.964 | 15,212 |

### RQ5: 다른 GCL Backbone

| Backbone | _w_org f1_1 | Behavioral k-NN f1_1 | Δ |
|----------|------------|----------------------|---|
| **BGRL** | 0.323 | **0.586** | **+0.263** |
| DGI-trs | 0.044 | 0.049 | +0.005 |
| MVGRL | 0.045 | 0.049 | +0.004 |

### k 값 Ablation (Behavioral k-NN, BGRL)

| k | f1_1 | auroc | auprc |
|---|------|-------|-------|
| **5** | **0.594 ± 0.025** | 0.985 | **0.568** |
| 10 | 0.586 ± 0.013 | 0.985 | 0.565 |
| 20 | 0.574 ± 0.020 | 0.986 | 0.557 |
| 50 | 0.561 ± 0.018 | 0.983 | 0.536 |

---

## 빠른 시작

```bash
# 단일 모델 실행 (BGRL + behavioral k-NN view)
python models/bgrl_w_knn.py \
  --model_name bgrl_behav \
  --gpu 0 --seed 2025 \
  --node_data_name HOFINET_NODE_FEAT \
  --edge_data_name HOFINET_EDGES \
  --knn_graph HOFINET_KNN_BEHAV_k10 \
  --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 \
  --loss BarlowTwins --skip_tsne

# k-NN 그래프 구축
python datasets/build_knn_graph.py --k 5 10 20 50

# 전체 RQ 실험 배치
bash scripts/run_rq_experiments.sh

# 동질성 측정
python analysis/homophily_knn.py
```

## 프로젝트 구조

```
models/
  bgrl_w_knn.py              # BGRL + k-NN 그래프 뷰 (메인 모델)
  dgi_transductive_w_knn.py   # DGI-transductive + k-NN 뷰
  mvgrl_w_knn.py              # MVGRL + k-NN 뷰
  *_w_cen.py                  # FraudCenGCL 변형 (중심성 피처 뷰)
  *_w_org.py                  # 베이스라인 변형 (augmentation only)

datasets/
  build_knn_graph.py          # k-NN 그래프 구축
  pp_hofinet.py               # HOFINET 전처리 파이프라인
  HOFINET_KNN_BEHAV_k*.csv    # Behavioral k-NN 그래프 (k=5,10,20,50)
  HOFINET_KNN_STRUCT_k10.csv  # Structural k-NN 그래프

analysis/
  homophily_knn.py            # RQ4: 동질성 측정

scripts/
  run_rq_experiments.sh       # 전체 RQ 실험 배치 스크립트
  hp_search.py                # 하이퍼파라미터 탐색

_docs/
  FraudCenGCL_paper_fulltext.md  # 기존 BigData 2025 논문 전문
  MLGCL_paper_fulltext.md        # 관련 연구: Multi-Level GCL
  GCPAL_paper_fulltext.md        # 관련 연구: GCL for AML

config.py                     # 공통 인자 파서
data_loader.py                # 데이터 로딩
utils.py                      # 평가, 시각화
```

## 관련 연구

| 논문 | 학회 | 관계 |
|------|------|------|
| [FraudCenGCL](https://github.com/sklim84/KA-003-FraudCenGCL) | BigData 2025 | 기존 연구 (중심성을 피처 뷰로 사용) |
| [MLGCL](https://arxiv.org/abs/2107.02639) | Neurocomputing 2024 | 임베딩 k-NN을 뷰로 사용 (인용 그래프) |
| [GCPAL](https://doi.org/10.1007/s44196-024-00720-4) | IJCIS 2024 | 원시 피처 k-NN을 AML에 적용 (Elliptic) |

**차별점**: MLGCL과 GCPAL 모두 피처를 behavioral/structural로 분리하여 비교하지 않았음. 본 연구에서 **behavioral 피처만으로 동질성을 복원**하여 k-NN view의 효과를 이끈다는 것을 최초로 검증.

## TODO

- [ ] 공개 AML 데이터셋 (Elliptic) — 일반성 검증, GCPAL 직접 비교
- [ ] GBT/GRACE backbone (PyTorch 2.10 torch_sparse 호환 해결 필요)
- [ ] SOTA AML 모델 비교 (CARE-GNN, PC-GNN, H2-FDetector)
- [ ] 라벨 비율 민감도 분석 (1%, 5%, 10%, 20%)
- [ ] AML 유형별 분석

## 인용

```bibtex
@inproceedings{lim2025fraudcengcl,
  title={FraudCenGCL: Role-Aware Graph Contrastive Learning for Low-Homophily Fraud Detection},
  author={Lim, Seonkyu and Choi, Jeongwhan and Lee, Jaehoon},
  booktitle={IEEE International Conference on Big Data},
  year={2025}
}
```

## 라이선스

MIT
