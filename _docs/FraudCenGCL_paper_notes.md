# FraudCenGCL: Role-Aware Graph Contrastive Learning for Low-Homophily Fraud Detection

**Venue**: BigData 2025
**Authors**: Seonkyu Lim (KFTC), Jeongwhan Choi (Yonsei Univ.), Jaehoon Lee (LG AI Research)

---

## I. INTRODUCTION

### Problem
- 금융 사기는 개별 거래가 아닌 계좌 네트워크의 **조정된 패턴**으로 나타남
- 기존 rule-based/ML 모델은 거래를 독립 이벤트로 처리 → 위치적/관계적 맥락 무시
- GNN은 거래 네트워크 분석에 유망하나, 노드의 다양한 구조적 역할 표현에 한계
- 금융 거래 네트워크는 **low homophily** (이상 계좌가 정상 계좌와 더 많이 연결) → GNN의 이웃 유사성 가정 약화

### Solution: FraudCenGCL
- 계좌 행동 피처 + 그래프 중심성(dc, cc, bc)을 **대조 학습의 듀얼 뷰**로 활용
- 기존 GCL 백본의 아키텍처 변경 없이 적용 가능
- HOFINET 실데이터에서 6개 백본 모두 성능 향상

### Contributions
1. 중심성 기반 구조 피처를 행동 피처와 통합하는 대조 학습 프레임워크 제안
2. 다양한 중심성 지표(dc, cc, bc)를 여러 GCL 백본(BGRL, GRACE, GBT, DGI, MVGRL)에 적용 가능
3. 실제 은행간 이체 데이터로 일관된 성능 향상 입증

---

## II. RELATED WORK

### GNN for Fraud Detection
- GCN, GAT, GraphSAGE 기반 사기 탐지 모델 (CARE-GNN, PC-GNN, STA-GNN, FraudRE 등)
- 위장 행동, 시간 패턴 포착에 초점

### Contrastive Learning
- SimCLR (augmentation), MoCo (memory-based negative sampling), SupCon (supervised+contrastive)
- PIRL, BYOL (negative sample 없이 학습)

### Graph Contrastive Learning
- DGI: global-local consistency
- InfoGraph: whole-graph ↔ subgraph
- MVGRL: multi-view
- GRACE: feature perturbation
- GraphCL: structural augmentation
- BGRL: bootstrapping without negatives
- GBT: Barlow Twins로 redundancy 감소

**Gap**: 기존 GCL은 노드 피처 또는 글로벌 구조에 집중, 개별 노드의 **구조적 역할** 간과

---

## III. PROPOSED METHOD

### A. Framework 구성요소 (4단계)

#### 1) Graph Construction
- 노드 = 고유 계좌 (은행코드:계좌번호)
- 엣지 = 출금→입금 방향 이체
- Multi-edge directed graph

#### 2) Graph Node Features (Dual-Source)

**Behavioral Features (x_agg)**:
- 계좌별 거래 집계: 빈도, 평균 금액, 분산
- 개별 활동 패턴 설명

**Structural Features (x_cen)**:
- Degree centrality: 연결 수
- Closeness centrality: 도달 효율성
- Betweenness centrality: 중개 역할

**Projection**: 별도 projection layer로 두 피처를 공유 잠재 공간에 매핑
- proj_agg: behavioral → input_dim
- proj_cen: centrality → input_dim

#### 3) GNN Encoder
- 두 뷰 각각 인코딩 → 임베딩 생성
- 그래프 구조를 통해 이웃 관계 학습
- 결과 임베딩을 concatenate하여 통합 표현 생성

#### 4) Loss Functions and Training
- 같은 계좌의 두 뷰 임베딩: 유사하게
- 다른 계좌의 임베딩: 멀리
- 학습된 임베딩 → downstream classification (LogisticRegression)

### B. Graph Centrality 정의

G = (V, E), directed graph

**Degree Centrality**:
```
DC(v) = Σ_{u∈V} [1_{(u,v)∈E} + 1_{(v,u)∈E}]
```

**Closeness Centrality**:
```
CC(v) = Σ_{u∈V} [1/d(u,v) + 1/d(v,u)]
```

**Betweenness Centrality**:
```
BC(v) = Σ_{u1,u2∈V} 1_{v∈p(u1,u2)}
```

---

## IV. EXPERIMENTS

### A. Experimental Setting

**환경**: Python 3.8, PyTorch 1.12, CUDA 11.4, NetworkX 2.8.4, NVIDIA Tesla T4

#### 1) Dataset: HOFINET (논문 사용 데이터)

| Field | Description |
|-------|-------------|
| Transaction Date | 거래일자 |
| Transaction Time | 거래시간대 |
| Amount | 거래금액 |
| Media Type | 매체구분 (e.g., Mobile) |
| Fund Type | 자금구분 (e.g., salary) |
| Withdrawal Bank Code | 출금금융회사일련번호 |
| Withdrawal Account Number | 출금계좌일련번호 |
| Deposit Bank Code | 입금금융회사일련번호 |
| Deposit Account Number | 입금계좌일련번호 |
| Suspicious Indicator | 이상거래여부 |

**논문에서 사용한 데이터 범위**:
- Time Range: **2024년 3월**만
- Accounts: 30,106
- Transfers: 145,023
- Suspicious: 253 (0.1745%)

**Node Features 구성**:
- Behavioral: mean, max, std, count (거래금액), entropy (fund_type, media_type)
- Structural: dc, cc, bc
- Label: 출금/입금 중 하나라도 suspicious이면 fraud

#### 2) Homophily Analysis
- 호모필리 비율 ϕ = |{(u,v)∈E : y_u = y_v}| / |E|
- Benign: ϕ ≈ 0.99 (정상끼리 연결)
- Fraud: ϕ ≈ 0.41 (이상 계좌는 정상과 더 많이 연결 → low homophily)

#### 3) Baselines (6개 GCL 백본)
1. **BGRL**: bootstrapping, no negative samples
2. **DGI-TRS**: mutual information, transductive
3. **DGI-IND**: mutual information, inductive with sampling
4. **GBT**: Barlow Twins, redundancy reduction
5. **GRACE**: dual-view, feature masking + edge dropping
6. **MVGRL**: multi-view, node+graph level contrast

#### 4) Evaluation Metrics (fraud class 기준)
- Precision, Recall, F1-score (fraud class only)
- AUROC, AUPRC

### B. RQ1: Performance Comparison

| Model | Precision | Recall | F1 | AUROC | AUPRC |
|-------|-----------|--------|----|-------|-------|
| BGRL | 0.1177 | 0.4211 | 0.1839 | 0.7514 | 0.0852 |
| + FraudCenGCL | **0.3214** | **0.6207** | **0.4235** | **0.8533** | **0.3096** |
| Improv. | 173% | 47% | 130% | 14% | 263% |
| DGI-IND | 0.1453 | 0.7083 | 0.2411 | 0.8562 | 0.3217 |
| + FraudCenGCL | **0.3177** | **0.8182** | **0.4576** | **0.9144** | **0.5986** |
| Improv. | 119% | 16% | 90% | 7% | 86% |
| DGI-TRS | 0.0382 | 0.7143 | 0.0725 | 0.7427 | 0.1770 |
| + FraudCenGCL | **0.1849** | **0.7857** | **0.2993** | **0.8738** | **0.4475** |
| Improv. | 384% | 10% | 313% | 18% | 153% |
| GBT | 0.0552 | 0.7083 | 0.1024 | 0.8690 | 0.2352 |
| + FraudCenGCL | **0.4314** | **0.7333** | **0.5432** | **0.9293** | **0.5084** |
| Improv. | 682% | 4% | 430% | 7% | 116% |
| GRACE | 0.0178 | 0.7917 | 0.0349 | 0.7410 | 0.0696 |
| + FraudCenGCL | **0.1539** | **0.9565** | **0.2651** | **0.9751** | **0.3201** |
| Improv. | 763% | 21% | 660% | 32% | 360% |
| MVGRL | 0.0800 | 0.3077 | 0.1270 | 0.6655 | 0.1223 |
| + FraudCenGCL | **0.1774** | **0.8800** | **0.2953** | **0.9458** | **0.3230** |
| Improv. | 122% | 186% | 133% | 42% | 164% |

**핵심 발견**:
- F1 향상: 89.79% (DGI-IND) ~ 660.36% (GRACE)
- 경량 백본(GRACE, GBT)에서 상대적 향상 가장 큼
- DGI-IND처럼 강한 baseline도 의미있는 향상

### C. RQ2: Account Centrality Analysis

**Degree Centrality**:
- 정상: long-tailed, 극단적 허브 존재 (max ~0.41)
- 이상: 극단 허브 없음 (max ~0.0033), 하지만 평균/중앙값 더 높음
- → 이상 계좌는 중간 수준 연결성에 분포

**Closeness Centrality**:
- 이상 계좌가 평균, 상위 사분위 더 높음
- → 전략적으로 네트워크에 효율적으로 접근할 수 있는 위치

**Betweenness Centrality**:
- 가장 뚜렷한 차이
- 정상: 거의 모두 0 (mean ~3.2×10⁻¹¹)
- 이상: 넓은 분포, 훨씬 큰 값 (mean ~6.1×10⁻⁸)
- → 일부 이상 노드가 거래 흐름의 중개/브리지 역할

**결론**: 이상 계좌는 극단적 허브 지위를 피하면서 중간 수준 연결성, 높은 도달성, 브리징 역할 활용

### D. RQ3: Impact of Centrality Measures (BGRL backbone)

| Centralities | Precision | Recall | F1 | AUROC | AUPRC |
|-------------|-----------|--------|----|-------|-------|
| Baseline | 0.1177 | 0.4211 | 0.1839 | 0.7514 | 0.0852 |
| **DC+CC+BC** | **0.3214** | 0.6207 | **0.4235** | 0.8533 | 0.3096 |
| DC | 0.2754 | 0.6786 | 0.3918 | **0.9430** | **0.3331** |
| CC | 0.2421 | 0.7419 | 0.3651 | 0.9172 | 0.3171 |
| BC | 0.2447 | 0.7419 | 0.3680 | 0.9171 | 0.3177 |
| DC+CC | 0.2453 | 0.5200 | 0.3333 | 0.8611 | 0.2433 |
| DC+BC | 0.2623 | 0.5926 | 0.3636 | 0.9168 | 0.2008 |
| CC+BC | 0.2391 | 0.4400 | 0.3099 | 0.8802 | 0.2788 |

**핵심 발견**:
- DC+CC+BC 전체 사용 시 F1 최고 (0.4235), Precision 최고 (0.3214)
- DC 단독: AUROC 최고 (0.9430), AUPRC 최고 (0.3331) → 랭킹 품질 우수
- CC, BC 단독: Recall 최고 (0.7419) → 민감도 우수
- 2개 조합은 불안정: DC+CC, DC+BC → F1 하락
- **3개 모두 통합이 가장 robust하고 균형잡힌 표현**

### E. RQ4: GNN Architecture Analysis (BGRL backbone)

- Shallow GNN (1~4 layers) + hidden_dim 128~512이 최적
- hidden_dim 128 근처에서 F1, AUPRC 일관적으로 높음
- hidden_dim 512 + 2 layers → Recall 최대화하나 Precision 감소
- 깊은 네트워크는 oversmoothing 문제

### F. RQ5: Embedding Space (t-SNE)

- FraudCenGCL 적용 후 모든 백본에서 fraud 노드 클러스터의 compactness와 separability 향상
- GBT, GRACE에서 가장 뚜렷한 개선
- DGI-IND, MVGRL처럼 이미 분리가 좋은 경우에도 클러스터 경계 개선

---

## V. CONCLUSION AND FUTURE WORK

### 결론
- FraudCenGCL은 행동 피처 + 중심성 구조 역할을 통합한 GCL 프레임워크
- HOFINET 실데이터에서 6개 GCL 백본 모두 일관된 성능 향상
- 이상 계좌는 중간 연결성, 높은 도달성, 브리징 역할로 구조적 차이를 보임
- 3개 중심성 전체 사용이 가장 robust

### 한계 및 향후 과제 (저널 확장 방향)
1. **거래 레벨 직접 모델링**: 현재 계좌 레벨 레이블링 → 거래 수준 suspicious 모델링
2. **시간적 역학 통합 (Temporal Dynamics)**: 시간에 따른 패턴 변화 반영
3. **기관 간 그래프 설정 (Cross-Institutional)**: 여러 금융기관에 걸친 그래프 탐색

---

## 논문 vs 현재 코드 비교

| 항목 | 논문 | 현재 프로젝트 |
|------|------|-------------|
| 데이터 범위 | 2024년 3월만 (30K 계좌, 145K 거래) | 전체 기간 2021.09~2024.12 (452K 계좌, 4.73M 거래) |
| 이상거래 수 | 253건 (0.17%) | 14,490건 (0.31%) |
| 이상거래 레이블 | Suspicious Indicator (구체적 형태 미명시) | 이상거래여부 (0/1 binary) |
| 이상거래 유형 | 미사용 | 이상거래유형 (1~7), 이상거래설명 (활용 가능) |
| 데이터 컬럼 | 영문 필드명 | 한글 컬럼 → pp_hofinet.py로 변환 |
| 평가 방식 | train 10%, test 80% split + LogReg | 동일 |
| 시각화 | t-SNE + ARI/Silhouette | 동일 |
