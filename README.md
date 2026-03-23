# Beyond Augmentation: Behavioral Subgraph Contrast for AML

자금세탁(AML) 탐지를 위한 그래프 대조학습(GCL)에서 **View 구성**과 **Contrastive Level**, 두 독립 축을 분석하고 최적 조합을 제안합니다.

> [FraudCenGCL (BigData 2025)](https://github.com/sklim84/KA-003-FraudCenGCL) 확장 연구

## 핵심 아이디어

| 축 | 질문 | 제안 |
|----|------|------|
| **View 구성** | 어떤 그래프를 contrastive view로 쓸 것인가? | Behavioral k-NN graph |
| **Contrastive Level** | 어떤 수준에서 대조할 것인가? | Subgraph-level pooling |

```
                        View 구성
                   Augmentation    Behavioral k-NN
Contrastive  Node │ (a) baseline  │ (b) view 변경    │
Level        Sub  │ (c) level 변경│ (d) 최종 제안    │
```

## 연구 질문

| RQ | 질문 | 비교 |
|----|------|------|
| RQ1 | Behavioral view가 효과적인가? | (b) vs (a) |
| RQ2 | 왜 효과적인가? | F-F/F-B 비율, homophily |
| RQ3 | Subgraph pooling이 효과적인가? | (c) vs (a) |
| RQ4 | 두 축의 결합에 시너지가 있는가? | (d) vs (b), (c) |
| RQ5 | 3개 backbone에서 일관되는가? | BGRL, DGI-trs, MVGRL |

## 데이터셋

### HOFINET (Primary) — 실제 은행간 이체

| 항목 | 값 |
|------|-----|
| 계좌 (노드) | 452,816 |
| 이체 (엣지) | 4,732,130 (유향, 멀티엣지) |
| 의심 계좌 | 9,644 (2.13%) |
| 기간 | 2021 Q3 ~ 2024 Q4 (40개월) |
| AML 유형 | 6가지 (structuring, layering 등) |
| 출처 | 전자금융공동망 (KFTC) |

### AMLworld (Secondary) — 일반성 검증용 합성 벤치마크

| 항목 | 값 |
|------|-----|
| 데이터셋 | HI-Small (GCPAL 비교용) |
| 구조 | 유향 멀티그래프 (HOFINET과 동일) |
| AML 패턴 | 8가지 (fan-out, scatter-gather, cycle 등) |
| 3단계 | Placement → Layering → Integration 전부 모델링 |
| 라벨 | 완전한 ground truth |
| 출처 | [NeurIPS 2023 Datasets & Benchmarks](https://arxiv.org/abs/2306.16424), Kaggle 공개 |

> GCPAL이 AMLworld HI-Small을 사용하여 직접 비교 가능. Elliptic(Bitcoin)과 달리 은행 간 이체 도메인에 정합.

## 피처 분류

| Category | 수 | 설명 | GNN 입력 | k-NN 구축 |
|----------|-----|------|----------|----------|
| **A. Behavioral** (x_behav) | 22 | 금액통계, entropy, temporal | O | Behavioral k-NN |
| **B. Structural** (x_struct) | 9 | centrality, degree | O (선택) | Structural k-NN |

## Fraud 연결성 분석 (RQ2)

| 그래프 | F-F/F-B 비율 | 의미 |
|--------|-------------|------|
| Transaction | 1:5.7 | fraud가 benign에 묻힘 |
| Structural k-NN | 1:9.7 | 더 심하게 묻힘 |
| **Behavioral k-NN** | **1:1.4** | **fraud 신호 보존** |

## 실험 결과

> (a)(b)(c)(d) ablation × 3 backbone 실험 진행 중. 결과 확정 후 업데이트 예정.

## 빠른 시작

```bash
# (d) BGRL + behavioral view + subgraph pooling
python models/bgrl_w_knn.py \
  --knn_graph HOFINET_KNN_BEHAV_k10 \
  --subgraph_pool --gpu 0 --seed 2025 \
  --lr 0.0005 --hidden_dim 256 --gconv_nlayers 2 \
  --loss BarlowTwins --skip_tsne

# Subgraph CL (통합 프레임워크)
python models/subgraph_cl.py \
  --knn_graph HOFINET_KNN_BEHAV_k10 \
  --encoder_type bgrl --gpu 0 --seed 2025

# 전체 ablation 실험
bash scripts/run_ablation_abcd.sh
```

## 프로젝트 구조

```
models/
  subgraph_cl.py               # Subgraph CL 통합 프레임워크
  bgrl_w_knn.py                # BGRL + k-NN view (+ --subgraph_pool)
  dgi_transductive_w_knn.py    # DGI-trs + k-NN view (+ --subgraph_pool)
  mvgrl_w_knn.py               # MVGRL + k-NN view (+ --subgraph_pool)
  *_w_org.py                   # Baseline (+ --subgraph_pool)
datasets/
  build_knn_graph.py           # k-NN 그래프 구축
  pp_hofinet.py                # HOFINET 전처리
analysis/
  homophily_knn.py             # F-F/F-B 비율 측정
scripts/
  run_ablation_abcd.sh         # (a)(b)(c)(d) ablation
```

## 관련 연구

| 논문 | 학회 | 관계 |
|------|------|------|
| [MLGCL](https://arxiv.org/abs/2107.02639) | Neurocomputing 2024 | k-NN view for GCL |
| [GCPAL](https://doi.org/10.1007/s44196-024-00720-4) | IJCIS 2024 | k-NN view for AML |
| [SUBG-CON](https://arxiv.org/abs/2009.10564) | ICDM 2020 | Subgraph CL |
| [ImGCL](https://ojs.aaai.org/index.php/AAAI/article/view/26319) | NeurIPS 2023 | Imbalanced GCL |

**차별점**: 기존 연구는 view 구성 또는 contrastive level을 개별적으로 다룸. 본 연구는 **두 축을 교차 분석**하여 AML에 최적인 조합을 도출.

## TODO

- [ ] (a)(b)(c)(d) ablation 결과 정리
- [ ] AMLworld HI-Small 데이터셋 — 일반성 검증, GCPAL 직접 비교
- [ ] SOTA AML 모델 비교

## 라이선스

MIT
