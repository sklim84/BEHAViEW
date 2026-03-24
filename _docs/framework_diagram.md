# Framework Diagram: Behavioral Subgraph Contrast for AML

## 1. 전체 프레임워크 개요

```
                    ┌──────────────────────────────────────────────┐
                    │         Input: Transaction Data              │
                    │  (452K accounts, 4.7M directed multi-edges)  │
                    └────────────────┬─────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                                  ▼
           ┌────────────────┐                ┌─────────────────────┐
           │ Transaction    │                │ Feature Engineering │
           │ Graph G_trans  │                │                     │
           │ (directed,     │                │ x_behav (22):       │
           │  multi-edge)   │                │  amount stats (8)   │
           │                │                │  entropy (2)        │
           └───────┬────────┘                │  temporal (12)      │
                   │                         └──────────┬──────────┘
                   │                                    │
                   │                         ┌──────────▼──────────┐
                   │                         │ Behavioral k-NN     │
                   │                         │ Graph G_knn         │
                   │                         │                     │
                   │                         │ cosine similarity   │
                   │                         │ on x_behav → top-k  │
                   │                         │ neighbors (k=10)    │
                   │                         └──────────┬──────────┘
                   │                                    │
          ┌────────▼─────────┐            ┌─────────────▼──────────┐
          │    View 1         │            │    View 2              │
          │  (Transaction     │            │  (Behavioral           │
          │   topology)       │            │   similarity)          │
          │                   │            │                        │
          │  "누구와           │            │  "누구와 행동이         │
          │   거래하는가"      │            │   비슷한가"             │
          │                   │            │                        │
          │  F-F/F-B = 1:5.7  │            │  F-F/F-B = 1:1.4      │
          └────────┬──────────┘            └──────────┬────────────┘
                   │                                  │
                   ▼                                  ▼
          ┌────────────────────────────────────────────────────────┐
          │              Shared GNN Encoder                        │
          │                                                        │
          │  GCNConv → BatchNorm → PReLU → Dropout  (per layer)    │
          │                                                        │
          │  ※ BN이 결정적: 없으면 F1 0.05, 있으면 F1 0.68        │
          └───────────┬────────────────────────┬───────────────────┘
                      │                        │
                      ▼                        ▼
          ┌───────────────────┐    ┌───────────────────┐
          │  z1 (node embeds  │    │  z2 (node embeds  │
          │   on G_trans)     │    │   on G_knn)       │
          └─────────┬─────────┘    └─────────┬─────────┘
                    │                        │
                    ▼                        ▼
          ┌───────────────────┐    ┌───────────────────┐
          │  Subgraph Pooling │    │  Subgraph Pooling │
          │                   │    │                   │
          │  s_i = mean(z_i   │    │  s_i = mean(z_i   │
          │    + neighbors    │    │    + neighbors     │
          │    of i in        │    │    of i in         │
          │    G_trans)       │    │    G_knn)          │
          └─────────┬─────────┘    └─────────┬─────────┘
                    │                        │
                    ▼                        ▼
          ┌────────────────────────────────────────────┐
          │         Projector + Predictor               │
          │  Linear → BN → PReLU → Linear               │
          └──────────────────┬─────────────────────────┘
                             │
                             ▼
          ┌────────────────────────────────────────────┐
          │        Bootstrap L2L Loss (BYOL)           │
          │                                            │
          │  online predictor(s1) ↔ target proj(s2)    │
          │  online predictor(s2) ↔ target proj(s1)    │
          │                                            │
          │  + Momentum target encoder (EMA 0.99)      │
          │  ※ Negative sample 불필요                    │
          └──────────────────┬─────────────────────────┘
                             │
                             ▼
          ┌────────────────────────────────────────────┐
          │        Downstream Evaluation               │
          │                                            │
          │  Frozen [s1 ∥ s2] → LogisticRegression     │
          │  10% train / 80% test                      │
          │  Metrics: F1_fraud, AUROC, AUPRC           │
          └────────────────────────────────────────────┘
```

## 2. (a)(b)(c)(d) 4-Setting 비교

```
                        View 구성
                ┌──────────────┬──────────────┐
                │ Augmentation │ Behavioral   │
                │ (same graph, │ k-NN         │
                │  diff aug)   │ (diff graph) │
   ┌────────────┼──────────────┼──────────────┤
   │ Node-Level │              │              │
   │ (no pool)  │  (a) 0.270   │  (b) 0.678   │
   │            │              │   (+151%)    │
C  ├────────────┼──────────────┼──────────────┤
o  │ Subgraph   │              │              │
n  │ Pooling    │  (c) 0.205   │  (d) 0.682   │
t  │            │   (-24%)     │   (+153%)    │
r  └────────────┴──────────────┴──────────────┘
a
s              ※ GBT encoder 기준, 4-seed avg F1_fraud
t
               ※ (c) < (a): subgraph pooling이 transaction
Level            graph의 낮은 F-F/F-B(1:5.7)를 증폭하여 악화
               ※ (d) ≈ (b): behavioral k-NN의 높은 F-F/F-B
                 (1:1.4)에서는 subgraph pooling이 소폭 추가 개선
```

## 3. 기존 모델 매핑

```
┌─────────────────────────────────────────────────────────────┐
│                    GCL Landscape for AML                     │
│                                                             │
│  View 구성 ──────────────────────────────────────────────►   │
│  (Augmentation)                        (Behavioral k-NN)    │
│                                                             │
│  │ ┌─────────────────────────┐  ┌─────────────────────┐    │
│  │ │ (a) Standard GCL        │  │ (b) + Behav View    │    │
│  │ │                         │  │                     │    │
│  │ │  L2L:                   │  │  + k-NN graph       │    │
│  │ │   GRACE (InfoNCE)       │  │    as View 2        │    │
│  │ │   GCA (adaptive aug)    │  │                     │    │
│C │ │   BGRL (bootstrap)      │  │  F1: 0.57~0.68      │    │
│o │ │   GBT (Barlow Twins)    │  │  (BN encoder)       │    │
│n │ │                         │  │                     │    │
│t │ │  G2L:                   │  │  cf. MLGCL, GCPAL   │    │
│r │ │   DGI (JSD+corruption)  │  │  (선행연구)          │    │
│a │ │   MVGRL (dual encoder)  │  │                     │    │
│s │ │                         │  │                     │    │
│t │ │  F1: 0.05~0.32          │  │                     │    │
│  │ └─────────────────────────┘  └─────────────────────┘    │
│  │                                                         │
│L │ ┌─────────────────────────┐  ┌─────────────────────┐    │
│e │ │ (c) + Subgraph Pool     │  │ (d) PROPOSED ★      │    │
│v │ │                         │  │                     │    │
│e │ │  subgraph pooling on    │  │  Behavioral k-NN    │    │
│l │ │  transaction graph      │  │  + Subgraph Pool    │    │
│  │ │                         │  │  + BN Encoder       │    │
│  │ │  F1: 0.12~0.25          │  │                     │    │
│  │ │  (BN encoder)           │  │  F1: 0.682 ★        │    │
│  │ │  ※ (a)보다 악화!        │  │  (+153% vs (a))     │    │
│  │ │                         │  │                     │    │
│  │ │  cf. SUBG-CON (ICDM'20) │  │  cf. SAMCL          │    │
│  │ │  (선행연구)              │  │  (subgraph CL,      │    │
│  ▼ │                         │  │   but not AML)      │    │
│    └─────────────────────────┘  └─────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 4. Behavioral k-NN이 효과적인 이유 (Fraud Connectivity)

```
  Transaction Graph              Behavioral k-NN Graph
  (원래 거래 관계)                (금액 패턴 유사성)

    B───B                           F───F
   / \ / \                         / \ / \
  B   F   B                       F   F   F
   \ / \ /                         \ / \ /
    B───B                           B───B
                                   / \
  F-F/F-B = 1:5.7                 B   B
  → fraud가 benign에
    5.7배 둘러싸임              F-F/F-B = 1:1.4
  → GNN이 fraud 신호 희석      → fraud끼리 연결 밀도 높음
                                → GNN이 fraud 패턴 학습 가능

  Homophily: 0.690               Homophily: 0.981
  Fraud-Fraud: 257K              Fraud-Fraud: 62K
  Fraud-Benign: 1,469K           Fraud-Benign: 88K
```

## 5. BN 효과 시각화

```
  F1_fraud
  0.70 ┤                          ★ GCN+BN (d): 0.682
       │     ■ ■ ■ ■              ★ GIN+BN (d): 0.570
  0.60 ┤
       │
  0.50 ┤
       │
  0.40 ┤
       │
  0.30 ┤     ●                    ● GCN+BN (a): 0.270
       │
  0.20 ┤
       │                          ▲ GIN+BN (a): 0.149
  0.10 ┤
       │     ○ ○ ○ ○              ○ GCN no-BN (d): 0.07
  0.00 ┤─────────────────────────────────────────────
       │    (a)    (b)    (c)    (d)
       │            Setting
       │
  ■ = GCN+BN (GBT/DGI+BN/MVGRL+BN/GRACE+BN)
  ○ = GCN no-BN (DGI/MVGRL/GRACE/GCA)
  ● = baseline (a), ★ = proposed (d)
```

## 6. Encoder 비교 (10종)

```
  ┌─────────────────────────────────────────────────────┐
  │              Encoder Architecture Space              │
  │                                                     │
  │  Conv Layer     BN          Activation    Dropout   │
  │  ──────────    ────         ──────────    ───────   │
  │                                                     │
  │  GCNConv ──┬── per-layer ── PReLU ─── ✓ : GBT      │ 0.682
  │            │                                        │
  │            ├── per-layer ── PReLU ─── ✓ : DGI+BN   │ 0.682
  │            │                                        │
  │            ├── per-layer ── PReLU ─── ✓ : MVGRL+BN │ 0.682
  │            │                                        │
  │            ├── per-layer ── ReLU ──── ✓ : GRACE+BN │ 0.681
  │            │                                        │
  │            ├── final-only ─ PReLU ─── ✓ : BGRL     │ 0.647
  │            │                                        │
  │            ├── none ─────── PReLU ─── ✗ : DGI      │ 0.074
  │            ├── none ─────── PReLU ─── ✗ : MVGRL    │ 0.071
  │            ├── none ─────── ReLU ──── ✓ : GRACE    │ 0.069
  │            └── none ─────── ReLU ──── ✗ : GCA      │ 0.085
  │                                                     │
  │  GINConv ──── per-layer ── ReLU ──── ✓ : GIN      │ 0.570
  │                                                     │
  │  ※ 결정 요인: BN 유무 (0.07 vs 0.68, ~10x)         │
  │  ※ Conv/Activation/Dropout는 부차적                 │
  └─────────────────────────────────────────────────────┘
```

## 논문 Figure 구성 제안

| Figure | 내용 | 섹션 |
|--------|------|------|
| Fig.1 | 전체 프레임워크 (Section 1 도식) | Method |
| Fig.2 | (a)(b)(c)(d) 2×2 매트릭스 (Section 2) | Method |
| Fig.3 | Fraud connectivity 비교 (Section 4) | Analysis |
| Fig.4 | Encoder BN 효과 bar chart (Section 5) | Experiments |
| Fig.5 | 기존 모델 landscape (Section 3) | Related Work |
