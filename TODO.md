# FraudCenGCL TODO

## 프로젝트 목표
FraudCenGCL 논문을 강화하여 저널에 제출. HOFINET.csv 데이터를 활용하여 그래프 중심성/구조 피처를 GCL의 보조 뷰로 사용하는 이상거래 탐지 성능 개선을 검증.

---

## 완료된 작업

### 1. 프로젝트 구조 재편 ✅
- 루트 레벨 파일 → `models/`, `scripts/`, `datasets/`, `analysis/`, `benchmarks/`, `visualize/` 으로 재구조화
- 공통 모듈 분리: `config.py`, `data_loader.py`, `utils.py`

### 2. HOFINET 데이터 전처리 파이프라인 (`datasets/pp_hofinet.py`) ✅
- HOFINET.csv (473만 거래) → 452,816 노드, 2,558,743 엣지, 9,644 fraud (2.13%)
- 노드 피처 (x_agg, 10개): out_mean/max/std/count, in_mean/max/std/count, md_type_entropy, fnd_type_entropy
- 그래프 피처 (x_cen, 7개): dc, cc, pagerank, hits_hub, hits_auth, kcore, triangle
  - GPU(cuGraph) 가속: dc, pagerank, hits, kcore, triangle (~2초)
  - CPU(NetworkX): cc (~11초)
  - 제외: betweenness centrality (452K 노드에서 99.99% 0), katz (fraud/benign 무차별), eigenvector (수렴 실패)

### 3. 모델 코드 수정 ✅
- TSNE 시각화: sklearn 1.7 호환 (`n_iter` → `max_iter`), 대규모 데이터 서브샘플링 (max 50K)
- DGI-IND, GRACE: `batch.n_id` GPU/CPU 디바이스 불일치 수정
- MVGRL: PPRDiffusion OOM → EdgeRemoving(pe=0.3) 대체

### 4. 초기 실험 (고정 하이퍼파라미터) ✅
결과 파일: `results/exp_results_rq5.csv`

| 모델 | _w_cen F1Ma | _w_org F1Ma | _w_cen AUROC | _w_org AUROC | _w_cen AUPRC | _w_org AUPRC |
|------|------------|------------|-------------|-------------|-------------|-------------|
| **BGRL** | **0.6622** | 0.6591 | **0.9508** | 0.9231 | **0.3388** | 0.3250 |
| DGI-IND | 0.1520 | 0.1531 | 0.9696 | **0.9800** | 0.5739 | **0.5864** |
| DGI-TRS | 0.0728 | **0.1195** | 0.8906 | **0.9010** | 0.2644 | **0.2882** |
| GBT | **0.6067** | 0.6057 | 0.9555 | **0.9581** | 0.3876 | **0.3939** |
| GRACE | 0.0808 | **0.1226** | 0.9066 | **0.9260** | **0.3399** | 0.3040 |
| MVGRL | 0.0618 | **0.1730** | 0.8956 | **0.9019** | 0.2653 | **0.3514** |

**문제점**: BGRL만 _w_cen 우세. 나머지 4개 모델은 _w_org 우세.
**원인 가설**: 하이퍼파라미터가 이전 3개 피처(dc/cc/bc)에 최적화됨. 특히 `input_dim=4`인 모델(GRACE, DGI-TRS)에서 7개 피처를 4차원으로 압축하여 정보 손실 발생.

---

## 진행 중인 작업

### 5. 하이퍼파라미터 서치 🔄
- 스크립트: `scripts/hp_search.py`
- 결과 파일: `results/exp_results_hp_search.csv`
- H100 80GB × 6장 병렬 실행
- 서치 공간 (6 _w_cen 모델 × 72 조합 = 432 jobs):
  - `input_dim`: [8, 16, 32]
  - `hidden_dim`: [128, 256, 512]
  - `lr`: [1e-4, 5e-4, 1e-3, 1e-2]
  - `gconv_nlayers`: [2, 3]
- 진행: 6/432 완료 (2026-03-06 기준)

---

## 앞으로 할 작업

### 6. HP 서치 결과 분석
- [ ] 모델별 최적 하이퍼파라미터 선정
- [ ] input_dim 영향 분석 (가설 검증: 큰 input_dim이 _w_cen 성능 개선하는지)
- [ ] 최적 HP로 _w_cen vs _w_org 재비교

### 7. _w_org 모델 HP 서치 (필요시)
- [ ] _w_cen 최적 HP 결과 확인 후, 공정한 비교를 위해 _w_org도 동일 서치 수행
- [ ] 또는 _w_cen 최적 HP를 _w_org에도 적용하여 비교

### 8. 피처 조합 실험 (Ablation Study)
- [ ] 피처 그룹별 기여도 분석:
  - dc + cc only (기존 centrality)
  - dc + cc + pagerank (importance 기반)
  - dc + cc + hits_hub + hits_auth (역할 기반)
  - dc + cc + kcore + triangle (구조 기반)
  - 전체 7개 피처
- [ ] 최적 피처 부분집합 선정

### 9. 결과 시각화
- [ ] 모델별 _w_cen vs _w_org 성능 비교 bar chart
- [ ] HP 서치 sensitivity heatmap (input_dim × hidden_dim, lr × gconv_nlayers 등)
- [ ] 피처 ablation 결과 시각화
- [ ] t-SNE embedding 시각화 (최적 모델)

### 10. 통계적 유의성 검증
- [ ] 다중 seed 실험 (seed=[2025, 42, 123, 456, 789])
- [ ] 평균 ± 표준편차 보고
- [ ] Wilcoxon signed-rank test (_w_cen vs _w_org)

### 11. 논문 작성 관련
- [ ] 실험 결과 테이블 정리 (LaTeX 형식)
- [ ] 그래프 피처 통계 테이블 (fraud vs benign 분포 차이)
- [ ] HOFINET 데이터셋 기술 통계
- [ ] 기존 논문 대비 변경점 정리 (BC → 다양한 그래프 구조 피처)

### 12. 추가 고려사항
- [ ] 대조 학습 loss 함수 비교 (InfoNCE, JSD, BarlowTwins, BootstrapLatent)
- [ ] 그래프 augmentation 전략 비교 (EdgeRemoving pe, FeatureMasking pf 등)
- [ ] 클래스 불균형 대응 (fraud 2.13% — weighted loss, oversampling 등)
