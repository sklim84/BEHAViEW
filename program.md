# FraudCenGCL Autoresearch

AI 에이전트가 자율적으로 GCL 모델 실험을 수행하는 프로그램.

## Setup

실험 시작 전 사용자와 협의:

1. **Run tag 결정**: 날짜 기반 태그 제안 (예: `mar20`). 브랜치 `autoresearch/<tag>` 생성.
2. **브랜치 생성**: `git checkout -b autoresearch/<tag>`
3. **컨텍스트 파악**: 아래 파일을 읽어 전체 구조 파악:
   - `CLAUDE.md` — 프로젝트 개요, 아키텍처, 실행 방법
   - `config.py` — 공통 argparse (모든 HP 인자 정의)
   - `utils.py` — 평가 함수, t-SNE, 결과 저장
   - `data_loader.py` — 데이터 로딩
   - `results/exp_results_hp_compare.csv` — 기존 실험 결과 (있으면)
4. **GPU 확인**: `nvidia-smi`로 사용 가능한 GPU 확인. 사용자가 지정한 GPU만 사용.
5. **results.tsv 초기화**: 헤더만 있는 `results.tsv` 생성.
6. **확인 후 시작**.

## 실험 대상

### 모델 (12개 = 6 아키텍처 × 2 변형)

| 아키텍처 | _w_cen (중심성 뷰) | _w_org (베이스라인) |
|----------|-------------------|-------------------|
| BGRL | `models/bgrl_w_cen.py` | `models/bgrl_w_org.py` |
| GBT | `models/gbt_w_cen.py` | `models/gbt_w_org.py` |
| GRACE | `models/grace_w_cen.py` | `models/grace_w_org.py` |
| DGI-IND | `models/dgi_inductive_w_cen.py` | `models/dgi_inductive_w_org.py` |
| DGI-TRS | `models/dgi_transductive_w_cen.py` | `models/dgi_transductive_w_org.py` |
| MVGRL | `models/mvgrl_w_cen.py` | `models/mvgrl_w_org.py` |

### 하이퍼파라미터 탐색 범위

| HP | 범위 | 기본값 |
|----|------|--------|
| `--lr` | 1e-4 ~ 1e-2 | 0.001 |
| `--input_dim` | 8, 16, 32, 64 | 16 |
| `--hidden_dim` | 128, 256, 512 | 256 |
| `--gconv_nlayers` | 2, 3, 4 | 3 |
| `--cen_feats` | 아래 세트 참조 | dc cc pagerank hits_hub hits_auth kcore triangle |

### cen_feats 세트 (_w_cen 모델만)

- **baseline (7)**: `dc cc pagerank hits_hub hits_auth kcore triangle`
- **top_discrim (7)**: `dc hits_auth hits_hub kcore triangle in_dc out_dc`
- **extended (13)**: baseline + `in_dc out_dc clustering avg_neigh_deg harmonic sq_clustering`
- 자유롭게 서브셋 시도 가능

### 고정 인자 (모든 실험 공통)

```
--seed 2025
--node_data_name HOFINET_NODE_FEAT
--edge_data_name HOFINET_EDGES
--skip_tsne
```

### 모델별 고정 loss

| 모델 | loss |
|------|------|
| BGRL | `--loss BarlowTwins` |
| GBT | `--loss BarlowTwins` |
| GRACE | `--loss InfoNCE --proj_dim 32` |
| DGI-IND, DGI-TRS | `--loss JSD` |
| MVGRL | `--loss BootstrapLatent` |

## 실험 실행 방법

모델 실행 커맨드 형식:

```bash
python -u models/<model>.py \
  --model_name <experiment_name> \
  --gpu <gpu_id> \
  --seed 2025 \
  --node_data_name HOFINET_NODE_FEAT \
  --edge_data_name HOFINET_EDGES \
  --skip_tsne \
  --metric_save_path ./results/exp_results_autoresearch.csv \
  --lr <lr> \
  --input_dim <input_dim> \
  --hidden_dim <hidden_dim> \
  --gconv_nlayers <gconv_nlayers> \
  --loss <loss> \
  --cen_feats <feats...> \
  > run.log 2>&1
```

_w_org 모델은 `--cen_feats` 인자를 생략한다.

## 평가 지표

**Primary**: `F1Ma` (Macro F1-Score) — 높을수록 좋음
**Secondary**: `auroc`, `auprc`

실험 결과는 stdout 마지막에 출력:
```
(E): Best test F1Mi=0.9821, F1Ma=0.6622
```

결과 추출:
```bash
grep "^(E):" run.log
```

## 결과 기록

`results.tsv` (탭 구분, 커밋하지 않음):

```
commit	model	variant	cen_feats	lr	input_dim	hidden_dim	gconv_nlayers	F1Ma	auroc	auprc	status	description
```

- status: `keep`, `discard`, `crash`
- description: 이 실험에서 시도한 것 (한 줄)

예시:
```
a1b2c3d	bgrl	w_cen	baseline	0.0001	16	256	3	0.6622	0.9508	0.3250	keep	baseline HP
b2c3d4e	bgrl	w_cen	baseline	0.0001	32	256	3	0.6757	0.9462	0.3617	keep	input_dim 32로 증가
c3d4e5f	grace	w_cen	baseline	0.001	16	256	3	0.0808	0.9066	0.2800	discard	GRACE 기본 HP
```

## 실험 루프

전용 브랜치에서 실행 (예: `autoresearch/mar20`).

**LOOP FOREVER:**

1. **이전 결과 분석**: `results.tsv`와 `results/exp_results_autoresearch.csv`를 읽고, 지금까지의 best F1Ma와 트렌드를 파악한다.

2. **실험 가설 수립**: 이전 결과를 기반으로 다음 실험을 결정한다:
   - 초기에는 각 모델의 baseline을 빠르게 확인 (BGRL, GBT 우선 — 이전 결과에서 가장 좋았음)
   - 좋은 모델을 찾으면 해당 모델의 HP를 집중 탐색
   - _w_cen vs _w_org 비교 실험
   - cen_feats 조합 변경 실험

3. **실험 실행**:
   ```bash
   python -u models/<model>.py <args> > run.log 2>&1
   ```
   stdout/stderr를 반드시 파일로 리다이렉트한다. 컨텍스트에 출력이 넘치지 않도록.

4. **결과 확인**:
   ```bash
   grep "^(E):\|F1Ma\|auroc\|auprc" run.log
   ```
   grep 결과가 비어있으면 crash. `tail -n 30 run.log`로 에러 확인.

5. **결과 기록**: `results.tsv`에 기록.

6. **판단**:
   - 해당 모델/변형의 **best F1Ma보다 개선** → `keep`
   - 동일하거나 악화 → `discard`
   - crash → `crash`, 간단한 버그면 수정 후 재시도

7. **다음 실험 결정**: 결과를 분석하고 2단계로 돌아간다.

## 탐색 전략

### Phase 1: 스카우팅 (모델별 baseline)
각 모델(12개)을 기본 HP로 한 번씩 실행하여 baseline 확보.
기본 HP: `lr=0.001, input_dim=16, hidden_dim=256, gconv_nlayers=3`

### Phase 2: 집중 탐색 (Top-3 모델)
Phase 1에서 F1Ma 상위 3개 모델에 대해:
- lr 탐색: 0.0001, 0.0005, 0.001, 0.01
- hidden_dim 탐색: 128, 256, 512
- input_dim 탐색: 8, 16, 32
- 한 번에 하나의 HP만 변경 (ablation 스타일)

### Phase 3: 심층 탐색
- Best HP 조합에서 cen_feats 변형 실험
- _w_cen vs _w_org 동일 HP 비교
- gconv_nlayers 변형

### Phase 4: 자유 탐색
- 이전 결과에서 패턴 발견 → 가설 검증
- 근소한 차이의 HP 조합 미세 조정
- 이전 near-miss 실험 재시도 (다른 조합과 결합)

## 제약사항

**할 수 있는 것:**
- 모델 선택, HP 변경, cen_feats 변경
- `results.tsv` 기록
- 실험 결과 분석

**할 수 없는 것:**
- 모델 소스코드 (`models/*.py`) 수정
- `config.py`, `utils.py`, `data_loader.py` 수정
- 데이터 파일 수정
- 새 패키지 설치
- 사용자가 지정하지 않은 GPU 사용

**시간 제한:**
- 모델별 예상 시간: BGRL ~5분, GBT ~15분, GRACE ~10분, DGI-IND ~3분, DGI-TRS ~5분, MVGRL ~10분
- 20분 이상 걸리면 kill 후 crash 처리

**NEVER STOP**: 실험 루프가 시작되면 사용자가 중단할 때까지 계속한다. "계속할까요?" 같은 질문은 하지 않는다. 아이디어가 바닥나면 이전 결과를 다시 분석하고, 새로운 조합을 시도한다.

## 기존 실험 결과 참고

이전 HP search v2 결과 (`results/exp_results_hp_search_v2.csv`)에서 알려진 사실:
- **BGRL이 최고 성능**: F1Ma max 0.6757
- **GBT 2위**: F1Ma ~0.60
- **lr=0.0001이 BGRL에 최적**
- **baseline cen_feats (7개)가 충분** — extended/top_discrim과 유의미한 차이 없음
- **DGI, GRACE, MVGRL은 F1Ma < 0.4** — 상대적으로 낮음

이 정보를 Phase 1 스카우팅의 우선순위에 활용한다.
