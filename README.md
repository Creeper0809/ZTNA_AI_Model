# ZTNA-UEBA 동적 필드 가중치 모델

이 디렉터리는 스키마마다 필드 수와 이름이 달라도 각 필드의 이상 근거와 가중치를 계산하고, 이를 하나의 Trust Score와 5단계 ZTNA 정책으로 변환하는 실행 가능한 모델을 담고 있다.

사람이 B/N/D/T 상위 가중치를 정하지 않는다. 모델은 실제 로그에 존재하는 세부 필드 각각을 평가한다. BNDT/C는 학습 입력이 아니라 결과를 읽기 쉽게 묶는 사후 설명 분류다.

## 모델이 계산하는 것

portable v2.1의 처리 순서는 다음과 같다.

1. 운영자가 승인한 정상 로그로 `(dataset, source_type, event_type)`별 기준선을 만든다.
2. 각 필드를 정상 기준선 대비 존재 희귀도, 범주 희귀도, 수치 편차, 타입 불일치로 변환한다.
3. Set Attention이 요청마다 필드 가중치와 여러 이벤트의 source/event 가중치를 동적으로 계산한다.
4. 필드 기여도를 합산해 raw anomaly score를 만들고, 해당 원천의 정상 score 분포로 0~1 risk score를 보정한다.
5. `Trust Score = 100 × (1 - calibrated risk score)`를 계산하고 allow/monitor/step-up/restrict/deny 정책을 제안한다.

필드 순서는 결과에 영향을 주지 않는다. categorical 원문은 모델 체크포인트에 저장하지 않고 BLAKE2 fingerprint와 집계 수만 저장한다. timestamp는 정확한 값의 희귀도가 아니라 요일·시간대 분포로 비교하며, 거의 매번 바뀌는 고유 식별자는 반복 가능한 범주값보다 낮게 평가한다. 단일 필드가 점수를 무제한 독점하지 못하도록 필드 evidence 배율도 제한한다.

## “어떤 로그라도 처리”의 정확한 의미

임의의 key-value 로그는 고정 28개 컬럼으로 다시 만들지 않아도 토큰화하고 필드 가중치를 계산할 수 있다. 다만 처음 본 원천은 신뢰할 정상 기준이 없으므로 Trust/risk를 `null`로 반환하고 `shadow/observe_only`를 강제한다.

자동 정책을 사용하려면 각 이벤트에 다음 profile metadata가 있어야 한다.

- `dataset`: 회사·테넌트·수집 원천 식별자
- `source_type`: auth, ndr, vpn, edr 같은 로그 종류
- `event_type`: authentication, network_flow, session 같은 이벤트 종류

그 후 운영자가 정상임을 확인한 burn-in 로그로 온보딩한다. 도구의 기술적 최소치는 profile당 20건이지만, 실제 운영에서는 업무 주기와 정상 행동 모드를 포함하도록 최소 수일~수주의 충분한 표본을 권장한다. 새 정상 모드나 drift가 발견되면 승인 후 기준선을 갱신해야 한다.

## 현재 검증 결과

대표 표본은 4개 공개 원천의 21,000행이며, 원본 `normalized_events_common.csv.gz` 155,737,906행을 끝까지 읽어 dataset-label별 최대 3,000건을 결정적으로 추출했다.

전체 원천을 학습한 최종 후보 `artifacts/final_portable_model/model.pt`의 분리 test 결과는 다음과 같다.

| 범위 | ROC-AUC | AP | F1 | Precision | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|
| 전체 calibrated | 0.9303 | 0.8652 | 0.8653 | 0.8415 | 0.8905 | 0.1268 |
| CICIDS2019 | 0.9989 | 0.9979 | 0.9681 | 0.9382 | 1.0000 | 0.0668 |
| LANL | 0.9428 | 0.9246 | 0.7561 | 0.8540 | 0.6783 | 0.1132 |
| UNSW-NB15 | 0.9891 | 0.9770 | 0.9845 | 0.9696 | 1.0000 | 0.0323 |

CERT r4.2 표본에는 공격 라벨이 없어 공격 탐지 지표를 계산할 수 없다. 정상-only test FPR은 0.2902로 높았으며 시간·행동 drift가 원인일 수 있다. 따라서 이 profile은 회사 로그 검증 전 자동 차단 근거로 쓰면 안 된다.

원천 전체를 supervised 학습에서 제외한 zero-shot holdout에서는 순위 AP가 v1의 0.33~0.40에서 약 0.59~0.87로 개선됐지만, 정상 분포 이동으로 고정 임계값의 오탐·미탐이 남았다. 이는 “가중치 계산 가능”이 “미지 원천 즉시 강제 가능”을 의미하지 않는다는 근거다.

## 설치

프로젝트 루트에서 실행한다.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

CPU 환경에서는 PyPI 기본 torch를 설치하고 명령의 `--device cpu`를 사용한다.

## 재학습

대표 표본을 다시 만들려면 다음을 실행한다. 이 단계는 전체 원본을 읽으므로 시간이 걸린다.

```powershell
.venv\Scripts\python.exe scripts\build_sample.py `
  --input path\to\normalized_events_common.csv.gz `
  --output artifacts\training_sample.csv.gz `
  --profile artifacts\training_sample_profile.json `
  --per-group 3000 --chunksize 100000 --seed 20260720
```

portable 모델 학습:

```powershell
.venv\Scripts\python.exe -m ztna_ueba.train `
  --data artifacts\training_sample.csv.gz `
  --output-dir artifacts\final_portable_model `
  --epochs 8 --batch-size 256 --learning-rate 0.0003 `
  --field-dropout 0.20 --attention-entropy-weight 0.05 `
  --portable --device cuda --seed 20260720
```

원천 holdout은 `--holdout-dataset cicids2019`처럼 지정한다. holdout 원천의 공격 라벨은 학습하지 않고 train-normal만 기준선 적합에 사용하며, 평가는 분리된 test split에서 수행한다.

## 신규 로그 온보딩

CSV, CSV.GZ, JSON, JSONL을 지원한다. 입력이 정상임을 운영자가 확인했다는 뜻으로 `--confirmed-normal`이 반드시 필요하며, 원본 체크포인트는 덮어쓰지 못한다.

```powershell
.venv\Scripts\python.exe scripts\onboard_normal_source.py `
  --checkpoint artifacts\final_portable_model\model.pt `
  --normal-data examples\new_source_normal.jsonl `
  --output artifacts\company_model\model.pt `
  --confirmed-normal --device cuda
```

입력 파일에 profile metadata가 없다면 `--dataset`, `--source-type`, `--event-type`으로 주입할 수 있다. 온보딩 결과에는 원문 categorical 값이 아니라 기준선 통계와 score calibration만 저장된다.

## 추론

```powershell
.venv\Scripts\python.exe -m ztna_ueba.predict `
  --checkpoint artifacts\company_model\model.pt `
  --input examples\new_source_anomalous.json `
  --device cuda
```

주요 출력은 다음과 같다.

- `trust_score`, `risk_score`: 정상 기준선으로 보정된 운영 점수. 미온보딩 원천은 `null`이다.
- `raw_model_risk_probability`: 모델 내부 비교·진단용 값이며 단독 정책 근거로 사용하지 않는다.
- `baseline_ready`, `score_calibration_ready`, `confidence`: 점수 사용 준비 상태다.
- `shadow_mode_required`: true이면 정책 집행이 금지된다.
- `policy`: shadow 또는 allow/monitor/step_up/restrict/deny 제안이다.
- `top_fields`, `source_weights`, `bndt_posthoc_contributions`: 판정의 필드·원천별 근거다.

## 룰베이스 모델 비교

Jeong과 Yang(2025)의 20개 세부 지표, 0/5/10/15/20점 rubric,
고정 가중치 `B/N/D/T=0.4/0.3/0.2/0.1`, 허용/MFA/차단 구간을
결정론적 비교 기준선으로 구현했다.

```powershell
.venv\Scripts\python.exe scripts\evaluate_rule_baseline.py `
  --input examples\composite_risk_case.json

.venv\Scripts\python.exe scripts\compare_rule_and_model.py `
  --checkpoint artifacts\onboarding_smoke\model.pt `
  --input examples\composite_risk_case.json
```

동일 복합위험 예시에서 룰베이스 모델은 `B=75, N=85, D=50, T=100`,
Trust 75.5로 추가 인증(MFA)을 요구하고, 제안 모델은 Trust 0.0으로 차단한다.
현재 VPN 예시는 논문의 20개 항목 중 10개만 직접 관측하거나 프로젝트 proxy로
매핑하며 나머지 10개는 정상 20점으로 가정한다. 따라서 이 비교는 판정 구조를
보여주는 재현 사례이며, 어느 정책이 더 적절한지는 회사 replay 정답과 비용으로 검증해야 한다.

## 테스트와 감사

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\audit_model.py `
  --checkpoint artifacts\final_portable_model\model.pt `
  --data artifacts\training_sample.csv.gz `
  --output artifacts\final_portable_model\audit.json `
  --device cuda
```

`audit.json`에는 calibrated/raw confusion, FPR, 원천별 지표, profile calibration 준비율, attention 집중도, 필드별 평균 기여도가 기록된다. `profile_diagnostics.py`는 특정 데이터셋의 split/label별 baseline feature와 기여도 이동을 비교한다.

## 주요 산출물

다음 항목은 학습·평가 시 로컬에서 생성되며, 체크포인트와 원본·파생 데이터가 포함된
`artifacts/` 디렉터리는 Git 저장소에 포함하지 않는다.

- `artifacts/final_portable_model/model.pt`: 현재 최종 후보 체크포인트
- `artifacts/final_portable_model/metrics.json`: 학습 이력과 test 지표
- `artifacts/final_portable_model/audit.json`: 운영·설명 감사 결과
- `artifacts/training_sample.csv.gz`: 21,000행 대표 표본
- `artifacts/training_sample_profile.json`: 표본 구성과 split 기록
- `examples/new_source_*`: 미지 스키마 온보딩·정상·이상 예시
- `examples/composite_risk_scorecard_comparison.json`: 룰베이스 모델과 제안 모델의 동일 요청 비교

## 남아 있는 한계

- 공개 표본은 개별 이벤트 단위이므로 여러 source를 한 요청으로 묶는 event attention은 구조와 단위 테스트만 검증됐고 회사 상관 로그로 성능 검증되지 않았다.
- 현재 risk score는 정상 분포 기반 운영 지수이지 실제 사고 확률의 통계적 보장이 아니다.
- 신규 원천은 충분한 정상 burn-in, offline replay, 관리자 승인, Shadow 관찰을 거쳐야 한다.
- 데이터 drift와 label delay를 반영한 주기적 재보정·rollback은 운영 control plane에서 추가해야 한다.
