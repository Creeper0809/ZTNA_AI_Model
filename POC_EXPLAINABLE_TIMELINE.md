# 설명 가능한 사용자 의심 이벤트 PoC

## 피드백을 기능으로 바꾼 결과

경쟁 제품에도 평문 행동 요약, 엔터티 위험 점수, 사용자 타임라인과 인과 관계 기능이 있다. 따라서
`AI를 쓴다`, `이유를 보여 준다`, `타임라인을 제공한다`만으로는 차별점이라고 할 수 없다.

이번 PoC의 차별화 방향은 다음 전체 경로가 같은 근거로 이어진다는 점이다.

```text
가변 스키마의 실제 필드
  → 사용자·서비스 계정, 단말, 동료 집단, 로그 유형 기준선 선택
  → 최초 AI 판정과 상위 필드 선택
  → 정상 기준선 대비 차이와 AI 칼럼 가중치
  → 수치로 확인한 한국어 판정 근거
  → 필드값 교체·가림 후 판정 영향 보조 검증
  → Trust Score와 5단계 ZTNA 정책
  → 사용자별 의심 이벤트 타임라인과 원본 이벤트 추적
```

시장 유일성을 주장하는 것이 아니라, 기존 프로젝트가 가진 portable field attention을 SOC 조사와
ZTNA 집행에 연결한 검증 가능한 제품 차별점이다.

## 경쟁 기능과의 경계

| 제품 | 공식 자료에서 확인한 기능 | 이번 PoC가 별도로 검증하는 결합 |
|---|---|---|
| Microsoft Sentinel UEBA | 기준선, 평문 행동 요약, 집계·순서형 행동, 원본 로그 연결 | 가변 칼럼의 기준선 편차와 AI 가중치를 Trust Score와 함께 반환 |
| Exabeam | 사용자 위험 추세, Risk Reasons, 세션 타임라인 | 요청별 칼럼 가중치와 ZTNA 5단계 정책을 직접 연결 |
| Splunk Enterprise Security | 엔터티 누적 risk event와 타임라인 | 관리자가 정한 점수 외에 현재 모델 판정에 영향을 준 칼럼을 보조 검증 |
| Cortex XSIAM | 프로세스 인과 체인과 포렌식 타임라인 | VPN·IAM·NDR 등 가변 로그의 기준선 근거를 사용자 타임라인으로 연결 |

공식 확인 자료:

- Microsoft Sentinel UEBA behaviors: https://learn.microsoft.com/en-us/azure/sentinel/entity-behaviors-layer
- Exabeam User Profile and Timeline: https://docs.exabeam.com/en/cloud-delivered-advanced-analytics/all/user-guide/153664-get-to-know-a-user-profile.html
- Splunk ES Risk Scoring: https://docs.splunk.com/Documentation/ES/latest/Admin/RiskScoring
- Cortex XSIAM Timeline: https://docs-cortex.paloaltonetworks.com/r/Cortex-XSIAM/Cortex-XSIAM-3.x-Documentation/Timeline

## API가 반환하는 근거

`POST /v1/assess`는 다음을 반환하고 이벤트를 SQLite에 저장한다.

- Trust Score, 위험 점수, confidence와 5단계 정책
- 사람이 바로 읽을 수 있는 `발생 행위 → 평소 행동 → 현재 행동 → UEBA 판단 → 정책 결과` 문장
- 정상 기준선 대비 차이, AI 칼럼 가중치와 위험 기여도
- 필드마다 실제 사용한 UEBA 기준선 범위, 정상 로그 수, 필드 관측 수와 대체 여부
- 위험 기여도 절댓값 순위로 해당 필드를 근거에 포함한 선택 이유
- 사람용 판정과 분리된 기술 검증에서 필드 단독·조합 재추론으로 확인한 모델 판정 영향
- 모델 버전, 기준선 준비 상태와 원본 이벤트 참조

한국어 판정 근거는 선택한 UEBA 정상 기준선 대비 차이와 AI 칼럼 가중치를 연결한다. 사용자·서비스
계정 표본이 부족하면 단말, 동료 집단, 로그 유형 순으로 대체한다. API는 최대 12개 후보의 단일·쌍 효과를
먼저 검사하고 최대 6개를 골라 64개 부분집합을 모두 재추론한다. 평균 한계 기여도와 쌍 조합 효과는 해당 칼럼이 모델 판정에
미친 영향을 확인하는 보조 검증값으로만 반환한다.
정답 라벨, 공격명, split과 source file은 모델 입력과 저장 snapshot에서 제외하고 비밀번호·토큰 계열
값은 마스킹한다.

`GET /v1/actors/{actor_id}/timeline`은 해당 사용자의 의심 이벤트를 시간순으로 반환한다.

- 각 이벤트의 사람이 읽는 전체 문장과 구조화된 이유
- 같은 사용자라는 연결과 공통 단말·세션·IP·자원 같은 추가 연결 근거
- 의심 이벤트 수, 최대 위험도, 관련 로그 원천 수와 정책별 건수
- `timeline://events/{event_id}` 형식의 원본 이벤트 참조

위험 점수를 임의로 더해 사고 확률처럼 보이게 하지 않는다. 타임라인 요약은 관측된 개수, 최대값과
연결 근거를 그대로 제공한다.

## 실행

실제 모델로 데모 결과를 생성한다.

```powershell
.venv\Scripts\python.exe scripts\demo_explainable_timeline.py `
  --checkpoint artifacts\complete_bndt_ueba\model.pt `
  --device cpu
```

계층형 기준선 선택과 대체 경로만 독립적으로 재현한다.

```powershell
.venv\Scripts\python.exe scripts\demo_hierarchical_ueba.py
```

HTTP API를 실행한다.

```powershell
.venv\Scripts\python.exe -m ztna_ueba.api `
  --checkpoint artifacts\complete_bndt_ueba\model.pt `
  --database artifacts\explainable_timeline.sqlite `
  --device cpu
```

PowerShell 요청 예시는 다음과 같다.

```powershell
$event = Get-Content -Raw examples\rule_allow_model_deny_event.json | ConvertFrom-Json
$event | Add-Member actor_alias svc_backup -Force
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/assess `
  -ContentType 'application/json' -Body (@{event=$event} | ConvertTo-Json -Depth 10)

Invoke-RestMethod http://127.0.0.1:8080/v1/actors/svc_backup/timeline
```

## PoC 한계

- 사용자 식별은 `actor_id`, `actor_alias`, `user`, `username` 등 후보 필드 우선순위에 따른다. 실제
  회사에서는 IAM의 불변 사용자 ID로 별도 entity resolution을 해야 한다.
- 사용자·단말·동료 집단 기준선은 같은 로그 유형 안에서 정상 로그와 해당 필드 관측이 각각 20건 이상일
  때 사용한다. 표본이 부족한 범위는 자동으로 다음 범위로 대체되며 응답에 그 경로가 기록된다.
- 이벤트 연결은 같은 사용자와 공통 단말·세션·IP·자원에 대한 결정적 연결이다. 공격 단계의 인과 관계를
  통계적으로 증명하는 기능은 아니다.
- 필드 교체·가림 결과는 해당 필드가 모델 판정에 준 영향을 보여 준다. 실제 공격의 원인을 증명하지는 않는다.
- 공개 데이터와 합성 연결 시나리오로 기능을 검증한 것이다. 회사 로그에서 조사 시간 단축률, 설명
  정확성, 오탐률과 정책 효과를 별도로 측정해야 한다.

## 검증 결과

- 전체 기존·신규 자동 테스트 35개 통과
- 실제 `complete_bndt_ueba` 체크포인트에서 승인 백업 1건은 사용자·서비스 계정 기준선으로 허용,
  연속 NDR 이벤트 2건은 같은 로그 유형 기준선으로 차단
- `svc_backup` 의심 타임라인에 2건이 시간순으로 표시되고 공통 session ID, 단말, 출발지 IP,
  자원 ID가 연결 근거로 반환됨
- API 응답, 타임라인과 저장 event snapshot에서 정답 라벨·공격명·split·source file 유출 0건
- 생성 결과: `artifacts/explainable_timeline_poc/demo_output.json`
- 계층형 기준선 선택 결과: `artifacts/hierarchical_ueba_poc/demo_output.json`
