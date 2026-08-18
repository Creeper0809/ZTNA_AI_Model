# ZTNA-UEBA 운영 API와 SOC 대시보드

이 문서는 PoC 모델을 단일 서버 운영 베타로 실행하는 방법을 설명한다. 운영 API는 가변 로그를 받아 Trust Score와 5단계 정책을 즉시 반환한다. 의심 이벤트의 필드 조합 영향 분석은 별도 작업으로 실행되므로 점수 응답을 지연시키지 않는다.

## 실행 구조

```text
로그 수집기·ZTNA 제어부
          │ JSON / batch JSON
          ▼
   FastAPI 운영 API ── API 키·입력 제한·요청 ID
          │
          ├─ 빠른 경로: 기준선 비교 → AI 필드 가중치 → Trust Score → 정책
          │                                      │
          │                                      └─ 즉시 응답·이벤트 저장
          │
          └─ 의심 이벤트: 후보 풀 → 단일·쌍 선별 → 최종 부분 조합 → 상세 근거 갱신
                                                       │
                                                       ▼
                                            SOC 대시보드·사용자 타임라인
```

기본 추론은 모든 부분 조합을 실행하지 않는다. `auto` 모드에서는 `관찰`, `추가 인증`, `접근 제한`, `차단` 정책이거나 신뢰도가 0.75 미만인 이벤트만 상세 분석 대기열에 넣는다. 정상 허용 이벤트는 기준선 편차와 AI 가중치 근거까지만 저장한다.

운영자가 처음 보는 판정 설명은 `발생 행위 → 평소 행동 → 현재 행동 → UEBA 판단 → 정책 결과` 순서다. 사용자별 정상 이력이 충분하면 개인 기준선을 사용하고, 부족하면 단말·동료 집단·동일 로그 유형 순으로 내려가 실제 사용한 기준을 설명한다. 반사실 재검사와 Shapley 조합값은 사람용 판정 이유가 아니라 접힌 `기술 검증`에서만 제공한다.

상세 분석은 모델 위험 기여도, AI 가중치, 기준선 이탈도를 함께 사용해 최대 12개 후보를 만든다. 후보의 단일 정상화와 모든 필드 쌍을 먼저 재검사해 함께 나타날 때만 커지는 약한 신호를 찾는다. 최종 최대 6개는 64개 부분 조합을 모두 평가해 필드별 Shapley 기여도와 쌍 상호작용을 계산한다. 실제 모델에서는 변형 로그를 배치로 처리한다.

수치형 필드는 선택된 정상 기준선의 중앙값으로 바꾼다. 범주형·시간형 필드는 원문 정상값을 저장하거나 복원하지 않고, 필드를 유지한 채 희귀도·편차 특징만 정상 상태로 중화한다. 설명 품질 등급은 정상 기준선 표본, 후보 점수 포착률, 정상 기준 개입 비율, 조합별 영향 방향 일치와 Shapley 합산 일치도를 반영한다. 이 등급은 탐지 정확도나 공격 확률을 뜻하지 않는다.

## 로컬 실행

`ai_model` 디렉터리에서 실행한다.

```powershell
.venv\Scripts\python.exe -m pip install -e ".[server,test]"
$env:ZTNA_CHECKPOINT="artifacts/final_portable_model/model.pt"
$env:ZTNA_DATABASE="artifacts/operations.sqlite"
$env:ZTNA_API_KEY="충분히-긴-운영용-비밀값"
.venv\Scripts\python.exe -m ztna_ueba.server --host 127.0.0.1 --port 8080
```

- 대시보드: `http://127.0.0.1:8080/`
- OpenAPI: `http://127.0.0.1:8080/docs`
- 준비 상태: `http://127.0.0.1:8080/health/ready`

외부 주소(`0.0.0.0` 등)에 바인딩할 때는 `ZTNA_API_KEY`가 없으면 서버가 시작되지 않는다. 대시보드는 상단의 `API 키` 버튼으로 키를 입력하며 키는 현재 브라우저 탭에만 저장된다.

## 단건 입력

```powershell
$headers = @{ "X-API-Key" = $env:ZTNA_API_KEY }
$body = @{
  explanation_mode = "auto"
  event = @{
    event_id = "vpn-20260808-0001"
    dataset = "company"
    source_type = "vpn"
    event_type = "session"
    event_time = "2026-08-08T03:00:00Z"
    user_id_hash = "user_7a24f1"
    device_id_hash = "device_81cc09"
    src_ip_hash = "ip_098ff2"
    authentication_method = "password+mfa"
    authentication_result = "success"
    auth_attempts = 7
    geo_zone = "unknown-region"
  }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8080/api/v1/assess" `
  -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

응답의 `explanation_status`는 다음 중 하나다.

| 상태 | 의미 |
|---|---|
| `skipped` | 기본 근거만 생성했다. 정상 허용 이벤트의 기본 상태다. |
| `pending` | 상세 조합 분석을 기다리고 있다. |
| `processing` | 상세 조합 분석을 실행 중이다. |
| `completed` | 조합 영향 분석이 저장됐다. |
| `deferred` | 대기열이 가득 차 작업이 미뤄졌다. 이벤트 상세에서 재시도할 수 있다. |
| `failed` | 상세 분석에 실패했다. 오류를 확인한 뒤 재시도할 수 있다. |

`explanation_mode`는 `auto`, `none`, `full`을 지원한다. `full`은 요청 스레드에서 조합 분석까지 끝내므로 검증·디버깅에만 사용한다.

## 배치 입력

`POST /api/v1/assess/batch`는 한 요청에서 최대 100개의 이벤트를 받는다.

```json
{
  "explanation_mode": "auto",
  "events": [
    {"event_id": "evt-1", "dataset": "company", "source_type": "iam", "event_type": "authentication"},
    {"event_id": "evt-2", "dataset": "company", "source_type": "ndr", "event_type": "network_flow"}
  ]
}
```

## 대시보드 기능

- 최근 24시간 전체 요청, 의심 이벤트, 평균 위험도, 상세 분석 대기 건수
- 정책 판정 및 로그 원천 분포
- 의심 이벤트 필터와 사용자 검색
- 이벤트별 발생 행위, 의심 이유, 정책 이유
- 필드 관측값, AI 가중치, 기준선 이탈도, 위험 기여도
- 같은 사용자의 의심 이벤트 타임라인
- 관제 상태와 메모 저장, 실패·지연된 상세 분석 재시도
- 정답 라벨과 비밀값을 제거한 원본 이벤트 조회

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `ZTNA_CHECKPOINT` | 없음 | 모델 체크포인트. 필수다. |
| `ZTNA_DATABASE` | `artifacts/operations.sqlite` | 이벤트 저장소 경로 |
| `ZTNA_API_KEY` | 빈 값 | 외부 주소 바인딩 시 필수 |
| `ZTNA_HOST` | `127.0.0.1` | 수신 주소 |
| `ZTNA_PORT` | `8080` | 수신 포트 |
| `ZTNA_DEVICE` | `auto` | `auto`, `cpu`, `cuda` |
| `ZTNA_EXPLANATION_WORKERS` | `1` | 조합 분석 작업자 수. CPU 서버에서는 1을 권장한다. |
| `ZTNA_EXPLANATION_QUEUE_CAPACITY` | `256` | 메모리에 대기시킬 상세 분석 작업의 최대 수 |
| `ZTNA_SCORING_CONCURRENCY` | `2` | 동시 모델 추론 수 |

## Docker 실행

```powershell
Copy-Item .env.example .env
# .env의 ZTNA_API_KEY를 변경한다.
docker compose up --build
```

체크포인트는 이미지에 포함하지 않고 읽기 전용으로 마운트한다. 이벤트 DB는 `operations-data` 볼륨에 보관한다.

## 운영 전 확인

현재 구현은 한 서버에서 운영하는 파일럿·운영 베타 범위다. 고객 트래픽에 자동 차단을 적용하기 전에는 회사 정상 로그로 기준선과 점수 보정을 완료하고, 모의 운영에서 오탐률을 검증해야 한다.

여러 서버로 수평 확장하는 정식 상용 배포에서는 SQLite와 프로세스 내부 작업 대기열을 각각 PostgreSQL과 Redis 기반 외부 작업자로 교체해야 한다. 고객 인증은 API 키 대신 사내 SSO/OIDC, 권한 분리, 비밀 관리 시스템과 연동하고 데이터 보존·삭제 정책 및 감사 로그 반출을 적용해야 한다.
