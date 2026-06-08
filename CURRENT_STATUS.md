# CURRENT STATUS — 세션 인수인계

> 매 세션 끝에 갱신. **현재값·진행 상태·다음 할 일**의 SSOT. 계획·작업 분할 = `TASK.md`, 상시 가이드 = `CLAUDE.md`, 결정 = `docs/wiki/decisions.md`, 지식 = `docs/wiki/`.
> (작성 규칙: 길게 쓰지 말 것 — 수치 + 포인터 위주. 끝난 건 줄이고 다음 액션을 맨 위에.)

_최종 갱신: {{YYYY-MM-DD}} — 세션 {{N}}_

## 🟢 현재 최고 (수치 SSOT = summarize, 수동 복제 금지)
- **`uv run python scripts/summarize.py`** 가 최신 리더보드(단일·스택·LB)를 출력 — **여기에 숫자를 베껴 적지 않는다**(드리프트 방지).
- 최종 제출 후보(사람 판단): `experiments/submissions/{{...}}.csv`
- 목표 격차 해석은 아래 🧭 에만 유지(summarize 가 안 주는 정보).

## 🧭 현재 위치
- 마일스톤: **{{M?}}** · 진행 중 작업: **{{T?.?}}** → `docs/tasks/{{...}}.md`
- 목표 격차: 현재 {{값}} vs 목표 {{값}} = {{격차}} ({{천장 게이트 판단}})

## ✅ 직전 세션 한 일 (수치 + 근거)
- {{한 일 1 — 결과 수치}}
- {{한 일 2}}

## 🔜 다음 할 일 (우선순위)
1. {{다음 액션 — TASK 포인터}} · verify: {{체크}}
2. {{...}}

## 🅿️ Parked / 기각 (재시도 금지)
- {{레버 — 사유(수치). 상세는 docs/tasks/ 또는 decisions.md}}

## ⚠️ 블로커 / 주의
- {{환경·쿼터·데이터·인프라 가드 등}}

## ⚡ 빠른 복귀
```bash
uv sync
uv run python scripts/summarize.py          # 실험 리더보드
# 진행 중 작업 detail: docs/tasks/{{...}}.md
```
