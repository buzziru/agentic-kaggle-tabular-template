# TASK — 프로젝트 계획 & 마일스톤

> 전체 계획의 **인덱스**. 각 작업의 세부계획은 `docs/tasks/<id>.md` 에 따로 두고 여기서는 **포인터**로만 관리한다.
> 진행 상태(현재값·다음 액션)는 [CURRENT_STATUS.md](CURRENT_STATUS.md)(세션 핸드오프), 결정 근거는 [docs/wiki/decisions.md](docs/wiki/decisions.md).

## 사용법
- 작업 = **독립 실행 가능한 단위**. `(∥)` 표시 = 다른 작업과 **병렬 가능**(서브에이전트/세션 분할 OK).
- 작업 착수 시: [docs/tasks/_TEMPLATE.md](docs/tasks/_TEMPLATE.md) 복사 → `docs/tasks/<id>-<slug>.md` 생성(목표·성공기준·단계·파일·의존성) → 본 표의 포인터 연결 → 완료 시 `[x]` + [CURRENT_STATUS.md](CURRENT_STATUS.md) 갱신.
- 상태: `[ ]` 대기 · `[~]` 진행 · `[x]` 완료 · `[-]` 기각(사유는 detail 파일).

## 마일스톤 개요
| M | 목표 | 게이트(다음으로 넘어가는 조건) |
|---|---|---|
| M0 | 셋업 | `uv sync` OK · `.env` · `config.py` 컬럼 채움 · 데이터 로드 |
| M1 | EDA | CV 전략 확정(`docs/setup_questions.md`) · `docs/eda.md` 수치 요약 |
| M2 | 베이스라인 | 파이프라인 닫힘(로드→피처→CV→OOF→제출) · OOF≈LB 1회 확인 |
| M3 | 피처 | 채택 피처 OOF 향상 검증(동일 fold) |
| M4 | 모델 다양성 | 어댑터로 ≥2 모델 추가 · OOF 상관 리포트 |
| M5 | 앙상블 | 스택 meta-OOF > 단일 최고(held-out 검증) |
| M6 | 튜닝·마무리 | seed avg · 최종 제출 · 회고(`docs/wiki/experiments/`) |

---

## M0 — 셋업
- [ ] **T0.1** 환경·의존성·인증 → `docs/tasks/T0.1-setup.md` · verify: `uv run python -c "import src.config"` OK
- [ ] **T0.2** `config.py` 컬럼/지표/대회 채움 + 데이터 다운로드 → `docs/tasks/T0.2-config.md` · verify: `data.load_train().shape`

## M1 — EDA `(∥ 주제별 노트북 병렬)`
- [ ] **T1.1** 개요·결측·카디널리티·타깃 분포 → `docs/tasks/T1.1-eda-overview.md` · verify: `docs/eda.md` 수치
- [ ] **T1.2 (∥)** 타깃 관계·수치 분포 → `docs/tasks/T1.2-eda-target.md` · verify: 피처 후보 등록
- [ ] **T1.3 (∥)** train/test 드리프트 + ⚠️누수 점검 → `docs/tasks/T1.3-leakage-drift.md` · verify: 누수 리스크 목록
- [ ] **T1.4** CV 전략 확정 → `docs/tasks/T1.4-cv-decision.md` · verify: `docs/setup_questions.md` + `config.py` 반영

## M2 — 베이스라인
- [ ] **T2.1** LGBM 베이스라인 풀 파이프라인 → `docs/tasks/T2.1-baseline.md` · verify: `experiments/oof/exp_001.csv` + 로그
- [ ] **T2.2** 제출 + OOF≈LB 확인 → `docs/tasks/T2.2-submit-check.md` · verify: `record_submission` 원장 + 갭 기록

## M3 — 피처 엔지니어링 `(∥ feature-smith 1개씩)`
- [ ] **T3.1** 고카디널리티 OOF 타깃 인코딩 → `docs/tasks/T3.1-target-encoding.md` · verify: OOF Δ vs base
- [ ] **T3.2 (∥)** 그룹/시계열 과거-only 파생 → `docs/tasks/T3.2-group-features.md` · verify: 누수검증 + OOF Δ

## M4 — 모델 다양성 `(∥ 어댑터별)`
- [ ] **T4.1 (∥)** XGB 어댑터(`train_xgb`) → `docs/tasks/T4.1-xgb.md` · verify: OOF + corr(base)
- [ ] **T4.2 (∥)** CatBoost/NN 어댑터 추가 → `docs/tasks/T4.2-more-models.md` · verify: 어댑터 ≤50줄 + OOF
- [ ] **T4.3** (필요 시) GPU 오프로드 셋업 → `docs/tasks/T4.3-gpu-offload.md` · verify: `kaggle_jobs`/`colab_jobs` 회수

## M5 — 앙상블
- [ ] **T5.1** 스택 풀 구성 + 메타러너 비교 → `docs/tasks/T5.1-stack.md` · verify: `scripts/summarize.py` meta 행
- [ ] **T5.2** 멤버 추가 판정(held-out/nested) → `docs/tasks/T5.2-member-gate.md` · verify: meta-overfit 비용 기록

## M6 — 튜닝 & 마무리
- [ ] **T6.1** seed averaging + 최종 제출 선택 → `docs/tasks/T6.1-finalize.md` · verify: 최종 제출 원장
- [ ] **T6.2** 회고 작성 → `docs/tasks/T6.2-retrospective.md` · verify: `docs/wiki/experiments/exp_*.md`

---
_백로그/기각 작업은 detail 파일에 사유와 함께 보존(삭제하지 말 것 — 재시도 방지). 새 마일스톤/작업은 위 규칙으로 추가._
