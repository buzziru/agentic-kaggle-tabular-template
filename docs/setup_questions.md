# 셋업 결정 — 질문 & 답 (프로젝트 시작 시 확정)

> 대회/과제 시작 시 한 번 정하고 끝까지 일관 유지하는 구조적 결정. 근거를 남긴다.
> ⚠️ 각 결정은 `src/config.py` 고정 전 **사용자 승인**을 받는다(AI 단독 확정 금지) — 항목별 `[ ] 사용자 승인` 체크 후 config 반영.

## Q. CV 전략은? (가장 중요)
- **답**: {{StratifiedKFold / GroupKFold / KFold}}, {{N}}-fold, seed=42. (`config.CV_STRATEGY`)
- **근거**: train/test 분할 방식 = {{row-level / group-level}}. → {{왜 이 CV 가 대회 셋업과 일치하는지}}.
- [ ] 사용자 승인 (config 고정 전 필수)
- ⚠️ 그룹 누수 위험: {{있으면 GroupKFold, config.GROUP_KEYS 채움}}.
- ⚠️ 시계열은 **공식 미지원**(확장형 윈도우가 full-OOF 스태킹 계약과 불일치) — 필요 시 `cv.get_folds` 에 직접 분기 추가.

## Q. 문제 유형 / 지표 / 제출 형식은?
- **답**: 문제 유형 = {{binary / regression / multiclass}}, 지표 = {{ROC-AUC / RMSE / ...}}, 제출 = {{확률 / 클래스 / 수치}}.
- **일치 점검(필수)** — 셋이 어긋나면 점수가 틀린다:
  - [ ] `config.PROBLEM_TYPE` ↔ `conf/model/*.yaml` 의 objective/metric ↔ `config.METRIC` 일치
  - [ ] `utils.get_scorer` 가 해당 지표를 지원(없으면 추가)
  - [ ] binary/regression/multiclass 는 `PROBLEM_TYPE` 으로 자동 분기(`train_<model>.predict`). multiclass=OOF K개 확률열(`oof_<label>`)·제출 단일 라벨·라벨 0..K-1 인코딩 자동·`balanced_accuracy` 등. ⚠️ 스태킹(`src.stack`)은 아직 multiclass 미지원(단일모델만)
- [ ] 사용자 승인 (config 고정 전 필수)

## Q. 컬럼 분류 (config.py)
- ID: `{{ID_COL}}` / 타깃: `{{TARGET_COL}}`
- 범주형: {{...}} / 정수범주: {{...}} / 수치형: {{...}}
- 그룹 키: {{GROUP_KEYS}} / 시퀀스 키: {{SEQUENCE_COL}}
- [ ] 사용자 승인 (config 고정 전 필수)

## Q. 클래스 불균형 처리?
- **답**: {{지표가 순위 기반이면 가중 기본 미사용, on/off 실험 비교}}.

## Q. 실험 ID 컨벤션?
- **답**: `exp_<NNN>_<short-slug>` 연번. **끝까지 일관**(중간 변경 금지).

## Q. 측정 검정력 (필수 인지)
- fold std σ≈{{측정값}} → SE≈{{ }}. **|Δ| < ~2·SE 결정은 단일 시드로 판정 금지** → 다중 시드 또는 stack-add/잔차 프레임.
