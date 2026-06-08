# 셋업 결정 — 질문 & 답 (프로젝트 시작 시 확정)

> 대회/과제 시작 시 한 번 정하고 끝까지 일관 유지하는 구조적 결정. 근거를 남긴다.

## Q. CV 전략은? (가장 중요)
- **답**: {{StratifiedKFold / GroupKFold / TimeSeriesSplit / KFold}}, {{N}}-fold, seed=42.
- **근거**: train/test 분할 방식 = {{row-level / group-level / 시간순}}. → {{왜 이 CV 가 대회 셋업과 일치하는지}}.
- ⚠️ 그룹/시간 누수 위험: {{있으면 GroupKFold/TimeSeriesSplit, cv.py 교체}}.

## Q. 지표와 제출 형식은?
- **답**: {{ROC-AUC / RMSE / ...}}, 제출 = {{확률 / 클래스 / 수치}}.

## Q. 컬럼 분류 (config.py)
- ID: `{{ID_COL}}` / 타깃: `{{TARGET_COL}}`
- 범주형: {{...}} / 정수범주: {{...}} / 수치형: {{...}}
- 그룹 키: {{GROUP_KEYS}} / 시퀀스 키: {{SEQUENCE_COL}}

## Q. 클래스 불균형 처리?
- **답**: {{지표가 순위 기반이면 가중 기본 미사용, on/off 실험 비교}}.

## Q. 실험 ID 컨벤션?
- **답**: `exp_<NNN>_<short-slug>` 연번. **끝까지 일관**(중간 변경 금지).

## Q. 측정 검정력 (필수 인지)
- fold std σ≈{{측정값}} → SE≈{{ }}. **|Δ| < ~2·SE 결정은 단일 시드로 판정 금지** → 다중 시드 또는 stack-add/잔차 프레임.
