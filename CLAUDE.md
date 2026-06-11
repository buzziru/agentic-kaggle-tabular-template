# CLAUDE.md — ML 프로젝트 템플릿

> **이 파일은 템플릿이다.** 새 ML/Kaggle 프로젝트를 시작할 때 이 파일을 프로젝트 루트로 복사하고
> `{{...}}` 플레이스홀더를 채운 뒤 이 인용 블록을 지운다. `{{...}}` 가 아닌 본문(워크플로우·프로세스
> 규율·검증 원칙·코딩 컨벤션)은 **이유 없이 바꾸지 않는다**. 프로젝트 성격에 안 맞는 섹션만 잘라낸다.
> 템플릿 자체에 동작하는 스캐폴드(`src/`, `conf/`, `.claude/agents/`, `docs/wiki/`)가 들어 있다.

---

{{프로젝트 한 줄 설명}}. 바이브 코딩 방식으로 진행한다.

## 역할

{{도메인}} 수준의 ML 엔지니어이자 프로젝트 아키텍트. (예: "Kaggle Grandmaster 수준의 ML 엔지니어")

## 프로젝트 요약

- **URL / 컨텍스트**: {{대회 URL 또는 과제 출처}}
- **문제**: {{문제 유형 — 예: 이진 분류 / 회귀 / 다중분류}}, 타깃 = `{{TARGET_COL}}`
- **지표**: {{평가 지표 — 예: ROC-AUC}} (제출 형식: {{확률값 / 클래스 / 수치}})
- **제출**: `{{id_col, target_col}}` / {{제출 방법 — 예: Kaggle CLI}}
- **데이터**: train {{행×열}} / test {{행×열}} / 결측 {{유무}} / 타깃 분포 {{예: 양성률 19.9%}}
- 데이터·컬럼·누수 분석 상세: [docs/data_dictionary.md](docs/data_dictionary.md), [docs/setup_questions.md](docs/setup_questions.md), [docs/eda.md](docs/eda.md)

## 워크플로우

1. **EDA** — 주제별 노트북 `notebooks/eda_<NN>_<주제>.ipynb`(분석마다 **새 노트북**, append 금지)에서 탐색하고 결론은 [docs/eda.md](docs/eda.md) 에 **수치 요약**으로 정리한다. 의존성 `uv sync --extra eda`, 셀·네이밍·sys.path 규칙은 [notebook_conventions](docs/wiki/notebook_conventions.md).
2. **피처/모델링** — `src/` 중심의 `.py` 작업. 피처는 **오직 `src/features.py` 한 곳**에서 train/test 에 공통 적용한다(아래 "피처 엔지니어링 — 코드 파편화 방지").
3. **실행**
  - 베이스라인·중간 실험은 **로컬** `.py` 중심으로 돌린다 (`uv run python -m src.train model=lgbm ...`).
  - 대형 모델·장시간 튜닝은 **GPU 환경**을 쓴다. 환경별 런북 = [kaggle_jobs](docs/wiki/kaggle_jobs.md)·[lightning_jobs](docs/wiki/lightning_jobs.md)·[colab_jobs](docs/wiki/colab_jobs.md). 환경 비교·선택 기준은 [README.ko.md](README.ko.md) 의 "GPU 실행 / 인프라" 표 참조(중복 방지: 표는 거기 한 곳).
  - ⚠️ 노트북 환경엔 `src/` 코드를 `.ipynb` 변환 또는 Dataset push 후 import 한다. 작성 규칙(생성기·단일 진실원·fast-fail)은 [notebook_conventions](docs/wiki/notebook_conventions.md). 반복 오류는 즉시 가드로 코드화(아래 "외부 인프라 가드").
4. **실험 결과** — `experiments/logs/<exp_id>.json` 에 구조화 로그를 남긴다(+ W&B, 아래 "실험 추적").

## 서브에이전트 & 의사결정 기록

커스텀 에이전트(`.claude/agents/`, git 추적)는 **격리형 탐색/검증에만** 쓴다. 학습 루프·실험 비교·최종 판단은 동일 fold/seed 를 보장하기 위해 **메인에서 순차**로 한다. 참조 구현이 쓰는 에이전트:

- `eda-explorer` — read-only EDA. 주제별 노트북을 생성하고 **수치 요약만 리턴**한다(토큰 절약).
- `feature-smith` — `src/features.py` 피처 구현 + 누수 검증 + OOF(Out-of-Fold) 측정. 단일 파일을 건드리므로 **동시 1개만**.
- `code-reviewer` — 구현 직후·풀 실행 전 [docs/checklists/code_review.md](docs/checklists/code_review.md) 항목별 pass/block 대조. 코드 미수정, 하나라도 block 이면 BLOCK. 작성자와 분리(self-review 방지).
- `result-reviewer` — 풀 실행 로그를 사전등록 expectation 과 대조해 4종 판정(confirmed/refuted/inconclusive/invalid). 측정 검정력 규칙 내장(|Δ|<~2·SE 는 단일시드 판정 금지). 기록=`docs/wiki/experiments/judgments/`. **다음 실험 제안 금지**.
- `premise-auditor` — 판정 5건마다 blind 전제 감사(숫자만 입력, rationale 미열람). 공격 가설 3개+최저비용 반증. 기록=`docs/wiki/audits/`. kill/continue 판정은 사용자 몫(천장 게이트 기계화).
- `exp-runner` — 헤드리스 원격 실행(기본 Kaggle GPU, Lightning/Colab 디스패치). 풀 실행 전 expectation 커밋 + code-reviewer PASS 확인. {{필요 시 도메인 리서치 등 추가}}

⚠️ 풀 실험 1사이클: 설계+expectation 커밋 → (구현) → **code-reviewer** PASS → **exp-runner** 실행 → **result-reviewer** 판정 → 5건마다 **premise-auditor**. 판정·감사 후 방향은 사용자가 정한다.

주요 결정은 **[docs/wiki/decisions.md](docs/wiki/decisions.md)(ADR-lite)** 에 기록한다 — 새 결정마다 "왜 그렇게 정했는지"를 남긴다.

## 실험 우선순위 — 천장 게이트 (과몰입/토끼굴 가드, 필수)

포화 영역에서 마진 레버에 과투자하는 것이 가장 흔한 재발 약점이다. EV(기대값)는 **채택** 판단뿐 아니라 **선택·우선순위**에도 적용한다.

- **레버군(트랙) 개시 전**: **천장 추정 vs 목표 격차**를 1줄로 등록한다. 천장 < 격차면 **보조로 강등**한다(싼 것·병렬만, 주스레드 금지 — 임계경로 보호).
- **트랙의 2번째 이후 실험 전**: 어시스턴트는 **"이 레버의 천장이 격차를 덮는가?"를 challenge** 하고 kill/continue 의견을 낸다(patience: N연속 < ε 이면 종료 권고, 타임박스 병기).
- ⚠️ **공개/SOTA 단일 점수보다 크게 뒤처진 멤버는 무조건 우선순위 최상위다.** "강화는 전이 0" 같은 **일괄 보류(park)**는 *천장 근처의 한계 튜닝*에만 적용한다(약체 멤버 방치 교훈).
- ⚠️ **결정 주체는 사용자다.** 어시스턴트는 규칙대로 결과를 보고하고 기각/park **의견만** 낸다. **임의 기각·중단·강등·발사는 금지** — 의견을 낸 뒤 사용자 결정을 기다린다.
- **케이던스**: 이 게이트의 기계화가 `premise-auditor`(blind 전제 감사)다 — `docs/wiki/experiments/judgments/` 판정 **5건마다** main 이 트리거하고, kill/continue 는 사용자가 감사 리포트를 읽고 정한다.

## 프로세스 규율 (필수) — 운영 부채 방지

- **커밋**: 커밋 단위는 **응집된 판정/기능/문서셋**으로 한다. 작은 변경 여럿은 **하나로 묶어** 커밋한다(per-task·per-edit 커밋 금지). 케이던스는 의미 단위 완료 시 또는 세션 끝이며, **세션 끝에는 미커밋 *의미* 변경이 0** 이어야 한다.
  - 작업 추적 SSOT = [TASK.md](TASK.md)(마일스톤·분할 작업) + `docs/tasks/<id>.md`(세부계획) {{또는 GitHub Issues}}.
  - 세션 핸드오프 = [CURRENT_STATUS.md](CURRENT_STATUS.md)(세션 끝마다 갱신).
  - 안전망: `.claude/settings.json` 의 Stop 훅이 `scripts/hooks/stop_reminder.py` 를 호출해, 미커밋 tracked 파일이 ≥8 이면 리마인드한다.
- **실험 ID 컨벤션**: 프로젝트 시작 시 규칙 하나를 고정하고(`exp_<NNN>_<short-slug>` 연번 권장) **끝까지 일관**되게 쓴다(중간 변경 금지).
- **회고 의무**: 레버/트랙을 종료할 때 해당 실험군 회고를 `docs/wiki/experiments/exp_*.md` 에 작성해야 트랙을 close 한다(가설→결과→결론, 수치+근거). 누락 금지.
- **외부 인프라 가드**: 반복되는 환경 오류(Kaggle/Colab/GPU 등)는 **1회 발생 시 즉시 재사용 가드로 코드화**한다(코드 생성기·모니터·fast-fail). 가드 없이 N회 반복은 금지다.
- **expectation 게이트**: 풀 실행은 `specs/<exp_id>/expectation.yaml`(mechanism/predicted/falsification) 을 **실행 전 커밋**해야 한다. `guard_bash.py` 훅이 커밋·작업트리 일치를 검사한다. ⚠️ **스크리닝(`max_folds=`)은 면제** — 사전등록은 풀 실행에만 적용. 템플릿은 [docs/templates/expectation.yaml](docs/templates/expectation.yaml). 작성 직전 [scripts/decision_card.py](scripts/decision_card.py) 로 같은 그룹/모델의 노이즈(~2·SE)를 확인해 임계 설정을 돕는다(읽기 전용·제안만·자동 채움 없음).
- **가드 티어**: 훅 가드는 T0(하드·우회불가)/T1(기록 우회)/T2(경고) 3티어다. T1은 `specs/<exp_id>/override_<guard>.md` 를 커밋하면 통과+로깅(`docs/wiki/guard_overrides.jsonl`) — 훅을 통째로 끄는 것보다 좁고 추적된다. 오차단은 훅을 끄지 말고 `conf/guard/*.txt` 패턴 한 줄로 고친다. 상세 = [docs/wiki/guard_tiers.md](docs/wiki/guard_tiers.md).

## 프로젝트 구조

전체 트리와 각 파일 역할은 [README.ko.md](README.ko.md), 전체 데이터 흐름 그림은 [ARCHITECTURE.md](ARCHITECTURE.md)(Mermaid) 를 참조한다. 핵심 아키텍처만:

- `src/train_common.py` ★ — 공유 OOF CV 스캐폴드. 모든 모델의 단일 엔진 `run_oof_cv(cfg, trainer)` 가 여기 있다. `src/train.py` 가 통합 진입점이고, `src/registry.py` 가 `model.name`→Trainer 클래스를 선택한다. `train_lgbm.py`(LGBM)·`train_xgb.py` 는 그 어댑터(`ModelTrainer` 구현)이고, `stack.py` 는 OOF 계약만 소비한다.
- `src/{config,data,features,encoders,cv,utils,eda_utils}.py` · `conf/`(Hydra 노브) · `scripts/`(게이트) · `docs/wiki/`(런북·결정·회고).
- `experiments/`(logs·oof·submissions)·`data/` 는 내용물을 git 에서 제외한다.
- [TASK.md](TASK.md)(계획·분할 작업) · [CURRENT_STATUS.md](CURRENT_STATUS.md)(세션 핸드오프) · [docs/tasks/](docs/tasks/)(작업별 세부계획).

## 검증 전략

- ⚠️ **CV 전략·fold 수·seed 를 데이터에 맞게 확정**하고 `src/config.py` 에 둔다(예: StratifiedKFold 5-fold, seed=42). `CV_STRATEGY` 는 `cv.get_folds` 가 디스패치하는 실제 선택자다 — 그룹 누수 위험이면 GroupKFold, 아니면 (Stratified)KFold(공식 지원: Stratified/KFold/Group; 시계열은 미지원 = full-OOF 계약과 불일치, 필요 시 직접 분기). 핵심은 **train/test 분할 방식과 일치**시키는 것 — 근거는 [docs/setup_questions.md](docs/setup_questions.md).
- 모든 모델 비교는 **동일 fold(동일 seed) 기준 OOF 점수**로 한다. `cv.get_folds` 는 레짐별 분할을 `data/splits/{strategy}_{n}fold_seed{seed}.parquet` 로 **직렬화(로드-우선)**해 공유 코드 리팩토링에도 분할을 동결한다(5/7/10-fold 공존). ⚠️ 원격 실행엔 splits 파일을 동반해 로드 경로를 태운다(sklearn 버전차로 즉석 생성 시 분할이 달라질 수 있음).
- ⚠️ **측정 검정력의 한계를 인지한다(필수).** fold 간 std 로 SE 를 추정하고, **|Δ| 가 단일-시드 탐지 임계(~2·SE)보다 작은 결정은 단일 시드로 판정하지 않는다** — 다중 시드로 SE 를 줄이거나, 잔차/stack-add 프레임(노이즈 위에서 판정)으로 본다. 작은 차이를 노이즈에서 '음성'으로 오판하는 것이 흔한 실수다.
- ⚠️ **OOF≈LB 는 단일 모델에 한정된 가정이다.** 스태커는 별개 레짐이라 meta-OOF 가 held-out 보다 낙관적일 수 있다. 따라서 스택 멤버 추가는 in-sample meta-OOF 가 아니라 held-out/nested 로 판정하고, 멤버 증가의 meta-overfit 비용을 함께 본다.

## 모델링

- **베이스라인**: 단순·빠른 모델 하나(예: LightGBM CPU)로 파이프라인 전체(로드→피처→CV→제출)를 먼저 닫는다.
- **고카디널리티 범주형**: 누수 방지 OOF 타깃 인코딩(TE, `src/encoders.py`, fold-내 fit)과 native categorical 중에서 실험으로 고른다.
- **클래스 불균형/가중**: 지표 특성에 따라 결정한다. 순위 기반 지표(AUC 등)에선 가중이 무의미하거나 해로울 수 있으니 on/off 를 실험으로 비교한다.
- **권장 순서**: 베이스라인 → 모델 다양성 → 스태킹/블렌딩 → (마지막) 개별 하이퍼파라미터 튜닝. 다양성과 앙상블의 ROI 가 단일 모델 튜닝보다 크다(참조 프로젝트 ADR).

## 스택 풀 & 트레이너 구조 — 모델 추가 = 어댑터 (필수)

모델마다 학습 루프(seed/CV/OOF-TE/증강/저장/로그)를 **복제**하면 풀이 커질수록 모델이 특정 코드에 묶여 리팩토링이 막힌다(참조 회고: LGBM 별도 경로 → 노브 divergence 반복 → 패리티 게이트까지 필요). 처음부터 어댑터로 막는다.

- **단일 스캐폴드 + 어댑터**: 공통 골격은 `src/train_common.py` 의 `run_oof_cv(cfg, trainer)` **한 곳**에만 둔다. 새 모델은 골격을 복사하지 말고 `src/train_<model>.py` 에서 `ModelTrainer`(`src/registry.py`) 인터페이스(`prepare`/`fit`/`predict`/`get_metadata`/`save_model`)를 구현하고 `_REGISTRY` 에 한 줄 등록한다 → 한 모델 추가 ≈ 40줄(LGBM·XGB 예시). 골격을 고치면 **모든 모델이 한 번에** 따라온다(별도 경로·모델 분기 금지 = divergence 근원). 전체 흐름은 [ARCHITECTURE.md](ARCHITECTURE.md).
- **OOF 계약 = 디커플링 경계**: 모든 모델은 동일 fold(seed)로 `experiments/oof/<exp_id>.csv`=`[id, oof]`, `submissions/<exp_id>.csv`=`[id, <target>]` 만 산출한다. `src/stack.py` 는 **이 계약만** 소비 → 멤버가 늘어도 스택에 모델별 특수 코드가 0.
- **frozen 멤버 OOF 불변(필수)**: 스택 풀에 든 OOF 는 **동결**된다 — `features.py`/`train_common` 리팩토링이 frozen 멤버 OOF 를 바꾸면 같은-fold 정합이 깨져 풀이 무효가 된다. 리팩토링 전후로 **`scripts/check_fold_inputs.py`** 로 fit/predict 입력이 바이트 동일한지 검증한다(GPU·실학습 불필요 — 입력 같으면 결정적 모델이라 OOF 동일 보장).
- **모델별 FE 는 conf 훅으로**: 모델 전용 피처는 `features.py` 에 `add_<x>_features` 로 두고 `conf/features/*.yaml` 의 `feature_builder` 로 켠다(코드 포크 금지).

## 피처 엔지니어링 — 코드 파편화 방지 (필수)

피처는 시행착오로 **수없이 수정·추가·폐기**된다. 흩뿌리면 어느 버전이 실제로 도는지 알 수 없게 된다(참조 회고의 "2중 사본 drift"·"복사-템플릿 상속" 재발 버그 — [docs/wiki/notebook_conventions.md](docs/wiki/notebook_conventions.md)). 아래 규율로 막는다.

- **단일 진입점 + 순수 함수**: 모든 피처는 `src/features.py` 의 `build_features()` 한 곳에만 둔다(노트북·학습 스크립트에 일회성 변환 금지 — train/test 가 갈리고 누수의 근원). 각 헬퍼는 df→df 부수효과 없는 작은 함수로 쪼개고 `build_features()` 는 조립만 한다.
- **코드 분기 대신 설정 토글**: 변형은 `build_features()` 복제·포크 대신 **`conf/features/*.yaml` 노브**로 켜고 끈다. 함수 내 분기도 `cfg` 로 받되 `config.X` 를 기본값으로 둔다.
- **죽은 피처는 즉시 제거**: 기각된 피처는 트랙 종료 시 `build_features()` 에서 삭제하고 결론만 [docs/feature_engineering.md](docs/feature_engineering.md)·회고에 남긴다("혹시 몰라" 주석 코드 금지).

## 실험 추적 — 단일 스트림 (필수)

base·스택·앙상블을 **한 로그 스트림**(`experiments/logs/<exp_id>.json`, `utils.log_experiment` 자동)으로 모아 비대칭을 없앤다. OOF/submission 균일 계약·Feature Registry(`params.feature_recipe`)·Submission History·통합 뷰(`scripts/summarize.py`)·W&B 경로 상세는 [docs/wiki/experiment_tracking.md](docs/wiki/experiment_tracking.md).

- ⚠️ **best_iter 로깅(필수)**: early-stopping 모델은 fold별 `best_iter` 를 기록·검수한다. cap 에 붙으면(미발화) **미완 학습 신호**이므로 cap 을 올려 재학습한다 — 점수 신뢰 전 수렴 확인.

## 토큰 절약 원칙 (필수 준수)

- DataFrame 출력은 `.head(5)` / `.shape` / `.dtypes` / `.isnull().sum()` 만 허용한다(→ `utils.resumetable(df)` 요약 표 사용).
- 플롯은 **EDA 단계에서만** 생성하고 이후엔 수치 요약으로 대체한다. EDA 플롯은 작게(figsize ≤ 8×4, dpi 72), 개수도 절제한다.
- 결론은 **이미지가 아니라 계산한 수치**에서 도출한다.

## 코딩 컨벤션

- **Python 버전 고정**(`.python-version`, 실행 환경과 동일). 의존성 관리는 **`uv`**(`uv sync` / `uv run`).
- **타입힌트 필수**, **Google 스타일 docstring**, 함수당 50줄 내외 권장.
- **하드코딩 금지** — 경로/시드/컬럼 등 구조적 상수는 `src/config.py` 에 둔다.
  - ⚠️ config 상수라도 로직에서 직접 참조하면(예: `x / config.N_FOLDS`) 그것도 하드코딩이다. 실험에서 바꿀 값은 **`cfg` 파라미터로 받되 `config.X` 를 기본값으로** 둔다(override 가능). 즉 config 는 기본값 공급원, 동작 분기는 cfg(Hydra)다.
- **노트북 셀 규칙**: `;` 다중문 금지(setup 셀 예외) · 논리 블록 사이 빈 줄 · 셀당 단일 책임 · full 실행 전 소규모 fast-fail.
- **재현성**: 모든 실험은 `seed_everything()` + 커밋 해시 로깅(`utils.log_experiment` 자동).
- **문서 가독성·간결**: 문서는 한 문장·한 불릿에 한 가지만 담는다. 이유·원칙은 완전한 문장으로, 참조·열거·명령은 간결체로 쓰고, 절을 `·` 로 길게 잇는 run-on 을 피한다. 링크·코드는 올바른 마크다운으로 표기한다. ⚠️ **CLAUDE.md 는 매 세션 로드되므로 비대화 금지** — 상세 설명·런북은 `docs/wiki/` 로 빼고 여기엔 포인터만 둔다.

## 실행 예시

설치·다운로드·제출 전체는 [README.ko.md](README.ko.md). 학습(Hydra: OOF + 제출 + JSON 로그 + W&B):

```bash
uv run python -m src.train model=lgbm exp_id=exp_001 "notes='baseline'"
#  오버라이드: model=<name>(xgb 등) · model.params.<key>=<val> · features=<yaml> · use_wandb=false
#  ⚠️ notes 의 공백·특수문자는 작은따옴표로: "notes='...'"
```

## 보안

- `.env` 와 인증 키 파일은 시크릿이다 → `.gitignore` 로 제외된다. **절대 커밋하지 않는다.**
- 인증 정보는 `.env` 에서 로드한다(`utils.load_env`). 코드·노트북·로그에 키를 노출하지 않는다.

