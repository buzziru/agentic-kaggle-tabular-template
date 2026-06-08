# ML / Kaggle 프로젝트 템플릿

[English](README.md) | **한국어**

Kaggle 대회 완주에서 검증한 구조와 규율을 일반화한 ML/Kaggle 프로젝트 템플릿이다.
빈 보일러플레이트가 아니라 **동작하는 스캐폴드**와 **실전 회고에서 도출한 프로세스 가드**를 함께 담았다.
새 프로젝트는 이 리포를 복제하고 플레이스홀더만 채우면 바로 시작할 수 있다.

> **이 README 는 템플릿 리포 자체를 설명한다.** 각 프로젝트가 쓰는 README 는
> [docs/PROJECT_README.template.md](docs/PROJECT_README.template.md) 에 따로 있다 (시작할 때 루트로 복사).

> **동작 확인:** `uv sync && uv run python examples/run_example.py` — 더미 데이터로
> 전체 파이프라인(로드→피처→CV→OOF→제출→로그)이 도는 것을 바로 확인한다. 자세히는 [examples/](examples/).

## 왜 이 템플릿인가

- **동작하는 스캐폴드.** CV·OOF(Out-of-Fold)·로깅·제출 같은 학습 공통 골격은 `src/train_common.py` 한 곳에만 둔다. 새 모델은 이 골격을 복제하지 않고, 모델별로 다른 두 콜백(`prepare`=범주형 전처리, `fit_predict`=학습/예측)만 정의하면 된다. 덕분에 **모델 하나 추가가 약 40줄**로 끝나고, 골격을 고치면 모든 모델에 한 번에 반영된다. LightGBM(`src/train_lgbm.py`)·XGBoost(`src/train_xgb.py`)가 그 예시다.
- **회고 기반 프로세스 가드.** 실전에서 반복된 실패를 코드와 체크 게이트로 박제했다. 예컨대 같은 코드의 2중 사본이 어긋나는 문제, 설정 노브 불일치, 동결돼야 할 OOF 가 바뀌는 문제, 효용 낮은 레버에 대한 과투자다. 대응 장치로 `scripts/check_fold_inputs.py`(OOF 불변 검증), Stop 훅 커밋 리마인더, 천장 게이트(과몰입 방지)를 둔다.
- **누수 안전 기본값.** fold 내부에서만 fit 하는 OOF 타깃 인코딩(`src/encoders.py`), 행 단위 피처 예시와 그룹/시계열 과거-only 레시피([docs/feature_engineering.md](docs/feature_engineering.md)), CV-분할 일치 원칙을 처음부터 강제한다.
- **단일 스트림 실험 추적.** 모든 모델·스택·앙상블이 동일한 JSON 로그와 OOF/제출 계약을 거친다. `scripts/summarize.py` 로 전체를 한 리더보드에 모아 본다.
- **Claude Code 네이티브.** 커스텀 서브에이전트(`.claude/agents/`), 토큰 절약 규율, 자동 리마인더 훅을 기본 탑재해 AI 에이전트와의 협업을 전제로 설계했다.
- **모던 툴링.** 의존성은 `uv`, 실험 노브는 `Hydra`, 실험 추적은 선택적으로 W&B.

## 핵심 설계 원칙


| 원칙                                          | 강제 위치                                              |
| ------------------------------------------- | -------------------------------------------------- |
| 피처는 단일 진입점에만 (`build_features`) — 코드 파편화 방지 | `src/features.py`, `conf/features/*.yaml` 노브       |
| 모델 추가 = 어댑터 (스캐폴드 복제 금지)                    | `src/train_common.py` + `src/train_<model>.py`     |
| frozen 멤버 OOF 불변                            | `scripts/check_fold_inputs.py`                     |
| 결정 근거 기록 (ADR-lite) + 트랙 종료 회고 의무           | `docs/wiki/decisions.md`, `docs/wiki/experiments/` |
| 측정 검정력 인지 (작은 Δ 단일 시드 판정 금지)                | `docs/setup_questions.md`, 검증 전략                   |


각 원칙의 배경과 근거는 [CLAUDE.md](CLAUDE.md) 에 모두 담겨 있다 (AI 에이전트·사람 공용 상시 가이드).

## 구조

```
CLAUDE.md          # 프로젝트 상시 가이드 (규칙·구조·실행법) — 새 프로젝트로 복사해 채움
TASK.md            # 마일스톤·분할 작업 인덱스
CURRENT_STATUS.md  # 세션 핸드오프 (현재값·다음 액션 SSOT)
conf/              # Hydra 설정 — 튜닝/실험 노브 (config.yaml, model/, features/)
src/               # config·data·features·encoders·cv·train_common·train_lgbm·train_xgb·stack·predict·utils
scripts/           # 게이트·집계 (check_fold_inputs, summarize, hooks/)
docs/              # data_dictionary · eda · feature_engineering · setup_questions
docs/wiki/         # 결정 기록(ADR-lite) · 실험 회고 · 인프라 런북(Kaggle/Colab/Lightning)
docs/PROJECT_README.template.md  # 새 프로젝트가 복사해 채우는 README 템플릿
.claude/agents/    # 커스텀 서브에이전트 (eda-explorer · feature-smith · kaggle-runner)
kaggle/            # 헤드리스 GPU 실행 자산 (gen_kernel·monitor·push)
experiments/       # logs(JSON) · oof · submissions  (내용물 git 제외)
```

## 이 템플릿으로 새 프로젝트 시작하기

```bash
# 1) 템플릿 사용 (GitHub 'Use this template' 또는 복제)
git clone <this-repo> my-project && cd my-project

# 2) 프로젝트 README 준비 — 템플릿을 루트로 복사 후 채움 (이 랜딩 README 는 덮어씀)
cp docs/PROJECT_README.template.md README.md

# 3) 의존성 (uv)
uv sync                              # eda/gpu 추가: uv sync --extra eda --extra gpu
#  → 생성된 uv.lock 을 프로젝트에 커밋 (lock 은 프로젝트별 산출물; 템플릿엔 미포함)

# 4) 인증
cp .env.example .env                 # KAGGLE_USERNAME/KAGGLE_KEY/WANDB_API_KEY 채우기

# 5) 채워야 할 곳
#    - CLAUDE.md      : 상단 인용 블록 안내대로 {{...}} 채우고 블록 삭제
#    - src/config.py  : ID/TARGET/컬럼 정의·METRIC·COMPETITION·CV 전략
#    - docs/setup_questions.md : CV 전략 근거 등 셋업 결정

# 6) 베이스라인 (Hydra: OOF + 제출파일 + JSON 로그 + W&B)
uv run python -m src.train_lgbm exp_id=exp_001 "notes='baseline'"
uv run python scripts/summarize.py   # 실험 리더보드
```

## 워크플로우

먼저 베이스라인으로 파이프라인 전체(로드→피처→CV→OOF→제출)를 한 번 닫는다. 그다음 아래 순서로 진행한다.

**EDA → 베이스라인 → 피처 → 모델 다양성 → 스태킹/블렌딩 → (마지막) 튜닝**

다양성과 앙상블이 단일 모델 튜닝보다 ROI 가 크다는 것이 참조 프로젝트의 핵심 교훈이다.
마일스톤·게이트는 [TASK.md](TASK.md), 세션 핸드오프는 [CURRENT_STATUS.md](CURRENT_STATUS.md) 를 참조한다.

## GPU 실행 / 인프라

베이스라인과 중간 실험은 로컬 CPU(`uv run python -m src.train_lgbm ...`)로 돌리고, 대형 모델·장시간 튜닝만 GPU 로 오프로드한다. 참조 프로젝트는 **Lightning AI Studio** 환경에서 진행했고, GPU 활용을 위해 세 가지 실행 경로를 문서화해 두었다. 같은 `src/` 코드가 환경만 바꿔 그대로 돌아가며, 각 런북에 운영 이슈와 실전 교훈이 정리돼 있다.


| 경로                   | GPU       | 비용           | 실행 방식                                      | 언제 쓰나                                  | 런북                                               |
| -------------------- | --------- | ------------ | ------------------------------------------ | -------------------------------------- | ------------------------------------------------ |
| **Lightning AI Job** | T4~H200   | 크레딧 과금       | 헤드리스 (`src/` 코드 그대로 Job 제출)                | wandb online·반복/통합 라운드 (베이스 프로젝트 주 환경) | [lightning_jobs.md](docs/wiki/lightning_jobs.md) |
| **Kaggle GPU 커널**    | T4 / P100 | 무료 쿼터(주간 한도) | 헤드리스 (`src` Dataset push 후 `kernels push`) | torch 외 모델·단발 실행                       | [kaggle_jobs.md](docs/wiki/kaggle_jobs.md)       |
| **Colab**            | L4 24GB   | Pro/PAYG     | 노트북 직접 업로드·UI 실행 (헤드리스 아님)                 | Kaggle T4 16GB 로 OOM 이고 L4 면 해결되는 모델   | [colab_jobs.md](docs/wiki/colab_jobs.md)         |


선택 기준 비교표는 `kaggle_jobs.md` 의 "Kaggle vs Lightning Job" 섹션이 단일 출처다. 노트북 변환·실행 규칙은 [notebook_conventions.md](docs/wiki/notebook_conventions.md) 를 참조한다.

## 보안

`.env`·`kaggle.json` 등 시크릿은 `.gitignore` 로 제외된다. **절대 커밋하지 않는다.**
인증 정보는 `.env` 에서 로드한다(`src/utils.load_env`).

## 라이선스

[MIT](LICENSE)