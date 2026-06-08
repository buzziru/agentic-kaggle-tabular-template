# {{프로젝트 이름}}

{{한 줄 설명}}. 바이브 코딩 방식.

> 이 저장소는 완성 참조 구현(Kaggle PS S6E5, Private 0.95460·상위 4.9%)에서 검증된 구조를 일반화한 **ML/Kaggle 프로젝트 템플릿**이다. 새 프로젝트는 `{{...}}` 플레이스홀더를 채우고 시작한다.

## 📊 문제 요약
| 항목 | 내용 |
|---|---|
| 문제 | {{이진 분류 / 회귀 / ...}} |
| 지표 | {{ROC-AUC / RMSE / ...}} |
| 데이터 | train {{행×열}} / test {{행×열}} |
| 타깃 | `{{TARGET_COL}}` |
| 제출 | `{{id, target}}` ({{Kaggle CLI}}) |

## 🗂️ 구조
```
CLAUDE.md          # 프로젝트 상시 가이드 (규칙·구조·실행법)
TASK.md            # 마일스톤·분할 작업 인덱스 (세부계획은 docs/tasks/ 포인터)
CURRENT_STATUS.md  # 세션 핸드오프 (현재값·다음 액션 SSOT)
conf/              # Hydra 설정 — 튜닝/실험 노브 (config.yaml, model/, features/)
src/               # config·data·features·encoders·cv·train·predict·utils·eda_utils
docs/              # data_dictionary · eda · feature_engineering · setup_questions
docs/wiki/         # 결정 기록(ADR-lite) · 실험 회고 · 인프라 런북
kaggle/            # 헤드리스 GPU 실행 자산 (gen_kernel·monitor·push)
experiments/       # logs(JSON) · oof · submissions  (내용물 git 제외)
```

## 🚀 시작하기
```bash
# 1) 의존성 (uv)
uv sync                              # eda/gpu 추가: uv sync --extra eda --extra gpu
#  → 생성된 uv.lock 을 이 프로젝트에 커밋한다 (lock 은 프로젝트별 산출물; 템플릿엔 미포함).

# 2) 인증
cp .env.example .env                 # KAGGLE_USERNAME/KAGGLE_KEY/WANDB_API_KEY 채우기

# 3) 데이터 다운로드 (.env 인증)
set -a; . ./.env; set +a
kaggle competitions download -c {{COMPETITION_SLUG}} -p data/ && \
  unzip -o "data/*.zip" -d data/ && rm data/*.zip

# 4) 학습 (Hydra: OOF + 제출파일 + JSON 로그 + W&B)
uv run python -m src.train exp_id=exp_001 "notes='baseline'"
#  타깃 인코딩: features=te_example / 파라미터: model.params.num_leaves=127 / W&B off: use_wandb=false

# 5) 제출
kaggle competitions submit -c {{COMPETITION_SLUG}} \
  -f experiments/submissions/exp_001.csv -m "exp_001 baseline"
```

## 🔬 검증 전략
- {{CV_STRATEGY}} {{N}}-fold, seed=42 (단일 seed → 최종에만 seed averaging)
- 모든 비교는 **동일 fold OOF 점수** 기준. CV 는 train/test 분할 방식과 일치시킨다 (`docs/setup_questions.md`).

## 📋 워크플로우
- **EDA** → 주제별 노트북 + `docs/eda.md` · **피처** → `src/features.py` 단일 진입점
- **모델링** → 베이스라인 → 다양성 → 스태킹 → (마지막) 튜닝
- 작업 추적 = {{GitHub Issues}} · 지식 = `docs/wiki/` · 상시 가이드 = `CLAUDE.md`

## 🔒 보안
`.env`·`kaggle.json` 은 시크릿 → `.gitignore` 제외. 절대 커밋 금지.
