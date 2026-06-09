"""프로젝트 전역 설정: 경로, 시드, 컬럼 정의, CV 파라미터 (구조적 상수).

모든 모듈은 하드코딩 대신 이 파일의 상수를 참조한다.
튜닝/스윕 대상 하이퍼파라미터(모델 params, 타깃 인코딩 smoothing 등)는 여기가 아니라
`conf/` (Hydra) 에 둔다 — `src/train_lgbm.py` 참조.

▶ 새 프로젝트에서 채울 것: 컬럼 정의(ID/TARGET/CATEGORICAL/NUMERIC/GROUP_KEYS),
  METRIC, COMPETITION, WANDB_PROJECT.
"""

from __future__ import annotations

from pathlib import Path

# ===== 경로 =====
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
EXPERIMENTS_DIR: Path = ROOT_DIR / "experiments"
LOG_DIR: Path = EXPERIMENTS_DIR / "logs"
OOF_DIR: Path = EXPERIMENTS_DIR / "oof"
SUBMISSION_DIR: Path = EXPERIMENTS_DIR / "submissions"
MODEL_DIR: Path = EXPERIMENTS_DIR / "models"  # 적합 모델 저장 (save_models=true 일 때만)

TRAIN_PATH: Path = DATA_DIR / "train.csv"
TEST_PATH: Path = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH: Path = DATA_DIR / "sample_submission.csv"
# (선택) 외부 증강 원본 — train 증강 전용(검증/제출엔 미사용). 없으면 augment 비활성.
SOURCE_AUG_PATH: Path = DATA_DIR / "external" / "source.csv"

# ===== 재현성 =====
SEED: int = 42

# ===== 컬럼 정의 (데이터에 맞게 채울 것) =====
ID_COL: str = "id"
TARGET_COL: str = "target"

# native categorical 로 처리할 범주형 (예: ["brand", "city"]).
CATEGORICAL_COLS: list[str] = []
# 저카디널리티 정수형 범주 (범주로 취급 가능, 예: ["year", "month"]).
CATEGORICAL_INT_COLS: list[str] = []
# 수치형 (예: ["amount", "duration"]).
NUMERIC_COLS: list[str] = []

# (선택) 시퀀스/그룹 식별 키 — 시계열·그룹 누수 점검과 그룹 기반 피처에 사용.
# 시계열/그룹 구조가 없으면 [] 로 둔다. (예: ["user_id", "session"])
GROUP_KEYS: list[str] = []
# (선택) 그룹 내 정렬 기준(과거만 참조하는 시퀀스 피처용, 예: "lap" / "timestamp").
SEQUENCE_COL: str | None = None

# ===== CV 설정 =====
# ⚠️ 정답은 데이터가 정한다 — train/test 분할 방식과 일치시킨다 (docs/setup_questions.md).
#    이 값은 cv.get_folds 가 디스패치하는 **실제 선택자**다(장식용 상수 아님). 지원 전략:
#   - "StratifiedKFold": 분류(계층화)              - "KFold": 회귀
#   - "GroupKFold": 그룹 누수 위험 (config.GROUP_KEYS 필요)
#   - "TimeSeriesSplit": 시계열 (시간순 정렬 전제)
# 아래 기본값은 이진분류 템플릿 가정일 뿐 — 프로젝트에 맞게 고른다.
CV_STRATEGY: str = "StratifiedKFold"
N_FOLDS: int = 5

# ===== 문제 유형 =====
# 템플릿은 binary/regression 을 1급 지원한다. 바꾸면 아래를 함께 맞춘다:
#   1) conf/model/*.yaml 의 objective/metric (예: regression+rmse)
#   2) METRIC (아래) + CV_STRATEGY(회귀는 보통 KFold) + 제출 형식
#   3) train_<model>.predict 출력: binary=predict_proba[:,1] / regression=predict (PROBLEM_TYPE 로 분기됨)
# ⚠️ multiclass 는 1-D OOF 계약(OOF 단일 열)을 넘어서므로 train_common 이 NotImplementedError 로
#    막는다 — 지원하려면 OOF/submission/stack 계약을 k열로 확장해야 한다(확장점).
# 점검 항목은 docs/setup_questions.md 체크리스트 참조.
PROBLEM_TYPE: str = "binary"  # binary / regression / (multiclass = 확장 필요)

# ===== 평가 지표 =====
# ⚠️ PROBLEM_TYPE 과 conf/model/*.yaml 의 objective/metric 을 일치시킬 것. scorer 는 utils.get_scorer 가 이 값으로 자동 결정.
METRIC: str = "auc"  # 예: auc / rmse / logloss / mae / accuracy

# ===== Kaggle =====
COMPETITION: str = "{{COMPETITION_SLUG}}"

# ===== W&B (실험 추적) =====
# 인증: .env 의 WANDB_API_KEY (utils.load_env 로 로드). 기본 활성, use_wandb=false 로 비활성.
WANDB_PROJECT: str = "{{WANDB_PROJECT}}"
WANDB_ENTITY: str | None = None  # None = 로그인 계정 기본 엔티티
