"""미니 스모크 인스턴스 — 더미 데이터로 템플릿 파이프라인 전체를 실제로 돌린다.

로드 → 피처 → CV → OOF → 제출파일 → JSON 로그까지, `src/` 코드를 **그대로** 실행해
"이 템플릿이 동작한다"를 한 명령으로 보여 준다. 경로만 `examples/_work/` 로 격리하므로
실제 `data/`·`experiments/` 는 건드리지 않는다.

실행:
    uv run python examples/run_example.py

이진분류 더미 데이터(2000/500행, 수치 피처 5개)를 결정적으로 생성하고
LightGBM 베이스라인(`src.train_lgbm`)을 5-fold OOF 로 학습한다. 신호가 심어져 있어
ROC-AUC 가 0.9 안팎으로 나오면 파이프라인이 정상이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Windows 콘솔(cp949)에서도 한글·이모지 출력이 깨지거나 크래시하지 않도록 UTF-8 강제.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402  (sys.path 설정 후 임포트)

# ── 1) 경로를 examples/_work 로 격리 (실제 data/·experiments/ 불변) ──────────────
WORK = ROOT / "examples" / "_work"
DATA = WORK / "data"
for d in (DATA, WORK / "oof", WORK / "submissions", WORK / "logs"):
    d.mkdir(parents=True, exist_ok=True)
config.DATA_DIR = DATA
config.TRAIN_PATH = DATA / "train.csv"
config.TEST_PATH = DATA / "test.csv"
config.SAMPLE_SUBMISSION_PATH = DATA / "sample_submission.csv"
config.OOF_DIR = WORK / "oof"
config.SUBMISSION_DIR = WORK / "submissions"
config.LOG_DIR = WORK / "logs"


def make_dummy(n: int, seed: int, *, with_target: bool) -> pd.DataFrame:
    """수치 피처 5개의 이진분류 더미 데이터를 결정적으로 생성한다.

    Args:
        n: 행 수.
        seed: 난수 시드 (재현성).
        with_target: True 면 `target` 컬럼 포함(train), False 면 제외(test).

    Returns:
        `id` + `f0..f4` (+ `target`) 컬럼의 DataFrame.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5))
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(5)])
    # 일부 피처에만 신호 → AUC 가 0.5 보다 확실히 높게 나오도록.
    logit = x @ np.array([1.3, -0.9, 0.6, 0.0, 0.4]) + rng.normal(scale=0.5, size=n)
    offset = 0 if with_target else 100_000  # train/test id 충돌 방지
    df.insert(0, "id", np.arange(n) + offset)
    if with_target:
        df["target"] = (logit > 0.0).astype(int)
    return df


def build_cfg() -> OmegaConf:
    """conf/ 기본값을 로드해 데모용 cfg 를 구성한다 (W&B off, 학습량 축소)."""
    model = OmegaConf.load(ROOT / "conf" / "model" / "lgbm.yaml")
    model.num_boost_round = 600  # 데모 속도 (기본 5000) — early-stopping 이 먼저 멈춘다
    model.early_stopping = 80
    features = OmegaConf.load(ROOT / "conf" / "features" / "base.yaml")
    return OmegaConf.create(
        {
            "exp_id": "exp_demo",
            "notes": "examples/ dummy smoke test",
            "use_wandb": False,  # 더미 실행은 W&B 끔 (인증·프로젝트 불필요)
            "n_folds": config.N_FOLDS,
            "max_folds": None,
            "kill_criterion": "",
            "seed": config.SEED,
            "augment": {"enabled": False, "weight": 1.0},
            "model": model,
            "features": features,
        }
    )


def main() -> None:
    # ── 2) 더미 데이터 생성 ──────────────────────────────────────────────
    train = make_dummy(2000, seed=0, with_target=True)
    test = make_dummy(500, seed=1, with_target=False)
    train.to_csv(config.TRAIN_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)
    test[["id"]].assign(**{config.TARGET_COL: 0.0}).to_csv(
        config.SAMPLE_SUBMISSION_PATH, index=False
    )
    print(f"[데이터] train={train.shape} test={test.shape} → {DATA}")
    print(f"[타깃] 양성률 = {train[config.TARGET_COL].mean():.3f}")

    # ── 3) 실제 파이프라인 실행 (src.train_lgbm) ─────────────────────────
    from src.train_lgbm import run  # noqa: E402  (경로 격리 후 임포트)

    result = run(build_cfg())

    # ── 4) 결과 요약 ─────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print(f"  OOF {config.METRIC.upper()} (mean) = {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    print("  산출물:")
    print(f"    OOF        : {config.OOF_DIR / 'exp_demo.csv'}")
    print(f"    submission : {config.SUBMISSION_DIR / 'exp_demo.csv'}")
    print(f"    log(JSON)  : {result['log_path']}")
    print("=" * 56)
    if result["cv_mean"] > 0.8:
        print("✅ 파이프라인 정상 (로드→피처→CV→OOF→제출→로그).")
    else:
        print("⚠️ AUC 가 예상보다 낮다 — 더미 신호/환경을 확인하라.")


if __name__ == "__main__":
    main()
