"""LightGBM 트레이너 — 공유 스캐폴드(train_common)의 어댑터. 베이스라인/표준.

⚠️ 모델 추가는 이 파일을 **복사하지 않는다**. 이 파일이 곧 "어댑터 패턴" 예시다 —
   새 모델은 src/train_<model>.py 에서 `ModelTrainer`(src/registry.py) 인터페이스를
   구현(prepare/fit/predict/get_metadata/save_model)하고 registry 에 등록한다 (예: src/train_xgb.py).
   공통 골격(seed/CV/OOF-TE/증강/저장/로그)은 train_common 단일 소스라,
   골격을 고쳐도 모든 모델이 한 번에 따라온다.

실행 (통합 진입점 — registry 가 model.name 으로 트레이너 선택):
    uv run python -m src.train model=lgbm exp_id=exp_001 "notes='lgbm baseline'"
    uv run python -m src.train model=lgbm exp_id=exp_002 features=te_example       # 타깃 인코딩
    uv run python -m src.train model=lgbm exp_id=exp_003 model.params.num_leaves=127
    uv run python -m src.train -m model=lgbm model.params.num_leaves=63,127,255     # 스윕

이 모듈 자체 실행도 가능(back-compat): uv run python -m src.train_lgbm exp_id=exp_001
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pandas.api.types import CategoricalDtype

from src import cat_prep, config
from src.train_common import run_oof_cv


class LGBMTrainer:
    """LightGBM 어댑터 (`ModelTrainer` 구현). native categorical 사용."""

    name = "lgbm"
    supports_weight = True

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.params = {
            **OmegaConf.to_container(cfg.model.params, resolve=True),
            "n_jobs": -1,
            "verbose": -1,
        }
        self.num_boost_round = cfg.model.num_boost_round
        self.early_stopping = cfg.model.early_stopping

    def log_extra(self) -> dict[str, Any]:
        return {"num_boost_round": self.num_boost_round}

    def prepare(
        self,
        x: pd.DataFrame,
        x_test: pd.DataFrame,
        x_src: "pd.DataFrame | None",
        cat_cols: list[str],
        aug_enabled: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, "pd.DataFrame | None", dict[str, CategoricalDtype]]:
        # native categorical 안정화 — 공유 헬퍼 단일 소스(복제 금지, src/cat_prep.py).
        return cat_prep.fix_categoricals(x, x_test, x_src, cat_cols)

    def fit(
        self,
        x_tr: pd.DataFrame,
        y_tr: pd.Series,
        x_va: pd.DataFrame,
        y_va: pd.Series,
        w_tr: "np.ndarray | None",
        cat_cols: list[str],
        cat_dtypes: dict[str, CategoricalDtype],
    ) -> lgb.Booster:
        x_tr = cat_prep.recast(x_tr, cat_cols, cat_dtypes)  # 증강 concat 후 고정 cat dtype 재적용
        dtrain = lgb.Dataset(x_tr, y_tr, categorical_feature=cat_cols, weight=w_tr)
        dvalid = lgb.Dataset(x_va, y_va, categorical_feature=cat_cols)
        return lgb.train(
            {**self.params, "seed": self.cfg.get("seed", config.SEED)},
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(self.early_stopping, verbose=False), lgb.log_evaluation(0)],
        )

    def predict(self, model: lgb.Booster, x: pd.DataFrame) -> np.ndarray:
        # binary objective=양성확률, regression objective=값 (objective 가 출력 형식을 결정 → PROBLEM_TYPE 무관).
        return model.predict(x, num_iteration=model.best_iteration)

    def get_metadata(self, model: lgb.Booster) -> dict[str, Any]:
        return {"best_iter": int(model.best_iteration)}

    def save_model(self, model: lgb.Booster, path: Path) -> None:
        model.save_model(f"{path}.txt")  # LightGBM 텍스트 포맷


def run(cfg: DictConfig) -> dict[str, Any]:
    """LightGBM OOF 파이프라인 (train_common 어댑터). back-compat 진입점."""
    return run_oof_cv(cfg, LGBMTrainer(cfg))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
