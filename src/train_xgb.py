"""XGBoost 트레이너 — **모델 추가 = 어댑터** 예시 (train_common 사용).

새 모델을 스택 풀에 넣는 표준 방법: 골격을 복사하지 말고 `ModelTrainer`(src/registry.py)
인터페이스를 구현(prepare/fit/predict/get_metadata/save_model)하고 registry 에 등록한다.
이 파일 전체가 한 모델을 추가하는 데 드는 코드량이다. OOF 산출 형식은 train_common 이
보장하므로 src.stack 이 그대로 소비한다.

deps: `uv sync --extra gpu` (xgboost).

실행 (통합 진입점):
    uv run python -m src.train model=xgb features=te_example exp_id=exp_010 \
        use_wandb=false "notes='xgb diversity'"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from pandas.api.types import CategoricalDtype

from src import cat_prep, config
from src.train_common import run_oof_cv


class XGBTrainer:
    """XGBoost 어댑터 (`ModelTrainer` 구현). enable_categorical 사용."""

    name = "xgb"
    supports_weight = True

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.params = OmegaConf.to_container(cfg.model.params, resolve=True)
        self.n_estimators = cfg.model.num_boost_round
        self.early_stopping = cfg.model.early_stopping

    def log_extra(self) -> dict[str, Any]:
        return {"n_estimators": self.n_estimators}

    def prepare(
        self,
        x: pd.DataFrame,
        x_test: pd.DataFrame,
        x_src: "pd.DataFrame | None",
        cat_cols: list[str],
        aug_enabled: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, "pd.DataFrame | None", dict[str, CategoricalDtype]]:
        # enable_categorical: train/valid/test/source category 집합 일치 필요 — 공유 헬퍼 단일 소스.
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
    ) -> "xgb.XGBModel":
        x_tr = cat_prep.recast(x_tr, cat_cols, cat_dtypes)  # 증강 concat 후 고정 cat dtype 재적용
        # PROBLEM_TYPE 으로 estimator 선택 (binary=Classifier·regression=Regressor). objective 는 conf 에서.
        est_cls = xgb.XGBRegressor if config.PROBLEM_TYPE == "regression" else xgb.XGBClassifier
        model = est_cls(
            n_estimators=self.n_estimators,
            early_stopping_rounds=self.early_stopping,
            enable_categorical=True,
            n_jobs=-1,
            random_state=self.cfg.get("seed", config.SEED),
            **self.params,
        )
        model.fit(x_tr, y_tr, sample_weight=w_tr, eval_set=[(x_va, y_va)], verbose=False)
        return model

    def predict(self, model: "xgb.XGBModel", x: pd.DataFrame) -> np.ndarray:
        rng = (0, model.best_iteration + 1)
        if config.PROBLEM_TYPE == "regression":
            return model.predict(x, iteration_range=rng)
        if config.PROBLEM_TYPE == "binary":
            return model.predict_proba(x, iteration_range=rng)[:, 1]
        return model.predict_proba(x, iteration_range=rng)  # multiclass: (n, K) 확률

    def get_metadata(self, model: "xgb.XGBModel") -> dict[str, Any]:
        return {"best_iter": int(model.best_iteration)}

    def save_model(self, model: "xgb.XGBModel", path: Path) -> None:
        model.save_model(f"{path}.json")  # XGBoost JSON 포맷


def run(cfg: DictConfig) -> dict[str, Any]:
    """XGBoost OOF 파이프라인 (train_common 어댑터). back-compat 진입점."""
    return run_oof_cv(cfg, XGBTrainer(cfg))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
