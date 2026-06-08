"""XGBoost 트레이너 — **모델 추가 = 어댑터** 예시 (train_common 사용).

새 모델을 스택 풀에 넣는 표준 방법: 골격을 복사하지 말고 prepare/fit_predict 만
정의해 `run_oof_cv` 에 넘긴다. 이 파일 전체가 한 모델을 추가하는 데 드는 코드량이다
(~40줄). OOF 산출 형식은 train_common 이 보장하므로 src.stack 이 그대로 소비한다.

deps: `uv sync --extra gpu` (xgboost).

실행:
    uv run python -m src.train_xgb exp_id=exp_010 model=xgb features=te_example \
        use_wandb=false "notes='xgb diversity'"
"""

from __future__ import annotations

from typing import Any

import hydra
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from pandas.api.types import CategoricalDtype

from src import config
from src.train_common import run_oof_cv


def run(cfg: DictConfig) -> dict[str, Any]:
    """XGBoost OOF 파이프라인 (train_common 어댑터)."""
    xgb_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    n_estimators = cfg.model.num_boost_round
    early_stopping = cfg.model.early_stopping

    def prepare(x, x_test, x_src, cat_cols, aug_enabled):
        # enable_categorical: train/valid/test/source 의 category 집합 일치 필요 → 고정 dtype.
        cat_dtypes: dict[str, CategoricalDtype] = {}
        for col in cat_cols:
            vals = set(x[col].dropna().unique()) | set(x_test[col].dropna().unique())
            if x_src is not None:
                vals |= set(x_src[col].dropna().unique())
            cat_dtypes[col] = CategoricalDtype(categories=sorted(vals))

        def cast(df):
            df = df.copy()
            for col in cat_cols:
                df[col] = df[col].astype(cat_dtypes[col])
            return df

        x, x_test = cast(x), cast(x_test)
        if x_src is not None:
            x_src = cast(x_src)
        return x, x_test, x_src, cat_dtypes

    def fit_predict(x_tr, y_tr, x_va, y_va, x_te, w_tr, cat_cols, cat_dtypes):
        x_tr = x_tr.copy()  # 증강 concat 후 고정 cat dtype 재적용
        for col in cat_cols:
            x_tr[col] = x_tr[col].astype(cat_dtypes[col])
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping,
            enable_categorical=True,
            n_jobs=-1,
            random_state=cfg.get("seed", config.SEED),
            **xgb_params,
        )
        model.fit(x_tr, y_tr, sample_weight=w_tr, eval_set=[(x_va, y_va)], verbose=False)
        rng = (0, model.best_iteration + 1)
        return (
            model.predict_proba(x_va, iteration_range=rng)[:, 1],
            model.predict_proba(x_te, iteration_range=rng)[:, 1],
            int(model.best_iteration),
        )

    return run_oof_cv(
        cfg,
        prepare=prepare,
        fit_predict=fit_predict,
        supports_weight=True,
        log_extra={"n_estimators": n_estimators},
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
