"""LightGBM 트레이너 — 공유 스캐폴드(train_common)의 어댑터. 베이스라인/표준.

⚠️ 모델 추가는 이 파일을 **복사하지 않는다**. 이 파일이 곧 "어댑터 패턴" 예시다 —
   새 모델은 src/train_<model>.py 에서 prepare/fit_predict 만 정의하고
   `run_oof_cv` 에 넘긴다 (예: src/train_xgb.py). 공통 골격(seed/CV/OOF-TE/증강/저장/로그)은
   train_common 단일 소스라, 골격을 고쳐도 모든 모델이 한 번에 따라온다.

실행:
    uv run python -m src.train_lgbm exp_id=exp_001 "notes='lgbm baseline'"   # notes 특수문자는 작은따옴표
    uv run python -m src.train_lgbm exp_id=exp_002 features=te_example          # 타깃 인코딩
    uv run python -m src.train_lgbm exp_id=exp_003 model.params.num_leaves=127  # 파라미터 오버라이드
    uv run python -m src.train_lgbm -m model.params.num_leaves=63,127,255       # 스윕(멀티런)
"""

from __future__ import annotations

from typing import Any

import hydra
import lightgbm as lgb
from omegaconf import DictConfig, OmegaConf
from pandas.api.types import CategoricalDtype

from src import config
from src.train_common import run_oof_cv


def run(cfg: DictConfig) -> dict[str, Any]:
    """LightGBM OOF 파이프라인 (train_common 어댑터)."""
    lgb_params = {
        **OmegaConf.to_container(cfg.model.params, resolve=True),
        "n_jobs": -1,
        "verbose": -1,
    }
    num_boost_round = cfg.model.num_boost_round
    early_stopping = cfg.model.early_stopping

    def prepare(x, x_test, x_src, cat_cols, aug_enabled):
        # native categorical 안정화: train∪test∪source 합집합으로 고정 CategoricalDtype.
        # (증강 concat 후 category 가 object 로 풀리는 것 방지 — fit_predict 에서 재적용.)
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
        dtrain = lgb.Dataset(x_tr, y_tr, categorical_feature=cat_cols, weight=w_tr)
        dvalid = lgb.Dataset(x_va, y_va, categorical_feature=cat_cols)
        model = lgb.train(
            {**lgb_params, "seed": cfg.get("seed", config.SEED)},
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)],
        )
        bi = model.best_iteration
        return (
            model.predict(x_va, num_iteration=bi),
            model.predict(x_te, num_iteration=bi),
            int(bi),
        )

    return run_oof_cv(
        cfg,
        prepare=prepare,
        fit_predict=fit_predict,
        supports_weight=True,
        log_extra={"num_boost_round": num_boost_round},
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
