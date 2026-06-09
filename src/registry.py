"""모델 트레이너 registry — 모델명 → Trainer 클래스 (단일 진입점 `src.train` 이 소비).

⚠️ 새 모델 추가 = `src/train_<model>.py` 에 `ModelTrainer` 구현 클래스를 만들고
   아래 `_REGISTRY` 에 (name, "모듈경로", "클래스명") 한 줄만 등록한다.
   `train_common.run_oof_cv` 는 **수정하지 않는다** (모델 분기 금지 — 어댑터 패턴).

OOF 계약(스택 풀의 디커플링 경계)·공통 골격은 train_common 한 곳에만 있다.
구현 클래스는 모델별 차이(범주형 전처리 + fit/predict)만 제공한다.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol

import numpy as np
import pandas as pd


class ModelTrainer(Protocol):
    """공유 스캐폴드(`train_common.run_oof_cv`)가 호출하는 모델별 인터페이스.

    train_common 이 seed/CV/fold loop/OOF/metric/artifact/logging 을 전부 통제하고,
    구현 클래스는 모델별 차이만 담당한다. 골격을 복사하지 않는다(노브 divergence 방지).

    cfg 에서 뽑은 파라미터(예: num_boost_round)는 `__init__(cfg)` 에 보관한다.
    """

    name: str
    supports_weight: bool  # 증강 sample_weight 지원 여부 (False 면 weight≠1.0 차단)

    def prepare(
        self,
        x: pd.DataFrame,
        x_test: pd.DataFrame,
        x_src: pd.DataFrame | None,
        cat_cols: list[str],
        aug_enabled: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, "pd.DataFrame | None", Any]:
        """모델별 범주형 전처리. (x, x_test, x_src, state) 를 반환한다."""
        ...

    def fit_predict(
        self,
        x_tr: pd.DataFrame,
        y_tr: pd.Series,
        x_va: pd.DataFrame,
        y_va: pd.Series,
        x_te: pd.DataFrame,
        w_tr: "np.ndarray | None",
        cat_cols: list[str],
        state: Any,
    ) -> tuple[np.ndarray, np.ndarray, "int | None"]:
        """fold 학습/예측. (oof_pred, test_pred, best_iter|None) 을 반환한다."""
        ...

    def log_extra(self) -> dict[str, Any]:
        """실험 로그 params 에 추가할 항목(예: num_boost_round). 없으면 빈 dict."""
        ...


# name -> (모듈경로, 클래스명). 지연 import — lgbm 만 쓸 때 xgboost 미설치여도 동작.
_REGISTRY: dict[str, tuple[str, str]] = {
    "lgbm": ("src.train_lgbm", "LGBMTrainer"),
    "xgb": ("src.train_xgb", "XGBTrainer"),
}


def get_trainer(name: str) -> "type[ModelTrainer]":
    """모델명으로 Trainer 클래스를 반환한다 (지연 import).

    Args:
        name: cfg.model.name (예: "lgbm", "xgb").

    Returns:
        ModelTrainer 를 구현한 클래스. `cls(cfg)` 로 인스턴스화한다.

    Raises:
        KeyError: 미등록 모델명.
    """
    if name not in _REGISTRY:
        raise KeyError(f"미등록 모델 '{name}'. 등록된 모델: {sorted(_REGISTRY)} (src/registry.py)")
    module_path, cls_name = _REGISTRY[name]
    return getattr(importlib.import_module(module_path), cls_name)
