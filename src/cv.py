"""검증 분할 (Cross-Validation).

기본은 StratifiedKFold(분류). ⚠️ 데이터 구조에 맞춰 교체한다 (config.CV_STRATEGY,
docs/setup_questions.md):
  - 그룹 누수 위험: GroupKFold(groups=...)
  - 시계열: TimeSeriesSplit
  - 회귀(계층화 불필요): KFold
train/test 분할 방식과 일치시키는 것이 핵심.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config


def get_folds(
    y: pd.Series | np.ndarray,
    *,
    n_folds: int = config.N_FOLDS,
    seed: int = config.SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedKFold fold 인덱스 목록을 생성한다.

    Args:
        y: 타깃 (계층화 기준).
        n_folds: fold 수.
        seed: 셔플 시드.

    Returns:
        (train_idx, valid_idx) 튜플의 리스트.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    dummy_x = np.zeros(len(y))
    return list(skf.split(dummy_x, y))
