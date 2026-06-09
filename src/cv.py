"""검증 분할 (Cross-Validation).

`get_folds` 는 `config.CV_STRATEGY` 로 전략을 **디스패치**한다(장식용 상수가 아니라 실제
선택자). 데이터 구조에 맞춰 config 에서 고른다 (docs/setup_questions.md) — train/test 분할
방식과 일치시키는 것이 핵심. 지원 전략:
  - "StratifiedKFold": 분류(계층화)         - "KFold": 회귀
  - "GroupKFold": 그룹 누수 위험(groups 필요 — config.GROUP_KEYS → make_groups)
⚠️ 시계열(TimeSeriesSplit)은 **공식 미지원**이다 — 확장형 윈도우는 전체 행을 검증하지
   못해(초기 구간 미검증) full-OOF 스태킹 계약과 어긋난다. 필요하면 여기 분기를
   직접 추가하되, OOF 가 부분적임을 감안해 stack 풀에서 분리 운용한다.
새 전략은 여기 분기에 추가한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from src import config


def get_folds(
    y: pd.Series | np.ndarray,
    *,
    n_folds: int = config.N_FOLDS,
    seed: int = config.SEED,
    groups: np.ndarray | None = None,
    strategy: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """config.CV_STRATEGY(또는 strategy)에 맞는 fold 인덱스 목록을 생성한다.

    Args:
        y: 타깃 (계층화 기준 — StratifiedKFold 에서만 사용).
        n_folds: fold 수.
        seed: 셔플 시드 (Stratified/KFold).
        groups: 그룹 키 배열 (GroupKFold 필수). `make_groups` 로 만든다.
        strategy: 전략명. None 이면 config.CV_STRATEGY.

    Returns:
        (train_idx, valid_idx) 튜플의 리스트.

    Raises:
        ValueError: 미지원 전략, 또는 GroupKFold 인데 groups 미제공.
    """
    strategy = strategy or config.CV_STRATEGY
    dummy_x = np.zeros(len(y))
    if strategy == "StratifiedKFold":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(skf.split(dummy_x, y))
    if strategy == "KFold":
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(kf.split(dummy_x))
    if strategy == "GroupKFold":
        if groups is None:
            raise ValueError(
                "GroupKFold 는 groups 가 필요하다 — config.GROUP_KEYS 를 채우고 cv.make_groups 로 전달하라"
            )
        return list(GroupKFold(n_splits=n_folds).split(dummy_x, y, groups))
    raise ValueError(
        f"미지원 CV 전략 '{strategy}' — 공식 지원: StratifiedKFold/KFold/GroupKFold. "
        "시계열은 공식 미지원(full-OOF 계약과 불일치) — 필요 시 cv.get_folds 에 직접 분기 추가."
    )


def make_groups(df: pd.DataFrame) -> np.ndarray | None:
    """config.GROUP_KEYS 로 복합 그룹 키 배열을 만든다 (GroupKFold 용).

    Args:
        df: 그룹 키 컬럼을 가진 DataFrame (보통 train_df).

    Returns:
        행별 그룹 키 배열. GROUP_KEYS 가 비어 있으면 None (그룹 전략 미사용).
    """
    if not config.GROUP_KEYS:
        return None
    return df[config.GROUP_KEYS].astype(str).agg("_".join, axis=1).to_numpy()
