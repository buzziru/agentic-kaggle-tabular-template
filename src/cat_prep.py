"""범주형 고정 dtype 헬퍼 — native categorical 모델(LGBM/XGB 등)의 공유 전처리.

⚠️ 단일 진실원: 이 로직은 여기 한 곳에만 둔다. 트레이너가 복제하지 않는다
   (2중 사본 drift = CLAUDE.md 가 지목한 #1 재발 버그). 새 native-categorical 모델은
   `prepare` 에서 `fix_categoricals` 를, `fit` 에서 `recast` 를 호출만 한다.

왜 고정 dtype 인가: train/test(/source) 의 category 집합을 합집합으로 고정해야
증강 concat 후 category 가 object 로 풀리거나 split 간 카테고리 집합이 어긋나는 것을
막는다 (LGBM native categorical · XGB enable_categorical 모두 동일 요구).
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import CategoricalDtype


def recast(
    df: pd.DataFrame, cat_cols: list[str], cat_dtypes: dict[str, CategoricalDtype]
) -> pd.DataFrame:
    """cat_cols 를 고정 dtype 로 (재)캐스팅한 사본을 반환한다.

    증강 concat 후 category 가 풀린 train fold 에 다시 적용할 때도 쓴다.

    Args:
        df: 입력 DataFrame.
        cat_cols: 범주형 컬럼명.
        cat_dtypes: 컬럼별 고정 CategoricalDtype (`fix_categoricals` 산출).

    Returns:
        cat_cols 가 고정 dtype 로 캐스팅된 df 사본.
    """
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype(cat_dtypes[col])
    return df


def fix_categoricals(
    x: pd.DataFrame,
    x_test: pd.DataFrame,
    x_src: "pd.DataFrame | None",
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, "pd.DataFrame | None", dict[str, CategoricalDtype]]:
    """train/test/source 합집합으로 cat_cols 를 고정 CategoricalDtype 로 캐스팅한다.

    Args:
        x: train 피처.
        x_test: test 피처.
        x_src: (선택) 외부 증강 피처. None 이면 무시.
        cat_cols: 범주형 컬럼명.

    Returns:
        (x, x_test, x_src, cat_dtypes) — 캐스팅된 df 들과 컬럼별 고정 dtype.
    """
    cat_dtypes: dict[str, CategoricalDtype] = {}
    for col in cat_cols:
        vals = set(x[col].dropna().unique()) | set(x_test[col].dropna().unique())
        if x_src is not None:
            vals |= set(x_src[col].dropna().unique())
        cat_dtypes[col] = CategoricalDtype(categories=sorted(vals))

    x = recast(x, cat_cols, cat_dtypes)
    x_test = recast(x_test, cat_cols, cat_dtypes)
    if x_src is not None:
        x_src = recast(x_src, cat_cols, cat_dtypes)
    return x, x_test, x_src, cat_dtypes
