"""피처 엔지니어링 — 단일 진입점.

⚠️ 코드 파편화 방지 (CLAUDE.md "피처 엔지니어링" 섹션):
  - 모든 피처는 `build_features()` (또는 아래 `add_*` 헬퍼)에 구현한다. 노트북/학습
    스크립트에 일회성 변환을 박지 않는다 — train/test 가 갈리고 누수의 근원이 된다.
  - 피처 변형은 함수를 복제하지 말고 `conf/features/*.yaml` 노브로 켜고 끈다
    (`feature_builder` 훅·`drop_cols`·`target_encode_cols`·`extra_categorical_cols`).
  - 각 피처는 부수효과 없는 작은 함수로 쪼개고, 기각된 피처는 즉시 제거한다.

⚠️ 누수 방지: 그룹/시계열 파생은 **과거 관측만** 참조한다(`shift(>0)`·expanding·cumcount).
  미래 행이나 그룹 전체 통계(타깃 사용)는 금지. 타깃 사용 인코딩은 fold-내 OOF
  (`src.encoders.OOFTargetEncoder`)로 처리한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """원본 컬럼으로부터 모델 입력 피처를 생성한다 (모든 모델 공통).

    베이스라인은 패스스루. 신규 피처는 train/test 에 동일 적용되도록 이 함수
    한 곳에서만 정의한다.

    Args:
        df: load_train/load_test 로 읽은 원본 DataFrame.

    Returns:
        피처가 추가된 DataFrame (원본 비파괴, copy 반환).
    """
    out = df.copy()
    # TODO: EDA 결과 기반 파생 피처를 여기(또는 add_* 헬퍼)에 추가.
    return out


def _group_past_expanding(df: pd.DataFrame, value_col: str, agg: str) -> pd.Series:
    """그룹(GROUP_KEYS) 내 SEQUENCE_COL 오름차순으로 **과거 관측행만** expanding 집계.

    누수 0: `shift(1)` 후 expanding 이므로 현재 행 직전까지의 과거만 본다. 그룹별
    정렬해 집계한 뒤 원본 index 로 복원한다(행 순서/index 불변, unique RangeIndex 가정).

    Args:
        df: GROUP_KEYS + SEQUENCE_COL + value_col 을 포함한 DataFrame.
        value_col: 집계 대상 수치 컬럼.
        agg: 'mean'|'std'|'max'|'min'|'sum' 중 하나.

    Returns:
        원본 index 정렬 Series (과거 없음 → NaN).
    """
    if not config.GROUP_KEYS or config.SEQUENCE_COL is None:
        raise ValueError("config.GROUP_KEYS 와 SEQUENCE_COL 을 먼저 설정해야 한다")
    keys = config.GROUP_KEYS
    order = df.sort_values(keys + [config.SEQUENCE_COL], kind="mergesort").index
    val = df.loc[order, value_col]
    grp = [df.loc[order, k] for k in keys]
    shifted = val.groupby(grp, observed=True).shift(1)  # 현재행 제외(과거만)
    res = getattr(shifted.groupby(grp, observed=True).expanding(), agg)()
    res = res.reset_index(level=list(range(len(keys))), drop=True)
    return res.reindex(df.index)


def add_example_group_features(df: pd.DataFrame) -> pd.DataFrame:
    """예시: 모델별/실험별 변형을 켜는 `feature_builder` 훅 (conf/features 에서 지정).

    그룹 내 과거 관측만 쓰는 누수 안전 파생의 패턴 예시. GROUP_KEYS/SEQUENCE_COL 이
    없는 프로젝트(독립 행)면 그대로 통과한다.

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        파생이 추가된 복사본.
    """
    if not config.GROUP_KEYS or config.SEQUENCE_COL is None:
        return df.copy()
    out = df.copy()
    # 예시 (실제 컬럼명으로 교체):
    # out["x_past_mean"] = _group_past_expanding(out, "x", "mean").fillna(0).astype("float32")
    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """모델에 투입할 피처 컬럼 목록을 반환한다 (id, target 제외).

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        피처 컬럼 이름 리스트.
    """
    drop = {config.ID_COL, config.TARGET_COL}
    return [c for c in df.columns if c not in drop]
