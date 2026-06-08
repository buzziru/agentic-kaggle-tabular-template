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


def add_example_features(df: pd.DataFrame) -> pd.DataFrame:
    """예시: conf 의 `feature_builder` 훅으로 켜는 모델별/실험별 피처 (conf/features/*.yaml).

    데이터 형태와 무관한 **행-단위(누수 안전) 파생** 패턴 예시 — 다른 행/타깃을 참조하지
    않으므로 누수 0. 실제 피처는 EDA 결과에 맞게 이 함수(또는 build_features)에서 정의하고,
    변형은 코드 포크 대신 conf 노브로 켠다. 그룹/시계열 과거-only 집계가 필요하면
    `docs/feature_engineering.md` 의 "그룹/시계열 과거-only 집계" 레시피를 가져다 쓴다.

    Args:
        df: build_features 적용 후 DataFrame.

    Returns:
        파생이 추가된 복사본.
    """
    out = df.copy()
    # 행-단위 변환만 (다른 행/타깃 미참조 → 누수 0). 실제 컬럼명으로 교체 후 사용:
    # out["num_missing"] = out[config.NUMERIC_COLS].isnull().sum(axis=1).astype("int16")
    # out["a_over_b"] = (out["a"] / (out["b"] + 1e-6)).astype("float32")
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
