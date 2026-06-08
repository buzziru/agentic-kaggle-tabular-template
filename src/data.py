"""데이터 로드/IO.

범주형 컬럼은 메모리 절약 및 native categorical 처리를 위해 category dtype 으로 변환한다.
"""

from __future__ import annotations

import pandas as pd

from src import config


def _set_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """범주형 컬럼을 category dtype 으로 변환한다.

    Args:
        df: 입력 DataFrame.

    Returns:
        범주형 변환이 적용된 DataFrame (원본 수정).
    """
    for col in config.CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def load_train() -> pd.DataFrame:
    """train.csv 로드."""
    return _set_categorical(pd.read_csv(config.TRAIN_PATH))


def load_test() -> pd.DataFrame:
    """test.csv 로드."""
    return _set_categorical(pd.read_csv(config.TEST_PATH))


def load_sample_submission() -> pd.DataFrame:
    """sample_submission.csv 로드."""
    return pd.read_csv(config.SAMPLE_SUBMISSION_PATH)


# train 과 정렬할 컬럼(피처 + 타깃). id 는 증강 전용이라 제외.
_AUG_COLS: list[str] = (
    config.CATEGORICAL_COLS
    + config.CATEGORICAL_INT_COLS
    + config.NUMERIC_COLS
    + [config.TARGET_COL]
)


def load_source_augmentation() -> pd.DataFrame:
    """(선택) 외부 원본 데이터를 train 스키마에 정렬해 로드한다 (train 증강 전용).

    train 의 피처 + 타깃 컬럼만 선택하고(`id` 제외) 범주형 dtype 을 동일 적용한다.
    검증/제출에는 절대 사용하지 않으며 fold 의 train 부분에만 합친다.
    augment 를 안 쓰면 이 함수와 `config.SOURCE_AUG_PATH` 는 무시해도 된다.

    Returns:
        train 과 동일한 피처+타깃 컬럼·dtype 으로 정렬된 DataFrame.

    Raises:
        ValueError: 원본에 필요한 컬럼이 누락된 경우.
    """
    df = pd.read_csv(config.SOURCE_AUG_PATH)
    missing = [c for c in _AUG_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"원본에 필요한 컬럼 누락: {missing}")
    return _set_categorical(df[_AUG_COLS].copy())
