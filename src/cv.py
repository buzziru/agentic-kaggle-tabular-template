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

레짐별 직렬화: `get_folds` 는 **로드-우선**이다. 같은 레짐(strategy·n_folds·seed·그룹해시)의
분할을 `config.DATA_DIR/splits/{strategy}_{n}fold_seed{seed}[_{ghash}].parquet`(스키마
`row_idx,fold`)에 한 번 만들어 두고, 다음부터는 로드해 재사용한다. 공유 코드(features·
train_common)가 리팩토링돼도 분할은 파일로 동결되므로 같은-fold 정합·frozen 멤버 OOF
불변이 보장된다. fold 다양성 축(5/7/10-fold)은 파일명이 달라 공존한다.
⚠️ 데이터 부재 환경(원격 커널)에 splits 파일이 없으면 기존처럼 즉석 생성하는데, **로컬과
   원격의 sklearn 버전이 다르면 같은 seed 라도 분할이 달라질 수 있다** — frozen 멤버를
   원격에서 재현할 땐 로컬에서 만든 splits 파일을 함께 올려 로드 경로를 타게 하라.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from src import config


def _splits_path(strategy: str, n_folds: int, seed: int, groups: np.ndarray | None) -> Path:
    """레짐별 splits 파일 경로. config.DATA_DIR 기준(절대경로 하드코딩 금지 — examples/_work 격리 유지).

    GroupKFold 는 분할이 groups 데이터에 의존하므로 파일명에 그룹 해시 8자를 붙여 충돌을 막는다.
    """
    name = f"{strategy}_{n_folds}fold_seed{seed}"
    if strategy == "GroupKFold" and groups is not None:
        ghash = hashlib.sha256(np.asarray(groups).astype(str).tobytes()).hexdigest()[:8]
        name += f"_{ghash}"
    return config.DATA_DIR / "splits" / f"{name}.parquet"


def _save_folds(path: Path, folds: list[tuple[np.ndarray, np.ndarray]], n_rows: int) -> None:
    """fold 리스트를 (row_idx, fold) 2열 parquet 로 저장한다.

    각 행이 정확히 한 fold 의 검증셋에 속하는 partition 임을 검증한 뒤에만 저장한다
    (지원 전략 Stratified/KFold/Group 은 항상 partition; 아니면 직렬화 자체가 무의미).
    """
    fold_arr = np.full(n_rows, -1, dtype=np.int64)
    for f, (_tr, va) in enumerate(folds):
        fold_arr[va] = f
    if (fold_arr < 0).any():
        raise ValueError(
            "splits 직렬화 불가: 일부 행이 어느 fold 의 검증셋에도 없다(non-partition). "
            "지원 전략(Stratified/KFold/Group)만 직렬화된다 — 커스텀 분할은 파일 캐시를 쓰지 마라."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_idx": np.arange(n_rows, dtype=np.int64), "fold": fold_arr}).to_parquet(
        path, index=False
    )


def _load_folds(path: Path, n_folds: int, n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """저장된 splits 를 (train_idx, valid_idx) 리스트로 복원하고 일관성을 자가검증한다.

    검증: 행수 일치·fold 범위(0..n_folds-1)·전체 커버·중복 없음. 행수가 데이터와 다르면
    데이터가 바뀐 신호이므로 **자동 재생성하지 않고** 명확한 에러로 중단한다(사람이 판단).
    재구성 인덱스는 ascending — sklearn split 순서와 동일하므로 입력 동등성(check_fold_inputs)이 깨지지 않는다.
    """
    df = pd.read_parquet(path).sort_values("row_idx")
    fold_arr = df["fold"].to_numpy()
    if len(fold_arr) != n_rows:
        raise ValueError(
            f"splits 파일 행수 불일치: 파일={len(fold_arr)} vs 데이터={n_rows} ({path.name}). "
            "데이터가 바뀐 신호다 — 자동 재생성하지 않는다. 파일을 지우고 의도적으로 재생성하라."
        )
    uniq = np.unique(fold_arr)
    if not np.array_equal(uniq, np.arange(n_folds)):
        raise ValueError(
            f"splits 파일 fold 라벨 불일치: {uniq.tolist()} vs 기대 0..{n_folds - 1} ({path.name})."
        )
    return [
        (np.where(fold_arr != f)[0], np.where(fold_arr == f)[0]) for f in range(n_folds)
    ]


def get_folds(
    y: pd.Series | np.ndarray,
    *,
    n_folds: int = config.N_FOLDS,
    seed: int = config.SEED,
    groups: np.ndarray | None = None,
    strategy: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """config.CV_STRATEGY(또는 strategy)에 맞는 fold 인덱스 목록을 **로드 또는 생성**한다.

    레짐별 splits 파일이 있으면 로드하고, 없으면 생성 후 저장한다(시그니처·반환 타입 불변).

    Args:
        y: 타깃 (계층화 기준 — StratifiedKFold 에서만 사용).
        n_folds: fold 수.
        seed: 셔플 시드 (Stratified/KFold).
        groups: 그룹 키 배열 (GroupKFold 필수). `make_groups` 로 만든다.
        strategy: 전략명. None 이면 config.CV_STRATEGY.

    Returns:
        (train_idx, valid_idx) 튜플의 리스트.

    Raises:
        ValueError: 미지원 전략, GroupKFold 인데 groups 미제공, 또는 splits 파일 불일치.
    """
    strategy = strategy or config.CV_STRATEGY
    n_rows = len(y)
    path = _splits_path(strategy, n_folds, seed, groups)
    if path.is_file():
        return _load_folds(path, n_folds, n_rows)

    dummy_x = np.zeros(n_rows)
    if strategy == "StratifiedKFold":
        folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(dummy_x, y))
    elif strategy == "KFold":
        folds = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(dummy_x))
    elif strategy == "GroupKFold":
        if groups is None:
            raise ValueError(
                "GroupKFold 는 groups 가 필요하다 — config.GROUP_KEYS 를 채우고 cv.make_groups 로 전달하라"
            )
        folds = list(GroupKFold(n_splits=n_folds).split(dummy_x, y, groups))
    else:
        raise ValueError(
            f"미지원 CV 전략 '{strategy}' — 공식 지원: StratifiedKFold/KFold/GroupKFold. "
            "시계열은 공식 미지원(full-OOF 계약과 불일치) — 필요 시 cv.get_folds 에 직접 분기 추가."
        )
    _save_folds(path, folds, n_rows)
    return folds


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
