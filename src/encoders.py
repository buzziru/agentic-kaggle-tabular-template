"""누수 방지 타깃 인코딩 (OOF target encoding).

핵심 원리:
- **train fold 행**: 내부 KFold 로 각 행을 *다른* train 행 통계로만 인코딩 → 모델이
  자기 행의 라벨을 기억하지 못하게 함 (train 과적합 방지).
- **valid / test 행**: 전체 train fold 통계로 인코딩 → 해당 행의 라벨이 통계에
  들어가지 않으므로 누수 없음.

스무딩(smoothing): 표본이 적은 범주는 전역 평균 쪽으로 수축시켜 분산을 줄인다.
고카디널리티 범주형(예: 사용자/상품 ID)에 native categorical 보다 효과적일 수 있다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src import config


class OOFTargetEncoder:
    """Out-of-fold 스무딩 타깃 인코더.

    fold 루프 안에서 fold 마다 새로 생성해 사용한다 (fold 의 train 부분으로만 fit).

    Attributes:
        cols: 타깃 인코딩 대상 컬럼.
        smoothing: 스무딩 강도(샘플 수 기준). 클수록 전역 평균으로 강하게 수축.
        n_inner: train 행 OOF 인코딩용 내부 fold 수.
        seed: 내부 KFold 시드.
    """

    def __init__(
        self,
        cols: list[str],
        *,
        smoothing: float = 20.0,
        n_inner: int = 5,
        seed: int = config.SEED,
    ) -> None:
        self.cols = cols
        self.smoothing = smoothing
        self.n_inner = n_inner
        self.seed = seed
        self.global_mean_: float = 0.0
        self.maps_: dict[str, pd.Series] = {}

    def _smoothed_map(self, x_col: pd.Series, y: pd.Series) -> pd.Series:
        """한 컬럼의 (범주 → 스무딩된 타깃 평균) 매핑을 만든다.

        Args:
            x_col: 범주형 컬럼 값.
            y: 대응 타깃.

        Returns:
            범주를 인덱스로 갖는 인코딩 값 Series.
        """
        df = pd.DataFrame({"c": x_col.astype(object).to_numpy(), "y": y.to_numpy()})
        agg = df.groupby("c")["y"].agg(["mean", "count"])
        return (agg["mean"] * agg["count"] + self.global_mean_ * self.smoothing) / (
            agg["count"] + self.smoothing
        )

    def fit_transform_train(self, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """train fold 를 OOF 인코딩하고, valid/test 용 전체 매핑을 적합한다.

        Args:
            x: train fold 피처 (대상 컬럼 포함).
            y: train fold 타깃.

        Returns:
            대상 컬럼이 OOF 인코딩(float)으로 치환된 복사본.
        """
        x = x.reset_index(drop=True).copy()
        y = y.reset_index(drop=True)
        self.global_mean_ = float(y.mean())

        # valid/test 용: 전체 train fold 로 적합
        for col in self.cols:
            self.maps_[col] = self._smoothed_map(x[col], y)

        # train 행: 내부 OOF (자기 자신 제외)
        encoded = {col: np.full(len(x), self.global_mean_, dtype=float) for col in self.cols}
        skf = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=self.seed)
        for inner_tr, inner_va in skf.split(x, y):
            for col in self.cols:
                m = self._smoothed_map(x[col].iloc[inner_tr], y.iloc[inner_tr])
                vals = x[col].iloc[inner_va].astype(object).map(m)
                encoded[col][inner_va] = vals.fillna(self.global_mean_).to_numpy()

        for col in self.cols:
            x[col] = encoded[col]
        return x

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """fit 된 전체 매핑으로 valid/test 를 인코딩한다.

        Args:
            x: 변환할 피처 (대상 컬럼 포함).

        Returns:
            대상 컬럼이 인코딩(float)으로 치환된 복사본. 미등장 범주는 전역 평균.
        """
        x = x.copy()
        for col in self.cols:
            x[col] = x[col].astype(object).map(self.maps_[col]).fillna(self.global_mean_).astype(float)
        return x
