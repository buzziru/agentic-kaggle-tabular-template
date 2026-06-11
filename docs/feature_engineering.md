# 피처 엔지니어링 — 후보 & 검증 로그

> 피처는 `src/features.py` 단일 진입점에서만 구현(CLAUDE.md "코드 파편화 방지"). 변형은 `conf/features/*.yaml` 노브로.

## 누수 안전 원칙
- 그룹/시계열 파생 = **과거 관측만**(`shift(>0)`·expanding·cumcount, 정렬 키 = `config.SEQUENCE_COL`).
- 타깃 사용 인코딩 = **fold-내 OOF**(`src.encoders.OOFTargetEncoder`). ⚠️ multiclass 미지원(스칼라 평균 무의미) → native categorical(`CATEGORICAL_COLS`)로 처리.
- 신규 피처는 미래 행 마스킹 불변성으로 누수 점검 후 채택.

## 레시피: 그룹/시계열 과거-only 집계 (선택)
독립 행 데이터면 불필요하다. 그룹(`config.GROUP_KEYS`)+시퀀스(`config.SEQUENCE_COL`) 구조가
있을 때만 아래 헬퍼를 `src/features.py` 에 붙여 `add_example_features`(또는 전용 `add_*` 빌더)에서
호출한다. `shift(1)` 후 expanding 이라 현재 행 직전까지의 **과거만** 보므로 누수 0.

```python
import numpy as np  # (features.py 에 추가)


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


# 사용 (add_example_features 등에서):
# out["x_past_mean"] = _group_past_expanding(out, "x", "mean").fillna(0).astype("float32")
```

## 후보 목록
| 피처 | 정의 | 근거(EDA) | 누수 안전 | 상태 |
|---|---|---|---|---|
| {{name}} | {{ }} | {{ }} | {{과거-only/타깃미사용}} | {{후보/구현/채택/기각}} |

## 검증 로그 (OOF 기준)
| exp_id | 피처셋(conf) | baseline 대비 Δ | 판정 | 비고 |
|---|---|---|---|---|
| {{exp_NNN}} | {{features=...}} | {{+0.000x}} | {{채택/기각}} | {{ }} |

## 기각 교훈
- {{기각된 피처 → 왜 (수치+근거). 코드는 제거, 결론만 여기.}}
