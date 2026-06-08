"""입력 동등성 게이트 — 트레이너/피처 리팩토링이 모델 fit/predict 입력을 안 바꿨는지 검증.

모델 레이어를 더미로 monkeypatch(실제 학습 X)해 각 트레이너 `run()` 을 돌리고, fold별
fit/predict 입력의 해시를 덤프한다. 리팩토링 前/後 해시가 바이트 일치하면 입력 동일 →
결정적 모델(seed 고정)이라 **frozen 스택 멤버의 OOF 도 동일**. GPU·실학습 불필요.

→ 스택 풀이 커진 뒤 train_common/features 를 리팩토링할 때, 동결된 멤버 OOF 가 바뀌지
  않음을 보장하는 안전망 (CLAUDE.md "스택 풀 & 트레이너 구조" 참조).

⚠️ 이 파일은 템플릿이다 — 본인이 추가한 모델/conf 에 맞춰 PATCHERS·CASES 를 채운다.
   실행에는 data/ 의 train/test(+augment 시 source)가 있어야 한다(학습은 안 함).

사용:
    uv run python scripts/check_fold_inputs.py before.json   # 리팩토링 前
    # ... 리팩토링 ...
    uv run python scripts/check_fold_inputs.py after.json
    diff before.json after.json && echo "입력 동등 — frozen 멤버 OOF 불변 보장"
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src import config

CAP: list[dict] = []


def _h(obj) -> str:
    """DataFrame/Series/ndarray/None 의 결정적 해시(컬럼순 정렬·index 포함)."""
    if obj is None:
        return "none"
    if isinstance(obj, pd.DataFrame):
        cols = sorted(map(str, obj.columns))
        v = pd.util.hash_pandas_object(obj[cols], index=True).values
        return hashlib.sha256(v.tobytes()).hexdigest()[:16]
    if isinstance(obj, pd.Series):
        v = pd.util.hash_pandas_object(obj, index=True).values
        return hashlib.sha256(v.tobytes()).hexdigest()[:16]
    return hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest()[:16]


class _DummyBooster:
    best_iteration = 0

    def predict(self, X, *a, **k):
        CAP.append({"role": "pred", "X": _h(X)})
        return np.zeros(len(X))


class _DummyXGB:
    best_iteration = 0

    def __init__(self, *a, **k):
        pass

    def fit(self, X, y, *a, sample_weight=None, eval_set=None, **k):
        xval = eval_set[0][0] if eval_set else None
        CAP.append({"role": "fit", "X": _h(X), "y": _h(y), "w": _h(sample_weight), "xval": _h(xval)})
        return self

    def predict_proba(self, X, *a, **k):
        CAP.append({"role": "pred", "X": _h(X)})
        return np.zeros((len(X), 2))


def _patch_lgbm():
    """LGBM(train_lgbm.py): lgb.train 을 더미로 (함수 패치)."""
    import src.train_lgbm as T

    def dummy_train(params, train_set, num_boost_round=0, valid_sets=None, callbacks=None, **k):
        xval = valid_sets[0].data if valid_sets else None
        CAP.append({"role": "fit", "X": _h(train_set.data), "y": _h(train_set.label),
                    "w": _h(train_set.weight), "xval": _h(xval)})
        return _DummyBooster()

    T.lgb.train = dummy_train
    return T


def _patch_xgb():
    """XGB(train_xgb.py): XGBClassifier 를 더미로 (클래스 패치)."""
    import src.train_xgb as T

    T.xgb.XGBClassifier = _DummyXGB
    return T


# 모델 키 -> 패처. 새 모델 추가 시 여기에 등록.
PATCHERS = {"lgbm": _patch_lgbm, "xgb": _patch_xgb}

# (모델, features yaml, augment) — 분기(no-aug/te 등)를 커버하도록 추가.
CASES = [
    ("lgbm", "base", False),
    ("xgb", "base", False),
    # ("lgbm", "te_example", False),         # TE 분기 (실제 컬럼 채운 yaml 로)
    # ("lgbm", "base", True),                # augment 분기 (data/external 있을 때)
]


def _cfg(model: str, feats: str, aug: bool) -> OmegaConf:
    return OmegaConf.create({
        "exp_id": f"check_{model}",
        "notes": "",
        "use_wandb": False,
        "seed": config.SEED,
        "n_folds": config.N_FOLDS,
        "max_folds": None,
        "kill_criterion": "",
        "model": OmegaConf.load(f"conf/model/{model}.yaml"),
        "features": OmegaConf.load(f"conf/features/{feats}.yaml"),
        "augment": {"enabled": aug, "weight": 1.0},
    })


def run_case(model: str, feats: str, aug: bool) -> list[dict]:
    CAP.clear()
    T = PATCHERS[model]()
    d = Path(tempfile.mkdtemp())
    config.OOF_DIR = config.SUBMISSION_DIR = config.LOG_DIR = d  # 실제 산출물 쓰기 회피
    T.run(_cfg(model, feats, aug))
    return list(CAP)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용: uv run python scripts/check_fold_inputs.py <출력json>")
        sys.exit(1)
    out = {}
    for m, f, a in CASES:
        key = f"{m}/{f}/aug={a}"
        out[key] = run_case(m, f, a)
        print(f"{key}: folds={sum(1 for c in out[key] if c['role'] == 'fit')}")
    json.dump(out, open(sys.argv[1], "w"), indent=2)
    print("saved", sys.argv[1])
