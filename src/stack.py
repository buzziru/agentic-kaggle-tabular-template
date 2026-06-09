"""스태킹/블렌딩 메타러너 — 스택 풀의 디커플링 경계.

⚠️ 이 모듈은 **OOF 계약만** 소비한다 — 모델 코드/내부에 의존하지 않는다:
  experiments/oof/<member>.csv        = [id, oof]   (동일 seed=42 fold 의 leak-free OOF)
  experiments/submissions/<member>.csv = [id, <target>]
어떤 모델을 추가하든 train_common 이 이 형식을 보장하므로, 스택 풀은 멤버 수가 늘어도
모델별 특수 코드가 0 이다 (= "모델 추가로 스택이 특정 코드에 묶이는" 문제 차단).

base OOF 를 메타 피처로, **같은 fold 로 메타를 CV 학습**해 meta-OOF 산출(누수 없음).
메타러너는 `config.PROBLEM_TYPE` 으로 갈린다(src 와 동일 방향 — binary/regression 1급):
  - binary:     equal / rank_mean / logistic / nnls(logloss 최소 블렌드)
  - regression: equal / linear / nnls(MSE 최소 블렌드)   ← rank_mean·logistic 은 binary 전용
⚠️ multiclass 는 1-D OOF 계약(단일 oof 열) 밖이라 막는다(확장점 — OOF 계약 확장 필요).
⚠️ GBDT 메타 금지(피처 소수 과적합). 스택 멤버 추가 판정은 in-sample meta-OOF 가 아니라
   held-out/nested 로(스태커는 단일모델과 별개 레짐 — CLAUDE.md 검증 전략).

실행:
    uv run python -m src.stack --members exp_001,exp_010,exp_011 --tag stack_v1
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

from src import config, cv, data, utils

_STRATEGIES = {"StratifiedKFold", "KFold", "GroupKFold"}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _resolve_regime(members: list[str]) -> tuple[str, int]:
    """멤버 로그의 cv_strategy 로 meta-CV 레짐(strategy, n_folds)을 결정·검증한다.

    멤버는 **동일 fold 분할**에서 나온 OOF 여야 스택이 유효하다. 각 멤버 로그의
    cv_strategy(예: 'StratifiedKFold_7')를 읽어 전원이 일치하는지 확인하고 그 레짐을
    쓴다 → base 의 실제 n_folds 를 따라간다(config.N_FOLDS 하드코딩으로 인한 override
    불일치 방지). 로그가 없거나 파싱 불가면 config 기본으로 폴백(경고).

    Args:
        members: 스택 멤버 exp_id 들.

    Returns:
        (strategy, n_folds) — cv.get_folds 에 그대로 전달.

    Raises:
        ValueError: 멤버 간 CV 레짐 불일치 (이질적 fold 는 스택 불가).
    """
    labels = []
    for m in members:
        p = config.LOG_DIR / f"{m}.json"
        if not p.exists():
            print(f"⚠️ [{m}] 로그 없음 — config 기본 레짐({config.CV_STRATEGY}_{config.N_FOLDS})으로 폴백")
            return config.CV_STRATEGY, config.N_FOLDS
        labels.append(json.loads(p.read_text()).get("cv_strategy", ""))

    if len(set(labels)) != 1:
        raise ValueError(
            f"멤버 CV 레짐 불일치 {dict(zip(members, labels))} — 동일 fold 분할 멤버만 스택 가능 "
            "(같은 strategy·n_folds·seed 로 재학습하라)."
        )
    strategy, _, nf = labels[0].rpartition("_")
    if strategy not in _STRATEGIES or not nf.isdigit():
        print(f"⚠️ cv_strategy='{labels[0]}' 파싱 불가(부분 실행 등) — config 기본 레짐으로 폴백")
        return config.CV_STRATEGY, config.N_FOLDS
    if strategy != config.CV_STRATEGY:
        print(f"⚠️ 멤버 전략 '{strategy}' ≠ config.CV_STRATEGY '{config.CV_STRATEGY}' — 멤버(base) 전략을 따른다")
    return strategy, int(nf)


def _load(members: list[str], train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """base OOF·test 행렬과 타깃을 정렬해 로드한다 (id 정합 = 계약 강제)."""
    y = utils.cast_target(train[config.TARGET_COL]).to_numpy()  # binary=int·regression=float
    sub_ids = data.load_sample_submission()[config.ID_COL]

    oof_cols, test_cols = [], []
    for m in members:
        o = utils.read_pred(config.OOF_DIR, m)
        assert o[config.ID_COL].equals(train[config.ID_COL]), f"{m} OOF id 불일치"
        oof_cols.append(o["oof"].to_numpy())
        s = utils.read_pred(config.SUBMISSION_DIR, m)
        assert s[config.ID_COL].equals(sub_ids), f"{m} submission id 불일치"
        test_cols.append(s[config.TARGET_COL].to_numpy())
    return np.column_stack(oof_cols), np.column_stack(test_cols), y


def _nnls_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """비음수·합=1 블렌드 가중을 손실 최소화로 적합한다.

    손실은 PROBLEM_TYPE 에 맞춘다 — regression=MSE, binary=logloss. 두 경우 모두
    가중 블렌드 `X @ w` 가 그대로 예측이라 볼록 결합(convex blend)으로 안전하다.
    """
    k = X.shape[1]
    w0 = np.full(k, 1 / k)
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bnds = [(0.0, 1.0)] * k
    if config.PROBLEM_TYPE == "regression":
        loss = lambda w: float(np.mean((X @ w - y) ** 2))  # noqa: E731
    else:
        loss = lambda w: log_loss(y, np.clip(X @ w, 1e-7, 1 - 1e-7))  # noqa: E731
    res = minimize(loss, w0, method="SLSQP", bounds=bnds, constraints=cons)
    return res.x


def _fit_predict(name: str, Xtr, ytr, Xva) -> np.ndarray:
    if name == "equal":  # 단순 평균 (binary·regression 공통)
        return Xva.mean(axis=1)
    if name == "rank_mean":  # binary 전용 — 순위 평균(회귀는 스케일을 파괴)
        return np.column_stack([pd.Series(Xva[:, j]).rank().to_numpy() for j in range(Xva.shape[1])]
                               ).mean(axis=1) / len(Xva)
    if name == "logistic":  # binary 전용 — logit 공간 로지스틱 메타
        m = LogisticRegression(C=1.0, max_iter=1000)
        m.fit(_logit(Xtr), ytr)
        return m.predict_proba(_logit(Xva))[:, 1]
    if name == "linear":  # regression 전용 — 비제약 선형 메타
        m = LinearRegression()
        m.fit(Xtr, ytr)
        return m.predict(Xva)
    if name == "nnls":  # 볼록 블렌드 (가중 손실은 PROBLEM_TYPE 별)
        w = _nnls_weights(Xtr, ytr)
        return Xva @ w
    raise ValueError(name)


def _cv_meta(name: str, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, folds: list, scorer) -> dict:
    """동일 fold 로 meta-OOF 산출 + 전체 재적합 test 예측."""
    oof = np.zeros(len(y))
    for tr, va in folds:
        oof[va] = _fit_predict(name, X[tr], y[tr], X[va])
    test = _fit_predict(name, X, y, X_test)  # 전체 재적합
    return {"name": name, "oof_score": scorer(y, oof), "oof": oof, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=str, required=True, help="쉼표구분 exp_id 들")
    ap.add_argument("--tag", type=str, default="stack")
    args = ap.parse_args()
    members = [m.strip() for m in args.members.split(",")]

    if config.PROBLEM_TYPE == "multiclass":
        raise NotImplementedError(
            "stack 은 1-D OOF 계약(단일 oof 열)만 소비한다 — multiclass 는 멤버 OOF 가 k열이라 "
            "_load/메타러너/출력 계약을 확장해야 한다(확장점). 템플릿은 binary/regression 1급."
        )
    utils.validate_problem_config()  # problem_type↔metric↔cv_strategy 명백한 불일치 차단

    train = data.load_train()
    X, X_test, y = _load(members, train)
    # ⚠️ base 학습과 **동일 검증 레짐**으로 meta-OOF 산출 — 멤버 로그에서 실제 strategy·n_folds
    #    를 읽어(override 반영) groups 와 함께 전달. seed 는 config.SEED 로 base 와 공통.
    strategy, n_folds = _resolve_regime(members)
    folds = cv.get_folds(y, n_folds=n_folds, groups=cv.make_groups(train), strategy=strategy)
    scorer = utils.get_scorer()                 # config.METRIC 기준
    greater = utils.greater_is_better()

    print(f"=== members: {members} ({config.METRIC}) ===")
    print("개별 OOF:")
    for j, m in enumerate(members):
        print(f"  {m}: {scorer(y, X[:, j]):.6f}")
    print("Pearson corr:")
    print(pd.DataFrame(np.corrcoef(X.T), index=members, columns=members).round(4).to_string())

    metas = (
        ["equal", "linear", "nnls"]
        if config.PROBLEM_TYPE == "regression"
        else ["equal", "rank_mean", "logistic", "nnls"]
    )
    results = [_cv_meta(n, X, y, X_test, folds, scorer) for n in metas]
    print("\n=== meta-OOF ===")
    for r in sorted(results, key=lambda d: d["oof_score"], reverse=greater):
        print(f"  {r['name']:>10}: {r['oof_score']:.6f}")

    w_nnls = _nnls_weights(X, y)
    print("\nnnls 가중:", {m: round(float(w), 4) for m, w in zip(members, w_nnls)})

    best = (max if greater else min)(results, key=lambda d: d["oof_score"])
    config.OOF_DIR.mkdir(parents=True, exist_ok=True)
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({config.ID_COL: train[config.ID_COL], "oof": best["oof"]}).to_csv(
        config.OOF_DIR / f"{args.tag}_{best['name']}.csv", index=False)
    sub = data.load_sample_submission()
    sub[config.TARGET_COL] = best["test"]
    sub.to_csv(config.SUBMISSION_DIR / f"{args.tag}_{best['name']}.csv", index=False)

    # 앙상블 기록을 base 모델과 동일한 로그 스트림에 영속화 (members·가중·meta-OOF).
    # → 스택도 experiments/logs 에 남아 summarize/리더보드에 함께 노출(비대칭 제거).
    utils.log_experiment(
        exp_id=f"{args.tag}_{best['name']}",
        model=f"stack:{best['name']}",
        features=members,
        cv_scores=[best["oof_score"]],
        params={
            "members": members,
            "meta_winner": best["name"],
            "meta_oof": {r["name"]: round(float(r["oof_score"]), 6) for r in results},
            "nnls_weights": {m: round(float(w), 4) for m, w in zip(members, w_nnls)},
        },
        notes=f"stack over {len(members)} members; meta={best['name']}",
    )
    print(f"\n최고 메타 = {best['name']} ({best['oof_score']:.6f}) → 저장: {args.tag}_{best['name']}.csv")


if __name__ == "__main__":
    main()
