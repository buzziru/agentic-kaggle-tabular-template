"""스태킹/블렌딩 메타러너 — 스택 풀의 디커플링 경계.

⚠️ 이 모듈은 **OOF 계약만** 소비한다 — 모델 코드/내부에 의존하지 않는다:
  experiments/oof/<member>.csv        = [id, oof]   (동일 seed=42 fold 의 leak-free OOF)
  experiments/submissions/<member>.csv = [id, <target>]
어떤 모델을 추가하든 train_common 이 이 형식을 보장하므로, 스택 풀은 멤버 수가 늘어도
모델별 특수 코드가 0 이다 (= "모델 추가로 스택이 특정 코드에 묶이는" 문제 차단).

base OOF 를 메타 피처로, **같은 fold 로 메타를 CV 학습**해 meta-OOF 산출(누수 없음).
메타러너 4종(equal/rank_mean/logistic/nnls) 비교 + corr 리포트.
⚠️ GBDT 메타 금지(피처 소수 과적합). 스택 멤버 추가 판정은 in-sample meta-OOF 가 아니라
   held-out/nested 로(스태커는 단일모델과 별개 레짐 — CLAUDE.md 검증 전략).

실행:
    uv run python -m src.stack --members exp_001,exp_010,exp_011 --tag stack_v1
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from src import config, cv, data, utils


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _load(members: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """base OOF·test 행렬과 타깃을 정렬해 로드한다 (id 정합 = 계약 강제)."""
    train = data.load_train()
    y = train[config.TARGET_COL].astype(int).to_numpy()
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
    """비음수·합=1 가중을 logloss 최소화로 적합 (블렌드 가중)."""
    k = X.shape[1]
    w0 = np.full(k, 1 / k)
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bnds = [(0.0, 1.0)] * k
    res = minimize(lambda w: log_loss(y, np.clip(X @ w, 1e-7, 1 - 1e-7)),
                   w0, method="SLSQP", bounds=bnds, constraints=cons)
    return res.x


def _fit_predict(name: str, Xtr, ytr, Xva) -> np.ndarray:
    if name == "equal":
        return Xva.mean(axis=1)
    if name == "rank_mean":
        return np.column_stack([pd.Series(Xva[:, j]).rank().to_numpy() for j in range(Xva.shape[1])]
                               ).mean(axis=1) / len(Xva)
    if name == "logistic":
        m = LogisticRegression(C=1.0, max_iter=1000)
        m.fit(_logit(Xtr), ytr)
        return m.predict_proba(_logit(Xva))[:, 1]
    if name == "nnls":
        w = _nnls_weights(Xtr, ytr)
        return Xva @ w
    raise ValueError(name)


def _cv_meta(name: str, X: np.ndarray, y: np.ndarray, X_test: np.ndarray, folds: list) -> dict:
    """동일 fold 로 meta-OOF 산출 + 전체 재적합 test 예측."""
    oof = np.zeros(len(y))
    for tr, va in folds:
        oof[va] = _fit_predict(name, X[tr], y[tr], X[va])
    test = _fit_predict(name, X, y, X_test)  # 전체 재적합
    return {"name": name, "oof_auc": roc_auc_score(y, oof), "oof": oof, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=str, required=True, help="쉼표구분 exp_id 들")
    ap.add_argument("--tag", type=str, default="stack")
    args = ap.parse_args()
    members = [m.strip() for m in args.members.split(",")]

    X, X_test, y = _load(members)
    folds = cv.get_folds(y)  # seed=42, base 와 동일 분할

    print(f"=== members: {members} ===")
    print("개별 OOF:")
    for j, m in enumerate(members):
        print(f"  {m}: {roc_auc_score(y, X[:, j]):.6f}")
    print("Pearson corr:")
    print(pd.DataFrame(np.corrcoef(X.T), index=members, columns=members).round(4).to_string())

    results = [_cv_meta(n, X, y, X_test, folds) for n in ["equal", "rank_mean", "logistic", "nnls"]]
    print("\n=== meta-OOF ===")
    for r in sorted(results, key=lambda d: -d["oof_auc"]):
        print(f"  {r['name']:>10}: {r['oof_auc']:.6f}")

    w_nnls = _nnls_weights(X, y)
    print("\nnnls 가중:", {m: round(float(w), 4) for m, w in zip(members, w_nnls)})

    best = max(results, key=lambda d: d["oof_auc"])
    config.OOF_DIR.mkdir(parents=True, exist_ok=True)
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    train = data.load_train()
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
        cv_scores=[best["oof_auc"]],
        params={
            "members": members,
            "meta_winner": best["name"],
            "meta_oof": {r["name"]: round(float(r["oof_auc"]), 6) for r in results},
            "nnls_weights": {m: round(float(w), 4) for m, w in zip(members, w_nnls)},
        },
        notes=f"stack over {len(members)} members; meta={best['name']}",
    )
    print(f"\n최고 메타 = {best['name']} ({best['oof_auc']:.6f}) → 저장: {args.tag}_{best['name']}.csv")


if __name__ == "__main__":
    main()
