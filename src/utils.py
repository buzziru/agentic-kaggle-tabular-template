"""공통 유틸: 시드 고정, git 해시, 구조화 JSON 실험 로깅, df 요약.

토큰 절약 원칙에 따라 DataFrame 전체 출력 대신 요약 헬퍼(`resumetable`)를 제공한다.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src import config


def read_pred(directory: Path, name: str) -> pd.DataFrame:
    """oof/submission 예측을 로드한다 — parquet 우선, csv 폴백.

    스택 멤버 예측은 디스크 절약을 위해 parquet(float32) 로 저장될 수 있다.
    parquet 이 있으면 그것을, 없으면 기존 csv 를 읽는다 (혼재 상태 호환).

    Args:
        directory: `config.OOF_DIR` 또는 `config.SUBMISSION_DIR`.
        name: 멤버/실험 ID (확장자 제외).

    Returns:
        예측 DataFrame.
    """
    pq = directory / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(directory / f"{name}.csv")


def load_env() -> None:
    """프로젝트 루트 `.env` 를 환경변수로 로드한다 (W&B/Kaggle 인증).

    이미 설정된 환경변수는 덮어쓰지 않는다.
    """
    load_dotenv(config.ROOT_DIR / ".env")


def seed_everything(seed: int = config.SEED) -> None:
    """난수 시드를 전역 고정한다.

    Args:
        seed: 고정할 시드 값.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_git_hash() -> str:
    """현재 커밋 해시를 반환한다 (재현성 로깅용).

    Returns:
        짧은 커밋 해시. git repo 가 아니거나 실패 시 "unknown".
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=config.ROOT_DIR,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def now_iso() -> str:
    """현재 UTC 시각을 ISO 문자열로 반환한다."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_experiment(
    exp_id: str,
    model: str,
    features: list[str],
    cv_scores: list[float],
    params: dict[str, Any],
    *,
    best_iters: list[int] | None = None,
    lb_score: float | None = None,
    notes: str = "",
    kill_criterion: str = "",
    cv_strategy: str | None = None,
    log_dir: Path | None = None,
) -> Path:
    """실험 결과를 구조화 JSON 으로 저장한다 (1 실험 = 1 파일).

    Args:
        exp_id: 실험 식별자 (예: "exp_001").
        model: 모델 이름 (예: "lgbm").
        features: 사용한 피처 목록.
        cv_scores: fold 별 검증 점수.
        params: 모델 하이퍼파라미터.
        best_iters: fold 별 best_iteration (early-stopping 모델). cap 미발화(미완 학습)
            점검용 — best_iter 가 num_boost_round 에 붙으면 cap 상향 필요 (CLAUDE.md 원칙).
        lb_score: 리더보드 점수 (제출 후 갱신, 기본 None).
        notes: 자유 메모.
        kill_criterion: 스파이크 전 사전 중단조건(과몰입 구조적 가드).
        cv_strategy: 실제 사용한 CV 라벨(예: "StratifiedKFold_7"). None 이면 config 기본값.
            ⚠️ n_folds 오버라이드·부분 실행을 정직하게 남기려면 호출부가 실제 값을 넘긴다.
        log_dir: 로그 저장 디렉터리. None 이면 호출 시점의 `config.LOG_DIR` 사용
            (기본인자 동결 방지 — Kaggle 등에서 런타임 오버라이드가 반영되게).

    Returns:
        저장된 JSON 파일 경로.
    """
    if log_dir is None:
        log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(cv_scores, dtype=float)
    record: dict[str, Any] = {
        "exp_id": exp_id,
        "timestamp": now_iso(),
        "git_hash": get_git_hash(),
        "model": model,
        "features": features,
        "cv_strategy": cv_strategy or f"{config.CV_STRATEGY}_{config.N_FOLDS}",
        "cv_scores": [round(float(s), 6) for s in scores],
        "cv_mean": round(float(scores.mean()), 6),
        "cv_std": round(float(scores.std()), 6),
        "best_iters": [int(b) for b in best_iters] if best_iters is not None else None,
        "lb_score": lb_score,
        "params": params,
        "notes": notes,
        "kill_criterion": kill_criterion,
    }
    path = log_dir / f"{exp_id}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def cast_target(y: pd.Series) -> pd.Series:
    """config.PROBLEM_TYPE 에 맞춰 타깃 dtype 을 캐스팅한다.

    회귀는 float, 분류(binary)는 int. 학습/OOF-TE 가 올바른 타깃 타입을 받게 한다
    (binary 전제 `.astype(int)` 하드코딩 제거 — 회귀 1급 지원).

    Args:
        y: 원본 타깃 Series.

    Returns:
        캐스팅된 타깃 Series.
    """
    if config.PROBLEM_TYPE == "regression":
        return y.astype(float)
    return y.astype(int)


def get_scorer(name: str | None = None) -> Callable[[Any, Any], float]:
    """config.METRIC(또는 name)에 맞는 (y_true, y_pred)->float scorer 반환.

    train_common/stack 의 점수 계산이 `config.METRIC` 하나로 자동 결정된다(하드코딩 제거).
    지원: auc·logloss·rmse·mae·accuracy. 새 지표는 여기 + `greater_is_better` 에 추가.

    Args:
        name: 지표명. None 이면 `config.METRIC`.

    Returns:
        (y_true, y_pred) -> float scorer.
    """
    from sklearn import metrics  # 지연 임포트

    name = (name or config.METRIC).lower()
    if name == "auc":
        return lambda y, p: float(metrics.roc_auc_score(y, p))
    if name == "logloss":
        return lambda y, p: float(metrics.log_loss(y, p))
    if name == "rmse":
        return lambda y, p: float(metrics.mean_squared_error(y, p) ** 0.5)
    if name == "mae":
        return lambda y, p: float(metrics.mean_absolute_error(y, p))
    if name == "accuracy":  # ⚠️ 이진분류 전제(임계 0.5). 다중분류는 argmax 로 바꿀 것.
        return lambda y, p: float(metrics.accuracy_score(y, (np.asarray(p) > 0.5).astype(int)))
    raise ValueError(f"미지원 지표 '{name}' — utils.get_scorer 에 추가하라")


def greater_is_better(name: str | None = None) -> bool:
    """지표가 높을수록 좋은가 (정렬·best 선택 방향)."""
    name = (name or config.METRIC).lower()
    return name not in {"logloss", "rmse", "mae"}


# 문제 유형별 허용 지표·CV 전략 (validate_problem_config 의 일치 검사용).
_METRIC_BY_PROBLEM: dict[str, set[str]] = {
    "binary": {"auc", "logloss", "accuracy"},
    "regression": {"rmse", "mae"},
    "multiclass": {"accuracy", "logloss"},
}
_CV_BY_PROBLEM: dict[str, set[str]] = {
    "binary": {"StratifiedKFold", "GroupKFold"},
    "regression": {"KFold", "GroupKFold"},
    "multiclass": {"StratifiedKFold", "GroupKFold"},
}


def _objective_problem_type(objective: str) -> str | None:
    """모델 objective 문자열에서 문제 유형을 추정한다 (불일치 검사용). 모호하면 None.

    lgbm/xgb objective 공통 키워드 기반 — "binary"→binary, "multi"→multiclass,
    회귀 키워드(reg/squared/poisson/tweedie/…)→regression. "reg:logistic" 처럼 "reg"
    접두가 있으면 회귀로 본다(바이너리 오탐 방지 — 'logistic' 단독은 키로 쓰지 않음).
    """
    o = objective.lower()
    if "multi" in o:
        return "multiclass"
    if "binary" in o:
        return "binary"
    reg_keys = ("reg", "squared", "rmse", "mae", "mse", "l1", "l2",
                "poisson", "tweedie", "gamma", "huber", "quantile", "absoluteerror")
    if any(k in o for k in reg_keys):
        return "regression"
    return None


def validate_problem_config(model_objective: str | None = None) -> None:
    """problem_type·metric·cv_strategy·model_objective 의 명백한 불일치를 차단한다.

    셋(+선택적 objective)이 어긋나면 점수가 조용히 틀린다(예: 회귀에 StratifiedKFold,
    binary 에 rmse, objective=regression 인데 PROBLEM_TYPE=binary). 학습/스택 시작 전에
    호출해 빠르게 실패시킨다 — 명백한 모순만 잡고 모호하면 통과시킨다(오탐 회피).

    Args:
        model_objective: conf 모델 objective (예: "binary"/"regression"). None 이면 생략.

    Raises:
        ValueError: 알려진 불일치(미지원 PROBLEM_TYPE 포함).
    """
    pt = config.PROBLEM_TYPE
    if pt not in _METRIC_BY_PROBLEM:
        raise ValueError(f"config.PROBLEM_TYPE='{pt}' 미지원 — binary/regression/multiclass 중 하나")
    if config.METRIC not in _METRIC_BY_PROBLEM[pt]:
        raise ValueError(
            f"PROBLEM_TYPE={pt} ↔ METRIC='{config.METRIC}' 불일치 — 허용: {sorted(_METRIC_BY_PROBLEM[pt])}"
        )
    if config.CV_STRATEGY not in _CV_BY_PROBLEM[pt]:
        raise ValueError(
            f"PROBLEM_TYPE={pt} ↔ CV_STRATEGY='{config.CV_STRATEGY}' 불일치 — "
            f"허용: {sorted(_CV_BY_PROBLEM[pt])} (예: 회귀에 StratifiedKFold 는 연속 타깃 계층화 불가)"
        )
    if model_objective:
        obj_pt = _objective_problem_type(model_objective)
        if obj_pt is not None and obj_pt != pt:
            raise ValueError(
                f"PROBLEM_TYPE={pt} ↔ model objective='{model_objective}'(→{obj_pt}) 불일치 — "
                "conf/model/*.yaml 의 objective 를 PROBLEM_TYPE 에 맞춰라"
            )


def load_logs(log_dir: Path | None = None) -> pd.DataFrame:
    """experiments/logs/*.json 전부를 한 표로 집계한다 (통합 리더보드/레지스트리 뷰).

    base 모델·스택 멤버·앙상블이 모두 동일 로그 스트림에 기록되므로, 이 한 함수가
    실험 로그·feature recipe(레지스트리)·lb_score(제출 결과)를 한눈에 모은다.

    Args:
        log_dir: 로그 디렉터리. None 이면 `config.LOG_DIR`.

    Returns:
        exp 별 1행 DataFrame. 로그가 없으면 빈 DataFrame.
    """
    if log_dir is None:
        log_dir = config.LOG_DIR
    rows: list[dict[str, Any]] = []
    for p in sorted(log_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        recipe = (d.get("params") or {}).get("feature_recipe") or {}
        rows.append(
            {
                "exp_id": d.get("exp_id"),
                "model": d.get("model"),
                "cv_mean": d.get("cv_mean"),
                "cv_std": d.get("cv_std"),
                "lb_score": d.get("lb_score"),
                "n_feat": len(d.get("features") or []),
                "feature_builder": recipe.get("feature_builder"),
                "notes": (d.get("notes") or "")[:60],
                "timestamp": d.get("timestamp"),
            }
        )
    return pd.DataFrame(rows)


def resumetable(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame 요약 테이블을 반환한다 (토큰 절약용 핵심 메타).

    피처별 데이터타입·결측값 수·고유값 수와 샘플값(첫/둘째)을 한 표로 정리한다.
    전체 출력 대신 이 표 하나로 데이터 개요를 파악한다.

    Args:
        df: 요약할 DataFrame.

    Returns:
        피처별 요약 행을 담은 DataFrame (피처, 데이터타입, 결측값 개수, 고유값 개수, 샘플값).
    """
    summary = pd.DataFrame(df.dtypes, columns=["데이터타입"])
    summary = summary.reset_index().rename(columns={"index": "피처"})
    summary["결측값 개수"] = df.isnull().sum().values
    summary["고유값 개수"] = df.nunique().values
    summary["첫번째 값"] = df.iloc[0].values if len(df) > 0 else None
    summary["두번째 값"] = df.iloc[1].values if len(df) > 1 else None
    return summary
