"""추론/제출 헬퍼 + Submission History 원장.

train_common 이 이미 experiments/submissions/<exp_id>.csv 를 생성하므로, 이 모듈은
Kaggle 제출 명령 구성 + 제출 이력 기록(원장)을 제공한다.

제출 (셸):
    set -a; . ./.env; set +a
    kaggle competitions submit -c <COMPETITION> \
        -f experiments/submissions/<exp_id>.csv -m "<메시지>"
제출 후 결과를 원장에 기록:
    uv run python -c "from src import predict; predict.record_submission('exp_001','baseline', lb_public=0.94)"
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config, utils

SUBMISSION_HISTORY: Path = config.EXPERIMENTS_DIR / "submission_history.csv"


def submission_path(exp_id: str) -> Path:
    """주어진 실험의 제출 파일 경로를 반환한다.

    Args:
        exp_id: 실험 식별자.

    Returns:
        제출 CSV 경로.
    """
    return config.SUBMISSION_DIR / f"{exp_id}.csv"


def kaggle_submit_command(exp_id: str, message: str) -> str:
    """Kaggle CLI 제출 명령 문자열을 생성한다 (.env 인증 전제).

    Args:
        exp_id: 실험 식별자.
        message: 제출 메시지.

    Returns:
        실행 가능한 셸 명령 문자열.
    """
    path = submission_path(exp_id)
    return (
        "set -a; . ./.env; set +a && "
        f"kaggle competitions submit -c {config.COMPETITION} "
        f'-f {path} -m "{message}"'
    )


def record_submission(
    exp_id: str,
    message: str,
    *,
    lb_public: float | None = None,
    lb_private: float | None = None,
) -> Path:
    """제출 1건을 Submission History 원장에 append 하고, 로그의 lb_score 도 동기화한다.

    원장(`experiments/submission_history.csv`) = 제출 시각·exp·메시지·Public/Private LB 의
    단일 이력. 동시에 해당 exp 의 JSON 로그 `lb_score` 를 갱신해 `utils.load_logs()`
    리더보드에 LB 가 함께 보이도록 한다 (실험 로그 ↔ 제출 결과 일괄 관리).

    Args:
        exp_id: 제출한 실험 식별자.
        message: Kaggle 제출 메시지.
        lb_public: Public LB 점수 (알면).
        lb_private: Private LB 점수 (대회 종료 후).

    Returns:
        원장 CSV 경로.
    """
    SUBMISSION_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": utils.now_iso(),
        "exp_id": exp_id,
        "message": message,
        "lb_public": lb_public,
        "lb_private": lb_private,
        "file": submission_path(exp_id).name,
    }
    df = pd.read_csv(SUBMISSION_HISTORY) if SUBMISSION_HISTORY.exists() else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(SUBMISSION_HISTORY, index=False)

    # 로그의 lb_score 동기화 (Private 우선, 없으면 Public).
    score = lb_private if lb_private is not None else lb_public
    log_path = config.LOG_DIR / f"{exp_id}.json"
    if score is not None and log_path.exists():
        d = json.loads(log_path.read_text())
        d["lb_score"] = score
        log_path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    return SUBMISSION_HISTORY
