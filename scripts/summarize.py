"""실험 로그 통합 뷰 — experiments/logs/*.json 을 한 리더보드 표로 집계.

base 모델·스택 멤버·앙상블이 모두 같은 로그 스트림(utils.log_experiment)에 남으므로,
이 한 명령이 실험 로그 + feature recipe(레지스트리) + lb_score(제출 결과)를 한눈에 모은다.
제출 이력 전체는 experiments/submission_history.csv (predict.record_submission).

사용:
    uv run python scripts/summarize.py            # cv_mean 내림차순 리더보드
    uv run python scripts/summarize.py --sort lb_score
"""

from __future__ import annotations

import argparse

from src import config, utils


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", default="cv_mean", help="정렬 컬럼 (cv_mean/lb_score/timestamp)")
    args = ap.parse_args()

    df = utils.load_logs()
    if df.empty:
        print(f"로그 없음: {config.LOG_DIR}")
        return

    by = args.sort if args.sort in df.columns else "cv_mean"
    df = df.sort_values(by, ascending=False, na_position="last")
    print(f"=== 실험 리더보드 ({len(df)}건, sort={by}) ===")
    print(df.to_string(index=False))

    hist = config.EXPERIMENTS_DIR / "submission_history.csv"
    print(f"\nSubmission History: {hist}" + ("" if hist.exists() else " (아직 없음)"))


if __name__ == "__main__":
    main()
