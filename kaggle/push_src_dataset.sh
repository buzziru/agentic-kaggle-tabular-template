#!/usr/bin/env bash
# src/ + conf/ 를 Kaggle Dataset({{KAGGLE_USER}}/{{SRC_DATASET}})으로 push.
# Kaggle GPU 노트북이 import 해 쓰는 코드 번들 (docs/wiki/kaggle_jobs.md).
#
# 사용:
#   bash kaggle/push_src_dataset.sh create            # 최초 생성
#   bash kaggle/push_src_dataset.sh version "변경 메모"  # 코드 변경 후 갱신
#
# 인증: .env 의 KAGGLE_USERNAME / KAGGLE_KEY.
# ▶ dataset-metadata.json 의 id 를 본인 {{KAGGLE_USER}}/{{SRC_DATASET}} 로 먼저 채울 것.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

MODE="${1:-version}"
MSG="${2:-update src bundle}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# 코드만 번들 (데이터·산출물·시크릿 제외)
cp -r src conf "$STAGE"/
cp kaggle/dataset-metadata.json "$STAGE"/
# frozen splits 동반 (원격 load-first 경로 — 큰 csv 는 제외, splits parquet 만).
if [ -d data/splits ]; then
  mkdir -p "$STAGE/data"
  cp -r data/splits "$STAGE/data/"
fi
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} +

echo "[push] staging:"; ls -1 "$STAGE"
case "$MODE" in
  create)  uv run kaggle datasets create  -p "$STAGE" -r zip ;;
  version) uv run kaggle datasets version -p "$STAGE" -m "$MSG" -r zip ;;
  *) echo "MODE 는 create|version"; exit 1 ;;
esac
echo "[push] 완료: https://www.kaggle.com/datasets/{{KAGGLE_USER}}/{{SRC_DATASET}}"
