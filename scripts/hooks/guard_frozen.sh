#!/usr/bin/env bash
# guard_frozen.sh — PreToolUse(Edit|MultiEdit|Write|NotebookEdit) 훅.
# frozen 스택 멤버의 "산출물"(experiments/{oof,submissions,logs}/<exp_id>.*)과
# 이미 생성된 data/splits/*.parquet 의 수정을 차단한다.
# ⚠️ 동결 대상은 산출물(아티팩트)이지 src/ 공유 코드가 아니다 — 코드는 차단하지 않는다.
#    공유 코드 변경은 scripts/check_fold_inputs.py 입력 동등성으로 보호한다(MERGE 원칙 1).
# exit 0: 허용 / exit 2: 차단 (stderr 가 Claude 에게 전달됨). 인프라 오류는 fail-open.

set -uo pipefail

INPUT="$(cat)"
PYBIN="$(command -v python3 || command -v python || true)"
[ -z "$PYBIN" ] && exit 0   # python 부재 환경: 가드 불가 → 통과(1차 방어는 에이전트 지시문)

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"

REASON="$("$PYBIN" - "$INPUT" "$PROJECT_DIR" <<'PY'
import json, sys, os, re

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    data = json.loads(sys.argv[1])
except Exception:
    sys.exit(0)
project = sys.argv[2]
ti = data.get("tool_input", {}) or {}
fp = ti.get("file_path") or ti.get("notebook_path") or ""
if not fp:
    sys.exit(0)

np = fp.replace("\\", "/")

# 규칙 1: frozen 스택 멤버 산출물 (exp_id 가 frozen.txt 에 등록된 경우)
m = re.search(r"(?:^|/)experiments/(?:oof|submissions|logs)/([^/]+)\.[^/]+$", np)
if m:
    exp_id = m.group(1)
    frozen = os.path.join(project, "frozen.txt")
    ids = set()
    if os.path.isfile(frozen):
        with open(frozen, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    ids.add(ln)
    if exp_id in ids:
        print("FROZEN_ARTIFACT::" + exp_id)
    sys.exit(0)

# 규칙 2: data/splits/*.parquet 은 생성 후 불변 (존재하는 파일만 차단, 생성은 허용)
m = re.search(r"(?:^|/)data/splits/[^/]+\.parquet$", np)
if m:
    cand = fp if os.path.isabs(fp) else os.path.join(project, fp)
    if os.path.exists(cand) or os.path.exists(np):
        print("SPLITS::existing")
    sys.exit(0)
PY
)" || exit 0

case "$REASON" in
    FROZEN_ARTIFACT::*)
        exp="${REASON#FROZEN_ARTIFACT::}"
        echo "BLOCKED: frozen 스택 멤버 산출물 ($exp) — 수정 불가." >&2
        echo "변경하려면 새 exp_id 로 재학습하고 풀을 갱신하라." >&2
        exit 2 ;;
    SPLITS::*)
        echo "BLOCKED: data/splits/*.parquet 은 생성 후 불변입니다 (frozen-OOF 정합 보호)." >&2
        echo "분할을 바꾸려면 새 레짐 파일명으로 생성하세요." >&2
        exit 2 ;;
esac

exit 0
