#!/usr/bin/env python
"""PreToolUse(Edit|MultiEdit|Write|NotebookEdit) 훅 — frozen 산출물/기존 splits 편집 차단.

가드 티어 [T1 기록 우회] (docs/wiki/guard_tiers.md):
  규칙1: frozen 스택 멤버 산출물 experiments/{oof,submissions,logs}/<exp_id>.*
         (exp_id 가 frozen.txt 에 등록된 경우).
  규칙2: 이미 생성된 data/splits/*.parquet (생성은 허용, 기존 수정만 차단).

⚠️ 동결 대상은 산출물(아티팩트)이지 src/ 공유 코드가 아니다 — 공유 코드는
   scripts/check_fold_inputs.py 입력 동등성으로 보호한다.
override(specs/<exp_id|_global>/override_<guard>.md) 커밋 시 통과 + 로깅.
exit 0 허용 / exit 2 차단. 인프라 오류는 fail-open.
"""

from __future__ import annotations

import json
import os
import re
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    project = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    ti = data.get("tool_input") or {}
    fp = ti.get("file_path") or ti.get("notebook_path") or ""
    if not fp:
        return 0

    sys.path.insert(0, os.path.join(project, "scripts", "hooks"))
    try:
        import _guardlib as gl
    except Exception:
        return 0  # 가드 인프라 부재 → fail-open
    gl.reconfigure_utf8()

    np = fp.replace("\\", "/")

    def t1(guard: str, exp_id: str, target: str, *msg: str) -> None:
        hint = gl.t1_gate(project, guard, exp_id, target)
        if hint is None:
            return  # 기록 우회 성립 → 통과
        for ln in list(msg) + hint:
            print(ln, file=sys.stderr)
        sys.exit(2)

    # 규칙 1: frozen 스택 멤버 산출물
    m = re.search(r"(?:^|/)experiments/(?:oof|submissions|logs)/([^/]+)\.[^/]+$", np)
    if m:
        exp_id = m.group(1)
        if exp_id in gl.load_frozen_ids(project):
            t1(
                "frozen_artifact", exp_id, np,
                "BLOCKED: frozen 스택 멤버 산출물 (%s) — 수정 불가 (T1)." % exp_id,
                "변경하려면 새 exp_id 로 재학습하고 풀을 갱신하세요.",
            )
        return 0

    # 규칙 2: data/splits/*.parquet 은 생성 후 불변 (존재하는 파일만 차단)
    m = re.search(r"(?:^|/)data/splits/[^/]+\.parquet$", np)
    if m:
        cand = fp if os.path.isabs(fp) else os.path.join(project, fp)
        if os.path.exists(cand) or os.path.exists(np):
            t1(
                "splits_edit", "", np,
                "BLOCKED: data/splits/*.parquet 은 생성 후 불변입니다 (frozen-OOF 정합, T1).",
                "분할을 바꾸려면 새 레짐 파일명으로 생성하세요.",
            )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
