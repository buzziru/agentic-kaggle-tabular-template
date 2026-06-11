"""Stop 훅 — 두 가지를 리마인드한다 (둘 다 비차단, 항상 exit 0).

  (1) 미커밋 tracked 변경이 임계 이상 — 운영 부채 방지(CLAUDE.md 프로세스 규율).
  (2) 가드 훅 비활성화 감지 — PreToolUse 가드가 settings.json 에서 사라졌거나
      스크립트 파일이 없으면 알린다("끄는 건 자유, 조용히 꺼지진 않게").

git 없거나 오류면 조용히 통과(non-blocking).
임계는 THRESHOLD 로 조정. `python` 미해결 환경이면 settings.json 명령을 `python3`/`uv run python` 으로 바꾼다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

THRESHOLD = 8  # 미커밋 tracked 변경이 이 수 이상이면 리마인드
EXPECTED_GUARDS = ["guard_bash.py", "guard_frozen.py", "lint_notebook.py"]  # 등록·존재 확인 대상

for _stream in (sys.stdout, sys.stderr):  # Windows 콘솔 cp949 한글 깨짐/에러 방지
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_uncommitted() -> None:
    """미커밋 tracked 변경 수가 THRESHOLD 이상이면 리마인드."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, timeout=10,
    )
    if out.returncode == 0:
        n = len([ln for ln in out.stdout.splitlines() if ln.strip()])
        if n >= THRESHOLD:
            print(
                f"[reminder] 미커밋 tracked 변경 {n}개 (≥{THRESHOLD}) — "
                "의미 단위로 묶어 커밋하고 CURRENT_STATUS.md 를 갱신하세요 (CLAUDE.md 프로세스 규율)."
            )


def check_guards_enabled() -> None:
    """가드 훅이 settings.json 에 등록돼 있고 스크립트가 존재하는지 확인."""
    project = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    settings = os.path.join(project, ".claude", "settings.json")
    try:
        with open(settings, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        return  # 파싱 불가 → 조용히 통과
    cmds = " ".join(
        h.get("command", "")
        for blocks in (cfg.get("hooks") or {}).values()
        for b in blocks
        for h in b.get("hooks", [])
    )
    missing_reg = [s for s in EXPECTED_GUARDS if s not in cmds]
    missing_file = [
        s for s in EXPECTED_GUARDS if not os.path.isfile(os.path.join(project, "scripts", "hooks", s))
    ]
    if missing_reg:
        print(
            "[reminder] 가드 훅 미등록: %s — settings.json 의 hooks 에서 사라졌습니다. "
            "의도된 비활성화가 아니면 복구하세요 (docs/wiki/guard_tiers.md)." % ", ".join(missing_reg)
        )
    if missing_file:
        print(
            "[reminder] 가드 스크립트 파일 없음: %s — 참조는 있으나 파일이 사라졌습니다." % ", ".join(missing_file)
        )


for _check in (check_uncommitted, check_guards_enabled):
    try:
        _check()
    except Exception:
        pass

sys.exit(0)  # 항상 non-blocking
