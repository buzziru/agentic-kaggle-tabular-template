"""Stop 훅 — 미커밋 tracked 변경이 임계 이상이면 리마인드 (운영 부채 자동 방지).

CLAUDE.md 프로세스 규율("세션 끝 = 미커밋 의미 변경 0")의 기계적 안전망.
`.claude/settings.json` 의 Stop 훅이 호출한다. git 없거나 오류면 조용히 통과(non-blocking).
임계는 THRESHOLD 로 조정. `python` 미해결 환경이면 settings.json 명령을 `python3`/`uv run python` 으로 바꾼다.
"""

from __future__ import annotations

import subprocess
import sys

THRESHOLD = 8  # 미커밋 tracked 변경이 이 수 이상이면 리마인드

try:
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
except Exception:
    pass

sys.exit(0)  # 항상 non-blocking
