"""PostToolUse(NotebookEdit) 훅 — 노트북 코드 셀의 `;` 다중문을 검출한다.

CLAUDE.md 코딩 컨벤션("`;` 다중문 금지, setup 셀 예외", L129)의 기계적 안전망.
편집된 셀의 new_source 만 검사한다(파일 전체 재파싱 불필요).

판정 규칙:
  - 문장 구분자 `;` (예: `a = 1; b = 2`) → 위반.
  - 트레일링 `;` (예: `plt.plot();` — Jupyter 출력 억제) → 허용.
  - 문자열·주석 안의 `;` → tokenize 로 무시.
  - setup 셀(import / sys.path 포함) → 예외(면제).

티어 T2(경고, docs/wiki/guard_tiers.md): 비차단이다. exit 0 으로 통과하되 JSON
hookSpecificOutput.additionalContext 로 Claude 에 위반을 알린다(자기수정 유도).
T2 는 별도 로그를 남기지 않는다(노이즈 방지) — 경고로 충분하다.
파싱 불가·인프라 오류는 조용히 통과(exit 0).
"""

from __future__ import annotations

import io
import json
import sys
import tokenize


def find_separator_semicolons(source: str) -> list[int]:
    """문장 구분자로 쓰인 `;` 의 줄 번호 목록을 반환한다.

    Args:
        source: 코드 셀 소스.

    Returns:
        위반 줄 번호(1-base) 목록. 트레일링 `;` 는 제외.
    """
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []  # 미완 셀 등 파싱 불가 → 면제

    violations: list[int] = []
    for i, tok in enumerate(toks):
        if tok.type != tokenize.OP or tok.string != ";":
            continue
        # 다음 유의미 토큰을 본다. NEWLINE/NL/COMMENT/ENDMARKER 면 트레일링(허용).
        nxt = next(
            (
                t
                for t in toks[i + 1 :]
                if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE)
            ),
            None,
        )
        if nxt is not None and nxt.type != tokenize.ENDMARKER:
            violations.append(tok.start[0])
    return violations


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    ti = data.get("tool_input", {}) or {}
    if (ti.get("cell_type") or "code") != "code":
        return 0
    source = ti.get("new_source") or ""
    if not source.strip():
        return 0

    # setup 셀 예외: import / sys.path 가 있으면 면제.
    if "import " in source or "sys.path" in source:
        return 0

    lines = sorted(set(find_separator_semicolons(source)))
    if not lines:
        return 0

    nums = ", ".join(str(n) for n in lines)
    msg = (
        f"[notebook lint, 비차단] 셀 {nums} 번째 줄에 문장 구분자 `;` 가 있습니다 — "
        "한 줄 한 문장으로 분리하세요(CLAUDE.md 코딩 컨벤션). "
        "출력 억제용 트레일링 `;` 는 무관합니다."
    )
    # T2: exit 0 + additionalContext 로 비차단 피드백(도구는 이미 실행됨).
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": msg,
                }
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
