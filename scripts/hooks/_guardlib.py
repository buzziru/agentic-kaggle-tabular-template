"""가드 공통 헬퍼 — guard_bash.py / guard_frozen.py 가 공유한다.

커밋 검사·패턴 로드·override(기록 우회) 처리를 한 곳에 둔다(중복 구현 금지).
T1 escape hatch 의 단일 진실원이다 — 가드 스크립트는 이 헬퍼만 호출한다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone


def reconfigure_utf8() -> None:
    """Windows 콘솔 cp949 깨짐 방지 — stdout/stderr 를 UTF-8 로 고정."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def git(project: str, *args: str) -> subprocess.CompletedProcess:
    """`git -C <project> <args>` 실행 결과를 반환한다."""
    return subprocess.run(["git", "-C", project, *args], capture_output=True, text=True)


def is_committed_unmodified(project: str, relpath: str) -> bool:
    """relpath 가 존재 + HEAD 에 커밋 + 작업트리=HEAD 이면 True.

    expectation 게이트와 override 검사가 공유하는 단일 헬퍼다.
    """
    full = os.path.join(project, relpath)
    if not os.path.isfile(full):
        return False
    if git(project, "cat-file", "-e", "HEAD:%s" % relpath).returncode != 0:
        return False
    if git(project, "diff", "--quiet", "HEAD", "--", relpath).returncode != 0:
        return False
    return True


def head_hash(project: str) -> str:
    """현재 HEAD 커밋 해시(실패 시 빈 문자열)."""
    r = git(project, "rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else ""


def load_patterns(project: str, name: str, default: list[str] | None = None) -> list[str]:
    """conf/guard/<name> 을 1줄 1패턴(# 주석)으로 읽는다. 없으면 default.

    오차단 대응 = "훅 끄기"가 아니라 이 txt 한 줄 추가/수정 (패턴 외부화).
    """
    path = os.path.join(project, "conf", "guard", name)
    if not os.path.isfile(path):
        return list(default or [])
    out: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                out.append(ln)
    return out or list(default or [])


def load_frozen_ids(project: str) -> list[str]:
    """frozen.txt 의 동결 exp_id 목록(주석·공백 제외)."""
    path = os.path.join(project, "frozen.txt")
    ids: list[str] = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    ids.append(ln)
    return ids


def override_relpath(guard: str, exp_id: str) -> str:
    """T1 override 파일 경로. exp_id 없으면 _global 키를 쓴다."""
    base = "specs/%s" % (exp_id if exp_id else "_global")
    return "%s/override_%s.md" % (base, guard)


def record_event(
    project: str, guard: str, action: str, exp_id: str, target_path: str, reason_path: str
) -> None:
    """docs/wiki/guard_overrides.jsonl 에 1줄 append (append-only, 추적 대상).

    action="override"(T1 우회 성립) / "blocked"(T0 시도 차단). 다음 단계(게이트
    붕괴 완화)가 이 로그에서 우회 빈도를 읽는다.
    """
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "guard": guard,
        "action": action,
        "exp_id": exp_id or "",
        "target_path": target_path,
        "reason_path": reason_path,
        "git_hash": head_hash(project),
    }
    log = os.path.join(project, "docs", "wiki", "guard_overrides.jsonl")
    try:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 로깅 실패가 가드 판정을 막지 않는다


def t1_gate(project: str, guard: str, exp_id: str, target_path: str) -> list[str] | None:
    """T1 기록 우회. override 가 커밋돼 있으면 record + None(허용) 반환.

    아니면 우회 안내 hint 라인 리스트를 반환한다(호출자가 차단 메시지에 덧붙임).
    """
    rel = override_relpath(guard, exp_id)
    if is_committed_unmodified(project, rel):
        record_event(project, guard, "override", exp_id, target_path, rel)
        return None
    return ["우회하려면 %s 에 사유를 적고 커밋한 뒤 재시도하세요 (T1 기록 우회)." % rel]
