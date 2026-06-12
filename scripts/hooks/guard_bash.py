#!/usr/bin/env python
"""PreToolUse(Bash) 훅 — 세 가지를 검사한다 (가드 티어: docs/wiki/guard_tiers.md).

  (a) expectation 게이트 [T1]: 풀 학습/push 명령이면 specs/<exp_id>/expectation.yaml
      존재+HEAD커밋+작업트리일치를 검사. exp_id 포맷(연번)도 검사 [T1].
      max_folds= 스크리닝·src.stack 은 면제. 풀 실행 패턴은 conf/guard/push_patterns.txt.
  (b) frozen 산출물/기존 splits 우회 쓰기 [T1, 휴리스틱]: shell 리다이렉트·sed -i·
      rm/mv/cp/tee 가 frozen 산출물 또는 기존 data/splits/*.parquet 를 대상으로 하면.
  (c) 시크릿 커밋 [T0 하드]: git add/commit 이 시크릿을 스테이징하면 우회 불가 차단 +
      시도 로깅. 시크릿 패턴은 conf/guard/secret_patterns.txt.

T1 은 override 파일(specs/<exp_id>/override_<guard>.md)을 커밋하면 통과 + 로깅한다.
exit 0 허용 / exit 2 차단(stderr 가 Claude 에 전달됨). 인프라 오류는 fail-open.
"""

from __future__ import annotations

import json
import os
import re
import sys

# conf/guard/*.txt 부재 시 폴백 기본값 (파일이 단일 진실원, 이건 안전망).
DEFAULT_SECRET = [
    r"(?:^|[\s=/'\"])[^\s'\";|&]*\.(?:env|key|pem|pfx|p12)\b",
    r"(?:^|[\s=/'\"/])(?:[^\s'\";|&]*[/.])?(?:credentials?|secrets?|service[_-]?account)\.(?:json|ya?ml|ini|cfg|conf|txt|env|pem|key)\b",
    r"kaggle\.json",
    r"id_rsa",
]
DEFAULT_PUSH = [
    r"python[0-9]?\s+-m\s+src\.train",
    r"kaggle\s+kernels\s+push",
    r"lightning\s+run\s+job",
]


def meta_exp_id(cmd: str, cwd: str) -> tuple[str | None, str]:
    """`kaggle kernels push` 의 -p/--path <dir>/kernel-metadata.json 에서 exp_id 를 읽는다.

    명령줄이 아니라 메타데이터(gen_kernel.py 가 KERNELS SSOT 에서 기록)를 진실원으로 쓴다 —
    명령줄 exp_id= 변수 주입 우회를 제거한다. push 게이트는 우회 방지가 목적이라 metadata 를
    못 읽으면 fail-closed(차단)한다.

    Returns:
        (exp_id, "") 성공 / (None, 사유) 실패.
    """
    md = re.search(r"(?:-p|--path)\s+(\S+)", cmd)
    d = md.group(1).strip("'\"") if md else "."
    base = d if os.path.isabs(d) else os.path.join(cwd, d)
    meta = os.path.join(base, "kernel-metadata.json")
    if not os.path.isfile(meta):
        return None, "kernel-metadata.json 을 찾지 못함: %s" % meta
    try:
        with open(meta, encoding="utf-8") as fh:
            exp_id = (json.load(fh) or {}).get("exp_id")
    except Exception as e:  # 파싱 실패도 fail-closed
        return None, "kernel-metadata.json 파싱 실패: %s" % e
    if not exp_id:
        return None, "kernel-metadata.json 에 exp_id 필드가 없음"
    return exp_id, ""


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    project = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    cwd = data.get("cwd") or project  # push -p <dir> 의 상대경로 기준
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if not cmd:
        return 0

    sys.path.insert(0, os.path.join(project, "scripts", "hooks"))
    try:
        import _guardlib as gl
    except Exception:
        return 0  # 가드 인프라 부재 → fail-open (1차 방어는 에이전트 지시문)
    gl.reconfigure_utf8()

    norm = cmd.replace("\\", "/")

    def block(*lines: str) -> None:
        for ln in lines:
            print(ln, file=sys.stderr)
        sys.exit(2)

    def t1(guard: str, exp_id: str, target: str, *msg: str) -> None:
        """T1 게이트: override 성립 시 통과, 아니면 차단."""
        hint = gl.t1_gate(project, guard, exp_id, target)
        if hint is None:
            return  # 기록 우회 성립 → 통과
        block(*(list(msg) + hint))

    # -----------------------------------------------------------------
    # (c) 시크릿 스테이징 (T0 하드) — 우회 불가, 시도 로깅
    # `git add` 만 검사한다 — 새 시크릿 파일 유입은 add 가 유일한 벡터이고,
    # commit -m/-F 의 메시지 텍스트가 패턴에 걸리는 오차단을 막는다(commit -a 는
    # 이미 tracked 인 파일만 스테이징하므로 새 시크릿 유입 불가).
    # -----------------------------------------------------------------
    if re.search(r"(?:^|[;&|]|\s)git\s+add\b", cmd):
        for pat in gl.load_patterns(project, "secret_patterns.txt", DEFAULT_SECRET):
            try:
                if re.search(pat, norm, re.IGNORECASE):
                    gl.record_event(project, "secret", "blocked", "", norm[:200], "")
                    block(
                        "BLOCKED: git add 가 시크릿(.env·키·자격증명)을 스테이징하려 합니다 (T0 하드).",
                        "시크릿은 절대 커밋 금지(CLAUDE.md 보안). 대상에서 제외하세요.",
                    )
            except re.error:
                continue
        # force-add 로 .gitignore 우회 + 일괄 스테이징(. / -A)은 시크릿 누출 벡터.
        if re.search(r"\s-{1,2}f(?:orce)?\b", cmd) and re.search(
            r"(?:\s\.(?:\s|$)|--all\b|\s-A\b)", cmd
        ):
            gl.record_event(project, "secret", "blocked", "", norm[:200], "")
            block(
                "BLOCKED: git add -f 일괄 스테이징(.gitignore 우회)은 시크릿 누출 위험입니다 (T0 하드).",
                "force-add 가 필요하면 파일을 개별 지정하세요(시크릿 제외 확인).",
            )

    # -----------------------------------------------------------------
    # (a) 풀 실행 expectation 게이트 (T1)
    # -----------------------------------------------------------------
    run_pats = gl.load_patterns(project, "push_patterns.txt", DEFAULT_PUSH)
    is_run = any(re.search(p, cmd) for p in run_pats)
    is_stack = bool(re.search(r"-m\s+src\.stack", cmd))
    has_maxfolds = bool(re.search(r"max_folds\s*=", cmd))
    is_push = bool(re.search(r"kaggle\s+kernels\s+push", cmd))

    if is_run and not is_stack and not has_maxfolds:
        if is_push:
            # push 는 명령줄이 아니라 kernel-metadata.json 의 exp_id 로 게이트한다.
            exp_id, err = meta_exp_id(cmd, cwd)
            if exp_id is None:
                block(
                    "BLOCKED: push exp_id 게이트 — %s (fail-closed)." % err,
                    "kaggle/gen_kernel.py 로 커널을 재생성해 kernel-metadata.json 의 exp_id 를 채우세요.",
                )
        else:
            m = re.search(r"exp_id=(exp_[A-Za-z0-9_]+)", cmd) or re.search(
                r"(?<![A-Za-z0-9_])(exp_[A-Za-z0-9_]+)", cmd
            )
            if not m:
                block(
                    "BLOCKED: 풀 실행 명령에서 exp_id 를 추출하지 못했습니다 (fail-closed).",
                    "exp_id=exp_NNN 를 명시하거나, 스크리닝이면 명령에 max_folds= 를 지정하세요.",
                )
            exp_id = m.group(1)
        # exp_id 포맷 [T1]: 연번 컨벤션 exp_<NNN>_<slug>.
        if not re.match(r"^exp_\d+_", exp_id):
            t1(
                "expid", exp_id, exp_id,
                "BLOCKED: exp_id '%s' 가 연번 컨벤션(exp_<NNN>_<slug>)에 맞지 않습니다 (T1)." % exp_id,
                "예: exp_001_baseline. 다른 규칙을 쓰면 guard_bash.py 의 정규식을 고치세요.",
            )
        # expectation 파일 [T1]: 존재 + 커밋 + 미수정.
        expect = "specs/%s/expectation.yaml" % exp_id
        if not gl.is_committed_unmodified(project, expect):
            t1(
                "expectation", exp_id, expect,
                "BLOCKED: %s 가 없거나 미커밋/수정됨 (expectation 게이트, T1)." % expect,
                "풀 실행 전 mechanism/predicted/falsification 을 작성·커밋하세요.",
                "스크리닝이면 명령에 max_folds= 를 지정하면 면제됩니다.",
            )

    # -----------------------------------------------------------------
    # (b) frozen 산출물 / 기존 splits 우회 쓰기 (T1, 휴리스틱)
    # -----------------------------------------------------------------
    write_re = re.compile(
        r"(>>?|sed\s+-i|(?:^|\s)rm(?:\s|$)|(?:^|\s)mv(?:\s|$)|(?:^|\s)cp(?:\s|$)|(?:^|\s)tee(?:\s|$)|truncate)"
    )
    if write_re.search(cmd):
        for exp_id in gl.load_frozen_ids(project):
            if re.search(
                r"experiments/(?:oof|submissions|logs)/" + re.escape(exp_id) + r"\.", norm
            ):
                t1(
                    "frozen_bypass", exp_id, exp_id,
                    "BLOCKED: frozen 산출물(%s) 우회 쓰기 시도 (T1, 휴리스틱)." % exp_id,
                    "frozen 스택 멤버 산출물은 읽기 전용입니다.",
                )
        for mm in re.finditer(r"data/splits/([^\s'\";|&>]+\.parquet)", norm):
            cand = os.path.join(project, "data", "splits", os.path.basename(mm.group(1)))
            if os.path.exists(cand):
                t1(
                    "splits_bypass", "", os.path.basename(mm.group(1)),
                    "BLOCKED: 기존 data/splits/*.parquet 우회 쓰기 시도 (T1, 휴리스틱).",
                    "분할 파일은 생성 후 불변입니다.",
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
