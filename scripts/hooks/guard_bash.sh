#!/usr/bin/env bash
# guard_bash.sh — PreToolUse(Bash) 훅. 두 가지를 검사한다.
#  (a) 풀 실행 게이트: 풀 학습/push 명령이면 specs/<exp_id>/expectation.yaml 의
#      존재 + git HEAD 커밋 + 작업트리=HEAD 일치를 검사. 미충족 시 차단.
#      ⚠️ max_folds= 가 있으면 스크리닝으로 보고 면제(MERGE 충돌해소 ④). src.stack 도 면제.
#  (b) frozen 산출물 우회 쓰기 차단(휴리스틱): > >> sed -i rm mv cp tee 등이
#      frozen exp_id 산출물 또는 기존 data/splits/*.parquet 를 대상으로 하면 차단.
#      ⚠️ 휴리스틱이다 — 1차 방어는 에이전트 지시문, 이 훅은 안전망이다.
# exit 0: 허용 / exit 2: 차단. 인프라 오류는 fail-open.

set -uo pipefail

INPUT="$(cat)"
PYBIN="$(command -v python3 || command -v python || true)"
[ -z "$PYBIN" ] && exit 0

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"

"$PYBIN" - "$INPUT" "$PROJECT_DIR" <<'PY'
import json, sys, os, re, subprocess

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
cmd = (data.get("tool_input", {}) or {}).get("command", "") or ""
if not cmd:
    sys.exit(0)


def block(*lines):
    for ln in lines:
        print(ln, file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------
# (a) 풀 실행 expectation 게이트
# ---------------------------------------------------------------
RUN_RE = re.compile(r"python[0-9]?\s+-m\s+src\.train|kaggle\s+kernels\s+push|lightning\s+run\s+job")
is_run = bool(RUN_RE.search(cmd))
is_stack = bool(re.search(r"-m\s+src\.stack", cmd))
has_maxfolds = bool(re.search(r"max_folds\s*=", cmd))

if is_run and not is_stack and not has_maxfolds:
    # exp_id 추출: exp_id=<token> 우선, 없으면 bare exp_ 토큰. 둘 다 실패 시 fail-closed.
    m = re.search(r"exp_id=(exp_[A-Za-z0-9_]+)", cmd) or re.search(r"(?<![A-Za-z0-9_])(exp_[A-Za-z0-9_]+)", cmd)
    if not m:
        block("BLOCKED: 풀 실행 명령에서 exp_id 를 추출하지 못했습니다 (fail-closed).",
              "exp_id=exp_NNN 를 명시하거나, 스크리닝이면 명령에 max_folds= 를 지정하세요.")
    exp_id = m.group(1)
    expect = "specs/%s/expectation.yaml" % exp_id
    if not os.path.isfile(os.path.join(project, expect)):
        block("BLOCKED: %s 가 없습니다 (expectation 게이트, 충돌해소 ④)." % expect,
              "풀 실행 전 mechanism/predicted/falsification 을 작성·커밋하세요.",
              "스크리닝이면 명령에 max_folds= 를 지정하면 면제됩니다.")

    def git(*args):
        return subprocess.run(["git", "-C", project, *args],
                              capture_output=True, text=True)

    if git("cat-file", "-e", "HEAD:%s" % expect).returncode != 0:
        block("BLOCKED: %s 가 git HEAD 에 커밋되지 않았습니다." % expect,
              "사전 등록은 커밋된 버전만 유효합니다. 먼저 커밋하세요.")
    if git("diff", "--quiet", "HEAD", "--", expect).returncode != 0:
        block("BLOCKED: %s 가 HEAD 커밋 이후 수정되었습니다." % expect,
              "실행 직전 expectation 수정은 사전 등록 위반입니다.")

# ---------------------------------------------------------------
# (b) frozen 산출물 / 기존 splits 우회 쓰기 차단 (휴리스틱)
# ---------------------------------------------------------------
WRITE_RE = re.compile(r"(>>?|sed\s+-i|(?:^|\s)rm(?:\s|$)|(?:^|\s)mv(?:\s|$)|(?:^|\s)cp(?:\s|$)|(?:^|\s)tee(?:\s|$)|truncate)")
if WRITE_RE.search(cmd):
    norm = cmd.replace("\\", "/")
    frozen = os.path.join(project, "frozen.txt")
    ids = []
    if os.path.isfile(frozen):
        with open(frozen, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln and not ln.startswith("#"):
                    ids.append(ln)
    for exp_id in ids:
        if re.search(r"experiments/(?:oof|submissions|logs)/" + re.escape(exp_id) + r"\.", norm):
            block("BLOCKED: frozen 산출물(%s) 우회 쓰기 시도 (휴리스틱)." % exp_id,
                  "frozen 스택 멤버 산출물은 읽기 전용입니다.")
    for mm in re.finditer(r"data/splits/([^\s'\";|&>]+\.parquet)", norm):
        cand = os.path.join(project, "data", "splits", os.path.basename(mm.group(1)))
        if os.path.exists(cand):
            block("BLOCKED: 기존 data/splits/*.parquet 우회 쓰기 시도 (휴리스틱).",
                  "분할 파일은 생성 후 불변입니다.")

sys.exit(0)
PY
exit $?
