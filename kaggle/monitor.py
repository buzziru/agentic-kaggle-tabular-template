"""Kaggle 커널 완료 모니터 — output-회수 기반(status 파싱 금지).

재발 버그 2종 근절용 재사용 유틸(/tmp 임시 스크립트 폐기):
  ⓐ **status-grep 오판**: `kernels status` 의 일시적 `500 Server Error` 문자열의
     "Error" 가 `grep -i error` 에 오매칭 -> 멀쩡한 RUNNING 커널을 FAILED 로 오판.
     -> 완료 감지는 **`kernels output` 회수 -> OOF 파일 출현 여부**로만 한다.
  ⓑ **동명파일 find 충돌**: OOF(`oof/<exp>.csv`)와 submission(`submissions/<exp>.csv`)이
     같은 basename 이라 `find -name <exp>.csv | head -1` 이 submission(test 예측)을
     OOF 로 잘못 회수. -> **명시적 서브디렉터리**(oof/·submissions/·logs/)에서 각각 회수.

slug·exp_id 는 gen_kernel 의 KERNELS 레지스트리(SSOT)에서 가져온다.

▶ 사용 전 채울 것: OWNER (본인 Kaggle 사용자명).

사용:
    uv run python kaggle/monitor.py <name> [<name> ...]
    # 백그라운드: 위를 run_in_background 로. 완료 시 experiments/{oof,submissions,logs}/ 회수.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_kernel import KERNELS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OWNER = "{{KAGGLE_USER}}"   # 본인 Kaggle 사용자명
POLL_SEC = 90
MAX_ITERS = 200  # ~5h 안전 상한


def _load_env() -> None:
    """.env 의 KAGGLE_USERNAME/KAGGLE_KEY 를 os.environ 에 주입."""
    import os

    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def _try_recover(name: str) -> bool:
    """커널 출력을 회수 시도. OOF 파일이 나오면 oof/submission/log 를 명시 경로로 복사.

    Args:
        name: KERNELS 레지스트리 키.

    Returns:
        OOF 회수 성공(=완료) 여부.
    """
    p = KERNELS[name]
    slug, exp = p["slug"], p["exp_id"]
    out = Path(f"/tmp/out_{slug}")
    if out.exists():
        shutil.rmtree(out)
    subprocess.run(
        ["kaggle", "kernels", "output", f"{OWNER}/{slug}", "-p", str(out)],
        capture_output=True,
        text=True,
    )
    oof = out / "oof" / f"{exp}.csv"
    if not oof.exists():  # 미완 — 다음 폴링
        return False
    # 완료 — 명시적 서브디렉터리에서 각각 회수(동명파일 충돌 차단)
    (REPO / "experiments" / "oof").mkdir(parents=True, exist_ok=True)
    (REPO / "experiments" / "submissions").mkdir(parents=True, exist_ok=True)
    (REPO / "experiments" / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copy(oof, REPO / "experiments" / "oof" / f"{exp}.csv")
    for sub, dst in [("submissions", "submissions"), ("logs", "logs")]:
        src = out / sub / (f"{exp}.csv" if sub == "submissions" else f"{exp}.json")
        if src.exists():
            shutil.copy(src, REPO / "experiments" / dst / src.name)
    return True


def main(names: list[str]) -> None:
    """주어진 커널들을 완료까지 폴링·회수."""
    if not names:
        print("사용: uv run python kaggle/monitor.py <name> [<name> ...]")
        print("등록:", ", ".join(KERNELS))
        return
    unknown = [n for n in names if n not in KERNELS]
    if unknown:
        raise KeyError(f"미등록 커널: {unknown}. 등록: {list(KERNELS)}")
    _load_env()
    done: set[str] = set()
    for i in range(MAX_ITERS):
        for n in names:
            if n in done:
                continue
            if _try_recover(n):
                done.add(n)
                print(f"[{i:03d}] {n} RECOVERED -> experiments/  ({len(done)}/{len(names)})", flush=True)
            else:
                print(f"[{i:03d}] {n} 미완", flush=True)
        if len(done) == len(names):
            print("ALL RECOVERED", flush=True)
            return
        time.sleep(POLL_SEC)
    print(f"TIMEOUT — 회수: {sorted(done)} / 미완: {[n for n in names if n not in done]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
