"""scripts/decision_card.py — expectation 작성 직전 '객관적 컨텍스트 카드'.

목적: `specs/<exp_id>/expectation.yaml` 을 사람이 쓰기 **직전**에, 같은 그룹/모델의
최근 실험 로그에서 **노이즈를 명시**해 작성 비용을 낮춘다. fold 간 분산으로 SE 를
추정하고, 단일-시드 탐지 임계(~2·SE)를 **제안만** 한다.

⚠️ 이 스크립트는 어떤 파일도 쓰지 않는다 — expectation.yaml 의 추정값을 **자동으로
   채우지 않는다**(작성은 사람이 직접). 임계는 노이즈 하한일 뿐 메커니즘 근거로
   사람이 정한다. 비교군이 없으면 노이즈 플로어를 생략하고 카드만 출력한다.

사용:
    uv run python scripts/decision_card.py --exp-id exp_012_x --model lgbm --baseline exp_005_y
    #  필터: --model <name> · --group <exp_id 부분문자열> · --baseline <앵커 exp_id>
    #  --n <개수>(기본 5) · --log-dir <경로>(테스트/더미 로그용)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo 루트 → `from src import ...`
from src import config, utils  # noqa: E402


def load_raw_logs(log_dir: Path) -> list[dict[str, Any]]:
    """experiments/logs/*.json 원본 레코드를 그대로 읽는다 (fold 점수 보존).

    Args:
        log_dir: 로그 디렉터리.

    Returns:
        파싱된 로그 레코드 리스트(읽기 실패 파일은 건너뜀).
    """
    rows: list[dict[str, Any]] = []
    for p in sorted(log_dir.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return rows


def fold_stats(rec: dict[str, Any]) -> dict[str, Any] | None:
    """한 실험의 fold 점수에서 평균·표본표준편차·평균의 SE 를 계산한다.

    fold 간 std 는 시드 노이즈의 proxy 다(정확한 시드 SE 는 다중 시드 필요 — CLAUDE.md).

    Args:
        rec: 로그 레코드.

    Returns:
        {k, mean, std, se, scores} 딕셔너리. fold 가 2개 미만이면 None.
    """
    scores = rec.get("cv_scores")
    if not scores or len(scores) < 2:
        return None
    arr = np.asarray(scores, dtype=float)
    k = int(arr.size)
    std = float(arr.std(ddof=1))  # 표본 std (모집단 std 가 아님 — SE 추정용)
    return {"k": k, "mean": float(arr.mean()), "std": std, "se": std / k**0.5, "scores": arr}


def _matches(rec: dict[str, Any], model: str | None, group_sub: str | None) -> bool:
    """레코드가 model/group 필터에 부합하는가."""
    if model and (rec.get("model") or "").lower() != model.lower():
        return False
    if group_sub and group_sub not in (rec.get("exp_id") or ""):
        return False
    return True


def _fmt(v: Any, width: int) -> str:
    """표 셀 포맷 — 숫자는 5자리, None 은 '-'."""
    if v is None:
        return "-".ljust(width)
    if isinstance(v, float):
        return f"{v:.5f}".ljust(width)
    return str(v)[:width].ljust(width)


def render_group_table(group: list[dict[str, Any]], n: int) -> str:
    """비교군 최근 n건을 표 문자열로 만든다."""
    head = f"  {'exp_id':<20} {'model':<8} {'cv_mean':<9} {'cv_std':<9} {'k':<3} {'lb':<9} when"
    lines = [head]
    for r in group[:n]:
        fs = fold_stats(r)
        k = fs["k"] if fs else len(r.get("cv_scores") or [])
        lines.append(
            f"  {_fmt(r.get('exp_id'), 20)} {_fmt(r.get('model'), 8)} "
            f"{_fmt(r.get('cv_mean'), 9)} {_fmt(r.get('cv_std'), 9)} "
            f"{str(k):<3} {_fmt(r.get('lb_score'), 9)} {(r.get('timestamp') or '')[:19]}"
        )
    return "\n".join(lines)


def render_noise(anchor: dict[str, Any], group: list[dict[str, Any]]) -> str:
    """앵커의 SE 기반 노이즈 플로어 + 그룹 cv_mean 분산을 문자열로 만든다."""
    fs = fold_stats(anchor)
    if fs is None:
        return "[노이즈]  생략 — 앵커에 fold 점수가 부족합니다(k<2)."
    two_se = 2 * fs["se"]
    out = [
        f"[노이즈]  anchor = {anchor.get('exp_id')}  (k={fs['k']})",
        f"  cv_mean={fs['mean']:.6f}  표본std={fs['std']:.6f}  SE=std/√k={fs['se']:.6f}",
        f"  단일-시드 탐지 임계 ~2·SE ≈ {two_se:.6f}",
    ]
    means = [r.get("cv_mean") for r in group if isinstance(r.get("cv_mean"), (int, float))]
    if len(means) >= 2:
        arr = np.asarray(means, dtype=float)
        out.append(
            f"  그룹 cv_mean 분산({len(means)}건): "
            f"min {arr.min():.6f} ~ max {arr.max():.6f} "
            f"(spread {arr.max() - arr.min():.6f}, std {arr.std(ddof=1):.6f})"
        )
    return "\n".join(out)


def build_card(
    exp_id: str,
    logs: list[dict[str, Any]],
    *,
    model: str | None,
    group_sub: str | None,
    baseline: str | None,
    n: int,
) -> str:
    """결정 카드 전문을 만든다 (읽기 전용 — 파일 쓰기 없음)."""
    direction = "↑ 높을수록 좋음" if utils.greater_is_better() else "↓ 낮을수록 좋음"
    group = sorted(
        (r for r in logs if _matches(r, model, group_sub) and r.get("exp_id") != exp_id),
        key=lambda r: r.get("timestamp") or "",
        reverse=True,
    )
    baseline_rec = next((r for r in logs if r.get("exp_id") == baseline), None) if baseline else None
    anchor = baseline_rec or (group[0] if group else None)

    bar = "=" * 64
    parts = [
        bar,
        f"DECISION CARD — {exp_id}",
        f"metric = {config.METRIC} ({direction})   |   대상: specs/{exp_id}/expectation.yaml",
        f"필터: model={model or '*'}  group~='{group_sub or ''}'  baseline={baseline or '-'}",
        "",
    ]
    if baseline and baseline_rec is None:
        parts.append(f"⚠ baseline '{baseline}' 로그를 찾지 못함 — 최근 그룹 멤버로 대체.")

    noise_available = anchor is not None and fold_stats(anchor) is not None
    if anchor is None and not group:
        # 비교군 없음 → 노이즈 플로어 생략, 카드는 출력
        parts += [
            "[비교군]  없음 — 매칭되는 과거 로그가 없습니다.",
            "[노이즈]  생략 (비교군 없음).",
        ]
    else:
        parts += [
            f"[비교군]  최근 {min(n, len(group))}건 (총 {len(group)}건 매칭)",
            render_group_table(group, n) if group else "  (그룹 비어있음 — baseline 단독 앵커)",
            "",
            render_noise(anchor, group),
        ]

    parts.append("")
    if noise_available:
        parts += [
            "[제안]  (참고용 — 그대로 베끼지 말 것)",
            "  predicted.metric_delta 의 |Δ| 가 위 ~2·SE 보다 커야 단일 시드로 판정 가능.",
            "  그 미만을 노릴 거면 다중 시드 또는 잔차/stack-add 프레임으로 설계하라.",
            "  → 이 임계는 노이즈 하한일 뿐, 값은 메커니즘 근거로 직접 정한다.",
        ]
    else:
        parts += [
            "[제안]  비교군/앵커가 없어 노이즈 임계를 제시할 수 없습니다.",
            "  첫 실험이면 mechanism·falsification 정의에 집중하라(--baseline 지정 시 노이즈 컨텍스트 생성).",
        ]
    parts += [
        "",
        "[작성]  이 카드는 컨텍스트만 제공한다. expectation.yaml 은 직접 작성·커밋한다.",
        "  템플릿: docs/templates/expectation.yaml",
        bar,
    ]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="expectation 작성 직전 노이즈 컨텍스트 카드 (읽기 전용)")
    ap.add_argument("--exp-id", default="exp_XXX", help="작성 대상 실험 ID (카드 헤더용)")
    ap.add_argument("--model", default=None, help="비교군 필터: 모델명")
    ap.add_argument("--group", default=None, help="비교군 필터: exp_id 부분문자열")
    ap.add_argument("--baseline", default=None, help="노이즈 앵커 exp_id (없으면 최근 그룹 멤버)")
    ap.add_argument("--n", type=int, default=5, help="표시할 최근 비교군 개수")
    ap.add_argument("--log-dir", default=None, help="로그 디렉터리 (기본 config.LOG_DIR)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else config.LOG_DIR
    if not log_dir.exists():
        print(f"로그 디렉터리 없음: {log_dir}")
        return
    logs = load_raw_logs(log_dir)
    print(
        build_card(
            args.exp_id, logs,
            model=args.model, group_sub=args.group, baseline=args.baseline, n=args.n,
        )
    )


if __name__ == "__main__":
    main()
