# Harness Evolution

This document tracks how the template evolves from real Kaggle runs. Each lesson traces back to a
documented case under [`case_studies/`](../case_studies/); the template is the accumulated, generalized
residue of those runs.

## Principles

A lesson is promoted into the template only if:

- it is likely to recur across future ML projects;
- it prevents structural failure, not just one-off score loss;
- it can be expressed as code, checklist, agent instruction, or workflow gate;
- it does not hard-code competition-specific tricks.

Competition-specific answers (data quirks, exact recipes, encoding branches) stay in the case study as
history — they are **not** baked into the scaffold.

## Evolution Log

Rows are grouped by template version. `v0.1.0` absorbed several lessons at once because it was distilled
from the first case; later cases should add focused rows under new versions.

| Version | Source case | Problem observed | Harness change |
|---|---|---|---|
| v0.1.0 | [Playground S6E5](../case_studies/playground-s6e5.md) | Rabbit-hole tuning of saturated single models — chasing the last +0.0005 in features/one model (7 failed experiments) | **Ceiling gate**: register "ceiling estimate vs target gap" before opening a lever track; demote low-ceiling levers to side threads (`CLAUDE.md` → 실험 우선순위) |
| v0.1.0 | [Playground S6E5](../case_studies/playground-s6e5.md) | Over-applying "single-model gains don't transfer" → left a model ~0.002 behind public/SOTA for several sessions | **Ceiling-gate caveat**: a member far behind public/SOTA is **P0**; the "no transfer" heuristic applies only to *near-ceiling* tuning (`CLAUDE.md`) |
| v0.1.0 | [Playground S6E5](../case_studies/playground-s6e5.md) | Members judged by individual AUC; weak members discarded; diversity axes exhausted | **Diversity-first judging**: measure by stack-add / residual (not individual AUC), keep weak-but-orthogonal members, treat **fold-split as a first-class diversity axis** (`CLAUDE.md` → 검증 전략·모델링) |
| v0.1.0 | [Playground S6E5](../case_studies/playground-s6e5.md) | Per-model trainers duplicated the training loop → config-knob divergence | **Single scaffold + adapter trainers** on one OOF/submission contract (`src/train_common.py`, `train_lgbm`/`train_xgb`) |
| v0.1.0 | [Playground S6E5](../case_studies/playground-s6e5.md) | Feature code fragmented across notebooks/scripts; a frozen member's OOF drifted on refactor | **Single-entry-point FE** + **OOF invariance check** (`src/features.py`, `scripts/check_fold_inputs.py`) |
| v0.1.0 | [Playground S6E5](../case_studies/playground-s6e5.md) | Small Δ judged on a single seed below the noise floor (σ≈0.0007) | **Measurement-power rule**: don't judge \|Δ\| < ~2·SE on one seed; use multi-seed or residual/stack-add framing (`CLAUDE.md`, `docs/setup_questions.md`) |
| v0.1.0 | [Playground S6E5](../case_studies/playground-s6e5.md) | Operational debt — irregular commits, experiment-ID drift, missing retrospectives, reactive infra guards | **Process guards**: commit-reminder Stop hook, fixed exp-ID convention, mandatory end-of-track retrospective, proactive infra guards (`CLAUDE.md` → 프로세스 규율, `scripts/hooks/`) |
| v0.2.0 | [ml-exp-workflow merge](wiki/merge_plan_ml_exp_workflow.md) | Designer and reviewer were the same agent (lenient self-review); experiment claims judged ad hoc against no pre-registered prediction | **Role separation + pre-registration gate**: `code-reviewer` (fixed checklist) and `result-reviewer` (4-verdict, measurement-power rule) split from the author; `specs/<exp_id>/expectation.yaml` committed before a full run, enforced by `guard_bash.sh` (screening exempt) |
| v0.2.0 | [ml-exp-workflow merge](wiki/merge_plan_ml_exp_workflow.md) | Premise auditing ("ceiling gate") stayed a manual habit and decayed; weak members parked under a blanket "no transfer" heuristic | **Mechanized premise audit**: `premise-auditor` runs blind (numbers only) every 5 judgments, emits 3 attack hypotheses + lowest-cost falsification; kill/continue stays with the human (`docs/wiki/audits/`) |
| v0.2.0 | [ml-exp-workflow merge](wiki/merge_plan_ml_exp_workflow.md) | CV split was regenerated per run → shared-code refactors could silently drift a frozen member's fold inputs; environment sklearn differences risked divergent splits | **Regime-serialized splits**: `cv.get_folds` is load-first to `data/splits/{strategy}_{n}fold_seed{seed}.parquet` (group-hashed for GroupKFold); freezes the partition across refactors while keeping the 5/7/10-fold diversity axis, with a remote-reproducibility warning |
| v0.2.0 | [ml-exp-workflow merge](wiki/merge_plan_ml_exp_workflow.md) | Imported workflow's "freeze the code directory" clashed with the adapter architecture | **Freeze artifacts, gate shared code**: `frozen.txt` lists frozen-member **exp_ids**; `guard_frozen.sh` blocks edits to their `experiments/{oof,submissions,logs}/<exp_id>.*` while shared-code changes are protected by `check_fold_inputs.py` input-equality, not a freeze |
| v0.3.0 | [guard tiers](wiki/guard_tiers.md) | Fail-closed guards accumulate false-blocks → the user disables the hook wholesale, losing every gate at once (the binary "blocked-and-stuck vs hook-off" trap) | **Guard-tier hardening**: classify guards T0(hard)/T1(record-bypass)/T2(warn); T1 false-blocks get a third option — commit `specs/<exp_id>/override_<guard>.md` to pass + log to `guard_overrides.jsonl` (append-only, feeds the next gate-erosion step); externalize match patterns to `conf/guard/*.txt` so a false-block is a one-line data fix, not a code edit; Stop hook detects guard de-registration. Bash guards ported to Python (`guard_bash.py`/`guard_frozen.py`, shared `_guardlib.py`) for Windows robustness |

## How to add a row

When a run surfaces a lesson that meets the promotion criteria:

1. Write/extend the case study in [`case_studies/`](../case_studies/) and link it from `case_studies/README.md`.
2. Land the harness change (code / checklist / agent / gate) under a new template version.
3. Add one row here: *version · source case · problem observed · harness change* (point to where it lives).
