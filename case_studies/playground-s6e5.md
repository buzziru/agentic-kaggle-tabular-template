# Case Study — Playground Series S6E5 (F1 Pit Stops)

> **Result:** Private ROC-AUC **0.95460** · ≈148 / 3023 teams (**top ~5%**) · template **v0.1.0**
>
> This is the run the template was distilled from. It records what worked, what failed, and — most
> importantly — **which lessons were promoted into the template** (see
> [docs/harness_evolution.md](../docs/harness_evolution.md)).

## The competition

- **Task:** binary classification (`PitNextLap` — does the car pit next lap), metric **ROC-AUC**, probability submission. train 439k / test 188k / no missing / 19.9% positive.
- **Data nature:** **synthetic (deep-learning-generated)** data, augmented from a smaller real source. train and test share the same `(Race, Year, Driver)` groups, so the correct validation was **row-level StratifiedKFold** (not GroupKFold) — matching how train/test were drawn.
- **What made it hard:**
  - `Driver` was high-cardinality (887) and **adversarial** (large train↔test distribution shift), so the encoding choice (target-encode / frequency / drop) became a key decorrelation lever.
  - **Synthetic artifacts** — sparse lap sampling and corrupted inherited deltas — meant **time-series / sequence feature engineering mostly added noise**. This was the single biggest source of failed experiments.
- **Field:** 3023 teams; top teams ~0.9549–0.9550. The decisive edge was **OOF pool diversity + a regularized meta-learner**, not any one strong model.

## Performance path (Private LB)

| Stage | Score | Lever |
|---|---:|---|
| LGBM baseline | ~0.9443 | native categorical |
| stack v4 | 0.95273 | add RealMLP + external-data row augmentation |
| stack v6 / v7 | 0.95386 / 0.95395 | encoding-branched GBDTs (combined-FE LGBM, freq-enc XGB) |
| stack v9 | 0.95400 | add a weak-but-orthogonal 5th member (TabICL) |
| **+ RealMLP recipe reproduced** | **0.95446** | faithfully reproducing a public strong model (+0.0017 single) |
| ridge-LR-logits meta | 0.95449 | swap meta-learner (greedy blend → ridge on logits) |
| **+ fold-split diversity** | **0.95460** | 7 / 10-fold variants as an orthogonal axis (diminishing returns) |

The final climb (0.95405 → 0.95460) came **not** from squeezing a single model, but from **(1) diagnosing an under-tuned model, (2) a better meta-learner, and (3) a new diversity axis (fold structure)**.

## What worked

1. **External raw-data row augmentation** — added to fold-train only (validation stays competition-only). The largest early jump.
2. **Model diversity + logit-space stacking** — encoding branches (Driver TE / XGB frequency / CatBoost native) lowered cross-correlation and beat public blenders.
3. **Reproducing a public strong model faithfully** — *the single biggest lever.* The team's RealMLP had been left at library defaults (architecture borrowed, recipe ignored), stuck at 0.9524 vs a public 0.9544. Cloning the full optimizer recipe + feature pipeline took it to 0.9540. **The ceiling was tuning, not data.**
4. **Regularized meta-learner (ridge on logits)** — extracts signal from weak-but-orthogonal members that a greedy blender discards.
5. **Fold-split diversity (5 / 7 / 10-fold)** — once feature and model axes had collapsed to ~0.99 correlation, **fold structure was still an independent axis** (residual ~0.53): cheap and reliable.
6. **Keeping weak-but-orthogonal members** — lowest individual score ≠ lowest value; they still raise effective diversity.

## What failed (the expensive lessons)

| Experiment | Outcome | Lesson |
|---|---|---|
| Time-series / cross-sectional FE (7 tries) | all net-negative | synthetic artifacts had corrupted the sequence signal |
| Heavy FE on GBDT (215 combos) | absorbed; residual random | **GBDTs absorb FE** — they reconverge on the same raw signal |
| Strengthening saturated members | correlation ↑, stack-add ≤ 0 | a **saturated** axis gains nothing from more of the same |
| Single-model racing | individual ↑, stack transfer ≈ 0 | tuning a saturated member doesn't transfer to the ensemble |

**Meta-lesson:** chasing the last +0.0005 in *features / single models* burned seven experiments. The real answer was the **meta-learner + an orthogonal diversity axis (fold structure)** — and the right unit of measurement was **stack-add / residual diversity**, not individual AUC (noise floor σ ≈ 0.0007).

## Lessons promoted into the template

Only **recurring, structural** lessons were promoted (per the [promotion criteria](../docs/harness_evolution.md)) — not competition-specific tricks:

- **Ceiling gate** (tunnel-vision guard) — register "ceiling estimate vs target gap" before opening a lever track; demote low-ceiling levers to side threads. → CLAUDE.md "experiment priority".
- **Both ceiling traps, named** — *don't over-tune saturated members*, **but also** *don't blanket-park every single-model effort*. A member far behind public/SOTA is **P0** (the costliest mistake here was leaving RealMLP ~0.002 behind for several sessions). → ceiling-gate caveat in CLAUDE.md.
- **Member diversity over single-model squeezing** — judge a member by stack-add / residual (not individual AUC), keep weak-but-orthogonal members, and treat **fold-split as a first-class diversity axis**. → workflow ordering + validation strategy.
- **OOF contract + adapter trainers** — every model emits the same OOF/submission contract on the same fixed split, so the stack pool scales without per-model special-casing. → `src/train_common.py`.
- **Single-entry-point FE + frozen-OOF discipline** — prevent the code fragmentation and OOF drift that recurred here. → `src/features.py`, `scripts/check_fold_inputs.py`.
- **Process-discipline guards** — cohesive commits, a fixed experiment-ID convention, a mandatory end-of-track retrospective, and proactive infra guards. These were the operational debt that hurt most, not score levers. → commit-reminder hook + CLAUDE.md process rules.

## What does *not* generalize

Competition-specific answers were deliberately **not** baked into the scaffold: the synthetic-artifact diagnosis, the exact encoding branches, and the specific RealMLP recipe live here as history. The template keeps the **workflow and guards, not the answers**.

---
_Distilled from the full internal post-mortem. Related: [docs/harness_evolution.md](../docs/harness_evolution.md) · [case_studies/README.md](README.md)._
