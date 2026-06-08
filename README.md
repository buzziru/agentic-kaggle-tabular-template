# Agentic Kaggle Tabular Template

**English** | [한국어](README.ko.md)

A **focused, opinionated** template for **Kaggle-style tabular experimentation with a coding agent** —
OOF cross-validation, stacking, leakage guards, and a post-mortem-driven workflow, distilled from a completed Kaggle run.
Not an empty boilerplate: it ships a **working scaffold** plus **process guards derived from real post-mortems**.
It is **not** a general-purpose ML starter — read **Scope** below before adopting.

> 🇰🇷 **Note on language:** This README is in English, but the core guide ([CLAUDE.md](CLAUDE.md)) and the
> detailed docs under `docs/` are written in **Korean** — they are the canonical source. This README is a
> translation of [README.ko.md](README.ko.md). If something here drifts, the Korean version wins.

> **This README describes the template repo itself.** The README each project actually uses lives separately in
> [docs/PROJECT_README.template.md](docs/PROJECT_README.template.md) (copy it to the root when you start).

> **Try it:** `uv sync && uv run python examples/run_example.py` — runs the whole pipeline
> (load → features → CV → OOF → submission → log) on dummy data so you can see it work. Details: [examples/](examples/).

## Scope — who this is for

**A good fit if you:**

- compete on **Kaggle-style tabular** problems (CSV in → OOF / submission out);
- work **with Claude Code** (or a similar coding agent) and want an agent-native workflow with built-in guards;
- value **experiment discipline** — OOF CV, stacking/blending, leakage prevention, reproducible single-stream logs.

**Probably *not* a fit if you:**

- need **production ML** — serving, pipelines, MLOps, monitoring. This targets offline experimentation, not deployment.
- work on **image / NLP / deep-learning-centric** tasks. The scaffold is **GBDT / tabular-first** (LightGBM / XGBoost adapters).
- are a **complete ML beginner**. It assumes working knowledge of cross-validation, leakage, ensembling, and `Hydra` / `uv`.
- **don't use a coding agent.** The code runs standalone, but the agents (`.claude/agents/`) and hooks are central to the design — without them you skip a main reason to pick this over a plainer template.

## Why this template

- **A working scaffold.** It's code that runs, not just docs. The shared training skeleton (CV · OOF · logging · submission) lives in `src/train_common.py` alone. A new model doesn't copy the skeleton — it only defines two model-specific callbacks (`prepare` = categorical preprocessing, `fit_predict` = train/predict). So **adding one model takes ~40 lines**, and editing the skeleton updates every model at once. LightGBM (`src/train_lgbm.py`) and XGBoost (`src/train_xgb.py`) are the worked examples.
- **Post-mortem-driven process guards.** Failures that recurred in real runs are baked into code and check gates: two copies of the same code drifting apart, config knobs diverging, a frozen OOF getting mutated, over-investing in low-ceiling levers. The guards: `scripts/check_fold_inputs.py` (OOF invariance check), a Stop-hook commit reminder, and a "ceiling gate" (tunnel-vision guard).
- **Leakage-safe by default.** OOF target encoding fit inside the fold only (`src/encoders.py`), row-wise feature examples plus a group/time-series past-only recipe ([docs/feature_engineering.md](docs/feature_engineering.md)), and a CV-must-match-the-split rule are enforced from the start.
- **Single-stream experiment tracking.** Every model, stack, and ensemble goes through the same JSON log and OOF/submission contract. `scripts/summarize.py` rolls them up into one leaderboard.
- **Claude Code native.** Custom subagents (`.claude/agents/`), token-saving discipline, and auto-reminder hooks ship by default — designed for collaboration with AI agents.
- **Modern tooling.** `uv` for dependencies, `Hydra` for experiment knobs, and W&B optionally for tracking.

## Core design principles


| Principle                                                                              | Enforced in                                        |
| -------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Features in a single entry point only (`build_features`) — prevents code fragmentation | `src/features.py`, `conf/features/*.yaml` knobs    |
| Adding a model = an adapter (no scaffold duplication)                                  | `src/train_common.py` + `src/train_<model>.py`     |
| Frozen member OOF is immutable                                                         | `scripts/check_fold_inputs.py`                     |
| Record decision rationale (ADR-lite) + mandatory end-of-track retrospective            | `docs/wiki/decisions.md`, `docs/wiki/experiments/` |
| Be aware of measurement power (don't judge a small Δ on a single seed)                 | `docs/setup_questions.md`, validation strategy     |


The rationale behind each principle is in [CLAUDE.md](CLAUDE.md) (the always-on guide shared by AI agents and humans; written in Korean).

## Structure

```
CLAUDE.md          # Always-on project guide (rules·structure·how-to-run) — copied into each new project
TASK.md            # Milestone / split-task index
CURRENT_STATUS.md  # Session handoff (current values · next actions, SSOT)
conf/              # Hydra config — tuning/experiment knobs (config.yaml, model/, features/)
src/               # config·data·features·encoders·cv·train_common·train_lgbm·train_xgb·stack·predict·utils
scripts/           # gates·aggregation (check_fold_inputs, summarize, hooks/)
docs/              # data_dictionary · eda · feature_engineering · setup_questions
docs/wiki/         # decision log (ADR-lite) · experiment retrospectives · infra runbooks (Kaggle/Colab/Lightning)
docs/PROJECT_README.template.md  # the README each new project copies and fills in
.claude/agents/    # custom subagents (eda-explorer · feature-smith · kaggle-runner)
kaggle/            # headless GPU run assets (gen_kernel·monitor·push)
experiments/       # logs(JSON) · oof · submissions  (contents git-ignored)
```

## Starting a new project from this template

```bash
# 1) Use the template (GitHub 'Use this template' or clone)
git clone <this-repo> my-project && cd my-project

# 2) Prepare the project README — copy the template to root and fill it in (overwrites this landing README)
cp docs/PROJECT_README.template.md README.md

# 3) Dependencies (uv)
uv sync                              # add eda/gpu: uv sync --extra eda --extra gpu
#  → commit the generated uv.lock to your project (the lock is a per-project artifact; not shipped with the template)

# 4) Credentials
cp .env.example .env                 # fill in KAGGLE_USERNAME/KAGGLE_KEY/WANDB_API_KEY

# 5) What to fill in
#    - CLAUDE.md      : fill the {{...}} per the top quote block, then delete that block
#    - src/config.py  : ID/TARGET/column definitions · METRIC · COMPETITION · CV strategy
#    - docs/setup_questions.md : setup decisions such as CV-strategy rationale

# 6) Baseline (Hydra: OOF + submission file + JSON log + W&B)
uv run python -m src.train_lgbm exp_id=exp_001 "notes='baseline'"
uv run python scripts/summarize.py   # experiment leaderboard
```

## Workflow

First, close the whole pipeline once with a baseline (load → features → CV → OOF → submission). Then proceed in this order:

**EDA → baseline → features → model diversity → stacking/blending → (last) tuning**

A key lesson from the reference project: diversity and ensembling have a bigger ROI than tuning a single model.
See [TASK.md](TASK.md) for milestones/gates and [CURRENT_STATUS.md](CURRENT_STATUS.md) for session handoff.

## GPU execution / infrastructure

Run baselines and mid-experiments on local CPU (`uv run python -m src.train_lgbm ...`); offload only large models and long tuning to GPU. The reference project ran on **Lightning AI Studio**, and three execution paths are documented for GPU use. The same `src/` code runs unchanged across environments — only the environment changes — and each runbook captures operational issues and field lessons.


| Path                  | GPU       | Cost                      | Execution                                               | When to use                                                               | Runbook                                          |
| --------------------- | --------- | ------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------ |
| **Lightning AI Job**  | T4~H200   | metered credits           | headless (submit `src/` code as-is as a Job)            | wandb online · iterative/integration rounds (the base project's main env) | [lightning_jobs.md](docs/wiki/lightning_jobs.md) |
| **Kaggle GPU kernel** | T4 / P100 | free quota (weekly limit) | headless (push `src` as a Dataset, then `kernels push`) | non-torch models · one-off runs                                           | [kaggle_jobs.md](docs/wiki/kaggle_jobs.md)       |
| **Colab**             | L4 24GB   | Pro/PAYG                  | upload & run the notebook in the UI (not headless)      | models that OOM on Kaggle's T4 16GB but fit on L4                         | [colab_jobs.md](docs/wiki/colab_jobs.md)         |


The single source for the selection comparison is the "Kaggle vs Lightning Job" section in `kaggle_jobs.md`. For notebook conversion/run rules, see [notebook_conventions.md](docs/wiki/notebook_conventions.md).

## Security

Secrets like `.env` and `kaggle.json` are excluded via `.gitignore`. **Never commit them.**
Credentials are loaded from `.env` (`src/utils.load_env`).

## License

[MIT](LICENSE)