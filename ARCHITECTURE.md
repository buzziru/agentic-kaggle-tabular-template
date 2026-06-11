# 아키텍처

이 템플릿의 전체 데이터 흐름과 책임 경계를 한 장으로 나타낸다. 핵심 원리는
**단일 스캐폴드 + 어댑터**(모델 추가 = 어댑터 1개)와 **OOF 계약**(스택 풀의 디커플링
경계)이다. 자세한 규율은 [CLAUDE.md](CLAUDE.md)·[README.ko.md](README.ko.md) 참조.

```mermaid
flowchart TD
    CONF["conf/*.yaml<br/>model · features<br/>(+ Hydra overrides)"]
    CFG["src/config.py<br/>SEED · COLS · METRIC<br/>PROBLEM_TYPE · CV_STRATEGY"]

    CONF --> TRAIN["src.train<br/>single entrypoint (hydra.main)"]
    SPEC["specs/&lt;exp&gt;/expectation.yaml<br/>(pre-registration)"]
    GATE["guard_bash.py<br/>expectation gate<br/>full run ⇒ HEAD-committed expectation<br/>(max_folds screening exempt)"]
    SPEC -. referenced .-> GATE
    GATE -. blocks full run w/o committed expectation .-> TRAIN
    TRAIN --> REG["registry<br/>model.name → Trainer class"]
    REG --> TC
    CFG -. global constants .-> TC

    subgraph TC["train_common.run_oof_cv — shared OOF CV scaffold"]
        direction TB
        V["validate_problem_config<br/>problem_type ↔ metric ↔ cv ↔ objective"]
        G["multiclass guard<br/>(binary / regression = 1급)"]
        LD["data.load_train / load_test"]
        FE["features.build_features<br/>(+ feature_builder hook)"]
        FD["cv.get_folds<br/>Stratified / KFold / Group<br/>(+ make_groups · load-first splits/*.parquet)"]
        subgraph LOOP["per-fold loop"]
            direction TB
            TE["OOFTargetEncoder<br/>fold-safe · group-aware"]
            AUG["source augmentation<br/>(train-fold only)"]
            FIT["trainer.prepare → fit → predict<br/>get_metadata · save_model"]
            TE --> AUG --> FIT
        end
        PART{"max_folds?<br/>(screening)"}
        OUT["save OOF / submission / log"]
        SKIP["no artifacts<br/>(decision-only)"]
        V --> G --> LD --> FE --> FD --> LOOP --> PART
        PART -- "no (full)" --> OUT
        PART -- "yes (partial)" --> SKIP
    end

    subgraph ADAPT["ModelTrainer adapters (cat_prep)"]
        direction LR
        LGBM["LGBMTrainer"]
        XGB["XGBTrainer"]
        DOTS["..."]
    end
    REG -- selects --> ADAPT
    ADAPT -. prepare/fit/predict .-> FIT

    GUARD["scripts/check_fold_inputs.py<br/>frozen-OOF invariance guard"]
    GUARD -. verifies fit/predict inputs .-> LOOP

    OUT --> CONTRACT
    subgraph CONTRACT["OOF contract — decoupling boundary"]
        direction LR
        OOFF["experiments/oof/&lt;exp&gt;.csv"]
        SUBF["experiments/submissions/&lt;exp&gt;.csv"]
        LOGF["experiments/logs/&lt;exp&gt;.json"]
    end
    FROZEN["guard_frozen.py + frozen.txt<br/>frozen 멤버 산출물 수정 차단"]
    FROZEN -. protects (frozen exp_id artifacts) .-> CONTRACT
    SPEC -. expectation_path (ref only) .-> LOGF

    CONTRACT --> STACK
    subgraph STACK["src.stack — meta learner"]
        direction TB
        RID["validate ids"]
        RR["_resolve_regime<br/>(member cv strategy · n_folds)"]
        MF["cv.get_folds (same regime as base)"]
        META["meta-CV<br/>binary: equal / rank_mean / logistic / nnls<br/>regression: equal / linear / nnls"]
        SOUT["save stacked OOF / submission / log"]
        RID --> RR --> MF --> META --> SOUT
    end
    SOUT -. re-enters pool (same contract) .-> CONTRACT

    LOGF --> SUM["scripts/summarize.py<br/>unified leaderboard"]
```

## 읽는 법

- **실선 화살표** = 데이터/제어 흐름. **점선 화살표** = 참조·검증·선택(흐름 외 의존).
- `train_common.run_oof_cv` 가 공통 골격(검증→로드→피처→CV→fold loop→산출)을 **단일 소스**로
  통제하고, 모델 차이는 `ModelTrainer` 어댑터(`prepare/fit/predict/get_metadata/save_model`)만
  제공한다. 골격을 고치면 모든 모델이 한 번에 따라온다(노브 divergence 차단).
- **OOF 계약**(`experiments/{oof,submissions,logs}/<exp>.csv|json`)이 base·stack·앙상블을
  잇는 디커플링 경계다. `src.stack` 은 이 계약만 소비하고 모델 내부에 의존하지 않으며,
  meta-CV 는 멤버 로그에서 읽은 **base 와 동일 검증 레짐**으로 돈다.
- `check_fold_inputs` 는 리팩토링 전후 fit/predict 입력의 바이트 동등성을 검증해
  frozen 스택 멤버의 OOF 불변을 보장한다(GPU·실학습 불필요).
- `summarize.py` 가 모든 로그(base·stack)를 한 리더보드로 집계한다(단일 스트림).
