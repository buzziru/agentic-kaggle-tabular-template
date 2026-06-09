"""공유 OOF CV 스캐폴드 — 모든 모델 트레이너의 단일 진입점.

⚠️ 모델 추가 = 이 스캐폴드를 **복사하지 말고** prepare/fit_predict 콜백만 제공한다.
   (스택 풀이 커질 때 모델별 트레이너가 각자 골격을 복제하면 리팩토링이 불가능해진다 —
    참조 프로젝트 회고: LGBM 만 별도 경로였다가 노브 divergence 가 반복돼 패리티
    게이트까지 필요했다. 템플릿은 처음부터 LGBM 포함 전 모델을 이 스캐폴드의 어댑터로 둔다.)

각 트레이너는 `ModelTrainer`(src/registry.py) 인터페이스를 구현해 모델별
**prepare**(범주형 전처리)·**fit/predict/get_metadata/save_model**만 제공하고, 나머지 공통
골격(seed/env/wandb · build_features+feature_builder 훅 · feat/te/cat 컬럼 ·
fold OOF-TE+증강 concat · OOF/submission/로그 · wandb)은 여기서 처리한다.

OOF 계약(스택 풀의 디커플링 경계): 모든 모델은 동일 fold(seed)로
  experiments/oof/<exp_id>.csv = [id, oof]
  experiments/submissions/<exp_id>.csv = [id, <target>]
형식만 산출한다. `src.stack` 은 이 계약만 소비하고 모델 내부 코드에 의존하지 않는다.

⚠️ 리팩토링 안전망: 스캐폴드/피처를 고친 뒤 frozen 스택 멤버의 OOF 가 안 바뀌었는지
   `scripts/check_fold_inputs.py`(입력 동등성, GPU 불필요)로 검증한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src import config, cv, data, encoders, features, utils

if TYPE_CHECKING:
    from src.registry import ModelTrainer


def run_oof_cv(cfg: DictConfig, trainer: "ModelTrainer") -> dict[str, Any]:
    """공유 OOF CV 파이프라인.

    Args:
        cfg: Hydra 설정.
        trainer: `ModelTrainer`(src/registry.py) 구현체. prepare/fit/predict/
            get_metadata/save_model 와 supports_weight·log_extra() 만 제공하고, fold
            loop·OOF·metric·artifact·logging 은 여기서 통제한다(모델 분기 금지).

    Returns:
        cv_mean, cv_std, fold_scores, log_path.
    """
    if config.PROBLEM_TYPE == "multiclass":
        raise NotImplementedError(
            "multiclass 는 1-D OOF 계약(OOF 단일 열·submission 단일 타깃)을 넘어선다 — "
            "OOF k열·submission 다열·stack 확장이 필요하다. 템플릿은 binary/regression 을 1급 "
            "지원하며 multiclass 는 train_common/stack 의 OOF 계약을 확장해야 한다(확장점)."
        )
    # problem_type↔metric↔cv_strategy↔objective 명백한 불일치를 학습 전에 차단(조용한 오채점 방지).
    utils.validate_problem_config(model_objective=cfg.model.params.get("objective"))
    supports_weight = trainer.supports_weight
    seed = cfg.get("seed", config.SEED)  # 모델 seed (seed averaging 노브). fold 분할은 config.SEED 고정.
    utils.seed_everything(seed)
    utils.load_env()
    scorer = utils.get_scorer()  # config.METRIC 기준 (하드코딩 제거)

    exp_id, notes, use_wandb = cfg.exp_id, cfg.notes, cfg.use_wandb
    te_smoothing = cfg.features.target_encode_smoothing
    aug_enabled, aug_weight = cfg.augment.enabled, cfg.augment.weight
    if aug_enabled and aug_weight != 1.0 and not supports_weight:
        raise ValueError(
            f"이 모델은 sample_weight 미지원 — augment.weight={aug_weight}≠1.0 불가 "
            "(plain concat 만). weight=1.0 으로 실행하라."
        )
    feature_builder = cfg.features.get("feature_builder", None)

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=exp_id,
            notes=notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def build(df: pd.DataFrame) -> pd.DataFrame:
        df = features.build_features(df)
        if feature_builder:
            df = getattr(features, feature_builder)(df)
        return df

    if feature_builder:
        print(f"[features] feature_builder={feature_builder} 적용")
    train_df = build(data.load_train())
    test_df = build(data.load_test())

    # 플레이스홀더 미채움 자동 차단 (템플릿 복사 후 config 미수정 silent 통과 방지).
    for _n, _v in [("ID_COL", config.ID_COL), ("TARGET_COL", config.TARGET_COL)]:
        if _v not in train_df.columns:
            raise ValueError(f"config.{_n}='{_v}' 가 데이터 컬럼에 없음 — src/config.py 를 데이터에 맞게 채웠는지 확인")
    if use_wandb and "{{" in config.WANDB_PROJECT:
        raise ValueError("config.WANDB_PROJECT 플레이스홀더 미치환 — 채우거나 use_wandb=false")

    feat_cols = features.get_feature_cols(train_df)
    drop_cols = list(cfg.features.drop_cols)
    feat_cols = [c for c in feat_cols if c not in drop_cols]

    te_cols = [c for c in cfg.features.target_encode_cols if c in feat_cols]
    cat_cols = [c for c in config.CATEGORICAL_COLS if c in feat_cols and c not in te_cols]
    # 모델별 추가 범주형 (extra_categorical_cols). 기본 없음 → 미지정 모델/실험은 불변.
    for c in cfg.features.get("extra_categorical_cols", []) or []:
        if c in feat_cols and c not in te_cols and c not in cat_cols:
            cat_cols.append(c)

    x = train_df[feat_cols]
    y = utils.cast_target(train_df[config.TARGET_COL])  # PROBLEM_TYPE 별 dtype (binary=int·regression=float)
    x_test = test_df[feat_cols]

    x_src = y_src = None
    if aug_enabled:
        src_df = build(data.load_source_augmentation())
        x_src = src_df[feat_cols]
        y_src = utils.cast_target(src_df[config.TARGET_COL])
        print(f"[augment] 원본 {len(x_src):,}행 추가 (weight={aug_weight})")

    # 모델별 범주형 전처리 (어댑터가 제공).
    x, x_test, x_src, state = trainer.prepare(x, x_test, x_src, cat_cols, aug_enabled)

    oof = np.zeros(len(train_df))
    test_pred = np.zeros(len(test_df))
    fold_scores: list[float] = []
    best_iters: list[int | None] = []

    n_folds = int(cfg.get("n_folds", config.N_FOLDS))  # split 다양성(7/10-fold) 지원, 기본 N_FOLDS
    folds = cv.get_folds(y, n_folds=n_folds, groups=cv.make_groups(train_df))
    max_folds = cfg.get("max_folds", None)
    partial = bool(max_folds)  # 스크리닝 = 의사결정용. 스택 멤버 아님 → 아래서 OOF/submission 미저장.
    if partial:
        folds = folds[:max_folds]
        print(f"[max_folds] 앞 {max_folds}/{n_folds} fold 만 실행 (스크리닝 — OOF/submission 미저장)")
    save_models = bool(cfg.get("save_models", False))  # 적합 모델 저장 (추론/재사용용, 기본 off)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
        x_va, y_va = x.iloc[va_idx], y.iloc[va_idx]
        x_te = x_test

        # ⚠️ 누수 방지: 타깃 인코딩은 fold 의 train 부분으로만 fit.
        enc = None
        if te_cols:
            enc = encoders.OOFTargetEncoder(te_cols, smoothing=te_smoothing)
            x_tr = enc.fit_transform_train(x_tr, y_tr)
            x_va = enc.transform(x_va)
            x_te = enc.transform(x_test)

        # 외부 증강: train fold 에만 추가 (검증/test 미포함).
        w_tr = None
        if aug_enabled:
            n_comp = len(x_tr)
            x_src_f = enc.transform(x_src) if enc is not None else x_src.copy()
            x_tr = pd.concat([x_tr, x_src_f], ignore_index=True)
            y_tr = pd.concat(
                [y_tr.reset_index(drop=True), y_src.reset_index(drop=True)], ignore_index=True
            )
            if supports_weight:
                w_tr = np.concatenate([np.ones(n_comp), np.full(len(x_src_f), aug_weight)])

        model = trainer.fit(x_tr, y_tr, x_va, y_va, w_tr, cat_cols, state)
        oof[va_idx] = trainer.predict(model, x_va)
        test_pred += trainer.predict(model, x_te) / len(folds)
        best_iter = trainer.get_metadata(model).get("best_iter")
        if save_models:
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            trainer.save_model(model, config.MODEL_DIR / f"{exp_id}_fold{fold}")
        score = scorer(y_va, oof[va_idx])
        fold_scores.append(score)
        best_iters.append(best_iter)
        bi = f" (best_iter={best_iter})" if best_iter is not None else ""
        print(f"[fold {fold}] score = {score:.6f}{bi}")
        if wandb_run is not None:
            log = {"fold": fold, "fold_score": score}
            if best_iter is not None:
                log["best_iter"] = best_iter
            wandb_run.log(log)

    if partial:
        oof_score = float("nan")  # 부분 실행 → 전체 OOF 무의미. fold 점수만 신뢰.
        print(f"\n[부분 실행 {len(folds)}/{n_folds}] fold mean={np.mean(fold_scores):.6f} (OOF 생략)")
    else:
        oof_score = scorer(y, oof)
        print(f"\nOOF = {oof_score:.6f} | mean={np.mean(fold_scores):.6f} std={np.std(fold_scores):.6f}")

    # ⚠️ 스크리닝(partial)은 의사결정용 — 부분 OOF(미실행 fold=0)는 스택 풀을 오염시키므로 저장 금지.
    if partial:
        print("[partial] OOF/submission 미저장 (스택 멤버 아님). 로그엔 부분 실행으로 표기.")
    else:
        config.OOF_DIR.mkdir(parents=True, exist_ok=True)
        config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({config.ID_COL: train_df[config.ID_COL], "oof": oof}).to_csv(
            config.OOF_DIR / f"{exp_id}.csv", index=False
        )
        sub = data.load_sample_submission()
        sub[config.TARGET_COL] = test_pred
        sub.to_csv(config.SUBMISSION_DIR / f"{exp_id}.csv", index=False)

    te_note = f"target_encode={te_cols}" if te_cols else "no target encoding"
    logged_iters = best_iters if any(b is not None for b in best_iters) else None
    model_params = OmegaConf.to_container(cfg.model.params, resolve=True)
    log_extra = trainer.log_extra() if hasattr(trainer, "log_extra") else {}
    # feature recipe = Feature Registry 키 (어떤 빌더·TE·drop·범주 조합으로 만든 피처셋인지).
    feature_recipe = {
        "feature_builder": feature_builder,
        "target_encode_cols": te_cols,
        "drop_cols": drop_cols,
        "extra_categorical_cols": list(cfg.features.get("extra_categorical_cols", []) or []),
    }
    # 실제 사용한 fold 수·부분 실행을 정직하게 라벨링 (config.N_FOLDS 하드코딩 불일치 방지).
    cv_label = f"{config.CV_STRATEGY}_{n_folds}" + (f"_partial{len(folds)}" if partial else "")
    default_note = (
        f"partial screening {len(folds)}/{n_folds}; {te_note}"
        if partial
        else f"OOF={oof_score:.6f}; {te_note}"
    )
    log_path = utils.log_experiment(
        exp_id=exp_id,
        model=cfg.model.name,
        features=feat_cols,
        cv_scores=fold_scores,
        params={**model_params, "seed": seed, "feature_recipe": feature_recipe, **log_extra},
        best_iters=logged_iters,
        notes=notes or default_note,
        kill_criterion=cfg.get("kill_criterion", ""),
        cv_strategy=cv_label,
    )
    print(f"로그 저장: {log_path}")

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "oof_score": oof_score,
                "cv_mean": float(np.mean(fold_scores)),
                "cv_std": float(np.std(fold_scores)),
            }
        )
        wandb_run.finish()

    return {
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "log_path": str(log_path),
    }
