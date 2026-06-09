"""통합 학습 진입점 — registry 가 `model.name` 으로 트레이너를 선택해 공유 스캐폴드에 넘긴다.

흐름: cfg.model.name → get_trainer(name) → 트레이너 인스턴스 → run_oof_cv(cfg, trainer).
fold loop·OOF·metric·artifact·logging 은 전부 train_common 이 통제한다(모델 분기 없음).

⚠️ Hydra 기반이므로 REFACTORING.md 의 `--model lgbm` 은 `model=lgbm` 으로 쓴다
   (스윕·오버라이드를 보존; argparse 면 잃는다).

실행:
    uv run python -m src.train model=lgbm exp_id=exp_001 "notes='lgbm baseline'"
    uv run python -m src.train model=xgb  exp_id=exp_010 features=te_example
    uv run python -m src.train -m model=lgbm model.params.num_leaves=63,127,255   # 스윕
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.registry import get_trainer
from src.train_common import run_oof_cv


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = get_trainer(cfg.model.name)(cfg)
    run_oof_cv(cfg, trainer)


if __name__ == "__main__":
    main()
