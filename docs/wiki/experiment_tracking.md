# 실험 추적 — 단일 스트림 (상세)

> 원칙·요약은 루트 [CLAUDE.md](../../CLAUDE.md) §실험 추적. 이 문서는 경로·필드 상세 레퍼런스다.

실험 로그·OOF·Feature Registry·Ensemble·Submission History 를 **흩지 말고 한 스트림**으로 모은다.
base 모델·스택·앙상블이 전부 같은 로그 시스템을 거치게 해 비대칭을 없앤다.

- **JSON 로그(단일 스트림)**: `experiments/logs/<exp_id>.json` (`utils.log_experiment`, 자동). train_common(모든 모델)과 `stack.py`(앙상블) **모두** exp_id·git_hash·cv_scores·params·notes 를 기록한다. 앙상블은 members·meta-OOF·가중까지 params 에 남긴다.
- **OOF/Submission 파일**: `experiments/{oof,submissions}/<exp_id>.csv` 균일 계약 (train_common 자동).
- **Feature Registry**: 각 로그의 `params.feature_recipe`(feature_builder·TE·drop·extra_categorical)에 피처셋 정체를 박제한다 → 어떤 레시피를 어떤 exp 가 썼는지 질의할 수 있다.
- **Submission History**: `experiments/submission_history.csv` 원장(`predict.record_submission`)에 제출 시각·메시지·Public/Private LB 를 남기고, 동시에 해당 로그의 `lb_score` 를 동기화한다.
- **통합 뷰**: `uv run python scripts/summarize.py` 가 전 로그를 한 리더보드(exp·model·cv·lb·feature_builder)로 집계한다.
- **W&B**(선택): project `{{WANDB_PROJECT}}`, 인증은 `.env` 의 `WANDB_API_KEY`. 기본 활성이며 `use_wandb=false` 로 끈다.
