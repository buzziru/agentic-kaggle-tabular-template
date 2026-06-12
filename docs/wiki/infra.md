# GPU 실행 / 인프라 (SSOT)

> 환경 비교·선택 기준의 단일 출처. CLAUDE.md·exp-runner·README 가 여기를 참조한다.

베이스라인과 중간 실험은 로컬 CPU(`uv run python -m src.train model=lgbm ...`)로 돌리고,
대형 모델·장시간 튜닝만 GPU 로 오프로드한다. 참조 프로젝트는 **Lightning AI Studio** 환경에서
진행했고, 세 가지 실행 경로를 문서화했다. 같은 `src/` 코드가 환경만 바꿔 그대로 돌아간다.

| 경로 | GPU | 비용 | 실행 방식 | 언제 쓰나 | 런북 |
|---|---|---|---|---|---|
| **Lightning AI Job** | T4~H200 | 크레딧 과금 | 헤드리스 (`src/` 코드 그대로 Job 제출) | wandb online·반복/통합 라운드 | [lightning_jobs.md](lightning_jobs.md) |
| **Kaggle GPU 커널** | T4 / P100 | 무료 쿼터(주간 한도) | 헤드리스 (`src` Dataset push 후 `kernels push`) | torch 외 모델·단발 실행 | [kaggle_jobs.md](kaggle_jobs.md) |
| **Colab** | L4 24GB | Pro/PAYG | 노트북 직접 업로드·UI 실행(헤드리스 아님) | Kaggle T4 16GB OOM·L4 면 해결 | [colab_jobs.md](colab_jobs.md) |

선택 기준 비교표는 [kaggle_jobs.md](kaggle_jobs.md) 의 "Kaggle vs Lightning Job" 섹션이 단일 출처다.
노트북 변환·실행 규칙은 [notebook_conventions.md](notebook_conventions.md) 를 참조한다.
