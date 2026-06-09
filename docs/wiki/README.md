# Project Wiki

작업 지식을 **오래 남는 형태**로 기록하는 공간. (일회성 할 일/진행상황은 이슈 트래커, 상시 가이드는 루트 `CLAUDE.md`.)

## 문서 역할 구분
| 위치 | 역할 | 수명 |
|---|---|---|
| `CLAUDE.md` | 프로젝트 상시 가이드 (규칙·구조·실행법) | 항상 최신 |
| 이슈 트래커 | 실행 단위 (task / experiment / bug) | 열림→닫힘 |
| `docs/wiki/` | 결정·발견·회고 등 **지식 베이스** | 영속 |
| `docs/{eda,feature_engineering}.md` | 영역별 살아있는 작업 노트 | 갱신형 |

## 구성
- [`decisions.md`](decisions.md) — 의사결정 기록 (ADR-lite). 왜 그렇게 정했는지.
- `experiments/` — 주요 실험 회고 (가설 → 결과 → 결론). exp 단위로 추가.
- [`kaggle_jobs.md`](kaggle_jobs.md) · [`lightning_jobs.md`](lightning_jobs.md) · [`colab_jobs.md`](colab_jobs.md) — 헤드리스/GPU 실행 런북.
- [`notebook_conventions.md`](notebook_conventions.md) — Kaggle/Colab 노트북 작성 규칙.
- [`experiment_tracking.md`](experiment_tracking.md) — 실험 로그·OOF·Registry·Submission 경로/필드 상세.

## 작성 규칙
- 한 문서 = 한 주제. 제목에 날짜/이슈번호를 남긴다.
- 결론은 **수치 + 근거** 중심 (토큰 절약 원칙과 동일).
- 관련 Issue/PR/로그 경로를 상호 링크한다.
- ⚠️ **회고 의무**: 레버/트랙 종료 시 `experiments/exp_*.md` 회고를 작성해야 트랙 close (CLAUDE.md 프로세스 규율).
