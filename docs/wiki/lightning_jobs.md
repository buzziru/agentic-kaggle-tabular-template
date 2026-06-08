# Lightning AI Jobs — GPU 백그라운드 훈련 (Kaggle 대안)

> `src/` 코드를 **그대로**(노트북 변환·Dataset push 없이) 별도 GPU 머신의 비동기 Job 으로 돌리는 방법. SSOT: 이 문서.
> CPU cloudspace 세션에서 GPU 훈련이 필요할 때 사용. 대안 경로 = [[kaggle_jobs]].
>
> **플레이스홀더**(프로젝트별로 채움): `{{STUDIO}}` = 스튜디오 이름, `{{TEAMSPACE}}` = teamspace 이름, `{{TEAMSPACE_OWNER}}` = teamspace owner 사용자명(로그인 계정과 다를 수 있음 — 아래 트러블슈팅), `{{LOGIN_USER}}` = 현재 로그인 계정, `{{PROJECT_ID}}` = `$LIGHTNING_CLOUD_PROJECT_ID`, `{{WANDB_PROJECT}}` = W&B 프로젝트명.

## 환경 사실
- 플랫폼: **Lightning AI Studio**(클라우드). `lightning` CLI + `lightning-sdk` 설치(시스템 conda env).
- 현재 teamspace = **`{{TEAMSPACE}}`**(project_id = `{{PROJECT_ID}}` = `$LIGHTNING_CLOUD_PROJECT_ID`). ⚠️ **owner = 사용자 `{{TEAMSPACE_OWNER}}`**, 현재 로그인 `{{LOGIN_USER}}` 는 *멤버*일 수 있다. 그 경우 owner 지정 시 **`--user {{TEAMSPACE_OWNER}}`**(로그인 username 아님 — 아래 트러블슈팅).
- 현재 스튜디오 이름 = **`{{STUDIO}}`**(`/teamspace/studios/this_studio` 는 디렉터리 별칭).

## 두 가지 "백그라운드"
1. **Studio 백그라운드 실행** — `run_in_background` bash 등. **현재 스튜디오 머신**(GPU 없는 cloudspace면 CPU)에서 돌고 세션을 닫아도 지속. CPU 작업(예: Optuna 스터디)에 적합. **GPU 불가.**
2. **Jobs (`lightning run job` / SDK `Job.run`)** — **별도 GPU 머신**을 띄워 비동기 실행, 세션과 분리. GPU 훈련은 이쪽.

## Jobs 핵심 동작
- **환경 스냅샷**(`--studio` 모드): 현재 스튜디오의 **설치 패키지 + 파일(.venv·src·conf·data 포함)을 스냅샷**해 GPU 머신에서 실행 → `.venv/bin/python -m src....` **그대로 동작**(Kaggle 처럼 `.ipynb` 변환/Dataset push 불필요). 작업 디렉터리 = 스튜디오 루트.
- **비동기·분리**: 제출 후 스튜디오를 꺼도 Job 은 계속 실행.
- **산출물 회수**: Job 은 별도 머신에서 돌아 출력이 라이브 스튜디오 FS 에 자동 병합되지 **않는다**. 대신 Job 작업디렉터리를 미러한 **artifact 경로**에 남고 스튜디오에서 접근 가능:
  - 경로 = **`/teamspace/jobs/<job-name>/artifacts/`** (SDK `job.artifact_path`). Job 이 `experiments/oof/x.csv` 에 쓰면 → `/teamspace/jobs/<job-name>/artifacts/experiments/oof/x.csv`.
  - 회수: 필요한 파일을 로컬 `experiments/...` 로 `cp`. (`artifacts_remote`/`path_mappings` 로 명시 매핑도 가능.)
  - 로그: `job.logs`(SDK). W&B 는 `-e WANDB_API_KEY` 면 Job 안에서 정상 동기화.
- **시크릿/환경변수**: `-e KEY=VALUE`. **wandb 키를 `-e WANDB_API_KEY=...` 로 주입**하면 GPU 실험 wandb-on 을 Kaggle Secrets 없이 해결.
- **머신 타입**(GPU): `T4`, `T4_X_2/4/8`, `L4`, `L4_X_2/4/8`, `L40S(_X_2/4/8)`, `A100`, `A100_80GB`(`_X_2/4/8`), `H100(_X_2/4/8)`, `H200`, `B200_X_8`. CPU: `CPU_SMALL`~`CPU_X_16`.
- **과금**: Job 실행 시간만 과금, 종료 시 머신 회수(상시 스튜디오의 idle 비용 없음). `--interruptible` = 스팟(저렴, 선점 가능 → 체크포인트 권장). `--max_runtime <초>` 상한.

## 사용법

### CLI
```bash
lightning run job \
  --name <unique-name> \
  --machine L4 \
  --studio {{STUDIO}} \
  --teamspace {{TEAMSPACE}} --user {{TEAMSPACE_OWNER}} \
  --command ".venv/bin/python -m src.train_<model> exp_id=... model=<model> features=<f> ..." \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  [--interruptible] [--max_runtime 14400]
```
- `--command` 은 Job 셸에서 스튜디오 루트 기준 실행. **`.venv/bin/python`** 으로 호출(uv PATH 의존 회피).
- 복잡한 Hydra 리스트 오버라이드(`features.x=[a,b]`)는 셸 인용이 까다로우니 **전용 conf 파일**로 빼는 게 안전.

### SDK (파이썬 제어 — 워크플로우 통합용)
```python
from lightning_sdk import Job, Machine
job = Job.run(
    name="<unique-name>",
    machine=Machine.L4,
    studio="{{STUDIO}}",
    teamspace="{{TEAMSPACE}}",   # owner=user '{{TEAMSPACE_OWNER}}' → Teamspace(name='{{TEAMSPACE}}', user='{{TEAMSPACE_OWNER}}')
    command=".venv/bin/python -m src.train_<model> exp_id=exp_NNN model=<model> features=<f> use_wandb=true",
    env={"WANDB_API_KEY": "..."},
    interruptible=False,
    max_runtime=14400,
)
```
`Job.run(...)` 주요 인자: `name, machine, command, studio, image, teamspace, env, interruptible, max_runtime, artifacts_local/remote, path_mappings, cloud_account`.

## Hydra 리스트 오버라이드는 전용 conf 로
셸 인용이 까다로운 리스트 노브(예: 추가 범주형 컬럼)는 CLI 브래킷 대신 **전용 `conf/features/*.yaml`** 로 빼고 `features=<name>` 만 넘긴다. 예: `conf/features/<variant>.yaml` 에 `extra_categorical_cols: [...]` 를 두면 CLI 에선 `features=<variant>` 만 주면 됨(브래킷 오버라이드 회피).

## 예: Job 1건 실행 → 회수 (패턴)
```bash
set -a; . ./.env; set +a   # WANDB_API_KEY 로드
lightning run job --name <name> --machine L4 \
  --studio {{STUDIO}} --teamspace {{TEAMSPACE}} --user {{TEAMSPACE_OWNER}} \
  --command ".venv/bin/python -m src.train_<model> exp_id=<exp> model=<model> features=<f> use_wandb=true" \
  -e WANDB_API_KEY=$WANDB_API_KEY
# 회수:
cp /teamspace/jobs/<name>/artifacts/experiments/{oof,submissions,logs}/<exp>.* experiments/...
```
- end-to-end 검증 절차: 제출 → Completed → artifact 회수 → W&B(`{{WANDB_PROJECT}}`) 동기화 확인.

## Kaggle 대비
→ 비교표는 [[kaggle_jobs]] "Kaggle vs Lightning Job" 섹션(단일 관리). 요약: **무료 쿼터 단발·torch 외 모델은 Kaggle, wandb-online·반복·통합 라운드는 Lightning**. 둘 다 헤드리스·병렬 가능.

## 트러블슈팅 — teamspace owner 해석
- 증상: `lightning run job`/`lightning list studios`/SDK `Teamspace()`/`Studio()` 가
  `ValueError: Teamspace <login>/<ts> does not exist ... member of organizations: []` 로 실패.
- 원인: teamspace 의 **owner 가 로그인 사용자가 아니라 다른 user**. 로그인 계정은 멤버일 뿐이라 `user=<login>` 으로는 안 잡힘. (rest client `projects_service_list_memberships` 로 `owner_type=user`, `owner_id` ≠ 현재 user.id 확인해 진단.)
- ✅ 해결: owner 를 **실제 owner 사용자명(`{{TEAMSPACE_OWNER}}`)** 으로 지정. CLI `--user {{TEAMSPACE_OWNER}}`, SDK `Teamspace(name='{{TEAMSPACE}}', user='{{TEAMSPACE_OWNER}}')`. (토큰/로그인이 정상이어도 발생 — `lightning login` 과 무관.)

## Sources
[Background execution](https://lightning.ai/docs/overview/ai-studio/background-execution) · [Job outputs](https://lightning.ai/docs/overview/batch-jobs/job-outputs) · [Artifacts](https://lightning.ai/docs/overview/artifacts) · [SDK reference](https://lightning.ai/docs/overview/sdk-reference).
