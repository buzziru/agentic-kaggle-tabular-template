# Kaggle GPU 커널 — 헤드리스 실행 (모델 무관 SSOT)

> `src/` 코드를 **Kaggle GPU 커널**에서 헤드리스(`kaggle kernels` CLI)로 돌리는 **모델 무관 실행 절차·교훈** SSOT.
> GPU 모델(신경망·CatBoost·XGB 등) 실행 시 이 문서를 따른다. 대안 경로 = [[lightning_jobs]].
> 헤드리스 GPU 러너 에이전트의 교훈 SSOT(= 이 문서).
>
> **플레이스홀더**(프로젝트별로 채움): `{{KAGGLE_USER}}` = 본인 Kaggle 사용자명, `{{SRC_DATASET}}` = 코드 push 용 Dataset 슬러그(예: `proj-src`), `{{COMPETITION_SLUG}}` = 대회 슬러그, `{{EXTERNAL_DATASET}}` = (있으면) 외부 데이터 공개 Dataset, `{{WANDB_PROJECT}}` = W&B 프로젝트명.

## 핵심 동작
- **코드 이관**: `src/`+`conf/` 를 **Kaggle Dataset(`{{KAGGLE_USER}}/{{SRC_DATASET}}`)으로 push** → 노트북에서 `import`.
  누수검증된 `encoders`·`cv`·`features`·`data` 를 중복 없이 재사용(인라인 복붙 대비 동기화 리스크 0).
- **실행 = headless `kaggle kernels` CLI**: `kernel-metadata.json` + `.ipynb` 를 `kernels push` 하면
  **서버에서 즉시 실행**(GPU·Internet·data source 메타 지정). 수동 업로드 불필요, 전 과정 로컬 셸.
- **데이터**: 대회 데이터 = Kaggle competition input, (있으면) 외부 증강 = 공개 Dataset `{{EXTERNAL_DATASET}}`.

## 절차

### 1) src Dataset push (코드 변경 시마다)
```bash
bash kaggle/push_src_dataset.sh create               # 최초 1회
bash kaggle/push_src_dataset.sh version "변경 메모"    # 코드/conf 변경 후 갱신
```
→ `https://www.kaggle.com/datasets/{{KAGGLE_USER}}/{{SRC_DATASET}}` (`-r zip` 업로드 → Kaggle 가 `src/`·`conf/` 폴더로 정상 추출). **새 conf 파일을 노트북이 로드하면 반드시 먼저 version push.**

### 2) kernel push & 실행
`kaggle/kernel-metadata.json` 핵심: `id={{KAGGLE_USER}}/<slug>`, `code_file=<nb>.ipynb`, `enable_gpu=true`,
`enable_internet=true`, `dataset_sources=[{{KAGGLE_USER}}/{{SRC_DATASET}}, {{EXTERNAL_DATASET}}]`,
`competition_sources=[{{COMPETITION_SLUG}}]`, `is_private=true`, GPU 종류 = `machine_shape`(아래 교훈 2).
```bash
set -a; . ./.env; set +a
uv run kaggle kernels push -p kaggle/                 # 업로드 + 서버 실행 시작
```
- 노트북 흐름: `pip install` deps → `src` import + 경로 override(`/kaggle/working/{oof,submissions,logs}` 하위 분리) → repo `conf/*` 로 `cfg` 구성 → `run(cfg)`(`use_wandb=false`) → 산출물 검증.

### 3) 산출물 회수 (완료 후)
```bash
set -a; . ./.env; set +a
kaggle kernels output {{KAGGLE_USER}}/<slug> -p /tmp/<out>
cp /tmp/<out>/{oof,submissions,logs}/<exp>.* experiments/{oof,submissions,logs}/
```

## ⚠️ 운영 이슈 (실측)
- **동시 GPU 실행 가능(≥2)** — 슬러그 다른 커널을 연달아 push 하면 **서버에서 동시 RUNNING**(서로 다른 슬러그의 두 커널 동시 RUNNING 실측). → 독립 GPU 잡 **병렬 발사로 wall-clock 절약**(주간 GPU-h·세션 시간 한도 내).
- **slug = title 케밥케이스** — `kernels push` 실제 slug 는 metadata `id` 가 아니라 **title 을 케밥**으로 만든 것일 수 있음(예 title "model screen fold0" → slug `model-screen-fold0`). 회수 전 `kernels list --mine` 으로 실제 slug 확인.
- **status/get API 간헐 500** — `kernels status`·`output` 이 `GetKernelSessionStatus 500` / `kernels.get denied` 로 막히는 구간이 있다. `kernels list --mine`(별개 엔드포인트)는 동작 → slug·완료(lastRunTime) 추정에 사용, 회복 후 `output` 회수. **라이브 진행 모니터는 불가**(아래 교훈 4).

## ⚡ 실전 교훈 (GPU 모델 공통)
1. **Kaggle PyTorch 가 P100(sm_60) 미지원** — 현 이미지 torch 는 `sm_70 75 80 86 90 100 120`. **P100=sm_60 → CUDA 커널 불가**("no kernel image") 학습 크래시. `cuda.is_available()`·`get_device_name` 은 True 라 **assert 미검출**(연산에서야 터짐). → **신경망(torch)은 T4(sm_75)**. (CatBoost/XGB GPU 는 자체 커널이라 무관.)
2. **GPU 종류 = kernel-metadata `machine_shape`**(또는 `push --accelerator`). 검증값 `"nvidiaTeslaT4"`·`"nvidiaTeslaP100"`. 서버 검증 — 틀리면 push 에러. `enable_gpu:true` 병기.
3. **`from src import config` 깨짐** — `sys.path.append` + 빈 `__init__.py` 면 `src` 가 namespace 패키지로 shadowing → `cannot import name 'config'`. **수정**: `sys.path.insert(0,…)` + `__init__.py` 비우지 않기(1줄이라도).
4. **로그·산출물은 완료 후에만 + 모니터는 output-회수로만 판정** — `kernels output`/`logs` 는 실행 중 빈 응답(라이브 로그 불가). 완료 감지 = **`kernels output -p /tmp/out` 시도 → 기대 OOF 파일(`find <exp_id>.csv`) 출현 여부**(또는 웹). ⚠️ **`kernels status` 파싱으로 완료/실패 판정 금지**: 일시적 `500 Server Error` 응답 문자열의 "Error" 가 `grep -i error|cancel` 에 **오매칭** → 멀쩡히 RUNNING 인 커널을 FAILED 로 오판·모니터 조기종료. 모니터 루프 = output 회수 시도 후 OOF 파일 있으면 recover, 없으면 계속 폴링. 실패해도 종료 후 `.log` 회수 가능 → 에러 원문 확인. ⚠️ **monitor 발사 주체 = exp-runner**(push 직후 자기 워크플로우 안에서 background 발사; 메인이 별도로 호출하지 않는다). 완료 판정 = **결과물 회수 가능 여부(OOF 출현)** 지 `kernels status` 가 아니다.
5. **`kernels push` = 업로드 + 즉시 실행**(쿼터 소모). **fast-fail 가드 권장**: 노트북 앞단에 GPU·데이터 assert → 잘못된 환경이면 setup 직후 에러로 끝나 쿼터 절약.
6. **운영**: `kaggle datasets files` 는 첫 페이지만(페이지네이션). dataset 변경 → `push version` 후 kernel push(input 최신 버전 자동 참조).
7. **read-only 파일시스템 가드(초회 실행 `OSError: [Errno 30]`)** — `/kaggle/working` 외 경로는 쓰기 불가다. 노트북 cell3 가 `config.EXPERIMENTS_DIR` 의 **모든 파생 디렉터리(`OOF/SUBMISSION/LOG/TEST_PRED/MODEL`) + `DATA_DIR/splits`** 를 `/kaggle/working` 하위로 override 해야 한다(하나라도 빠지면 학습 완료 후 저장에서 OSError 재발). frozen splits 는 로컬에서 먼저 만들어(아무 실행 1회) `push_src_dataset.sh` 가 동반 push 하고, cell3 가 load-first 로 복사한다(원격 즉석 생성 시 sklearn 버전차로 분할 달라짐). ⓒ **write-probe(fast-fail)**: cell3 가 경로 override 직후 각 출력 디렉터리에 임시 파일을 써보고 read-only 면 즉시 `SystemExit` → 30분 학습 후가 아니라 5초 만에 누락 디렉터리를 지목(gen_kernel.py 코드화).

## Kaggle vs Lightning Job
| | Kaggle GPU 커널 | Lightning Job([[lightning_jobs]]) |
|---|---|---|
| 코드 | `.ipynb` 변환 + src Dataset push | `.venv/bin/python -m src...` 그대로 |
| GPU | T4/P100(무료 쿼터, 주간 한도) | T4/L4/A100… (크레딧 과금) |
| wandb | 헤드리스 online 어려움 → `use_wandb=false` | `-e WANDB_API_KEY` 한 줄(online ✓) |
| 회수 | `kernels output` | `/teamspace/jobs/<name>/artifacts/` 에서 `cp` |
| 라이브 로그 | ✗(완료 후만) | `job.logs`(SDK) |
→ 무료 쿼터 단발·torch 외 모델은 Kaggle, wandb-online·반복·통합 라운드는 Lightning. 둘 다 헤드리스·병렬 가능.

## 자산
`kaggle/` : 노트북 `.ipynb` 들, `kernel-metadata.json`(push마다 id/code_file 교체), `dataset-metadata.json`, `push_src_dataset.sh`, 커널 생성기 `gen_kernel.py`, 모니터 `monitor.py`, `README.md`. 노트북 작성 규칙 = [[notebook_conventions]].

## 예시 런북 — 첫 GPU 이관 (참조 구현에서 도출한 재사용 절차)
신경망(torch) 모델을 Kaggle GPU 로 처음 이관할 때의 검증된 순서. 모델·대회 무관 재사용.
1. **로컬 1-fold CPU 벤치로 비용 추정** — epoch당 초 × epoch × fold 로 5-fold wall-clock 산정. 로컬 비현실적이면 GPU 이관 확정.
2. **학습 코드 + conf 정비** — `src/train_<model>.py`(기존 학습 미러) + `conf/model/<model>.yaml`. 모델별 차이(범주형 처리·NaN 플레이스홀더·`sample_weight` 지원 여부·`best_iter` 해당 여부)를 코드에 명시.
3. **로컬 스모크**(2 epoch·소표본·full 과 동일 cfg 플래그)로 `run()` 전 경로 통과 확인 — OOF/submission/log shape 검수.
4. **src Dataset push** → `-r zip` 추출이 `src/`·`conf/` 폴더로 정상인지 1회 확인.
5. **(있으면) 외부 Dataset 무결성 검증** — 행수·파일명 `assert` 로 로컬본과 동일 확인.
6. **⚠️ 경로 충돌 주의** — `/kaggle/working` 에 OOF·submission·log 를 같이 두면 동일 exp_id 파일명 충돌 → `working/{oof,submissions,logs}` **하위 디렉터리로 분리**.
7. **fast-fail 가드**(교훈 5)로 잘못된 GPU·데이터를 setup 직후 차단.
- ⚠️ torch 모델은 **교훈 1[T4 필수]** 특히 해당(P100=sm_60 미지원).

## Sources
Kaggle API(`kaggle` CLI) · 헤드리스 GPU 실행 실측.
