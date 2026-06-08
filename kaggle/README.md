# Kaggle 헤드리스 GPU 실행 — 자산

> 절차·교훈 SSOT = [`docs/wiki/kaggle_jobs.md`](../docs/wiki/kaggle_jobs.md). 노트북 작성 규칙 = [`docs/wiki/notebook_conventions.md`](../docs/wiki/notebook_conventions.md). 이 폴더 = 실행 자산.
> 실행은 **headless `kaggle kernels` CLI** — 노트북 수동 업로드 불필요. 전 과정 로컬 셸에서.
>
> **플레이스홀더**: `{{KAGGLE_USER}}` `{{SRC_DATASET}}` `{{COMPETITION_SLUG}}` — `gen_kernel.py` 상단 상수와 `*-metadata.json` 에서 채운다.

## 구성
- `gen_kernel.py` — **커널 노트북 생성기(SSOT)**. `KERNELS` 레지스트리에 파라미터를 등록하고 `python kaggle/gen_kernel.py <name>` 으로 `<name>/<name>.ipynb` + `kernel-metadata.json` 단일쌍을 fresh 생성. **손복사·손편집 금지.**
- `monitor.py` — 완료 모니터. `kaggle/monitor.py <name> ...` 로 output 회수 기반 폴링 → `experiments/{oof,submissions,logs}/` 자동 회수(status 파싱 안 함).
- `push_src_dataset.sh` — `src/`+`conf/` 를 코드 번들 Dataset 으로 push/version.
- `dataset-metadata.json` — 코드 번들 Dataset 메타.
- `kernel-metadata.json` — 단일 커널용 메타 예시(생성기를 안 쓸 때만; 보통 `gen_kernel.py` 가 자동 생성).

## 절차 (모두 로컬에서)
```bash
set -a; . ./.env; set +a
```
### 1. src 코드 Dataset push (1회 + 코드 변경 시)
```bash
bash kaggle/push_src_dataset.sh create               # 최초
bash kaggle/push_src_dataset.sh version "변경 메모"    # 코드 변경 후 갱신
```

### 2. 커널 생성 → push → 모니터 (GPU 쿼터 소모)
```bash
python kaggle/gen_kernel.py <name>                    # 노트북+메타 fresh 생성
uv run kaggle kernels push -p kaggle/<name>/          # 업로드 + 서버 실행 시작
uv run python kaggle/monitor.py <name>                # 완료까지 폴링·회수 (run_in_background 권장)
```

### 3. 산출물
`monitor.py` 가 `oof/<exp>.csv`→`experiments/oof/`, `submissions/<exp>.csv`→`experiments/submissions/`, `logs/<exp>.json`→`experiments/logs/` 로 회수.

## 주의
- 코드 변경 시 **반드시 `push version` 후** kernel push(노트북이 최신 Dataset 버전 자동 참조).
- `.env`/`kaggle.json` 시크릿은 업로드 번들에서 제외됨(스크립트가 `src`+`conf` 만 복사).
- 발사 전 체크리스트 = [`docs/wiki/notebook_conventions.md`](../docs/wiki/notebook_conventions.md) 하단.
