# 노트북 작성 컨벤션 (Kaggle / Colab)

> GPU 노트북(`kaggle/*.ipynb`) 빌드 규칙. 실행 인프라 = [[kaggle_jobs]]·[[lightning_jobs]]. 코딩 컨벤션(Python)은 CLAUDE.md.
>
> ⚠️ 아래는 모두 **실제 빌드 실수**에서 도출 — 재발 방지 필수.

## 작성 = 생성기(필수, 손편집·복사 금지)
0. **Kaggle 커널 노트북 작성 = `KERNELS` 레지스트리 등록이 1단계. `kaggle/gen_kernel.py` 로 생성한다. 이전 노트북을 복사해 손으로 고치지 않는다.**
   - **🔑 등록 먼저(必)**: 새 커널은 **노트북을 쓰기 전에** `gen_kernel.py` 의 `KERNELS` 에 항목부터 추가하고 `python kaggle/gen_kernel.py <name>` 로 생성한다. `monitor.py` 는 **레지스트리 키(slug·exp_id)로 산출물을 회수**하므로 — **미등록 손작성 노트북은 monitor.py 가 회수 못 한다**(레지스트리 밖 손미러로 만들어 회수 불가 → 사후 등록 retrofit 사고). 발사 체크리스트 0순위 = `gen_kernel.py --list` 에 해당 키가 보이는지 확인.
   - **왜**: 손복사 워크플로의 재발 버그 근절 — ⓐ **2중 사본 drift**(최상위 `kaggle/<n>.ipynb` + push되는 `kaggle/<n>/<n>.ipynb` 사본이 따로 놀아 **편집이 stale 사본에 안 반영된 채 push**됨 → `use_wandb=True` 잔존 사고), ⓑ **복사-템플릿 상속**(이전 노트북 복사 후 exp_id·conf·gpu·use_wandb 중 하나를 놓쳐 **원본 설정 silent override**), ⓒ **미등록 = monitor 회수 불가**(위).
   - **단일 진실원 = `KERNELS` 레지스트리 dict**(gen_kernel.py 내). 커널 변경 = **파라미터만 고치고 재생성**: `python kaggle/gen_kernel.py <name>`(또는 `--all`). dir 에 `<name>.ipynb` + `kernel-metadata.json` **단일쌍**만 fresh 생성(중복 사본 없음).
   - ⚠️ **`use_wandb=False` 는 파라미터가 아니라 cfg 템플릿에 하드코딩** → 헤드리스 `kernels push` 에서 **구조적으로 True 가 될 수 없게** 한다(룰 9·[[kaggle_jobs]]). wandb 필요 시 Colab/Lightning 경로.
   - 새 모델 패밀리(NN 등)는 레지스트리에 항목 추가(+필요 시 템플릿에 deps/gpu-check/import 분기 확장). 레거시 손작성 노트북은 신규부터 생성기로 점진 이관.

## 가독성
1. **`;` 한 줄 다중 코드 금지.** `t0=time.time(); result=run(cfg)` 같은 한 줄 다중문 금지 — 디버깅·diff·셀 매칭이 나빠진다. 한 문장 = 한 줄.
2. **구분되는 내용은 한 줄 띄움.** 한 셀 안에서 논리 블록(로드/전처리/학습/출력)이 바뀌면 빈 줄로 구분.
3. **단계 주석 `# N) 설명`** 으로 셀 첫 줄 표기(예: `# 5) cfg + run`).

## 구조
3.5 **노트북 파일명 = `exp_id`(cfg `exp_id` 와 일치), `kaggle/<exp_id>.ipynb`.** **새로운 실험(config·피처·방향 변경)일 때만 새 노트북 생성**(공용 이름 금지). ⚠️ **마이너 수정(빌드 버그픽스·동일 config 재발사)은 기존 노트북 재사용/재push OK** — 새 exp_id 만들지 않는다. (예: 인코딩 변경=새 실험→새 노트북 / 빌드 버그픽스=기존 재사용.)
4. **셀당 단일 책임.** setup / import / 경로 override / config / run / save 를 셀로 분리. config 와 run 을 한 셀에 합칠 땐 **반드시 한 셀 안에서 cfg 정의 → run 순서**로 두고, **중복 config 셀을 만들지 말 것**.
   - (실수: config+run 통합 셀이 빌드 루프의 두 `if` 에 중복 매칭돼 cfg 정의가 run 셀로 덮여 `NameError: cfg not defined`.)
5. **config 정의가 run 보다 먼저.** 빌드 스크립트로 셀을 교체할 때 cfg 정의 셀이 누락/뒤섞이지 않았는지 검증(코드 셀에 `cfg = OmegaConf.create` 1회, `run(cfg)` 1회, 같은 위치인지).

## 안전
6. **full 전 소규모 fast-fail 셀.** 본 실행 전 작은 표본(예: 10k행)으로 fit/predict 1회 — API·GPU 메모리·의존성을 미리 검증해 쿼터/시간을 보호. ⚠️ **스모크는 풀 실행과 동일 cfg 플래그(특히 데이터 증강 같은 경로 분기 플래그)로** — 다른 플래그면 해당 경로가 미검증 통과(로컬 스모크 OFF·서버 ON 으로 증강-소스 NaN 경로 사망 사례).
   - (실수: 대형 입력 GPU OOM·임베딩 옵션 무효를 소규모로 사전 차단 가능했음.)
7. **의존성 설치 셀에 누락 금지.** `src.train_*` 가 import 하는 것 전부 설치 — hydra-core·모델 라이브러리 등. import 단계 실패는 GPU 도달 전 죽는다.
   - (실수: hydra 누락·모델 wrapper 라이브러리 누락으로 import 사망.)

## 토큰·출력
8. **출력 최소.** DataFrame 은 `.head(5)`/`.shape`/요약만, print 절제(CLAUDE.md 토큰 절약 원칙과 일관). 플롯은 노트북 빌드에 넣지 않는다.

## wandb
9. **인프라별 `use_wandb` 디폴트:** **Colab(사용자 UI 실행)·Lightning = `true`**(online, WANDB_API_KEY 선결 — Colab Secrets `userdata`/Lightning `-e`). **Kaggle 헤드리스(`kernels push`) = `false` 유지**(secret attach 미유지로 online 불가). 로컬 CPU 는 기본 on.

## 발사 전 체크리스트
- [ ] **0순위: `KERNELS` 레지스트리에 등록됨** — `python kaggle/gen_kernel.py --list` 에 키가 보임(미등록 = monitor.py 회수 불가)
- [ ] **`gen_kernel.py` 로 생성**(손복사·손편집 아님) · dir 에 노트북+메타 단일쌍
- [ ] `;` 다중문 없음 · 논리 블록 빈 줄 구분
- [ ] cfg 정의 셀 1개 + run 셀 위치 정상(중복 config 셀 없음)
- [ ] 소규모 fast-fail 셀 포함
- [ ] 설치 셀에 모든 의존성 포함
- [ ] 피처·파라미터 매니페스트 confirm(GPU 발사 전 피처 confirm 원칙)
