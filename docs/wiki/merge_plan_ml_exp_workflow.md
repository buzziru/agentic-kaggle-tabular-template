# MERGE_PLAN — ml-exp-workflow → 템플릿 병합 지침

> 이 문서는 Claude Code 가 `ml-exp-workflow/` 폴더의 워크플로를 기존 템플릿에
> 병합하는 작업 지침이다. 작업 전 반드시 읽을 것: 이 문서 전체, `CLAUDE.md`,
> `ARCHITECTURE.md`, `ml-exp-workflow/CLAUDE.md`, `docs/harness_evolution.md`.
> 병합 완료 후 이 문서는 `docs/wiki/` 로 이동하고 루트에서 제거한다.

---

## 0. 병합의 대원칙 (전 단계에 적용)

1. **아티팩트는 동결, 공유 코드는 검증 게이트.** 워크플로의 "코드 동결"을
   그대로 가져오지 않는다. 템플릿의 어댑터 아키텍처(단일 스캐폴드 +
   `ModelTrainer` 어댑터)는 유지하며, 동결 대상은 frozen 스택 멤버의
   **산출물**(`experiments/{oof,submissions,logs}/<exp_id>.*`)이다.
   공유 코드(`src/train_common.py`, `src/features.py`) 변경은 동결이 아니라
   `scripts/check_fold_inputs.py` 입력 동등성 검증으로 보호한다.
2. **템플릿의 기존 설계를 깨는 변경 금지.** 어댑터 패턴, OOF 계약,
   단일 로그 스트림, fold 다양성 축(5/7/10-fold), `max_folds` 스크리닝 루프,
   gen_kernel 생성기 규율은 그대로 유지한다.
3. **충돌 시 이 문서의 해소 방향이 우선한다.** ml-exp-workflow/ 의 파일과
   템플릿이 충돌하면 아래 §1 의 해소안을 따른다. ml-exp-workflow/ 원본을
   무비판적으로 복사하지 않는다.
4. 각 Phase 완료 시 검증(verify)을 실행하고, 통과 후 응집된 1커밋으로 남긴다.

## 1. 충돌 해소 (확정 사항 — 재논의 불필요)

| # | 충돌 | 해소 |
|---|---|---|
| ① | 단일 `splits.parquet` ↔ fold 다양성 축(5/7/10) | **레짐별 직렬화**: `data/splits/{strategy}_{n}fold_seed{seed}.parquet`. `cv.get_folds` 를 로더로 전환(파일 있으면 로드, 없으면 생성·저장). 기존 시그니처·호출부 불변 |
| ② | 실험별 코드 디렉토리 동결 ↔ 어댑터 패턴 | 코드 동결 폐기. `frozen.txt` 에는 **exp_id**(frozen 스택 멤버)를 등록하고 훅이 해당 산출물 파일의 수정·덮어쓰기를 차단. 공유 코드 변경은 code-reviewer 가 `check_fold_inputs` before/after 해시 일치 증거를 요구 |
| ③ | runs/ append-only ↔ 동일 exp_id 재실행 관행 | 전면 append-only 폐기. **frozen 등록된 exp_id 만** 덮어쓰기 차단. 풀 등록 전 실험은 기존처럼 재실행·덮어쓰기 허용 |
| ④ | expectation 게이트 ↔ `max_folds` 스크리닝 | **스크리닝(`max_folds` 지정) 면제, 풀 실행만 게이트.** `kill_criterion` cfg 필드는 expectation 의 falsification 으로 흡수하고 deprecated 주석 처리(하위호환 유지, 제거 금지) |

## 2. Phase 1 — 훅 + frozen 아티팩트 가드

### 2.1 디렉토리·파일
- `scripts/hooks/guard_frozen.sh`, `scripts/hooks/guard_bash.sh` 생성
  (`ml-exp-workflow/scripts/` 의 것을 기반으로 아래 적응 적용).
- 루트 `frozen.txt` 생성. 형식 변경: 경로 prefix 가 아니라 **exp_id 1줄 1개**
  (+ 주석). 헤더 주석에 "스택 풀 등록 시 exp_id 추가" 명시.
- `.claude/settings.json` 에 PreToolUse 훅 2개 추가. **기존 Stop 훅 보존.**

### 2.2 guard_frozen.sh 적응 (Edit|MultiEdit|Write|NotebookEdit 매처)
차단 규칙을 템플릿 구조로 교체:
- 대상 경로가 `experiments/oof|submissions|logs/<exp_id>.*` 이고 `<exp_id>` 가
  `frozen.txt` 에 있으면 exit 2. 차단 메시지: "frozen 스택 멤버 산출물 —
  수정 불가. 변경하려면 새 exp_id 로 재학습하고 풀을 갱신하라."
- `data/splits/*.parquet` 은 **존재하는 파일에 한해** 수정 차단(생성은 허용).
- 그 외 경로는 통과. (src/ 코드는 차단하지 않는다 — 원칙 1.)

### 2.3 guard_bash.sh 적응 (Bash 매처)
- (a) **풀 실행 게이트**: 명령이 아래 패턴이고 `max_folds` 인자가 없으면
  `specs/<exp_id>/expectation.yaml` 의 존재 + git HEAD 커밋 + HEAD 와
  작업트리 일치(`git cat-file -e HEAD:` / `git diff --quiet HEAD --`)를 검사.
  - 패턴: `python -m src.train`(또는 `src.train_*`) · `kaggle kernels push` ·
    `lightning run job`. exp_id 는 `exp_[A-Za-z0-9_]+` 토큰으로 추출,
    추출 실패 시 차단(fail-closed).
  - `max_folds=` 가 명령에 있으면 면제(스크리닝). `src.stack` 실행은 면제.
- (b) **frozen 산출물 우회 쓰기 차단**: `>` `>>` `sed -i` `rm` `mv` `cp` `tee`
  등이 frozen exp_id 의 산출물 경로 또는 기존 `data/splits/*.parquet` 를
  대상으로 하면 차단. (휴리스틱임을 주석으로 명시 — 1차 방어는 에이전트 지시문.)

### 2.4 expectation 보관 위치
`specs/<exp_id>/expectation.yaml`. **`experiments/` 가 아닌 이유를 주석으로
남길 것**: experiments/ 내용물은 .gitignore 제외 대상인데 expectation 은
커밋이 게이트 조건이므로 추적 디렉토리에 둬야 한다. `.gitignore` 에
`specs/` 가 추적되도록 확인. `ml-exp-workflow/templates/expectation.yaml` 을
`docs/templates/expectation.yaml` 로 복사하고 `kill_criterion` 흡수 주석 추가.

### verify (Phase 1)
임시 git 환경에서 최소 5케이스: frozen exp_id 산출물 Edit 차단 /
비 frozen exp_id 통과 / expectation 미커밋 풀 실행 차단 / 커밋 후 통과 /
`max_folds=1` 스크리닝 면제. 기존 Stop 훅이 여전히 동작하는지 확인.

## 3. Phase 2 — expectation 게이트의 코드 연결

- `conf/config.yaml`: `kill_criterion` 에 deprecated 주석
  ("→ specs/<exp_id>/expectation.yaml 의 falsification 으로 이동").
  필드 자체는 유지(하위호환).
- `src/train_common.py`: 풀 실행(`partial` 아님)이고
  `specs/<exp_id>/expectation.yaml` 이 존재하면 로그 JSON 에
  `expectation_path` 키를 추가 기록(내용 복사 아님 — 경로 참조만).
  존재하지 않아도 학습은 막지 않는다(차단은 훅의 책임 — 이중 차단 금지,
  Kaggle/Lightning 원격 환경에는 specs/ 가 없을 수 있음).

### verify (Phase 2)
`examples/run_example.py` 무수정 통과(expectation 없는 데모 실행이 깨지지
않아야 함). 풀 실행 로그에 `expectation_path` 키 기록 확인.

## 4. Phase 3 — 에이전트 3종 추가 + 1종 개명

`ml-exp-workflow/.claude/agents/` 의 정의를 기반으로 하되 아래 적응을 적용해
`.claude/agents/` 에 생성한다. 기존 `eda-explorer`·`feature-smith` 는 불변.

### 4.1 code-reviewer (신규)
- 체크리스트는 새로 쓰지 말고 템플릿 기존 규율을 항목화해
  `docs/checklists/code_review.md` 로 작성:
  1. CV 분할: `cv.get_folds` 경유(자체 KFold 인스턴스화 금지) — Phase 4 이후
     "splits 파일 로드 경유"로 문구 갱신
  2. 타깃 인지 인코딩: fold 내 fit (`OOFTargetEncoder` 경유)
  3. 증강 격리: train fold 에만 병합 (`augment` 경로)
  4. **공유 코드 변경 게이트**: diff 가 `src/train_common.py`·`src/features.py`·
     `src/cat_prep.py`·`src/encoders.py`·`src/cv.py` 를 건드리면
     `check_fold_inputs.py` before/after JSON diff 일치 증거 필수. 불일치 또는
     증거 미첨부 = block
  5. 산출 계약: OOF/submission/log 3종 스키마 + git hash
  6. 스모크: 프로드 경로 동일 cfg 플래그(특히 `augment.enabled`) 1-fold —
     기존 feature-smith 규칙과 동일 문구 사용
  7. 어댑터 순수성: 새 모델이 골격을 복제하지 않고 `ModelTrainer` 구현 +
     registry 등록만 하는가
- 에이전트 규칙: 코드 수정 금지, 항목별 근거(파일:라인/실행 출력) 필수,
  하나라도 block 이면 전체 BLOCK. 리포트는 `specs/<exp_id>/review_report.md`.

### 4.2 result-reviewer (신규)
- 판정 4종(confirmed/refuted/inconclusive/invalid) + **측정 검정력 규칙 내장**:
  |Δ| < ~2·SE(fold std 기반)인 비교는 단일 시드로 confirmed/refuted 판정
  금지 → inconclusive 로 강제하고 다중 시드 또는 stack-add 프레임을 명기.
  (CLAUDE.md 검증 전략의 기존 규칙을 판정 로직으로 승격.)
- 입력: `specs/<exp_id>/expectation.yaml`(HEAD 버전) +
  `experiments/logs/<exp_id>.json`. 기록: `docs/wiki/experiments/judgments/<exp_id>.md`.
- 금지: 다음 실험 제안. 판정 후 judgments/ 파일 수가 5의 배수면 main 에
  premise-auditor 트리거를 알린다.

### 4.3 premise-auditor (신규)
- 기존 "천장 게이트"(CLAUDE.md 실험 우선순위)의 기계화임을 정의 문서에 명시.
- blind 입력: `scripts/summarize.py` 출력 + OOF 상관 행렬 + LB 분포 +
  CV-LB 갭 궤적**만**. 읽기 금지: `docs/wiki/decisions.md` 의 이유 섹션,
  specs/*/spec 류 서사, main 대화 내역.
- 출력: 공격 가설 3개(수치 근거 + 최저비용 반증 실험), `docs/wiki/audits/`.
- kill/continue 판정 금지 — 기존 천장 게이트 규칙과 동일하게 **결정 주체는
  사용자**. 트리거: judgments/ 5건마다.

### 4.4 kaggle-runner → exp-runner (개명)
- 파일명·name 만 변경, Kaggle 절차·교훈 내용은 전부 보존. 상단에 인프라
  디스패치 1문단 추가: Kaggle(이 문서 본문) / Lightning·Colab(각 런북 링크).
- 실행 전제 추가: 풀 실행은 expectation 커밋 확인(훅이 차단하지만 발사 전
  자체 확인), code-reviewer PASS 확인.
- CLAUDE.md·README 양쪽의 `kaggle-runner` 언급을 모두 `exp-runner` 로 갱신.

### verify (Phase 3)
에이전트 4파일 frontmatter 유효(name/description/tools), 체크리스트 항목이
CLAUDE.md 기존 규율과 모순 없는지 대조 1회.

## 5. Phase 4 — splits 레짐별 직렬화 (가장 침습적 — 마지막)

- `src/cv.py` 의 `get_folds` 내부에 로드-우선 로직 추가:
  1. `data/splits/{strategy}_{n_folds}fold_seed{seed}.parquet` 존재 → 로드해
     fold 리스트로 복원(스키마: `row_idx`, `fold` 2열)
  2. 부재 → 기존 로직으로 생성 후 동일 경로에 저장하고 반환
  - 시그니처·반환 타입·호출부(`train_common`, `stack`, `check_fold_inputs`)
    불변. GroupKFold 는 groups 가 데이터 의존이므로 파일명에 그룹 해시 8자를
    추가해 충돌 방지.
- 직렬화 후 일관성 자가검증: 로드한 fold 의 (n_folds, 전체 행 커버, 중복 없음)
  assert. 행 수가 데이터와 불일치하면 명확한 에러로 중단(자동 재생성 금지 —
  데이터가 바뀌었다는 신호이므로 사람이 판단).
- 데이터 부재 환경(원격 커널)에서 splits 파일이 없으면 기존처럼 생성 — 단
  로컬과 원격의 sklearn 버전 차이 위험을 docstring 에 경고로 명시.
- `examples/run_example.py` 가 `examples/_work/` 격리를 유지하도록
  splits 저장 경로도 `config.DATA_DIR` 기준으로(절대경로 하드코딩 금지).

### verify (Phase 4)
`uv run python examples/run_example.py` 2회 연속 실행: 1회차 생성, 2회차
로드 경로를 타는지 + OOF 점수 동일. `check_fold_inputs.py` before/after 가
Phase 4 전후로 해시 동일(분할 자체가 안 바뀌었음을 증명).

## 6. 문서 갱신 (Phase 1~4 와 함께 커밋)

- `CLAUDE.md`: ① 서브에이전트 절에 3종 추가 + exp-runner 개명 반영,
  ② "검증 전략"에 splits 직렬화 1줄, ③ "실험 우선순위" 절에 premise-auditor
  케이던스(judgments 5건마다) 1줄, ④ 프로세스 규율에 expectation 게이트
  (스크리닝 면제 명시) 1줄. **비대화 금지 원칙 준수 — 각 1~2줄 + 포인터만.**
- `ARCHITECTURE.md`: Mermaid 에 expectation 게이트와 frozen 가드 훅 노드 추가.
- `docs/harness_evolution.md`: v0.2.0 행 추가 — source case 는 본 병합,
  problem observed 는 "설계-리뷰 동일 주체(self-review)·전제 감사의 비기계화·
  분할의 환경 의존", harness change 는 본 문서 요약.
- `README.ko.md`/`README.md`: 구조 트리에 specs/·docs/checklists/ 반영,
  에이전트 목록 갱신. 한국어판 먼저, 영어판은 번역 동기화.
- `ml-exp-workflow/` 폴더는 병합 완료 후 삭제 (필요 내용은 전부 이식됨).

## 7. 금지 사항 (위반 = 병합 실패)

- 어댑터 패턴 훼손: 실험별 학습 코드 복제 구조 도입 금지.
- 스크리닝 루프에 expectation 요구 금지.
- `experiments/` 산출물의 전면 append-only 강제 금지 (frozen exp_id 한정).
- 기존 Stop 훅·`check_fold_inputs.py`·gen_kernel 생성기 규율 제거 금지.
- CLAUDE.md 비대화 금지 — 상세는 docs/ 로, 본문은 포인터.
- `kill_criterion` 필드 즉시 제거 금지 (deprecate 만).

## 8. 최종 수용 기준

1. Phase 1~4 verify 전부 통과 + `examples/run_example.py` 정상.
2. 훅 시나리오 테스트(§2 verify 5케이스) 기록이 커밋 메시지 또는
   `docs/wiki/` 에 남아 있음.
3. `harness_evolution.md` v0.2.0 행 존재, README 구조 트리 일치.
4. `git grep kaggle-runner` 결과 0건 (런북 본문 역사 서술 제외).
5. 새 에이전트가 기존 에이전트(eda-explorer·feature-smith)의 파일을
   변경하지 않았음.
