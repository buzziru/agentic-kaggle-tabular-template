# 가드 티어 체계

훅 가드를 3티어로 분류한다. 목적은 **오차단을 0으로 만드는 것이 아니라**(그러면 게이트도 0이 된다), **사용자가 훅을 통째로 끌 이유를 없애는 것**이다. 비가역·치명은 우회 불가로 막고, 중요·복구가능은 "이번 건만 사유 남기고 우회"하는 세 번째 선택지를 준다.

## 티어 정의

| 티어 | 이름 | 동작 | escape hatch |
|---|---|---|---|
| **T0** | 하드 | `exit 2` 차단. 우회 불가. 시도를 로그에 남김 | 없음 (불변) |
| **T1** | 기록 우회 | override 파일이 **존재 + git HEAD 커밋**되면 통과 + `guard_overrides.jsonl` append, 아니면 `exit 2` 차단 | `specs/<exp_id>/override_<guard>.md` (사유 기재 후 커밋) |
| **T2** | 경고 | 비차단. stderr 로 사유만 출력하고 통과 | 불필요 (애초에 통과) |

분류 원칙: **비가역·치명 = T0 · 중요·복구가능 = T1 · 위생 = T2.** 시크릿은 어떤 경우에도 T0에서 내리지 않는다.

## 현 가드 분류

| 가드 | 티어 | 검사 대상 | 구현 위치 |
|---|---|---|---|
| 시크릿 커밋 | **T0** | `git add/commit` 의 `.env`·키·자격증명·`kaggle.json`·`id_rsa`, `add -f` 일괄 | `guard_bash.py` (c) |
| frozen 산출물 수정 | **T1** | `experiments/{oof,submissions,logs}/<exp_id>.*` (exp_id ∈ `frozen.txt`) | `guard_frozen.py` 규칙1 |
| splits 편집 불변 | **T1** | 기존 `data/splits/*.parquet` 편집 | `guard_frozen.py` 규칙2 |
| frozen/splits 우회 쓰기 | **T1** | shell 리다이렉트·`sed -i`·`rm/mv/cp/tee` 가 frozen 산출물/기존 splits 대상 (휴리스틱) | `guard_bash.py` (b) |
| expectation 게이트 | **T1** | 풀 실행 시 `specs/<exp_id>/expectation.yaml` 존재+HEAD커밋+작업트리일치 | `guard_bash.py` (a) |
| exp_id 포맷 | **T1** | 풀 실행 exp_id 가 `^exp_\d+_` 불일치 | `guard_bash.py` (a) |
| 노트북 `;` 린트 | **T2** | 편집 셀의 문장구분 `;` (트레일링·setup·문자열 면제) | `lint_notebook.py` |

비훅 보조: `stop_reminder.py`(미커밋 리마인더, Stop, 비차단)는 티어 밖이며 비활성화 감지의 호스트다. `scripts/check_fold_inputs.py`(수동 입력동등성 게이트)는 훅이 아니다.

## escape hatch (T1)

차단 시 가드가 우회 경로를 안내한다: `specs/<exp_id>/override_<guard>.md` 에 사유를 적고 커밋한 뒤 재시도. 가드는 override 파일의 존재 + git HEAD 커밋을 확인하고(expectation 게이트의 커밋 검사 헬퍼 재사용), 성립하면 통과시키며 `docs/wiki/guard_overrides.jsonl` 에 1줄 append 한다(timestamp, guard, exp_id, target_path, reason_path, git_hash). 이 로그는 추적 대상이며 append-only다 — 다음 단계(게이트 붕괴 완화)가 우회 빈도를 여기서 읽는다.

splits 등 exp_id 가 없는 대상은 override 키를 `specs/_global/override_<guard>.md` 로 둔다.

## 순환 방지

`conf/guard/*.txt`(패턴 데이터)·`frozen.txt`·`guard_overrides.jsonl` 은 **frozen 경로에 포함되면 안 되고 gitignore 되면 안 된다** — 그러면 갱신이 막혀 순환에 빠진다. guard_frozen 이 이 파일들을 자기 차단 대상으로 삼지 않는지 명시적으로 확인한다.

오차단 발생 시 대응은 "훅 끄기"가 아니라 "패턴 txt 한 줄 추가"다 — CLAUDE.md "외부 인프라 가드: 1회 발생 시 즉시 코드화" 원칙의 연장.
