# 코드 리뷰 체크리스트 (고정)

> code-reviewer 에이전트가 이 표를 항목별로 대조한다. 판정은 항목별 **pass/block**
> 만 존재하며, **하나라도 block 이면 전체 BLOCK**(부분 통과 없음). 항목 추가·삭제는
> `docs/wiki/decisions.md`(ADR-lite)로만 한다 — 리뷰 중 임의 변경 금지.
>
> 이 체크리스트는 새 규율이 아니라 **CLAUDE.md 기존 규율의 항목화**다. 각 항목은
> 템플릿의 어댑터 아키텍처·OOF 계약·누수 방지·frozen 불변 원칙을 게이트로 옮긴 것이다.

| # | 항목 | block 기준 |
|---|------|-----------|
| 1 | CV 분할 | 분할을 `cv.get_folds`(레짐별 splits 파일 로드-우선) 경유로 얻지 않고 KFold 류를 코드에서 직접 인스턴스화하면 block. (모든 모델은 동일 fold/seed 의 직렬화된 분할을 공유해야 한다 — `data/splits/{strategy}_{n}fold_seed{seed}.parquet`.) |
| 2 | 타깃 인지 인코딩 fit 위치 | target encoding·타깃 통계·supervised 스케일링이 fold **외부**(전체 train)에서 fit 되면 block. fold 내부 fit(`OOFTargetEncoder` 경유)만 허용. 타깃 비인지 변환(frequency/count 등)은 전체 fit 허용 |
| 3 | 증강 격리 | 외부·원본 증강 데이터가 검증 fold·OOF·test 산출에 섞이면 block. **train fold 에만 병합**(`augment` 경로, train-fold-only)인 경우만 pass |
| 4 | 공유 코드 변경 게이트 | diff 가 `src/train_common.py`·`src/features.py`·`src/cat_prep.py`·`src/encoders.py`·`src/cv.py` 중 하나라도 건드리면, `scripts/check_fold_inputs.py` 의 **before/after JSON diff 일치 증거**가 필수. 불일치 또는 증거 미첨부 = block. (frozen 스택 멤버의 OOF 불변 보호 — 공유 코드는 동결이 아니라 입력 동등성으로 지킨다.) |
| 5 | 산출 계약 | OOF/submission/log 3종을 스키마대로 쓰지 않으면 block: `experiments/oof/<exp_id>.csv`=`[id, oof]`, `experiments/submissions/<exp_id>.csv`=`[id, <target>]`, `experiments/logs/<exp_id>.json`. 로그에 git hash 누락도 block |
| 6 | 스모크 | **프로드 경로 동일 cfg 플래그**(특히 `augment.enabled`)로 1-fold 스모크가 실패하거나, OOF 인덱스가 train 과 정합하지 않으면 block. (1-fold 라도 실제 풀 실행과 같은 경로를 태워야 미검증 경로가 통과하지 않는다.) |
| 7 | 어댑터 순수성 | 새 모델이 `train_common` 골격을 복제하면 block. `ModelTrainer`(`src/registry.py`) 인터페이스(prepare/fit/predict/get_metadata/save_model) 구현 + `_REGISTRY` 등록 + 모델 차이를 conf 훅으로만 표현한 경우 pass. 별도 학습 경로·모델 분기 = divergence 근원 |
| 8 | 그룹·시계열 누수 (해당 시) | 그룹/시계열 구조 데이터에서 미래 정보 또는 fold 간 그룹 중복이 피처·인코딩에 유입되면 block. `CV_STRATEGY=GroupKFold` 인데 내부 OOF-TE 가 같은 그룹으로 분할되지 않으면 block. 해당 없으면 근거(데이터 구조)와 함께 n/a |

비고: 항목 4·7 이 이 체크리스트의 존재 이유다. "공유 코드 한 줄을 바꿔 다른 frozen
멤버의 OOF 재현이 깨지는" 변경과 "모델별 학습 경로 복제로 노브가 divergence 하는"
구조를 **실행 전에** 잡는 게이트다.
