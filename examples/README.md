# examples — 미니 스모크 인스턴스

템플릿이 **실제로 동작**하는지 한 명령으로 확인하는 더미 예시다. 별도 데이터·설정 없이
`src/` 파이프라인(로드 → 피처 → CV → OOF → 제출파일 → JSON 로그)을 그대로 돌린다.

```bash
uv sync
uv run python examples/run_example.py
```

## 무엇을 하나

- 이진분류 **더미 데이터**(train 2000 / test 500행, 수치 피처 5개)를 결정적으로 생성한다.
- LightGBM 베이스라인(`src.train_lgbm`)을 **5-fold OOF** 로 학습한다(W&B off).
- 산출물을 `examples/_work/` 에 남긴다(실제 `data/`·`experiments/` 는 건드리지 않음, git 제외).

## 기대 출력

신호가 심어져 있어 **OOF ROC-AUC 가 0.9 안팎**으로 나오면 파이프라인이 정상이다.

```
[데이터] train=(2000, 7) test=(500, 6) → .../examples/_work/data
========================================================
  OOF AUC (mean) = 0.9x ± 0.0x
  산출물:
    OOF        : .../examples/_work/oof/exp_demo.csv
    submission : .../examples/_work/submissions/exp_demo.csv
    log(JSON)  : .../examples/_work/logs/exp_demo.json
========================================================
✅ 파이프라인 정상 (로드→피처→CV→OOF→제출→로그).
```

## 실제 프로젝트와의 차이

이 예시는 경로만 `examples/_work/` 로 격리할 뿐, **코드 경로는 실 프로젝트와 동일**하다.
새 프로젝트를 시작하는 방법은 루트 [README.ko.md](../README.ko.md) 의 "이 템플릿으로 새 프로젝트 시작하기" 참조.
