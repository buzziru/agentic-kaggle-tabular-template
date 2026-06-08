# EDA 결론 (수치 요약)

> 주제별 노트북(`notebooks/eda_<NN>_<주제>.ipynb`)의 **결론만 수치로** 누적한다 (토큰 절약).
> 플롯·전체 출력은 노트북에, 여기는 결론 표/불릿만.

## 체크리스트 (EDA 전 점검)
- [ ] 타깃 분포 / 클래스 불균형
- [ ] 컬럼별 결측·고유값·dtype (`utils.resumetable`)
- [ ] 수치형 분포 (타깃별)
- [ ] 범주형 카디널리티 + 타깃률 (고카디널리티 → OOF TE 후보)
- [ ] train/test 드리프트 (adversarial validation, seed 고정)
- [ ] ⚠️ 파생/시퀀스 컬럼의 미래 정보 누수
- [ ] CV 전략이 train/test 분할과 일치하는지 (`docs/setup_questions.md`)

## 발견 (수치)
- {{핵심 발견 1 (수치)}}
- {{핵심 발견 2 (수치)}}

## 누수/리스크
- {{있으면}}

## 피처 후보
- {{후보 → docs/feature_engineering.md 로}}
