"""Kaggle 커널 노트북 생성기 — 단일 템플릿 + 파라미터로 결정론적 생성.

재발 버그(2중 사본 drift·복사-템플릿 상속 override) 근절용. 노트북을 손으로
복사·편집하지 않는다. 대신 `KERNELS` 레지스트리의 파라미터 dict + 본 템플릿에서
`kaggle/<name>/<code_file>` 와 `kaggle/<name>/kernel-metadata.json` 을 매번 fresh
생성한다. 커널을 바꾸려면 레지스트리 파라미터만 고치고 재생성한다.

⚠️ `use_wandb=False` 는 **파라미터가 아니라 cfg 템플릿에 하드코딩**돼 있다 —
헤드리스 `kernels push` 는 WANDB_API_KEY secret 미유지라 online 불가하므로
(SSOT docs/wiki/kaggle_jobs.md 교훈·notebook_conventions.md 룰9), 구조적으로
True 가 될 수 없게 막는다. GPU 모델로 wandb 가 필요하면 Colab/Lightning 경로를 쓴다.

▶ 사용 전 채울 것: 아래 OWNER / SRC_DATASET / COMPETITION (+ 필요 시 EXTERNAL_DATASETS),
  그리고 KERNELS 레지스트리(예시 2개를 본인 model/features 로 교체·추가).

사용:
    python kaggle/gen_kernel.py <name>     # 한 커널 생성
    python kaggle/gen_kernel.py --all      # 레지스트리 전체 생성
    python kaggle/gen_kernel.py --list     # 등록 커널 목록
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent

# ── 프로젝트별 상수 (채울 것) ───────────────────────────────────────────
OWNER = "{{KAGGLE_USER}}"                       # 본인 Kaggle 사용자명
SRC_DATASET = f"{OWNER}/{{{{SRC_DATASET}}}}"    # 코드 번들 Dataset 슬러그 (예: f"{OWNER}/proj-src")
COMPETITION = "{{COMPETITION_SLUG}}"            # 대회 슬러그
# (선택) 외부 보조 데이터 공개 Dataset. 없으면 빈 리스트로 둔다.
EXTERNAL_DATASETS: list[str] = []              # 예: ["someuser/some-aux-dataset"]
DATASET_SOURCES = [SRC_DATASET, *EXTERNAL_DATASETS]


# ── 커널 레지스트리 (SSOT) ──────────────────────────────────────────────
# 노트북마다 다른 건 아래 소수 파라미터뿐. 이 dict 가 단일 진실원.
# 아래 2개는 패턴 예시 — 본인 model/features 로 교체·확장한다.
# ⚠️ 트레이너는 src.registry 가 model yaml 의 name 으로 선택한다(여기 trainer 필드 불필요).
KERNELS: dict[str, dict] = {
    "example_lgbm_cpu": dict(
        slug="example-lgbm-cpu",
        title="example lgbm cpu",
        display="LGBM 예시 — CPU OOF",
        exp_id="exp_example_lgbm",
        features="base",          # conf/features/base.yaml
        model="lgbm",             # conf/model/lgbm.yaml — registry 가 model.name 으로 트레이너 선택
        notes="예시 커널: LGBM baseline OOF",
        gpu=False,
        deps=["lightgbm", "hydra-core", "omegaconf", "python-dotenv"],
        deps_comment="CPU — torch 없음",
    ),
    "example_nn_gpu": dict(
        slug="example-nn-gpu",
        title="example nn gpu",
        display="신경망 예시 — GPU(T4) OOF",
        exp_id="exp_example_nn",
        features="base",
        model="nn",               # src/train_nn.py 작성 후 src/registry.py 에 "nn" 등록 필요
        notes="예시 커널: torch 모델 GPU OOF",
        gpu=True,
        needs_torch=True,         # cell2 가 P100 cu121 torch 처리
        deps=["pytabkit", "hydra-core", "python-dotenv"],
        deps_comment="GPU torch 모델",
    ),
}

# 기본값 — 레지스트리에서 생략 가능
DEFAULTS = dict(
    external_data=False,  # True=외부 Dataset 마운트해 train 증강(검증/test 미포함). cell1/3 가 해당 경로 검증.
    external_rows=None,   # int 면 external csv 행수를 assert(무결성 가드). None=skip.
    max_folds=None,       # None=풀 n_folds, int=fold0 스크리닝
    num_boost_round_cap=5000,
    needs_torch=False,    # True=pytabkit 등 torch 의존 → cell2 가 P100 cu121 torch 처리
    model_overrides={},   # {param: value} → model yaml 로드 후 mc.params 에 주입(스윕용, 예: lr)
    n_folds=5,            # split 다양성: 7/10-fold OOF 는 5-fold 와 직교(d_eff 축)
)


# ── 셀 템플릿 ───────────────────────────────────────────────────────────
_GPU_CHECK = """
# GPU 환경 확인
import subprocess
gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                          capture_output=True, text=True)
if gpu_info.returncode == 0:
    print('GPU:', gpu_info.stdout.strip())
else:
    print('WARNING: GPU 없음 — GPU 학습 실패 가능')
"""


# pytabkit 등 torch 의존 GPU 커널용 — torch import 前 실행. Kaggle 기본 GPU=P100(sm_60)이고
# Kaggle 기본 torch 는 sm_70+ 만 빌드 → P100 이면 cu121 torch trio 재설치 후 CUDA 실연산 검증.
_TORCH_P100 = """
_gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                      capture_output=True, text=True).stdout.strip()
print('GPU:', _gpu)
if 'P100' in _gpu:
    print('P100(sm_60) -> cu121 torch trio 재설치 (Kaggle 기본 torch 는 sm_70+ 만)')
    pip('torch==2.5.1', 'torchvision==0.20.1', 'torchaudio==2.5.1',
        '--index-url', 'https://download.pytorch.org/whl/cu121')
else:
    print('sm_70+ -> Kaggle 기본 torch 유지')
import torch
assert torch.cuda.is_available(), 'CUDA 불가'
_x = torch.randn(64, 64, device='cuda'); float((_x @ _x).sum())
print('torch', torch.__version__, '| CUDA matmul OK |', torch.cuda.get_device_name(0))
"""


def _cell(source: str) -> dict:
    """nbformat code 셀 dict 를 만든다."""
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.strip("\n") + "\n",
    }


def _md_cell(source: str) -> dict:
    """nbformat markdown 셀 dict 를 만든다."""
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n")}


def _build_cells(p: dict) -> list[dict]:
    """파라미터 dict 로 노트북 본문(마크다운 + 5±α 코드 셀)을 렌더한다.

    Args:
        p: 병합된 파라미터(레지스트리 + DEFAULTS).

    Returns:
        nbformat 셀 dict 리스트.
    """
    kind = "GPU" if p["gpu"] else "CPU"
    fold_desc = f"풀 {p['n_folds']}-fold" if p["max_folds"] is None else f"fold0({p['max_folds']})"

    md = f"# {p['display']}\n{p['exp_id']}: features={p['features']} model={p['model']}, {fold_desc}, {kind}"

    # 1) 입력 자동탐색 + fast-fail 가드 (비싼 설치 前)
    cell1 = f"""# 1) 입력 자동탐색 + fast-fail 가드 (비싼 설치 前)
import sys, os, glob
from pathlib import Path

print('/kaggle/input:', os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'NONE')

c = glob.glob('/kaggle/input/**/src/config.py', recursive=True)
assert c, 'src/config.py 못 찾음'
SRC_ROOT = str(Path(c[0]).parents[1])
print('SRC_ROOT:', SRC_ROOT)

cc = glob.glob('/kaggle/input/**/{COMPETITION}', recursive=True)
assert cc, '대회 폴더 못 찾음'
COMP = Path(cc[0])
print('COMP:', COMP)
"""
    if p["external_data"]:
        cell1 += """
ac = glob.glob('/kaggle/input/**/*.csv', recursive=True)   # ⚠️ 외부 보조 데이터 패턴으로 좁혀라
ac = [f for f in ac if 'train.csv' not in f and 'test.csv' not in f and 'sample_submission' not in f]
assert ac, '외부 보조 csv 못 찾음'
AUX = Path(ac[0])
print('AUX:', AUX)
"""
    if p["gpu"]:
        cell1 += _GPU_CHECK
    cell1 += "\nprint('--- fast-fail 가드 통과 ---')"

    # 2) deps 설치
    deps_args = ", ".join(repr(d) for d in p["deps"])
    torch_block = _TORCH_P100 if p["needs_torch"] else ""
    cell2 = f"""# 2) 프로젝트 deps 설치 ({p['deps_comment']})
import subprocess
def pip(*a):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *a], check=True)
{torch_block}
pip({deps_args})
print('deps 설치 완료')"""

    # 3) import + 경로 override
    cell3 = f"""# 3) import + 경로 override
import pandas as pd

sys.path.insert(0, SRC_ROOT)
from src import config
from src.registry import get_trainer
from src.train_common import run_oof_cv
print('import OK:', config.__file__)

config.TRAIN_PATH = COMP / 'train.csv'
config.TEST_PATH = COMP / 'test.csv'
config.SAMPLE_SUBMISSION_PATH = COMP / 'sample_submission.csv'

out = Path('/kaggle/working')
# ⚠️ EXPERIMENTS_DIR 파생 디렉터리를 전부 writable 로 — 하나라도 빠지면 학습 완료 후 저장에서 OSError 재발.
config.EXPERIMENTS_DIR = out
config.OOF_DIR = out / 'oof'
config.SUBMISSION_DIR = out / 'submissions'
config.LOG_DIR = out / 'logs'
config.TEST_PRED_DIR = out / 'test_pred'
config.MODEL_DIR = out / 'models'
for d in [config.OOF_DIR, config.SUBMISSION_DIR, config.LOG_DIR, config.TEST_PRED_DIR, config.MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

import shutil
config.DATA_DIR = out / 'data'
(config.DATA_DIR / 'splits').mkdir(parents=True, exist_ok=True)
for _sp in glob.glob(str(Path(SRC_ROOT) / 'data' / 'splits' / '*.parquet')):
    shutil.copy(_sp, config.DATA_DIR / 'splits' / Path(_sp).name)
print('splits:', [p.name for p in (config.DATA_DIR / 'splits').glob('*.parquet')] or 'NONE(원격 생성됨)')
"""
    # write-probe(fast-fail): 출력 디렉터리가 read-only 면 30분 학습 전에 5초 만에 잡는다.
    # 비-f-string 으로 추가(내부 중괄호는 생성 노트북에서 평가).
    cell3 += '''
for _d in [config.OOF_DIR, config.SUBMISSION_DIR, config.LOG_DIR,
           config.TEST_PRED_DIR, config.MODEL_DIR, config.DATA_DIR / 'splits']:
    _t = _d / '.wtest'
    try:
        _t.write_text('ok')
        _t.unlink()
    except OSError as _e:
        raise SystemExit(f'[FATAL] 쓰기 불가(read-only): {_d} -> {_e}. cell3 경로 override 누락')
print('write-probe OK — 모든 출력 디렉터리 writable')
'''
    if p["external_data"]:
        cell3 += "\nconfig.SOURCE_AUG_PATH = AUX   # ⚠️ 프로젝트 config 의 외부 데이터 경로 상수명에 맞춰라"
        if p["external_rows"] is not None:
            cell3 += (
                "\n_a = pd.read_csv(config.SOURCE_AUG_PATH)"
                "\nprint('AUX shape:', _a.shape)"
                f"\nassert len(_a) == {p['external_rows']}, f'외부 행수 불일치: {{len(_a)}}'"
            )
    cell3 += (
        "\nassert config.TRAIN_PATH.exists(), f'train.csv 없음: {config.TRAIN_PATH}'"
        "\nprint('경로 override 완료')"
    )

    # 4) cfg + run
    aug = ("{'enabled': True, 'weight': 1.0}" if p["external_data"]
           else "{'enabled': False, 'weight': 1.0}")
    mf = "None" if p["max_folds"] is None else str(p["max_folds"])
    # model_overrides: model yaml 로드 후 mc.params 주입(스윕). 한 줄씩 명시 → diff·재현 명확.
    ovr_lines = "".join(f"\nmc.params['{k}'] = {v!r}" for k, v in (p.get("model_overrides") or {}).items())
    # ⚠️ use_wandb 는 여기 하드코딩 — 파라미터 아님(헤드리스 push online 불가).
    cell4 = f"""# 4) cfg + run
from omegaconf import OmegaConf
import time, json, os

CONF = Path(SRC_ROOT) / 'conf'
EXP_ID = '{p['exp_id']}'
NUM_BOOST_ROUND_CAP = {p['num_boost_round_cap']}

mc = OmegaConf.load(CONF / 'model' / '{p['model']}.yaml'){ovr_lines}
cfg = OmegaConf.create({{
    'exp_id': EXP_ID,
    'notes': '{p['notes']}',
    'use_wandb': False,
    'seed': 42,
    'max_folds': {mf},
    'n_folds': {p['n_folds']},
    'kill_criterion': '',
    'model': mc,
    'features': OmegaConf.load(CONF / 'features' / '{p['features']}.yaml'),
    'augment': {aug},
}})

t0 = time.time()
result = run_oof_cv(cfg, get_trainer(cfg.model.name)(cfg))   # registry 가 model.name 으로 트레이너 선택
dt = time.time() - t0

log_file = config.LOG_DIR / f'{{EXP_ID}}.json'
best_iters = None
if log_file.exists():
    log = json.load(open(log_file))
    best_iters = log.get('best_iters', log.get('fold_best_iters'))
    if best_iters:
        capped = [i for i in best_iters if i >= NUM_BOOST_ROUND_CAP - 1]
        if capped:
            print(f'WARNING [미수렴] best_iter cap({{NUM_BOOST_ROUND_CAP}}) 접촉: fold={{capped}} -> 미완 학습')
        else:
            print(f'[수렴 OK] best_iters={{best_iters}} (모두 < cap)')

print(f'cv_mean={{result.get("cv_mean"):.6f}} '
      f'folds={{[f"{{s:.6f}}" for s in result.get("fold_scores", [])]}} '
      f'best_iters={{best_iters}} {{dt:.0f}}s')"""

    # 5) 산출물 확인
    cell5 = """# 5) 산출물 확인
for subdir in ['oof', 'submissions', 'logs', 'test_pred', 'models']:
    p = Path('/kaggle/working') / subdir
    files = list(p.glob('*')) if p.exists() else []
    print(f'{subdir}/: {[f.name for f in files]}')"""

    return [_md_cell(md), _cell(cell1), _cell(cell2), _cell(cell3), _cell(cell4), _cell(cell5)]


def _metadata(name: str, p: dict) -> dict:
    """kernel-metadata.json dict 를 만든다."""
    return {
        "id": f"{OWNER}/{p['slug']}",
        "title": p["title"],
        "code_file": f"{name}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(p["gpu"]),
        "enable_internet": True,
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
    }


def generate(name: str) -> Path:
    """레지스트리의 커널 한 개를 kaggle/<name>/ 에 fresh 생성한다.

    Args:
        name: KERNELS 키.

    Returns:
        생성된 커널 디렉터리 경로.
    """
    if name not in KERNELS:
        raise KeyError(f"미등록 커널: {name}. 등록: {list(KERNELS)}")
    p = {**DEFAULTS, **KERNELS[name]}

    kdir = KAGGLE_DIR / name
    kdir.mkdir(exist_ok=True)
    nb = {
        "cells": _build_cells(p),
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # ensure_ascii=False 로 한글 유지 → Windows 기본 cp949 회피 위해 utf-8 명시.
    (kdir / f"{name}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    (kdir / "kernel-metadata.json").write_text(
        json.dumps(_metadata(name, p), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return kdir


def main(argv: list[str]) -> None:
    """CLI 진입점."""
    if not argv or argv[0] == "--list":
        for n, p in KERNELS.items():
            print(f"  {n:18s} -> {OWNER}/{p['slug']}  (gpu={p['gpu']}, model={p['model']})")
        return
    names = list(KERNELS) if argv[0] == "--all" else argv
    for n in names:
        kdir = generate(n)
        print(f"[gen] {n} -> {kdir}/  (use_wandb=False 고정)")


if __name__ == "__main__":
    main(sys.argv[1:])
