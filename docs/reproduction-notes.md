# Reproduction notes

Run from the repository root using Python 3.12.11. For the measured Windows CUDA reproduction, install the pinned environment using uv:

```powershell
uv venv --python 3.12.11
uv pip sync --python .venv\Scripts\python.exe --extra-index-url https://download.pytorch.org/whl/cu128 requirements-windows.lock
uv pip check --python .venv\Scripts\python.exe
$env:PATH = "$((Resolve-Path .venv\Scripts).Path);$env:PATH"
$env:PYTHONPATH = (Resolve-Path src).Path
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
```

Create a fresh output directory or use the notebook, which creates one. An independently decoded CSV must match the release hash to reproduce the exact submission. CPU or different CUDA/library versions can produce valid output that differs near a threshold; the exact-byte claim is bounded to the measured environment.

With the official raw archive unpacked to `data/MAGFiLO_1.0_Kaggle_2026`, use the included checkpoint and selected config listed in `artifacts/release.json`. The notebook invokes the same CLI and supplies the full input and artifact identities. To run it without opening a browser:

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute reproduce.ipynb --output reproduced.ipynb --output-dir output
```

The experimental training recipe is independent of that frozen checkpoint:

```powershell
.\.venv\Scripts\python.exe -m filament train --data data/MAGFiLO_1.0_Kaggle_2026 --split artifacts/split.json --config configs/baseline.json --output runs/retrain --device cuda
```

Training starts from random weights. The exact-byte check uses the included trained artifact rather than depending on kernel-level training determinism.

The release does not include competition images, annotations or submission rows. The SHA-256 manifest verifies the same original data can be obtained from the competition.

Run the selected inference path directly:

```powershell
New-Item -ItemType Directory -Force output/cli-reproduction | Out-Null
.\.venv\Scripts\python.exe -m filament infer --config artifacts/selected.json --checkpoint artifacts/model.pt --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images --output output/cli-reproduction/submission.csv --coverage output/cli-reproduction/coverage.json --receipt output/cli-reproduction/inference.json --device cuda
.\.venv\Scripts\python.exe qa/verify_entry.py export --csv output/cli-reproduction/submission.csv --coverage output/cli-reproduction/coverage.json --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images --inventory artifacts/test-files.sha256.json
```

Report source is supplied for transparency. The organizer-facing PDF includes a local qualitative plate from competition validation data. To rebuild that plate, regenerate the matched confirmation overlays and run `scripts/report_examples.py` from the repo root. The repository deliberately omits the image pixels and original annotations.
