# Reproducing the combined refit and verifier

Obtain the competition archive with official Kaggle tooling and place it at `data/MAGFiLO_1.0_Kaggle_2026/`. Use Python 3.12.11 and the checked-in Windows lock for the measured CUDA 12.8 reproduction:

```powershell
uv venv --python 3.12.11 .venv
uv pip sync --python .venv\Scripts\python.exe --extra-index-url https://download.pytorch.org/whl/cu128 requirements-windows.lock
```

Run all cells in `reproduce.ipynb` from the checkout root. The direct Windows commands are:

```powershell
$env:PYTHONPATH = 'src'
$data = 'data/MAGFiLO_1.0_Kaggle_2026'
.\.venv\Scripts\python.exe -m filament.small_proposal --config artifacts/selected.json --checkpoint artifacts/model.pt --verifier artifacts/small-proposal-selected.json --images "$data/test/test_images" --inventory artifacts/test-files.sha256.json --output output/refit-verifier/submission.csv --coverage output/refit-verifier/coverage.json --receipt output/refit-verifier/inference.json --device cuda
.\.venv\Scripts\python.exe qa/verify_entry.py export --csv output/refit-verifier/submission.csv --coverage output/refit-verifier/coverage.json --images "$data/test/test_images" --inventory artifacts/test-files.sha256.json --output output/refit-verifier/qa.json
```

Compare the CSV and coverage SHA-256 with `artifacts/release.json`. The exact byte match was verified on the RTX 4090 in the pinned environment. CPU and other CUDA or library builds may change decisions near a threshold. The frozen 419-image checkpoint and verifier are included. To retrain either component, use the [training lineage](refit-verifier-transfer.md). When refitting the verifier, point the map-generation commands at the independently rebuilt 336-image checkpoint, whose development predictions were out of its training set.

The report in `report/`, if present, documents the original baseline experiment. The organizer-facing combined report is a separate four-page PDF, and its exact submission identity is recorded in the handoff materials.
