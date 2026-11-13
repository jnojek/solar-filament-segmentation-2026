# Reproducing the combined refit and verifier

Obtain the competition archive with official Kaggle tooling and place it at `data/MAGFiLO_1.0_Kaggle_2026/`. Install `requirements.txt` in Python 3.12 with the matching CUDA 12.8 PyTorch build. From the checkout root, execute every cell in `reproduce.ipynb`. For the direct Windows equivalent:

```powershell
$env:PYTHONPATH = 'src'
$data = 'data/MAGFiLO_1.0_Kaggle_2026'
python -m filament.small_proposal --config artifacts/selected.json --checkpoint artifacts/model.pt --verifier artifacts/small-proposal-selected.json --images "$data/test/test_images" --inventory artifacts/test-files.sha256.json --output output/refit-verifier/submission.csv --coverage output/refit-verifier/coverage.json --receipt output/refit-verifier/inference.json --device cuda
python qa/verify_entry.py export --csv output/refit-verifier/submission.csv --coverage output/refit-verifier/coverage.json --images "$data/test/test_images" --inventory artifacts/test-files.sha256.json --output output/refit-verifier/qa.json
```

Compare the CSV and coverage SHA-256 with `artifacts/release.json`. The frozen checkpoint and verifier are included. The [training lineage](refit-verifier-transfer.md) describes independent retraining of the 419-image network and refitting the verifier from the 336-image model's out-of-training maps.

The report in `report/`, if present, documents the original baseline experiment. The organizer-facing combined report is a separate four-page PDF, and its exact submission identity is recorded in the handoff materials.
