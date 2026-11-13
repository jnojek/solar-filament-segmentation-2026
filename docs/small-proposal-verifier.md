# Small filament proposal verification

The baseline discards connected components smaller than 700 native pixels. The verifier considers components of 150–699 pixels from the same frozen probability plane and threshold. It keeps the original instances in their original order and appends only verified small components. The eight input features measure component area, median and upper model probability, local H-alpha contrast, missing contrast, spine aspect, projected radius and missing disk geometry. A logistic model with fixed L2 weight 0.02 accepts a proposal at probability 0.50 or above.

The verifier was fitted on the 115 frozen development images. Its target is the fraction of annotator views in which the proposal adds a new match above 0.50 IoU. All verifier fit and evaluation images were excluded from the original network's training. The 80 confirmation images and 101 Udaipur images were already exposed in prior campaign comparisons, so they are reported as follow-up evaluations. Training only used the competition-supplied labels. Inference needs only the supplied H-alpha JPEGs and frozen models.

## Refit the verifier from the separate 336-image model

Use the competition training data and pinned environment. Rebuild the original 336-image network using the original split. The shipped `artifacts/model.pt` is the 419-image inference checkpoint. It was trained on 83 of the 115 verifier fit images, so its development maps would change the verifier's fitting protocol. The commands below use PowerShell and new output paths.

```powershell
$env:PYTHONPATH = 'src'
$data = 'data/MAGFiLO_1.0_Kaggle_2026'
python -m filament train --data $data --split artifacts/split.json --config configs/baseline.json --output runs/rebuild-336 --device cuda
if ($LASTEXITCODE) { throw 'Original 336-image training failed' }
$checkpoint336 = 'runs/rebuild-336/model.pt'
# This score command uses the 0.65 / 700 thresholds in artifacts/selected.json.
foreach ($fold in @('dev', 'confirm', 'site_holdout')) {
    python -m filament score --data $data --split artifacts/split.json --config artifacts/selected.json --checkpoint $checkpoint336 --fold $fold --output "runs/verifier-$fold-score.json" --cache "runs/verifier-$fold-maps" --device cuda
    if ($LASTEXITCODE) { throw "Probability generation failed for $fold" }
}
python analysis/small_proposal_verifier.py --data $data --split artifacts/split.json --dev-cache runs/verifier-dev-maps --confirm-cache runs/verifier-confirm-maps --site-cache runs/verifier-site_holdout-maps --output runs/verifier-refit
```

The command first saves the fitted coefficients, then scores the fixed decision rule on confirmation and Udaipur. The included 419-image checkpoint and frozen verifier reproduce the selected inference artifact. Native probability maps are local intermediates.

## Reproduce the export

The finalist notebook checks the frozen code and model hashes before inference, then checks the complete CSV, coverage, and test-image manifest. The equivalent command is:

```powershell
python -m filament.small_proposal --config artifacts/selected.json --checkpoint artifacts/model.pt --verifier artifacts/small-proposal-selected.json --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images --inventory artifacts/test-files.sha256.json --output runs/verifier-export/submission.csv --coverage runs/verifier-export/coverage.json --receipt runs/verifier-export/inference.json --device cuda
```
