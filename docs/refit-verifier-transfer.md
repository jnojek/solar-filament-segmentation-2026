# Small filament verifier on the 419-image refit

The 419-image U-Net was trained independently with the baseline architecture and objective. The eight-feature verifier was fitted on 115 images using out-of-training predictions from the separate 336-image checkpoint. The 419-image model trained on 83 of those 115 images. The verifier coefficients and 0.50 acceptance cutoff transferred without fitting to any 419-model predictions. The refit uses the same foreground probability threshold of 0.65 and minimum area of 700 pixels. The verifier appends accepted 150–699 pixel masks. Exact mask comparison verified that every original refit mask was preserved in the same order and the added masks were disjoint.

| Previously exposed evaluation | Refit PQ | Original model with verifier | Refit with verifier |
| --- | ---: | ---: | ---: |
| Confirmation, 80 images | 0.31466 | 0.30734 | 0.32536 |
| Udaipur, 101 images | 0.34152 | 0.34578 | 0.35107 |
| Pooled | 0.32975 | 0.32913 | 0.33984 |

The pooled gain was 0.01009 over the stronger parent, with paired 95% interval [0.00647, 0.01425] using 68 observation-date blocks. Observations in the same date block stay together across both evaluation partitions. The [independent scoring receipt](../artifacts/small-proposal-refit-transfer-independent.json) records the bootstrap seed, matched metric checks, and interval. The transfer added 48 reference matches and reduced complete misses by 58 versus the refit. It introduced nine zero-overlap predictions, five meaningful splits, and zero new meaningful merges. Thin matches rose from 62 to 73 on confirmation and 73 to 83 on Udaipur. The original verifier retained more Udaipur thin matches (87) and near-limb matches (83 versus 78). The two folds remained outside both checkpoints' training and the verifier fit, and were already scored in earlier experiments. They are paired follow-up evidence.

To rebuild the 419-image checkpoint and calibration from the competition data:

```powershell
$env:PYTHONPATH = 'src'
$data = 'data/MAGFiLO_1.0_Kaggle_2026'
python -m filament train --data $data --split artifacts/refit-split.json --config artifacts/refit-config.json --output runs/refit-rebuild --device cuda
python -m filament.calibrate --data $data --split artifacts/refit-split.json --config artifacts/refit-config.json --checkpoint runs/refit-rebuild/model.pt --cache runs/refit-rebuild/dev-probs --output-config runs/refit-rebuild/frozen.json --output-score runs/refit-rebuild/calibration.json
```

Use the included frozen verifier for exact reproduction. To fit its coefficients again, first recreate the separate 336-image checkpoint using `configs/baseline.json` and `artifacts/split.json`, then follow the map-generation and fitting commands in [small-proposal-verifier.md](small-proposal-verifier.md) with the recreated 336-image checkpoint and the fixed 0.65/700 configuration. The deployed 419-image checkpoint and verifier are both included, so reproducing the submitted prediction bytes requires neither training run. The notebook verifies source, model and input identity before full inference.
