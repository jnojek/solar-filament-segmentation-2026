# Kaggle result

The native YOLO11s detector and box-prompted ResNet18 U-Net scored **0.36
public Panoptic Quality (PQ)** on November 14, 2026, at 19:44 UTC
(Kaggle submission **58353757**). The previous all-data semantic-ownership
pipeline scored 0.32 (58346328).

## Matched validation

Both local arms fit the same 364 competition H-alpha disks. Three separate
grouped out-of-fold detectors supplied refiner crop prompts. The refiner used
the regressed box, kept every crop at native placement and learned shape
within that box. The deployed rule uses a 0.30 detector confidence floor,
0.30 mask probability, 150-pixel minimum and fixed exclusive mask assembly.
Crop inference runs with the same 0.20 input floor used during calibration.

| Split | Disks | Strict detector + semantic | Box-prompted refiner |
|---|---:|---:|---:|
| Development | 32 | 0.3714 | 0.4171 |
| Fit-disjoint confirmation | 55 | 0.3002 | 0.3827 |
| Previously examined folds combined | 181 | 0.3474 | 0.4021 |

The 55 confirmation disks were excluded from both final 364-disk fits and all
out-of-fold donors. Earlier campaign checkpoints used these observations.
The fit-disjoint confirmation gain was 0.0826 PQ, with a paired 95% bootstrap interval
of [0.0476, 0.1181] over 15 observing-date blocks. The refiner matched 348 of
681 reference instances, up from 275. Meaningful merges increased from 4 to 8,
and unsupported masks increased from 48 to 56. The previously examined folds
showed 141 additional matches, 126 fewer false positives, 30 meaningful merges
(up from 5) and 324 unsupported masks (up from 306).

## Reproduce the submitted output

The [inference notebook](reproduce.ipynb) reads the supplied H-alpha images and
loads the two released fitted checkpoints. Independent RTX 4090 execution
reproduced the CSV and coverage byte-for-byte. The first export took 124 seconds
and the notebook replay took 131 seconds. Native RLE validation covered 180
disks of 2048 × 2048 pixels, with 1,260 masks and 3 empty disks.

CSV SHA-256: `fc6264ef4fe851b71a0796f0047653411e9c68b90eccdf25e66403c2b2237612`.

The [training guide](training/README.md) documents the grouped out-of-fold
detector fits and refiner training. Checkpoint and score identities are in the
[calibration receipt](artifacts/official-calibration.json). The four-page report
is prepared for the organizer's separate final form.

The integrated detector and refiner source, including training, is under
AGPL-3.0. The independent compact U-Net source remains MIT. [NOTICE](NOTICE)
credits the refiner source and records the TorchVision initialization and
fitted-state provenance.
