# Kaggle result

The all-707 box-prompted refiner scored **0.37 public Panoptic Quality (PQ)** on November 15, 2026 (Kaggle submission **58359830**). Submission 58353757 scored 0.36 with the separate 364-disk fit. The historical fit-disjoint confirmation improved from 0.30015 to 0.38274 on 55 disks. Those measures and the 0.36 Kaggle score belong to the separate 364-disk checkpoints.

The all-data detector and refiner use all 707 competition disks, including observations in the earlier evaluation splits. The refiner training draws from 1,154 annotator views and 8,199 supplied annotations. Three grouped out-of-fold detector donors generate training prompts. There is no independent local PQ for the all-707 weights.

The frozen export covers all 180 supplied 2048-by-2048 H-alpha images. It contains 1,263 exclusive native masks and 3 empty observations. A separate clean notebook reproduced the 349,958-byte CSV and coverage byte for byte. Native export inference took 110.665 seconds and replay inference took 110.001 seconds.

CSV SHA-256: `f6ed31a0a2cac84026332d76bbdef3a6c316478805c89b7a3e464d4ab7e656ce`.

Use the [inference notebook](reproduce.ipynb) or [training guide](training/README.md). The [calibration receipt](artifacts/official-calibration.json) records the exact output, checkpoint and report hashes. [NOTICE](NOTICE) records software licensing and fitted-state provenance. The four-page report is prepared for the organizer's separate final form.
