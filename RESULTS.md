# Solar Filament Segmentation 2026 results

Kaggle submission **58346328** completed with **0.32 public Panoptic Quality (PQ)**, verified at 15:31 UTC on November 14, 2026. Submission 58330929 scored 0.31 public PQ using the earlier 419-disk fit. The final refit uses all 707 competition training disks and has no independent local holdout.

The submitted CSV contains 1,220 instances across all 180 test images, including three images with no predicted instances. Export took 116.7 seconds on an RTX 4090. Independent notebook replay took 114.6 seconds and reproduced the CSV and coverage byte-for-byte.

- CSV SHA-256: 70ff70f038ac010f494476d0c913f714d00bde7fb2d0b7f5b6221aebaf85467e
- [Reproduce inference with the released checkpoints](reproduce.ipynb)
- [Rebuild the training dataset and run the fixed training recipe](training/README.md)
- [Official score and report identity](artifacts/official-calibration.json)

The training dry run reproduced every input selection and derived label, along with the exact bytes of all 6,363 native tiles. Fresh CUDA training was not repeated for the release check.

The YOLO detector, its trained weights, and the combined training entrypoints in training/refit.py and training/detector.py are covered by [AGPL-3.0](LICENSE-AGPL-3.0). The independent compact U-Net and its source retain the [MIT license](LICENSE). See [detector attribution](DETECTOR-LICENSE.md).
