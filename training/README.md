# Corresponding training source

This recipe reconstructs the native YOLO11s-seg detector, three genuine group-held inner
out-of-fold detector donors, and the 512-pixel ResNet18 U-Net refiner. Use the
official competition data and the pinned CUDA environment in the release,
including segmentation-models-pytorch 0.5.0.

From the repository root, use an unused directory on the data volume:

    python -B training/refit.py --data <competition_dataset_root> --output <dry_run_dir> --dry-run

The dry run verifies the 707 training IDs and dated inner groups. It computes
every original image, chosen-view annotation, and native crop/label hash on
CPU. It does not fit any checkpoint and requires no pretrained asset.

To train, obtain the official TorchVision ResNet18 ImageNet1K V1 initializer
from https://download.pytorch.org/models/resnet18-f37072fd.pth . The expected
SHA-256 is f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
and the expected size is 46,830,571 bytes. Use the audited local file:

    python -B training/refit.py --data <competition_dataset_root> --output <fit_dir> --resnet18-imagenet <local_exact_pth>

A single process owns the GPU at a time; independent child processes run
each donor from its own fresh scratch initialization. Have 12 GiB available
host RAM and 5 GiB free disk for native tiles and fit artifacts. The job
reconstructs the existing full-data final detector from its exact published
36+8 recipe, independently of prompt-donor fitting. Each donor uses only its
purged inner-fold training membership. It keeps the original 36 full-disk then eight native detector epochs, using
only the final EMA from its own preceding stage. Donors produce training
prompts only for their omitted dated groups. The mask model runs 24 epochs
and saves just its final state. The released checkpoints remain the
reference for exact inference replay. Retraining was not repeated as a
byte-identical weight-archive test. The 707 deployment fits all original
labeled observations, so it has no independent local PQ. Its separately
fit 364 predecessor provides historical recipe validation.

Corresponding source license and attribution are in NOTICE and the license
files at repository root. The original ImageNet checkpoint is excluded.
