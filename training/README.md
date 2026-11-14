# Recreate the all-707 training recipe

Use an extracted copy of the official MAGFiLO_1.0_Kaggle_2026 competition
dataset. Use Python 3.12 and the pinned CUDA-enabled PyTorch and TorchVision
versions listed in the inference release. Install the rest of the pinned
requirements in requirements.txt. The original run used an RTX 4090 and six
host workers with one native thread per worker.

From the release root, verify all training JPEGs and official COCO labels,
reconstruct their exact view selection and inspect fixed budgets on CPU:

    python -B training/refit.py --data <extracted_dataset_root> --output <unused_dry_run_dir> --dry-run

The dry-run computes every PNG and label hash, writes derived labels and a
local provenance receipt, and does not run a model or read test data. It
requires roughly 200 MB of free disk for temporary label and proof files.
For the full reproduction, use an unused output folder on the data volume,
with at least 5 GiB free and 12 GiB available host RAM:

    python -B training/refit.py --data <extracted_dataset_root> --output <unused_fit_dir>

The launcher rebuilds the native pixels from the original 707 disks and
hardlinks the original JPEGs. Startup may need Internet access for
installing the pinned dependencies; model fitting uses local architecture
configuration and requires no pretrained weights or downloaded images.
Both detector stages use a fixed selected annotator view per physical disk.
The U-Net uses original per-view mask averaging. The three stages run in separate processes and keep final
scratch epoch 36, its own final native epoch 8 and compact U-Net epoch 19.
Expected optimizer updates are 465, 896 and 5,605. Validation iteration,
checkpoint selection by validation and early stopping are disabled. Use
an unoccupied GPU for this standalone command.

Each output stage has a training.json receipt. scratch and native keep
engine/weights/last.pt; semantic keeps model.pt. Stock Ultralytics weight
archives contain ordinary timestamp and path metadata, so retraining can
produce different archive hashes. The included original release weights,
inference notebook and 180-image replay are the byte-exact reference.
All 707 labels contribute to fitting, so these checkpoints have no
independent validation partition.
