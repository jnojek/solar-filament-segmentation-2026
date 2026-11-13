# Data and software provenance

The competition API lists 888 files totaling 750,973,628 bytes: 707 training JPEGs, one training COCO JSON and 180 test JPEGs. The Kaggle CLI downloaded a 703,574,877-byte archive. Its SHA-256 and the per-file identities are in `artifacts/`. Archive members were validated against the live file listing before unpacking. The input and test hash lists are provenance only; the model does not use dates, site codes or identifiers as features.

Only the supplied training polygons are used as labels. The published MAGFiLO paper was read for context; no dataset archive, test labels, adjacent images, third-party predictions or pretrained weights were used. No competition images, original annotations, predictions or private credentials belong in the public Git release. The final CSV and PDF are separate owner deliverables.

Original implementation: the neural model follows the published U-Net architecture concept; official scorer behavior is cited and compared with an independent implementation. The official scorer notebook remains a local reference and is not vendored in the public repo. The report source is adapted from the organizer-provided Overleaf template for this competition only.

The runtime uses PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), SciPy (BSD-3-Clause), Pillow (HPND), pycocotools (BSD-2-Clause), pandas (BSD-3-Clause), psutil (BSD-3-Clause) and scikit-image (BSD-3-Clause). Jupyter packages support notebook execution. Package sources are the upstream Python and PyTorch distributions; the local CUDA PyTorch build was already present and reused without any pretrained model files.
