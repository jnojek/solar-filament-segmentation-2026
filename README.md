# Solar Filament Segmentation

This entry uses a native YOLO11s-seg detector trained on 364 competition disks
and a box-prompted U-Net refiner trained on the same disks. The refiner learns
individual mask shapes from grouped out-of-fold detector proposals. At inference,
the released model reads only the supplied H-alpha images.

Run reproduce.ipynb with Python 3.12 in the frozen CUDA environment, or set
PYTHONPATH to src and run:

    python -B -m filament.deployment_box_refiner --root . --images <test_images> --output <new_directory>

Use the pinned CUDA builds of Torch and TorchVision. The frozen package
versions and exact checkpoint hashes are listed in artifacts/release.json.
The pinned dependencies are listed in requirements.txt.

The source-aligned refiner credits Artem Shelamanov's ArtDoctor solar-kaggle-public
repository (commit ea487495ab55aa375fcc36eeb47adc701e79f60b, AGPL-3.0) for cropped
box conditioning, true out-of-fold prompts and empty crop negatives. The integrated
detector and refiner software is released under AGPL-3.0. The independent earlier
compact U-Net source remains MIT. See NOTICE for full attribution, fitted-state
provenance and donor terms, and training/README.md for corresponding source.
TorchVision ImageNet1K V1 initialized the ResNet18 encoder for training. The
original donor checkpoint is excluded. It is unnecessary for inference.
