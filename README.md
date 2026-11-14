# Solar Filament Segmentation

This package reproduces the fixed all-label MAGFiLO deployment. The detector was trained from
scratch for 36 epochs and continued for eight native-tile epochs. The compact U-Net used
19 epochs. Both fits used all 707 competition training disks. Reproduce inference with
the released final detector and U-Net without training or loading the annotations.
See artifacts/release.json for the exact checkpoint identities and DETECTOR-LICENSE.md
for attribution and licensing.

The release uses Python 3.12 and PyTorch 2.10.0+cu128. Install the matching CUDA build of
PyTorch and TorchVision and the pinned dependencies in requirements.txt. The exact
replay reference environment is an RTX 4090. GPU/software changes can alter numerics.

Extract the official supplied test JPEGs into a local folder. Open and run reproduce.ipynb,
or set PYTHONPATH to this repository's src and run:

    python -B -m filament.deployment_native --root . --images <test_images> --output <new_directory>

submission.csv stores each instance as a native COCO RLE. Empty disks have no CSV row
and appear explicitly in coverage.json. The notebook checks mask decoding and coverage
against the exact supplied JPEG inventory.
