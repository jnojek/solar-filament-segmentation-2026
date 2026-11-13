# Solar filament instance segmentation

This project segments individual solar filaments in native 2048 x 2048 grayscale GONG H-alpha observations. It uses only the training images and annotations provided by the [Kaggle competition](https://www.kaggle.com/competitions/filament-segmentation-2026). Inference uses only the supplied test JPEGs and the checkpoint trained from scratch here.

## Reproduce

1. Download the official data using the Kaggle CLI after joining the competition and accepting its rules. Unpack it as `data/MAGFiLO_1.0_Kaggle_2026/`. The repository does not redistribute any competition inputs or annotations.
2. Create a Python 3.12 environment and install the pinned packages in `requirements.txt`. Select the corresponding PyTorch CUDA 12.8 build for GPU reproduction. Use the shipped weights; there is no need to train.
3. Open `reproduce.ipynb` from the repository root and run all cells. It verifies the selected configuration and checkpoint, runs the complete test inference, and independently validates the native RLE output. See [reproduction notes](docs/reproduction-notes.md) for the exact CLI and [release manifest](artifacts/release.json) for artifact hashes.

## Selected result

The frozen baseline achieved local PQ of 0.3365 on 115 development disks, 0.2931 on 80 confirmation disks, and 0.3385 on 101 held-out Udaipur disks. The tested local contrast filter removed unsupported predictions but introduced too many complete misses under the predeclared rule. See [the selection record](artifacts/SELECTION.md) and [validation evidence](artifacts/validation.json).

Independent reproduction matched the exact CSV and coverage bytes in a fresh notebook process on the RTX 4090. The output contains 1,293 masks from 180 verified images. The separate owner handoff contains the report and the exact submission CSV.

## How to read the evidence

Development estimates select settings. Confirmation and the held-out observatory measure transfer after the selection is frozen. Local PQ mirrors the organizer's published self-evaluation notebook; a Kaggle score, if reported, comes from an accepted upload of the exact CSV. See [the solar rationale](docs/mechanism.md) for the mechanism and related work.

Organizers use compressed COCO RLE in column-major mask order. The submission has one row per predicted instance, and a separate local coverage record accounts for every source image, including images with no predicted instances.

## Attribution and provenance

- GONG observations and MAGFiLO annotations are sourced through Kaggle. See [Ahmadzadeh et al. (2024)](https://doi.org/10.1038/s41597-024-03876-y).
- Model design follows the encoder/decoder and skip connections of [U-Net](https://doi.org/10.1007/978-3-319-24574-4_28). The implementation and fitted weights are from this project.
- Evaluation follows the [organizer notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook), with original local implementation and fixtures. Obtain the official notebook directly to run independent parity checks.
- No other entrants' code, weights, or predictions were used. Public articles informed the scientific rationale only.
