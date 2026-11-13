# Solar filament instance segmentation

This project segments individual solar filaments in native 2048 × 2048 GONG H-alpha images. A U-Net trained on 419 competition images proposes instance masks. A small-component verifier, fitted on predictions from a separately trained 336-image model, adds supported 150–699 pixel masks. The export preserves every original refit detection.

Download the official data through Kaggle, unpack it as `data/MAGFiLO_1.0_Kaggle_2026/`, and install the pinned requirements in a Python 3.12 environment. Run `reproduce.ipynb` from the repository root. It verifies artifact and image hashes, reconstructs the 180-image submission, and independently checks the decoded masks. The included checkpoint and verifier are sufficient for exact inference. See [reproduction notes](docs/reproduction-notes.md) and the [release manifest](artifacts/release.json).

On the previously evaluated confirmation and Udaipur groups, the combined pipeline scored 0.32536 and 0.35107 PQ. Pooled PQ was 0.33984, compared with 0.32975 for the refit and 0.32913 for the original model with verifier. The paired 95% interval for the gain over the refit was [0.00647, 0.01425], grouping observations in 68 date blocks across the two partitions. The combined pipeline added 48 matches and reduced complete misses by 58 versus the refit. [Matched transfer analysis](docs/refit-verifier-transfer.md)

The local export contains 1,307 masks, including 50 additions, over 180 images. Three images have no predicted instances. The PDF and submission CSV are provided separately. The [release manifest](artifacts/release.json) records official submission identity and any available public score.

The project uses only the competition-supplied images and labels. The published [MAGFiLO paper](https://doi.org/10.1038/s41597-024-03876-y), [U-Net paper](https://doi.org/10.1007/978-3-319-24574-4_28), and [organizer scorer notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook) document the dataset, architecture, and evaluation. No other entrants' code, weights, predictions, or test annotations were used.

The [MIT License](LICENSE) covers the original project code, notebook, configurations, checkpoint and verifier coefficients. Competition images and annotations, the organizer report template, and third-party dependencies retain their respective terms.
