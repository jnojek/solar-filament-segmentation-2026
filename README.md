# Solar filament instance segmentation

An independently trained YOLO11s segmentation detector and native-resolution U-Net segment individual filaments in the Kaggle GONG H-alpha images. The frozen ownership rule and held-out comparison are in [the release manifest](artifacts/release.json) and [experiment notes](docs/detector-pivot.md). The release uses competition-supplied training images and annotations, scratch detector initialization, and test JPEGs at inference.

To recreate the competition CSV, use the [pinned setup](docs/reproduction-notes.md) and run [the replay notebook](reproduce.ipynb) from the repository root. The notebook regenerates all predictions, checks native mask geometry and verifies the exact output hash.

See [detector licensing](DETECTOR-LICENSE.md) and the earlier [MIT license](LICENSE). Competition data and the organizer template retain their source terms.
