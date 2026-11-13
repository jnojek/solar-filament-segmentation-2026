"""Add verified small components using the frozen baseline probability maps."""

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco
from scipy.special import expit

from .challenger import component_contrast, disk_geometry
from .infer import label_instances, label_rle
from .morphology_filter import export, spine_aspect


THRESHOLD, LARGE_AREA, SMALL_AREA = 0.65, 700, 150
FEATURES = (
    "log_area", "median_probability", "p90_probability", "contrast", "missing_contrast",
    "log_spine_aspect", "radial_fraction", "missing_disk",
)


def proposals(probability, image):
    labels, components = label_instances(probability, THRESHOLD, SMALL_AREA)
    disk = disk_geometry(image) if components else None
    large, small, features = [], [], []
    for label, bounds in components:
        rle = label_rle(labels, label, bounds)
        area = int(coco.area(rle))
        if area >= LARGE_AREA:
            large.append(rle)
            continue
        mask = labels[bounds] == label
        contrast, radial = component_contrast(labels, label, bounds, image, disk)
        median, p90 = np.quantile(probability[bounds][mask].astype(np.float32), [0.50, 0.90])
        aspect = spine_aspect(mask)
        small.append(rle)
        features.append((np.log(area), median, p90, contrast or 0.0, contrast is None,
                         np.log1p(aspect or 0.0), radial or 0.0, radial is None))
    return large, small, np.asarray(features, dtype=float).reshape((-1, len(FEATURES)))


def decisions(features, model):
    center = np.asarray(model["center"])
    scale = np.asarray(model["scale"])
    coefficients = np.asarray(model["weights"])
    if (model["feature_names"] != list(FEATURES) or center.shape != (len(FEATURES),)
            or scale.shape != center.shape or coefficients.shape != (len(FEATURES) + 1,)
            or not all(np.isfinite(values).all() for values in (center, scale, coefficients))
            or np.any(scale <= 0) or not 0 < model["keep_probability"] < 1):
        raise ValueError("Unexpected verifier features, coefficients or decision probability")
    standardized = np.clip((features - center) / scale, -5.0, 5.0)
    probability = expit(coefficients[0] + standardized @ coefficients[1:])
    return probability >= model["keep_probability"]


def candidate_rles(probability, image, threshold, min_area, config):
    if threshold != THRESHOLD or min_area != LARGE_AREA:
        raise ValueError("The frozen verifier requires the original probability and area thresholds")
    large, small, features = proposals(probability, image)
    accepted = [rle for rle, keep in zip(small, decisions(features, config)) if keep]
    return large + accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--verifier", dest="candidate", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache-in", type=Path)
    cache.add_argument("--cache-out", type=Path)
    args = parser.parse_args()
    receipt = export(args, predictor=candidate_rles)
    print(json.dumps({key: value for key, value in receipt.items() if key != "cached_input_sha256"}), flush=True)


if __name__ == "__main__":
    main()
