"""Measure development sensitivity to shifting the native inference tile grid."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from filament.data import DATASET_DIR, file_sha256, load_image, read_annotations, save_new_json
from filament.infer import open_model, probability_map, profile_device, runtime_receipt
from filament.train import score_probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data") / DATASET_DIR)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    alternate = {**config, "inference_stride": 256}
    ids = [split["dev"][int(len(split["dev"]) * fraction)] for fraction in (0.1, 0.35, 0.6, 0.85)]
    observations = read_annotations(args.data)
    torch.set_num_threads(4)
    model, device = open_model(config, args.checkpoint, "cuda")
    started = profile_device(device)
    rows = []
    for image_id in ids:
        observation = observations[image_id]
        stored = np.load(args.probabilities / f"{image_id}.npy", allow_pickle=False)
        image = load_image(observation.path)
        repeated = probability_map(model, image, config, device)
        if not np.array_equal(stored, repeated):
            raise ValueError(f"Frozen cached probability map failed exact replay: {image_id}")
        shifted = probability_map(model, image, alternate, device)
        diff = np.abs(repeated.astype(np.float32) - shifted.astype(np.float32))
        # Restrict the error summary to the vicinity of predicted foreground.
        support = (repeated > 0.25) | (shifted > 0.25)
        threshold, min_area = config["threshold"], config["min_area"]
        before, after = repeated >= threshold, shifted >= threshold
        union = np.count_nonzero(before | after)
        rows.append({
            "image_id": image_id,
            "foreground_iou": float(np.count_nonzero(before & after) / union) if union else None,
            "mean_abs_probability_change_on_support": float(diff[support].mean()) if support.any() else None,
            "baseline_pq": score_probability(repeated, observation, threshold, min_area).pq,
            "alternate_pq": score_probability(shifted, observation, threshold, min_area).pq,
        })
    receipt = {
        **runtime_receipt(device, started),
        "split_sha256": file_sha256(args.split),
        "config_sha256": file_sha256(args.config),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "baseline_stride": config["inference_stride"],
        "alternate_stride": alternate["inference_stride"],
        "disks": rows,
    }
    save_new_json(args.output, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
