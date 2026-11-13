"""Freeze the baseline's threshold and area using development observations only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco
import torch

from .data import DATASET_DIR, annotation_path, file_sha256, load_image, read_annotations, save_new_json
from .infer import label_instances, label_rle, open_model, probability_map, profile_device, runtime_receipt
from .metrics import aggregate, evaluate_rles
from .train import summarize


THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.76, 0.82, 0.88, 0.94, 0.97)
AREAS = (100, 220, 400, 700, 1000, 1600)


def calibrate_image(item, cache, thresholds, areas):
    image_id, observation = item
    probability = np.load(cache / f"{image_id}.npy", allow_pickle=False)
    result = {}
    for threshold in thresholds:
        labels, keep = label_instances(probability, threshold, min(areas))
        encoded = []
        for label, bounds in keep:
            rle = label_rle(labels, label, bounds)
            encoded.append((int(coco.area(rle)), rle))
        for area in areas:
            predictions = [rle for pixels, rle in encoded if pixels >= area]
            result[threshold, area] = aggregate(
                evaluate_rles(list(truth), predictions) for truth in observation.views.values()
            )
    return image_id, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data") / DATASET_DIR)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-score", type=Path, required=True)
    args = parser.parse_args()
    if args.output_config.exists() or args.output_score.exists():
        raise FileExistsError("Choose new output paths to preserve the frozen baseline")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if file_sha256(annotation_path(args.data)) != split["annotation_sha256"]:
        raise ValueError("Training annotation identity differs from the frozen split")
    observations = read_annotations(args.data)
    dev = [(image_id, observations[image_id]) for image_id in split["dev"]]
    torch.set_num_threads(4)
    model, device = open_model(config, args.checkpoint, "cuda")
    torch.cuda.set_per_process_memory_fraction(0.79, torch.cuda.current_device())
    started = profile_device(device)
    args.cache.mkdir(parents=True)
    for index, (image_id, observation) in enumerate(dev):
        probability = probability_map(model, load_image(observation.path), config, device)
        np.save(args.cache / f"{image_id}.npy", probability, allow_pickle=False)
        if index == 0 or (index + 1) % 20 == 0:
            print(f"Cached development maps: {index + 1}/{len(dev)}", flush=True)
    thresholds = sorted(set(THRESHOLDS) | {config["threshold"]})
    areas = sorted(set(AREAS) | {config["min_area"]})
    pairs = [(threshold, area) for threshold in thresholds for area in areas]
    pooled = {pair: [] for pair in pairs}
    per_image = {}
    evaluate = partial(calibrate_image, cache=args.cache, thresholds=thresholds, areas=areas)
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        for image_id, result in pool.map(evaluate, dev):
            per_image[image_id] = result
            for pair, metrics in result.items():
                pooled[pair].append(replace(metrics, ious=(), dices=()))
    rows = [{"threshold": threshold, "min_area": area, **summarize(aggregate(pooled[threshold, area]))}
            for threshold, area in pairs]
    ranked = sorted(rows, key=lambda row: (-row["pq"], row["merges"], row["fp"], row["min_area"]))
    best = ranked[0]
    winner = best["threshold"], best["min_area"]
    selected_metrics = {image_id: result[winner] for image_id, result in per_image.items()}
    receipt = {
        **runtime_receipt(device, started),
        **summarize(aggregate(selected_metrics.values())),
        "fold": "dev",
        "observations": len(dev),
        "split_sha256": file_sha256(args.split),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "training_config_sha256": file_sha256(args.config),
        "ranked": ranked,
        "by_image": {image_id: summarize(metrics) for image_id, metrics in selected_metrics.items()},
    }
    config.update(threshold=winner[0], min_area=winner[1])
    save_new_json(args.output_config, config)
    receipt["config_sha256"] = file_sha256(args.output_config)
    save_new_json(args.output_score, receipt)
    summary = {key: receipt[key] for key in (
        "pq", "tp", "fp", "fn", "splits", "merges", "mean_overlapping_pair_dice",
        "mean_overlapping_pair_iou", "observations", "elapsed_seconds",
    )}
    print(json.dumps({"threshold": winner[0], "min_area": winner[1], **summary}), flush=True)


if __name__ == "__main__":
    main()
