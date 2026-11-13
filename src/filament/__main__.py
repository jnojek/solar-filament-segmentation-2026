"""Command-line entry points for baseline training, inference and scoring."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco
import torch

from .data import ANNOTATIONS, DATASET_DIR, annotation_path, file_sha256, grouped_split
from .data import image_paths, load_image, read_annotations, save_new_json
from .format import read_rows, write_submission
from .infer import connected_instances, open_model, probability_map, profile_device, runtime_receipt
from .metrics import aggregate
from .train import run_training, score_probability, summarize

DEFAULT_DATA = Path("data") / DATASET_DIR


def make_split(args):
    observations = read_annotations(args.data)
    split = grouped_split(observations, args.seed)
    split["annotation_sha256"] = file_sha256(annotation_path(args.data))
    save_new_json(args.output, split)
    return {key: len(split[key]) for key in ("train", "dev", "confirm", "site_holdout", "purged")}


def model_and_config(args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(min(6, config["workers"]))
    model, device = open_model(config, args.checkpoint, args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.79, torch.cuda.current_device())
    return model, device, config


def score(args):
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if file_sha256(annotation_path(args.data)) != split["annotation_sha256"]:
        raise ValueError(f"{ANNOTATIONS} differs from the frozen split")
    observations = read_annotations(args.data)
    model, device, config = model_and_config(args)
    started = profile_device(device)
    if args.cache is not None:
        args.cache.mkdir(parents=True, exist_ok=False)
    results = {}
    ids = split[args.fold]
    if args.limit:
        ids = ids[:args.limit]
    for index, image_id in enumerate(ids):
        observation = observations[image_id]
        probability = probability_map(model, load_image(observation.path), config, device)
        if args.cache is not None:
            np.save(args.cache / f"{image_id}.npy", probability, allow_pickle=False)
        results[image_id] = score_probability(probability, observation, config["threshold"], config["min_area"])
        if index == 0 or (index + 1) % 10 == 0:
            print(f"{args.fold}: {index + 1}/{len(ids)}", flush=True)
    receipt = {
        **runtime_receipt(device, started),
        **summarize(aggregate(results.values())),
        "fold": args.fold,
        "observations": len(ids),
        "partial": bool(args.limit),
        "split_sha256": file_sha256(args.split),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "by_image": {image_id: summarize(metrics) for image_id, metrics in results.items()},
    }
    save_new_json(args.output, receipt)
    return {key: value for key, value in receipt.items() if key != "by_image"}


def infer(args):
    paths = image_paths(args.images)
    if args.sample is not None:
        expected = {filament_id.split("_", 1)[0] for filament_id, _ in read_rows(args.sample)}
        if expected != set(paths):
            raise ValueError("Sample IDs differ from the test image stems")
    if args.inventory is not None:
        expected = set(json.loads(args.inventory.read_text(encoding="utf-8")))
        if expected != set(paths):
            raise ValueError("Test image stems differ from the verified Kaggle inventory")
    destinations = [args.output, args.coverage, args.receipt]
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Choose a new artifact path: {path}")
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("Output, coverage and receipt paths must differ")
    model, device, config = model_and_config(args)
    started = profile_device(device)
    if args.cache is not None:
        args.cache.mkdir(parents=True, exist_ok=False)

    def generate():
        for index, (image_id, path) in enumerate(paths.items()):
            image = load_image(path)
            probability = probability_map(model, image, config, device)
            if args.cache is not None:
                np.save(args.cache / f"{image_id}.npy", probability, allow_pickle=False)
            threshold, min_area = config["threshold"], config["min_area"]
            if config.get("postprocess"):
                from .challenger import candidate_rles

                rles = candidate_rles(probability, image, threshold, min_area, config["postprocess"])
                masks = (coco.decode(rle) for rle in rles)
            else:
                masks = connected_instances(probability, threshold, min_area)
            yield image_id, masks
            if index == 0 or (index + 1) % 20 == 0:
                print(f"infer: {index + 1}/{len(paths)}", flush=True)

    write_submission(args.output, generate(), coverage_path=args.coverage)
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    row_count = sum(item["instances"] for item in coverage["images"].values())
    empty_count = sum(item["instances"] == 0 for item in coverage["images"].values())
    size = args.output.stat().st_size
    receipt = {
        **runtime_receipt(device, started),
        "observations": len(paths),
        "empty_observations": empty_count,
        "instances": row_count,
        "csv_bytes": size,
        "fits_final_form": size < 10_000_000,
        "csv_sha256": file_sha256(args.output),
        "coverage_sha256": file_sha256(args.coverage),
        "config_sha256": file_sha256(args.config),
        "checkpoint_sha256": file_sha256(args.checkpoint),
    }
    save_new_json(args.receipt, receipt)
    if size >= 10_000_000:
        raise RuntimeError(f"CSV is {size} bytes and exceeds the final form cap")
    return receipt


def add_model_flags(parser):
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--data", type=Path, default=DEFAULT_DATA)
    split.add_argument("--seed", type=int, default=271828)
    split.add_argument("--output", type=Path, required=True)
    split.set_defaults(run=make_split)
    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train.add_argument("--split", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    train.add_argument("--profile-steps", type=int)
    train.set_defaults(run=lambda args: run_training(
        args.data, args.split, args.config, args.output, args.device, args.profile_steps,
    ))
    evaluate = commands.add_parser("score")
    add_model_flags(evaluate)
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA)
    evaluate.add_argument("--split", type=Path, required=True)
    evaluate.add_argument("--fold", choices=("dev", "confirm", "site_holdout"), required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--cache", type=Path)
    evaluate.add_argument("--limit", type=int)
    evaluate.set_defaults(run=score)
    predict = commands.add_parser("infer")
    add_model_flags(predict)
    predict.add_argument("--images", type=Path, required=True)
    predict.add_argument("--sample", type=Path)
    predict.add_argument("--inventory", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--coverage", type=Path, required=True)
    predict.add_argument("--receipt", type=Path, required=True)
    predict.add_argument("--cache", type=Path)
    predict.set_defaults(run=infer)
    args = parser.parse_args()
    result = args.run(args)
    if result is not None:
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
