"""Export the frozen detector ownership rule at the competition's native size."""

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as coco
import torch

from .__main__ import model_and_config
from .data import file_sha256, image_paths, load_image, save_new_json
from .detector_ownership import fuse_detector
from . import detector_ownership
from .format import SHAPE, write_submission
from .infer import instance_rles, probability_map, profile_device, runtime_receipt
from .small_proposal import candidate_rles


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(args):
    paths = image_paths(args.images)
    manifest = read_json(args.inventory)
    if manifest != {path.name: file_sha256(path) for path in paths.values()}:
        raise ValueError("Input images differ from the frozen Kaggle manifest")
    predictions = read_json(args.detections)
    if set(predictions) != set(paths):
        raise ValueError("Detector prediction IDs differ from the verified image IDs")
    detector_receipt = read_json(args.detector_receipt)
    detector_checkpoint_hash = file_sha256(args.detector_checkpoint)
    detector_predictions_hash = file_sha256(args.detections)
    inventory_hash = file_sha256(args.inventory)
    if (
        detector_receipt["checkpoint_sha256"] != detector_checkpoint_hash
        or detector_receipt["prediction_sha256"] != detector_predictions_hash
        or detector_receipt["observations"] != len(paths)
        or detector_receipt["fold"] != "test"
        or detector_receipt["inventory_sha256"] != inventory_hash
    ):
        raise ValueError("Detector receipt differs from the frozen test prediction artifact")
    freeze = read_json(args.rule)
    if any(detector_receipt[key] != value for key, value in freeze["detector"].items()
           if key != "inventory_sha256"):
        raise ValueError("The test detector differs from the one calibrated on development")
    if file_sha256(Path(detector_ownership.__file__)) != freeze["identity"]["ownership_source_sha256"]:
        raise ValueError("Detector fusion source differs from the calibrated rule")
    rule = freeze["selected"]
    mode = rule["mode"]
    parent_name = rule.get("parent", "verifier" if mode == "append" else None)
    if mode == "append" and parent_name not in {"refit", "verifier"}:
        raise ValueError("An append rule must name its parent")
    if parent_name == "verifier" and args.verifier is None:
        raise ValueError("The verifier parent requires --verifier")
    verifier = read_json(args.verifier) if parent_name == "verifier" else None
    if verifier is not None and file_sha256(args.verifier) != freeze["identity"]["verifier_sha256"]:
        raise ValueError("Verifier coefficients differ from the calibrated rule")

    destinations = [args.output, args.coverage, args.receipt]
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("CSV, coverage and receipt paths must differ")
    if any(path.exists() for path in destinations):
        raise FileExistsError("Use fresh CSV, coverage and receipt paths")

    if mode == "detector":
        model, device, config = None, torch.device("cpu"), None
        native_shape = np.broadcast_to(np.float16(0), SHAPE)
    else:
        if args.config is None or args.checkpoint is None:
            raise ValueError("Fusion requires --config and --checkpoint")
        model, device, config = model_and_config(args)
    started = profile_device(device)

    def generate():
        for index, (image_id, path) in enumerate(paths.items(), 1):
            if model is None:
                probability = native_shape
            else:
                image = load_image(path)
                probability = probability_map(model, image, config, device)
            detections = []
            for prediction in predictions[image_id]:
                if tuple(prediction["size"]) != SHAPE:
                    raise ValueError(f"{image_id}: detector mask is not at native size")
                rle = {"size": list(SHAPE), "counts": prediction["counts"].encode("ascii")}
                detections.append((float(prediction["score"]), rle))
            parent = ()
            if parent_name == "verifier":
                parent = candidate_rles(
                    probability, image, config["threshold"], config["min_area"], verifier,
                )
            elif parent_name == "refit":
                parent = instance_rles(probability, config["threshold"], config["min_area"])
            instances = fuse_detector(
                probability, detections, mode=mode, confidence=rule["confidence"],
                semantic_threshold=rule["semantic_threshold"], parent=parent,
            )
            yield image_id, (coco.decode(rle) for rle in instances)
            if index == 1 or index % 20 == 0:
                print(f"export: {index}/{len(paths)}", flush=True)

    write_submission(args.output, generate(), coverage_path=args.coverage)
    images = read_json(args.coverage)["images"]
    size = args.output.stat().st_size
    runtime = runtime_receipt(device, started)
    receipt = {
        **runtime,
        "total_inference_seconds": round(detector_receipt["elapsed_seconds"] + runtime["elapsed_seconds"], 3),
        "total_ram_peak_bytes": max(detector_receipt["ram_peak_bytes"], runtime["ram_peak_bytes"]),
        "total_cuda_peak_reserved_bytes": max(
            detector_receipt["cuda_peak_reserved_bytes"], runtime["cuda_peak_reserved_bytes"] or 0,
        ),
        "observations": len(images),
        "empty_observations": sum(not image["instances"] for image in images.values()),
        "instances": sum(image["instances"] for image in images.values()),
        "csv_bytes": size,
        "fits_final_form": size < 10_000_000,
        "csv_sha256": file_sha256(args.output),
        "coverage_sha256": file_sha256(args.coverage),
        "config_sha256": file_sha256(args.config) if model is not None else None,
        "checkpoint_sha256": file_sha256(args.checkpoint) if model is not None else None,
        "detector_checkpoint_sha256": detector_checkpoint_hash,
        "detector_predictions_sha256": detector_predictions_hash,
        "detector_receipt_sha256": file_sha256(args.detector_receipt),
        "rule_sha256": file_sha256(args.rule),
        "test_manifest_sha256": inventory_hash,
        "verifier_sha256": file_sha256(args.verifier) if verifier is not None else None,
    }
    save_new_json(args.receipt, receipt)
    if not receipt["fits_final_form"]:
        raise ValueError("The CSV exceeds the final form upload limit")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "images", "inventory", "detector-checkpoint", "detector-receipt",
        "detections", "rule", "output", "coverage", "receipt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--verifier", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    print(json.dumps(export(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
