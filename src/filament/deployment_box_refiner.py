"""Reproduce the selected box-prompted 364-fit entry from supplied H-alpha images."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

from pycocotools import mask as coco
import torch

from .box_prompt_r18 import open_refiner, refine
from .data import file_sha256, load_image, save_new_json
from .deployment_native import read, verified_inputs
from .detector_native_predict import load_module, native_proposals
from .exclusive_mask_assembly import exclusive_masks
from .format import write_submission
from .infer import profile_device, runtime_receipt


def checked_rule(rule):
    if (set(rule) != {"confidence", "mask_probability", "flip_mode", "network_box_floor", "minimum_area", "nms_iou"}
            or rule["network_box_floor"] != .20 or rule["minimum_area"] != 150 or rule["nms_iou"] != .50
            or rule["confidence"] not in (.20, .30, .40) or rule["mask_probability"] not in (.30, .40, .50)
            or rule["flip_mode"] not in ("identity", "flips4")):
        raise ValueError("Exact selected refiner rule differs")
    return rule


def candidate_masks(model, gray, proposals, rule):
    # Use identical .20 admission/batching to the development run, then filter
    # the original regressed proposals at the chosen confidence.
    rows = refine(model, gray, proposals, rule["network_box_floor"], rule["mask_probability"], rule["flip_mode"])
    accepted = []
    for row in rows:
        index = row["proposal_index"]
        if type(index) is not int or not 0 <= index < len(proposals) or row["score"] != proposals[index]["score"]:
            raise ValueError("Refined output lost exact original detector ownership")
        if row["score"] >= rule["confidence"]:
            mask = {"size": row["size"], "counts": row["counts"].encode("ascii")}
            if coco.area(mask) < rule["minimum_area"]:
                raise ValueError("Native mask is smaller than the selected rule")
            accepted.append((row["score"], index, mask))
    return [mask for _, _, mask in exclusive_masks(accepted).accepted]


def predict(root, images, output):
    contract_path = root / "artifacts/release.json"
    cfg = read(contract_path)
    if cfg["type"] != "r18_box_prompt_707_fit" or cfg["training_observations"] != 707:
        raise ValueError("Expected the all-707 refiner deployment manifest")
    rule = checked_rule(cfg["rule"])
    for name, digest in cfg["artifact_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Frozen release bytes changed for {name}")
    for name, wanted in cfg["packages"].items():
        if version(name) != wanted:
            raise ValueError(f"Exact released package version differs for {name}")
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    paths = verified_inputs(root, images)
    scratch = load_module("yolo11s_scratch", root / "scripts/yolo11s_scratch.py")
    native = load_module("yolo11s_native_tiles", root / "scripts/yolo11s_native_tiles.py")
    scratch.RUN = output
    YOLO, yaml = scratch.setup()
    if file_sha256(yaml) != cfg["model_yaml_sha256"]:
        raise ValueError("Installed detector architecture differs")
    device = torch.device("cuda", torch.cuda.current_device())
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.75, 12 * 2**30 / total), device.index)
    start = profile_device(device)
    detector = YOLO(str(root / "artifacts/detector.pt"), task="segment")
    detector_config = read(root / "artifacts/detector-native-config.json")
    detector_options = read(root / "artifacts/detector-config.json")["prediction"]
    refiner = open_refiner().to(device, memory_format=torch.channels_last).eval()
    refiner.load_state_dict(torch.load(root / "artifacts/refiner.pt", weights_only=True, map_location="cpu"), strict=True)

    def instances():
        for i, (stem, path) in enumerate(paths.items(), 1):
            gray = load_image(path)
            proposals = native_proposals(detector, gray, detector_config, detector_options, native)
            masks = candidate_masks(refiner, gray, proposals, rule)
            yield stem, (coco.decode(mask) for mask in masks)
            if i == 1 or i % 20 == 0:
                print(f"Native box refiner {i}/{len(paths)}", flush=True)

    csv, coverage = output / "submission.csv", output / "coverage.json"
    write_submission(csv, instances(), coverage_path=coverage)
    covered = read(coverage)["images"]
    runtime = runtime_receipt(device, start)
    if set(covered) != set(paths) or not 0 < csv.stat().st_size < 10_000_000 or runtime["cuda_peak_reserved_bytes"] > 12 * 2**30:
        raise ValueError("Native export completeness, form size or GPU limit failed")
    receipt = {**runtime, "type": cfg["type"], "release_sha256": file_sha256(contract_path),
               "observations": len(covered), "instances": sum(row["instances"] for row in covered.values()),
               "empty_observations": sum(not row["instances"] for row in covered.values()),
               "csv_bytes": csv.stat().st_size, "csv_sha256": file_sha256(csv), "coverage_sha256": file_sha256(coverage),
               "test_inventory_sha256": file_sha256(root / "artifacts/test-files.sha256.json"),
               "checkpoint_sha256": cfg["checkpoint_sha256"], "rule": rule}
    save_new_json(output / "inference.json", receipt)
    print(json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path.cwd(), type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    predict(args.root.resolve(), args.images.resolve(), args.output.resolve())
