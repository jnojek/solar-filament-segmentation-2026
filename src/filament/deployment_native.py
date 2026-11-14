"""Reproduce the fixed all-label deployment from a hash-bound portable release."""

import argparse
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from pycocotools import mask as coco
import torch

from .data import file_sha256, image_paths, load_image, save_new_json
from .detector_native_predict import load_module, native_proposals
from .detector_ownership import DILATION_PX, MIN_AREA, fuse_detector
from .format import write_submission
from .infer import probability_map, profile_device, runtime_receipt
from .model import FilamentUNet


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified_inputs(root, images, expected=180):
    manifest = read(root / "artifacts/test-files.sha256.json")
    paths = image_paths(images)
    if len(paths) != expected or {path.name for path in paths.values()} != set(manifest):
        raise ValueError("Exact official test JPEG filename set differs")
    for path in paths.values():
        if file_sha256(path) != manifest[path.name]:
            raise ValueError(f"Exact supplied H-alpha image changed: {path.name}")
    return dict(sorted(paths.items()))


def create_models(root, contract, output):
    donor = load_module("yolo11s_scratch", root / "scripts/yolo11s_scratch.py")
    tiled = load_module("yolo11s_native_tiles", root / "scripts/yolo11s_native_tiles.py")
    donor.RUN = output
    detector_config = read(root / "artifacts/detector-config.json")
    native_config = read(root / "artifacts/detector-native-config.json")
    semantic_config = read(root / "artifacts/semantic-config.json")
    for key, wanted in contract["packages"].items():
        if importlib.metadata.version(key) != wanted:
            raise ValueError(f"Exact portable version mismatch for {key}")
    rule = contract["rule"]
    if rule != {"mode": "strict", "confidence": .30, "semantic_threshold": .65,
                "dilation_px": DILATION_PX, "min_area": MIN_AREA}:
        raise ValueError("Original strict rule changed")
    YOLO, yaml = donor.setup()
    if file_sha256(yaml) != contract["model_yaml_sha256"]:
        raise ValueError("Original installed YOLO architecture changed")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(min(.75, 12 * 2**30 / props.total_memory), device.index)
    torch.manual_seed(271828)
    det = YOLO(str(root / "artifacts/detector.pt"), task="segment")
    unet = FilamentUNet(semantic_config["channels"]).to(device, memory_format=torch.channels_last).eval()
    unet.load_state_dict(torch.load(root / "artifacts/semantic.pt", weights_only=True, map_location="cpu"), strict=True)
    return SimpleNamespace(det=det, unet=unet, native_cfg=native_config, detector_options=detector_config["prediction"],
                           sm_cfg=semantic_config, tiled=tiled, device=device)


def predict(root, images, output):
    contract = read(root / "artifacts/release.json")
    for name, digest in contract["artifact_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Frozen release artifact changed: {name}")
    if output.exists():
        raise FileExistsError("Use an unused portable output directory")
    paths = verified_inputs(root, images)
    kit = create_models(root, contract, output)
    started = profile_device(kit.device)
    rule = contract["rule"]

    def generate():
        for index, (stem, path) in enumerate(paths.items(), 1):
            gray = load_image(path)
            proposals = native_proposals(kit.det, gray, kit.native_cfg, kit.detector_options, kit.tiled)
            probability = probability_map(kit.unet, gray, kit.sm_cfg, kit.device)
            detections = [(p["score"], {"size": p["size"], "counts": p["counts"].encode("ascii")}) for p in proposals]
            masks = fuse_detector(probability, detections, mode=rule["mode"], confidence=rule["confidence"],
                                  semantic_threshold=rule["semantic_threshold"])
            yield stem, (coco.decode(rle) for rle in masks)
            if index == 1 or index % 20 == 0:
                print(f"Exact all-707 replay {index}/{len(paths)}", flush=True)

    csv, coverage = output / "submission.csv", output / "coverage.json"
    write_submission(csv, generate(), coverage_path=coverage)
    seen = read(coverage)["images"]
    run = runtime_receipt(kit.device, started)
    if (set(seen) != set(paths) or csv.stat().st_size >= 10_000_000
            or run["cuda_peak_reserved_bytes"] > 12 * 2**30):
        raise ValueError("Export coverage, resource or file size check failed")
    receipt = {**run, "type": "deployment_replay", "release_sha256": file_sha256(root / "artifacts/release.json"),
               "observations": len(seen), "instances": sum(row["instances"] for row in seen.values()),
               "empty_observations": sum(not row["instances"] for row in seen.values()),
               "csv_bytes": csv.stat().st_size, "csv_sha256": file_sha256(csv), "coverage_sha256": file_sha256(coverage),
               "test_inventory_sha256": file_sha256(root / "artifacts/test-files.sha256.json"),
               "checkpoint_sha256": contract["checkpoint_sha256"], "rule": rule}
    save_new_json(output / "inference.json", receipt)
    print(json.dumps(receipt, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict(args.root.resolve(), args.images.resolve(), args.output.resolve())
