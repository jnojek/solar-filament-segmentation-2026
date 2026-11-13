"""Keep faint elongated components while filtering weak compact predictions."""

import math
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .challenger import component_contrast, disk_geometry
from .infer import label_instances, label_rle


def spine_aspect(mask):
    """Estimate aspect as skeleton pixel count divided by median spine diameter."""
    spine = skeletonize(mask)
    if not spine.any():
        return None
    distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    diameter = 2.0 * float(np.median(distance[spine]))
    return float(spine.sum()) / diameter


def candidate_rles(probability, image, threshold, min_area, config):
    """Discard only components with measured low contrast and a compact spine."""
    if image.ndim != 2 or probability.shape != image.shape:
        raise ValueError("Expected a grayscale image and probability map of equal shape")
    if config["method"] != "morphology_supported_contrast":
        raise ValueError(f"Unsupported method: {config['method']}")
    floor = float(config["contrast_floor"])
    minimum_aspect = float(config["spine_aspect_floor"])
    if not all(math.isfinite(value) and value > 0 for value in (floor, minimum_aspect)):
        raise ValueError("Contrast and aspect floors must be positive and finite")

    labels, components = label_instances(probability, threshold, min_area)
    disk = disk_geometry(image) if components else None
    retained = []
    for label, bounds in components:
        contrast, _ = component_contrast(labels, label, bounds, image, disk)
        if contrast is not None and contrast < floor:
            aspect = spine_aspect(labels[bounds] == label)
            if aspect is not None and aspect < minimum_aspect:
                continue
        retained.append(label_rle(labels, label, bounds))
    return retained


def export(args, *, predictor=None):
    from pycocotools import mask as coco

    from .__main__ import model_and_config
    from .data import file_sha256, image_paths, load_image, save_new_json
    from .format import SHAPE, read_rows, write_submission
    from .infer import probability_map, profile_device, runtime_receipt

    if predictor is None:
        predictor = candidate_rles
    paths = image_paths(args.images)
    if args.sample is not None:
        ids = {filament_id.split("_", 1)[0] for filament_id, _ in read_rows(args.sample)}
        if ids != set(paths):
            raise ValueError("Sample IDs differ from image stems")
    if args.inventory is not None:
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        if isinstance(inventory, dict):
            if inventory != {path.name: file_sha256(path) for path in paths.values()}:
                raise ValueError("Image hashes differ from the frozen file manifest")
        elif not isinstance(inventory, list) or set(inventory) != set(paths):
            raise ValueError("Inventory IDs differ from image stems")
    destinations = [args.output, args.coverage, args.receipt]
    if len(set(path.resolve() for path in destinations)) != len(destinations):
        raise ValueError("Output, coverage and receipt paths must differ")
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Choose a new artifact path: {path}")
    if args.cache_in is not None:
        if {path.stem for path in args.cache_in.glob("*.npy")} != set(paths):
            raise ValueError("Cached probability IDs differ from image stems")
        baseline = json.loads(args.config.read_text(encoding="utf-8"))
        model = device = None
        started = perf_counter()
    else:
        model, device, baseline = model_and_config(args)
        started = profile_device(device)
    if args.cache_out is not None:
        args.cache_out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.candidate.read_text(encoding="utf-8"))
    cached_input_hashes = {}

    def generate():
        for index, (image_id, path) in enumerate(paths.items()):
            image = load_image(path)
            if args.cache_in is not None:
                cache_file = args.cache_in / f"{image_id}.npy"
                cached_input_hashes[image_id] = file_sha256(cache_file)
                probability = np.load(cache_file, mmap_mode="r", allow_pickle=False)
                if probability.shape != SHAPE or probability.dtype != np.float16 or not np.isfinite(probability).all():
                    raise ValueError(f"Invalid native float16 map for {image_id}")
            else:
                probability = probability_map(model, image, baseline, device)
            if args.cache_out is not None:
                np.save(args.cache_out / f"{image_id}.npy", probability, allow_pickle=False)
            instances = predictor(probability, image, baseline["threshold"], baseline["min_area"], config)
            yield image_id, (coco.decode(rle) for rle in instances)
            if index == 0 or (index + 1) % 20 == 0:
                print(f"infer: {index + 1}/{len(paths)}", flush=True)

    write_submission(args.output, generate(), coverage_path=args.coverage)
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))["images"]
    size = args.output.stat().st_size
    if device is None:
        import psutil

        memory = psutil.Process().memory_info()
        runtime = {
            "elapsed_seconds": round(perf_counter() - started, 3),
            "ram_rss_bytes": memory.rss,
            "ram_peak_bytes": getattr(memory, "peak_wset", memory.rss),
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
            "device": "cpu (cached probabilities)",
        }
    else:
        runtime = runtime_receipt(device, started)
    receipt = {
        **runtime,
        "observations": len(paths),
        "empty_observations": sum(not item["instances"] for item in coverage.values()),
        "instances": sum(item["instances"] for item in coverage.values()),
        "csv_bytes": size,
        "fits_final_form": size < 10_000_000,
        "csv_sha256": file_sha256(args.output),
        "coverage_sha256": file_sha256(args.coverage),
        "config_sha256": file_sha256(args.config),
        "candidate_config_sha256": file_sha256(args.candidate),
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
        "cached_input_sha256": cached_input_hashes if args.cache_in is not None else None,
    }
    save_new_json(args.receipt, receipt)
    if not receipt["fits_final_form"]:
        raise RuntimeError(f"CSV is {size} bytes and exceeds the final form cap")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--cache-in", type=Path)
    cache.add_argument("--cache-out", type=Path)
    args = parser.parse_args()
    if args.cache_in is None and args.checkpoint is None:
        parser.error("--checkpoint is required for full image inference")
    receipt = export(args)
    print(json.dumps({key: value for key, value in receipt.items() if key != "cached_input_sha256"}), flush=True)


if __name__ == "__main__":
    main()
