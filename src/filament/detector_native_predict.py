"""Portable native-tile detector inference using the frozen training implementation."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .data import file_sha256, load_image, save_new_json
from .detector_predict import input_paths, read
from .format import SHAPE
from .infer import profile_device, runtime_receipt


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load the verified source at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initialize(args):
    training = read(args.training_receipt)
    identity = training["identity"]
    source_paths = {
        "source_sha256": args.training_source,
        "config_sha256": args.config,
        "split_sha256": args.split,
        "original_detector_source_sha256": args.original_detector_source,
        "original_detector_config_sha256": args.original_detector_config,
    }
    if (training["profile"] or training["checkpoint_sha256"] != file_sha256(args.checkpoint)
            or any(identity[key] != file_sha256(path) for key, path in source_paths.items())):
        raise ValueError("Native inference inputs differ from the frozen training receipt")
    base = load_module("yolo11s_scratch", args.original_detector_source)
    tiled = load_module("yolo11s_native_tiles", args.training_source)
    # These paths are explicit at release time and never read competition training annotations.
    base.CONFIG = args.original_detector_config
    base.SPLIT = args.split
    tiled.CONFIG = args.config
    tiled.RUN = args.settings_dir.resolve()
    YOLO, architecture = tiled.setup()
    if tiled.identity(architecture, identity) != identity:
        raise ValueError("Native architecture or installed packages differ from the training receipt")
    model = YOLO(str(args.checkpoint.resolve()), task="segment")
    return model, training, tiled


def native_proposals(model, image, config, settings, tiled):
    candidates = []
    for offset in range(0, len(tiled.TILES), 3):
        group = tiled.TILES[offset:offset + 3]
        inputs = [np.repeat(image[top:top + tiled.SIZE, left:left + tiled.SIZE, None], 3, axis=2)
                  for top, left in group]
        results = model.predict(source=inputs, device=0, verbose=False, **settings)
        for (top, left), result in zip(group, results, strict=True):
            if result.masks is None:
                continue
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            if masks.shape[1:] != (tiled.SIZE, tiled.SIZE) or not len(masks) == len(boxes) == len(scores):
                raise ValueError("Native tile geometry differs from prediction")
            candidates.extend(candidate for mask, box, score in zip(masks, boxes, scores)
                              if (candidate := tiled.lift(mask, box, score, top, left,
                                                         config["seam_margin_px"])) is not None)
    return tiled.deduplicate(candidates, config["native_mask_nms_iou"], settings["max_det"])


def predict(args):
    if args.output.exists() or args.receipt.exists() or args.output.resolve() == args.receipt.resolve():
        raise ValueError("Use two distinct new detector output paths")
    paths, ids, reference = input_paths(args)
    model, training, tiled = initialize(args)
    settings = read(args.original_detector_config)["prediction"]
    config = read(args.config)
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.79)
    started = profile_device(device)
    predictions = {}
    for index, stem in enumerate(ids, 1):
        rows = native_proposals(model, load_image(paths[stem]), config, settings, tiled)
        if reference is not None and rows != reference[stem]:
            raise ValueError(f"Portable native detector differs from the original on {stem}")
        predictions[stem] = rows
        if index == 1 or index % 20 == 0:
            print(f"Native detector {index}/{len(ids)}", flush=True)
    save_new_json(args.output, predictions)
    receipt = {**runtime_receipt(device, started), **training["identity"],
               "adapter_source_sha256": file_sha256(Path(__file__)), "checkpoint_sha256": training["checkpoint_sha256"],
               "prediction_sha256": file_sha256(args.output), "inventory_sha256": file_sha256(args.inventory),
               "fold": args.fold, "observations": len(predictions), "confidence_floor": settings["conf"],
               "nms_iou": settings["iou"], "native_shape": list(SHAPE), "tile_origins": tiled.TILES,
               "proposals": sum(map(len, predictions.values()))}
    if reference is not None:
        receipt.update(reference_sha256=file_sha256(args.reference), exact_reference_parity=True,
                       observation_ids=ids)
    save_new_json(args.receipt, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "inventory", "config", "split", "training-receipt", "training-source",
                 "checkpoint", "settings-dir", "output", "receipt", "original-detector-config",
                 "original-detector-source"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fold", choices=("dev", "test"), default="test")
    parser.add_argument("--reference", type=Path, help="Original development cache for exact subset parity")
    parser.add_argument("--sample-count", type=int, default=4)
    print(json.dumps(predict(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
