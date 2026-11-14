"""Prepare and evaluate one 1024-pixel native-tile detector continuation."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from pycocotools import mask as coco
import torch
import yaml

import yolo11s_scratch as parent
from filament.data import file_sha256, image_paths, load_image, save_new_json
from filament.infer import profile_device, runtime_receipt


ROOT = parent.ROOT
RUN = ROOT / "runs/yolo11s-native-tiles"
PARENT = parent.RUN
SOURCE = Path(__file__).resolve()
CONFIG = ROOT / ".local/yolo11s-native-tiles-config.json"
SIZE = 1024
NATIVE = (2048, 2048)
ORIGINS = (0, 512, 1024)
TILES = tuple((top, left) for top in ORIGINS for left in ORIGINS)


def identity(model_yaml, recorded=None):
    provenance = {
        "split_sha256": file_sha256(parent.SPLIT),
        "annotation_sha256": file_sha256(parent.annotation_path(parent.DATA)),
    } if recorded is None else {key: recorded[key] for key in ("split_sha256", "annotation_sha256")}
    return {
        "source_sha256": file_sha256(SOURCE), "config_sha256": file_sha256(CONFIG),
        **provenance, "model_yaml_sha256": file_sha256(model_yaml),
        "original_detector_config_sha256": file_sha256(parent.CONFIG),
        "original_detector_source_sha256": file_sha256(parent.SOURCE),
        "packages": {name: version(name) for name in ("torch", "torchvision", "ultralytics", "opencv-python",
                                                       "numpy", "pycocotools")},
    }


def clipping(polygon, top, left, cache):
    xs, ys = polygon[::2], polygon[1::2]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if x1 <= left or y1 <= top or x0 >= left + SIZE or y0 >= top + SIZE:
        return [], False
    offset = np.tile((left, top), len(polygon) // 2)
    if left <= x0 and x1 < left + SIZE and top <= y0 and y1 < top + SIZE:
        return [(polygon - offset) / SIZE], True
    if cache[0] is None:
        cache[0] = coco.decode(coco.frPyObjects([polygon.tolist()], *NATIVE)[0])
    cropped = np.ascontiguousarray(cache[0][top:top + SIZE, left:left + SIZE])
    contours, _ = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = [(item.reshape(-1).astype(np.float64) + 0.5) / SIZE for item in contours
              if len(item) >= 3 and cv2.contourArea(item) >= 32]
    return result, False


def prepare():
    dataset = RUN / "dataset"
    dataset.mkdir(parents=True)
    cv2.setNumThreads(1)
    source_receipt = parent.read(PARENT / "dataset/dataset-receipt.json")
    split = parent.read(parent.SPLIT)
    settings = parent.read(CONFIG)
    if (source_receipt["split_sha256"] != file_sha256(parent.SPLIT)
            or source_receipt["annotation_sha256"] != file_sha256(parent.annotation_path(parent.DATA))):
        raise ValueError("Parent dataset membership or annotations differ")
    for kind in ("images", "labels"):
        for fold in ("train", "val"):
            (dataset / kind / fold).mkdir(parents=True)
    result = {}
    counts = Counter()
    for fold in ("train", "dev"):
        sub = "train" if fold == "train" else "val"
        source = PARENT / "dataset"
        members = source_receipt["membership"][fold]
        if set(members) != set(split[fold]):
            raise ValueError("Tile data must preserve the exact physical split")

        def materialize(stem):
            original = source / "images" / sub / f"{stem}.jpeg"
            label = source / "labels" / sub / f"{stem}.txt"
            recorded = members[stem]
            if file_sha256(original) != recorded["image_sha256"] or file_sha256(label) != recorded["label_sha256"]:
                raise ValueError(f"Changed parent input {stem}")
            image = load_image(original)
            polygons = [np.fromstring(row[2:], sep=" ", dtype=np.float64) * 2048
                        for row in label.read_text(encoding="ascii").splitlines()]
            caches = [[None] for _ in polygons]
            contained = np.zeros(len(polygons), dtype=bool)
            tile_rows = {}
            stats = Counter()
            for top, left in TILES:
                rows = []
                for index, (polygon, cache) in enumerate(zip(polygons, caches)):
                    parts, full = clipping(polygon, top, left, cache)
                    contained[index] |= full
                    rows.extend("0 " + " ".join(f"{value:.10f}" for value in part) for part in parts)
                    if parts:
                        stats["full_instances" if full else "clipped_instances"] += len(parts)
                name = f"{stem}-{top:04d}-{left:04d}"
                pixels = image[top:top + SIZE, left:left + SIZE]
                destination = dataset / "images" / sub / f"{name}.png"
                image_bytes = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
                destination.write_bytes(image_bytes)
                target = dataset / "labels" / sub / f"{name}.txt"
                target.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="ascii", newline="\n")
                tile_rows[name] = {"png_sha256": hashlib.sha256(image_bytes).hexdigest(),
                                   "label_sha256": file_sha256(target), "polygons": len(rows),
                                   "bytes": len(image_bytes)}
                stats["empty_tiles"] += not rows
            stats["polygons"] = len(polygons)
            stats["contained_polygons"] = int(contained.sum())
            return stem, tile_rows, stats

        result[fold] = {}
        with ThreadPoolExecutor(max_workers=settings["preparation_workers"]) as pool:
            for index, (stem, tiles, stats) in enumerate(pool.map(materialize, split[fold]), 1):
                result[fold][stem] = tiles
                counts.update(stats)
                if index == 1 or index % 40 == 0:
                    print(f"{fold} {index}/{len(split[fold])}", flush=True)
    specification = {"path": dataset.as_posix(), "train": "images/train", "val": "images/val",
                     "names": {0: "filament"}}
    path = dataset / "data.yaml"
    path.write_text(yaml.safe_dump(specification), encoding="utf-8")
    digest = file_sha256(PARENT / "dataset/dataset-receipt.json")
    receipt = {"membership": result, "statistics": dict(counts), "split_sha256": file_sha256(parent.SPLIT),
               "source_dataset_receipt_sha256": digest, "config_sha256": file_sha256(CONFIG),
               "yaml_sha256": file_sha256(path), "positions": TILES, "native": list(NATIVE), "size": SIZE,
               "derived_image_bytes": sum(item["bytes"] for fold in result.values()
                                          for disk in fold.values() for item in disk.values())}
    save_new_json(dataset / "dataset-receipt.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "membership"}), flush=True)


def verify_dataset():
    dataset = RUN / "dataset"
    path = dataset / "data.yaml"
    receipt_path = dataset / "dataset-receipt.json"
    receipt = parent.read(receipt_path)
    if (receipt["split_sha256"] != file_sha256(parent.SPLIT)
            or receipt["source_dataset_receipt_sha256"] != file_sha256(PARENT / "dataset/dataset-receipt.json")
            or receipt["config_sha256"] != file_sha256(CONFIG)
            or receipt["yaml_sha256"] != file_sha256(path)):
        raise ValueError("Native detector dataset provenance changed")
    for fold, disks in receipt["membership"].items():
        sub = "train" if fold == "train" else "val"
        files = {stem: metadata for disk in disks.values() for stem, metadata in disk.items()}
        if set(files) != {p.stem for p in (dataset / "images" / sub).glob("*.png")}:
            raise ValueError("Native detector tile inventory changed")
        for stem, item in files.items():
            if (item["png_sha256"] != file_sha256(dataset / "images" / sub / f"{stem}.png")
                    or item["label_sha256"] != file_sha256(dataset / "labels" / sub / f"{stem}.txt")):
                raise ValueError(f"Native detector tile changed: {stem}")
    return path, receipt_path


def setup():
    # Reuse the audited offline/integration controls in a separate settings directory.
    with patch.object(parent, "RUN", RUN):
        return parent.setup()


def train(profile):
    YOLO, architecture = setup()
    path, dataset_receipt = verify_dataset()
    parent_receipt = parent.read(PARENT / "training/training-receipt.json")
    checkpoint = ROOT / parent_receipt["checkpoint"]
    if (parent_receipt["profile"] or parent_receipt["identity"] != parent.identity(architecture)
            or parent_receipt["checkpoint_sha256"] != file_sha256(checkpoint)):
        raise ValueError("Continuation parent differs from the exact scratch detector")
    config = parent.read(CONFIG)
    identities = identity(architecture)
    name = "profile" if profile else "training"
    if (RUN / name).exists():
        raise FileExistsError(RUN / name)
    if not profile:
        prior = parent.read(RUN / "profile/training-receipt.json")
        if prior["identity"] != identities or prior["cuda_peak_reserved_bytes"] > 17 * 2**30:
            raise ValueError("Matching CUDA profile under 17 GiB required")
    torch.cuda.set_per_process_memory_fraction(0.79)
    started = profile_device(torch.device("cuda"))
    model = YOLO(str(checkpoint), task="segment")
    settings = {**parent.read(parent.CONFIG)["training"], **config["training"]}
    if profile:
        settings.update(epochs=1, fraction=config["profile_fraction"])
    from ultralytics.data import utils as data_utils

    with patch.object(data_utils, "check_font", lambda *_: None):
        model.train(data=str(path), project=str(RUN), name=name, device=0, **settings)
    weights = RUN / name / "weights/last.pt"
    result = {**runtime_receipt(torch.device("cuda"), started), "identity": identities,
              "dataset_receipt_sha256": file_sha256(dataset_receipt), "settings": settings,
              "initial_checkpoint_sha256": file_sha256(checkpoint), "checkpoint_sha256": file_sha256(weights),
              "checkpoint": weights.relative_to(ROOT).as_posix(), "profile": profile,
              "selection": "last epoch EMA weights"}
    save_new_json(RUN / name / "training-receipt.json", result)
    print(json.dumps(result), flush=True)


def lift(mask, box, score, top, left, seam_margin):
    rows, cols = np.nonzero(mask)
    if not rows.size:
        return None
    r0, c0, r1, c1 = rows.min(), cols.min(), rows.max(), cols.max()
    if ((top > 0 and r0 < seam_margin) or (left > 0 and c0 < seam_margin)
            or (top + SIZE < NATIVE[0] and r1 >= SIZE - seam_margin)
            or (left + SIZE < NATIVE[1] and c1 >= SIZE - seam_margin)):
        return None
    native = np.zeros(NATIVE, dtype=np.uint8, order="F")
    native[top:top + SIZE, left:left + SIZE] = mask
    rle = coco.encode(native)
    shift = np.asarray((left, top, left, top), dtype=np.float32)
    return (score, rle, [float(v) for v in box + shift])


def deduplicate(candidates, threshold, cap):
    keep = []
    for candidate in sorted(candidates, key=lambda item: (-item[0], *item[2])):
        if keep and (coco.iou([candidate[1]], [item[1] for item in keep], [0] * len(keep)) >= threshold).any():
            continue
        keep.append(candidate)
        if len(keep) == cap:
            break
    return [{"score": float(score), "size": rle["size"], "counts": rle["counts"].decode("ascii"),
             "box_xyxy": box} for score, rle, box in keep]


def predict(fold, image_directory):
    YOLO, architecture = setup()
    config = parent.read(CONFIG)
    receipt = parent.read(RUN / "training/training-receipt.json")
    checkpoint = ROOT / receipt["checkpoint"]
    recorded = receipt["identity"] if fold == "test" else None
    if receipt["profile"] or receipt["identity"] != identity(architecture, recorded) or receipt["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("Native detector checkpoint changed")
    if fold == "test":
        if image_directory is None:
            raise ValueError("Specify the official test image directory")
        paths = image_paths(image_directory)
        ids = sorted(paths)
        inventory = ROOT / "artifacts/test-files.sha256.json"
        if len(ids) != 180 or parent.read(inventory) != {p.name: file_sha256(p) for p in paths.values()}:
            raise ValueError("Test images differ from official manifest")
    else:
        if image_directory is not None:
            raise ValueError("Validation always uses the frozen training image inventory")
        paths = image_paths(parent.DATA / "train/train_images")
        ids = parent.read(parent.SPLIT)[fold]
        inventory = ROOT / "artifacts/train-files.sha256.json"
        manifest = parent.read(inventory)
        if any(file_sha256(paths[stem]) != manifest[paths[stem].relative_to(parent.DATA).as_posix()] for stem in ids):
            raise ValueError("Validation images differ from official manifest")
    output, output_receipt = (RUN / f"predictions-{fold}{suffix}.json" for suffix in ("", "-receipt"))
    if output.exists() or output_receipt.exists():
        raise FileExistsError(output)
    model = YOLO(str(checkpoint), task="segment")
    torch.cuda.set_per_process_memory_fraction(0.79)
    started = profile_device(torch.device("cuda"))
    settings = parent.read(parent.CONFIG)["prediction"]
    predictions = {}
    for index, stem in enumerate(ids, 1):
        image = load_image(paths[stem])
        candidates = []
        for offset in range(0, len(TILES), 3):
            group = TILES[offset:offset + 3]
            inputs = [np.repeat(image[top:top + SIZE, left:left + SIZE, None], 3, axis=2) for top, left in group]
            for (top, left), result in zip(group, model.predict(source=inputs, device=0, verbose=False, **settings)):
                if result.masks is None:
                    continue
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                if masks.shape[1:] != (SIZE, SIZE) or not len(masks) == len(boxes) == len(scores):
                    raise ValueError("Native tile geometry differs from prediction")
                candidates.extend(candidate for mask, box, score in zip(masks, boxes, scores)
                                  if (candidate := lift(mask, box, score, top, left, config["seam_margin_px"])) is not None)
        predictions[stem] = deduplicate(candidates, config["native_mask_nms_iou"], settings["max_det"])
        if index == 1 or index % 20 == 0:
            print(f"{fold} {index}/{len(ids)}", flush=True)
    save_new_json(output, predictions)
    result = {**runtime_receipt(torch.device("cuda"), started), **receipt["identity"], "fold": fold,
              "checkpoint_sha256": receipt["checkpoint_sha256"], "prediction_sha256": file_sha256(output),
              "observations": len(predictions), "confidence_floor": settings["conf"], "nms_iou": settings["iou"],
              "inventory_sha256": file_sha256(inventory), "native_shape": list(NATIVE),
              "proposals": sum(map(len, predictions.values())), "tile_origins": TILES}
    save_new_json(output_receipt, result)
    print(json.dumps(result), flush=True)


def fixture():
    coverage = np.zeros(NATIVE, dtype=np.uint8)
    for top, left in TILES:
        coverage[top:top + SIZE, left:left + SIZE] += 1
    assert coverage.min() == 1 and coverage.max() == 4
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[31:42, 8:29] = 1
    box = np.array((8, 31, 29, 42), dtype=np.float32)
    candidate = lift(mask, box, .50, 512, 512, 4)
    decoded = coco.decode(candidate[1])
    assert decoded[543:554, 520:541].all() and decoded.sum() == 231
    assert candidate[2] == [520, 543, 541, 554]
    mask[:, :8] = 0
    mask[31:42, 2:8] = 1
    assert lift(mask, box, .5, 512, 512, 4) is None
    assert lift(mask, box, .5, 512, 0, 4) is not None
    rows = deduplicate([candidate, (.3, candidate[1], candidate[2])], .5, 100)
    assert len(rows) == 1 and rows[0]["score"] == .50
    polygon = np.array((780, 780, 900, 780, 900, 850, 780, 850), dtype=np.float64)
    whole, full = clipping(polygon, 512, 512, [None])
    assert full and np.allclose(whole[0], (polygon - 512) / SIZE)
    parts, full = clipping(polygon, 0, 800, [None])
    assert not full and parts and all(np.all((0 <= part) & (part <= 1)) for part in parts)
    print("Nine-tile coverage, translation, native COCO, seam rejection, NMS and polygon clipping passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "profile", "train", "predict", "fixture"))
    parser.add_argument("--fold", choices=("dev", "confirm", "site_holdout", "test"))
    parser.add_argument("--images", type=Path)
    args = parser.parse_args()
    if args.stage == "fixture":
        fixture()
    elif args.stage == "prepare":
        prepare()
    elif args.stage == "predict":
        if args.fold is None:
            parser.error("Specify the frozen prediction fold")
        predict(args.fold, args.images)
    else:
        train(args.stage == "profile")


if __name__ == "__main__":
    main()
