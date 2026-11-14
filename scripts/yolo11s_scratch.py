"""Fit one scratch YOLO11s segmentation detector on the official grouped data."""

import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from pycocotools import mask as coco
import torch

from filament.data import DATASET_DIR, annotation_path, file_sha256, image_paths, load_image, save_new_json
from filament.infer import profile_device, runtime_receipt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/yolo11s-scratch"
DATA = ROOT / "data" / DATASET_DIR
SPLIT = ROOT / "runs/higher-data-refit-plan/split.json"
CONFIG = ROOT / ".local/yolo11s-scratch-config.json"
SOURCE = Path(__file__).resolve()
NATIVE = (2048, 2048)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def setup():
    (RUN / "settings").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(RUN / "settings")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_VERBOSE"] = "false"
    from ultralytics import YOLO, settings

    settings.update({key: False for key in (
        "sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune",
        "tensorboard", "wandb", "vscode",
    ) if key in settings})
    model_yaml = Path(__import__("ultralytics").__file__).parent / "cfg/models/11/yolo11-seg.yaml"
    return YOLO, model_yaml


def identity(model_yaml):
    return {"config_sha256": file_sha256(CONFIG), "split_sha256": file_sha256(SPLIT),
            "model_yaml_sha256": file_sha256(model_yaml), "source_sha256": file_sha256(SOURCE),
            "annotation_sha256": file_sha256(annotation_path(DATA)),
            "packages": {name: version(name) for name in (
                "torch", "torchvision", "ultralytics", "opencv-python", "numpy", "pycocotools",
            )}}


def prepare():
    from yaml import safe_dump

    split = read(SPLIT)
    config = read(CONFIG)
    doc = read(annotation_path(DATA))
    if file_sha256(annotation_path(DATA)) != split["annotation_sha256"]:
        raise ValueError("Competition annotation identity changed")
    folds = ("train", "dev", "confirm", "site_holdout", "purged")
    expected = (419, 32, 80, 101, 75)
    if tuple(len(split[fold]) for fold in folds) != expected:
        raise ValueError("Expected the exact 419-disk refit split")
    if len({stem for fold in folds for stem in split[fold]}) != sum(expected):
        raise ValueError("Physical images overlap across split")
    official = image_paths(DATA / "train/train_images")
    records = defaultdict(list)
    labels = defaultdict(list)
    for row in doc["images"]:
        stem = Path(row["file_name"]).stem
        if (row["height"], row["width"]) != NATIVE or row["id"].split("-", 1)[-1] != stem:
            raise ValueError(f"Malformed annotator record {row['id']}")
        records[stem].append(row["id"])
    for row in doc["annotations"]:
        polygon, = row["segmentation"]
        if (row["iscrowd"] or len(polygon) < 6 or len(polygon) % 2
                or any(not math.isfinite(point) or not 0 <= point <= 2048 for point in polygon)):
            raise ValueError(f"Unsupported COCO polygon {row['id']}")
        labels[row["image_id"]].append("0 " + " ".join(f"{point / 2048:.10f}" for point in polygon))
    view_index = {view for views in records.values() for view in views}
    if set(records) != set(official) or not set(labels).issubset(view_index):
        raise ValueError("JPEG and COCO view identities differ")

    dataset = RUN / "dataset"
    dataset.mkdir(parents=True)
    manifest = {}
    train_manifest = read(ROOT / "artifacts/train-files.sha256.json")
    for fold in ("train", "dev"):
        name = "train" if fold == "train" else "val"
        images, output = (dataset / sub / name for sub in ("images", "labels"))
        images.mkdir(parents=True)
        output.mkdir(parents=True)
        manifest[fold] = {}
        for stem in split[fold]:
            views = sorted(records[stem])
            choice = hashlib.sha256(f"{config['view_seed']} {stem}".encode("ascii")).digest()
            view = views[int.from_bytes(choice[:8], "big") % len(views)]
            path = official[stem]
            source_sha = file_sha256(path)
            if train_manifest[path.relative_to(DATA).as_posix()] != source_sha:
                raise ValueError(f"Changed official training JPEG {stem}")
            os.link(path, images / path.name)
            rows = labels[view]
            text_path = output / (stem + ".txt")
            text_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="ascii", newline="\n")
            manifest[fold][stem] = {"view": view, "polygons": len(rows), "image_sha256": source_sha,
                                    "label_sha256": file_sha256(text_path)}
    specification = {"path": dataset.as_posix(), "train": "images/train", "val": "images/val",
                     "names": {0: "filament"}}
    yaml = dataset / "data.yaml"
    yaml.write_text(safe_dump(specification), encoding="utf-8")
    save_new_json(dataset / "dataset-receipt.json", {
        "config_sha256": file_sha256(CONFIG), "split_sha256": file_sha256(SPLIT),
        "annotation_sha256": file_sha256(annotation_path(DATA)), "yaml_sha256": file_sha256(yaml),
        "membership": manifest,
    })
    print(json.dumps({fold: {"images": len(rows), "polygons": sum(row["polygons"] for row in rows.values())}
                      for fold, rows in manifest.items()}), flush=True)


def verify_dataset():
    dataset = RUN / "dataset"
    receipt = read(dataset / "dataset-receipt.json")
    yaml = dataset / "data.yaml"
    if (receipt["yaml_sha256"] != file_sha256(yaml) or receipt["config_sha256"] != file_sha256(CONFIG)
            or receipt["split_sha256"] != file_sha256(SPLIT)
            or receipt["annotation_sha256"] != file_sha256(annotation_path(DATA))):
        raise ValueError("Prepared dataset identity changed")
    split = read(SPLIT)
    for fold, rows in receipt["membership"].items():
        sub = "train" if fold == "train" else "val"
        images, labels = dataset / "images" / sub, dataset / "labels" / sub
        if set(rows) != set(split[fold]) or {p.stem for p in images.glob("*.jpeg")} != set(rows):
            raise ValueError("Prepared data membership differs from the frozen split")
        for stem, row in rows.items():
            if (file_sha256(images / (stem + ".jpeg")) != row["image_sha256"]
                    or file_sha256(labels / (stem + ".txt")) != row["label_sha256"]):
                raise ValueError(f"Prepared image or label changed for {stem}")
    return yaml, dataset / "dataset-receipt.json"


def train(profile):
    YOLO, model_yaml = setup()
    yaml, dataset_receipt = verify_dataset()
    config = read(CONFIG)
    identities = identity(model_yaml)
    name = "profile" if profile else "training"
    if (RUN / name).exists():
        raise FileExistsError(RUN / name)
    if not profile:
        check = read(RUN / "profile/training-receipt.json")
        if check["identity"] != identities or check["cuda_peak_reserved_bytes"] > 17 * 2**30:
            raise ValueError("Matching CUDA profile under 17 GiB required")
    torch.cuda.set_per_process_memory_fraction(0.79)
    started = profile_device(torch.device("cuda"))
    torch.manual_seed(config["training"]["seed"])
    model = YOLO("yolo11s-seg.yaml", task="segment", verbose=False)
    if model.ckpt or model.ckpt_path is not None or model.model.yaml.get("scale") != "s":
        raise ValueError("Expected the randomly initialized YOLO11s segmentation architecture")
    settings = dict(config["training"])
    if profile:
        settings.update(epochs=1, fraction=config["profile_fraction"])
    from ultralytics.data import utils as data_utils

    # Stock AMP preflight loads public weights and a sample image. Use full precision.
    with patch.object(data_utils, "check_font", lambda *_: None):
        model.train(data=str(yaml), project=str(RUN), name=name, device=0, **settings)
    weights = RUN / name / "weights/last.pt"
    result = {**runtime_receipt(torch.device("cuda"), started), "identity": identities,
              "dataset_receipt_sha256": file_sha256(dataset_receipt), "settings": settings,
              "checkpoint": weights.relative_to(ROOT).as_posix(), "checkpoint_sha256": file_sha256(weights),
              "selection": "last epoch EMA weights", "profile": profile}
    save_new_json(RUN / name / "training-receipt.json", result)
    print(json.dumps(result), flush=True)


def candidates(result):
    if result.masks is None:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    if masks.shape[1:] != NATIVE or len(masks) != len(boxes) or len(scores) != len(boxes):
        raise ValueError("Retina masks must align with the original native image")
    rows = []
    for mask, box, score in zip(masks, boxes, scores):
        if not np.any(mask):
            continue
        rle = coco.encode(np.asfortranarray(mask, dtype=np.uint8))
        rows.append({"score": float(score), "size": rle["size"], "counts": rle["counts"].decode("ascii"),
                     "box_xyxy": [float(value) for value in box]})
    return rows


def predict(fold, image_directory=None):
    YOLO, model_yaml = setup()
    training = read(RUN / "training/training-receipt.json")
    config = read(CONFIG)
    checkpoint = ROOT / training["checkpoint"]
    if training["profile"] or training["identity"] != identity(model_yaml) or training["checkpoint_sha256"] != file_sha256(checkpoint):
        raise ValueError("Checkpoint does not match frozen scratch run")
    output = RUN / f"predictions-{fold}.json"
    receipt_path = RUN / f"predictions-{fold}-receipt.json"
    if output.exists() or receipt_path.exists():
        raise FileExistsError(output)
    if fold == "test":
        if image_directory is None:
            raise ValueError("Specify the official test image directory")
        paths = image_paths(image_directory)
        inventory = ROOT / "artifacts/test-files.sha256.json"
        if len(paths) != 180 or read(inventory) != {p.name: file_sha256(p) for p in paths.values()}:
            raise ValueError("Detector input differs from the official test manifest")
        ids = sorted(paths)
    else:
        if image_directory is not None:
            raise ValueError("Evaluation predictions use the frozen official training directory")
        paths = image_paths(DATA / "train/train_images")
        ids = read(SPLIT)[fold]
        inventory = ROOT / "artifacts/train-files.sha256.json"
        expected = read(inventory)
        if any(file_sha256(paths[stem]) != expected[paths[stem].relative_to(DATA).as_posix()] for stem in ids):
            raise ValueError("Evaluation images differ from the official manifest")
    model = YOLO(str(checkpoint), task="segment")
    torch.cuda.set_per_process_memory_fraction(0.79)
    started = profile_device(torch.device("cuda"))
    settings = config["prediction"]
    predictions = {}
    for index, stem in enumerate(ids, 1):
        image = load_image(paths[stem])
        source = np.repeat(image[:, :, None], 3, axis=2)
        result, = model.predict(source=source, device=0, verbose=False, **settings)
        predictions[stem] = candidates(result)
        if index == 1 or index % 20 == 0:
            print(f"{fold} {index}/{len(ids)}", flush=True)
    save_new_json(output, predictions)
    receipt = {**runtime_receipt(torch.device("cuda"), started), **training["identity"],
               "checkpoint_sha256": training["checkpoint_sha256"], "prediction_sha256": file_sha256(output),
               "fold": fold, "observations": len(predictions), "confidence_floor": settings["conf"],
               "inventory_sha256": file_sha256(inventory),
               "nms_iou": settings["iou"], "native_shape": list(NATIVE),
               "proposals": sum(map(len, predictions.values()))}
    save_new_json(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "profile", "train", "predict"))
    parser.add_argument("--fold", choices=("dev", "confirm", "site_holdout", "test"))
    parser.add_argument("--images", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "predict":
        if args.fold is None:
            parser.error("Specify the frozen prediction fold")
        predict(args.fold, args.images)
    else:
        train(args.stage == "profile")


if __name__ == "__main__":
    main()
