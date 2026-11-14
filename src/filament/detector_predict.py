"""Reproduce native YOLO11s proposals using the frozen scratch checkpoint."""

import argparse
import json
from importlib.metadata import version
import os
from pathlib import Path

import numpy as np
from pycocotools import mask as coco
import torch

from .data import file_sha256, image_paths, load_image, save_new_json
from .format import SHAPE
from .infer import profile_device, runtime_receipt


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode_proposals(result):
    if result.masks is None:
        return []
    masks = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    if masks.shape[1:] != SHAPE or len(masks) != len(boxes) or len(scores) != len(boxes):
        raise ValueError("Detector mask shape or box count differs from native geometry")
    rows = []
    for mask, box, score in zip(masks, boxes, scores):
        if np.any(mask):
            rle = coco.encode(np.asfortranarray(mask, dtype=np.uint8))
            rows.append({"score": float(score), "size": rle["size"], "counts": rle["counts"].decode("ascii"),
                         "box_xyxy": [float(value) for value in box]})
    return rows


def initialize(args):
    args.settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(args.settings_dir.resolve())
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_VERBOSE"] = "false"
    import ultralytics

    training = read(args.training_receipt)
    original = training["identity"]
    architecture = Path(ultralytics.__file__).parent / "cfg/models/11/yolo11-seg.yaml"
    actual = {
        "config_sha256": file_sha256(args.config),
        "split_sha256": file_sha256(args.split),
        "model_yaml_sha256": file_sha256(architecture),
        "source_sha256": file_sha256(args.training_source),
        "packages": {name: version(name) for name in original["packages"]},
    }
    if (training["profile"] or training["checkpoint_sha256"] != file_sha256(args.checkpoint)
            or any(original[key] != value for key, value in actual.items())):
        raise ValueError("Portable inference inputs differ from the scratch training receipt")
    settings = ultralytics.settings
    settings.update({key: False for key in (
        "sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune",
        "tensorboard", "wandb", "vscode",
    ) if key in settings})
    return ultralytics.YOLO(str(args.checkpoint.resolve()), task="segment"), training, read(args.config)["prediction"]


def input_paths(args):
    paths = image_paths(args.images)
    inventory = read(args.inventory)
    if args.reference is not None:
        if args.fold == "test":
            raise ValueError("Parity mode is limited to the frozen development data")
        reference = read(args.reference)
        expected = read(args.split)["dev"]
        if set(reference) != set(expected) or args.sample_count < 1:
            raise ValueError("Parity reference must cover the complete frozen development set")
        nonempty = [stem for stem in expected if reference[stem]]
        empty = [stem for stem in expected if not reference[stem]]
        ids = (nonempty[:args.sample_count - 1] + empty[:1]) if nonempty and empty and args.sample_count > 1 else (
            nonempty[:args.sample_count] if nonempty else empty[:args.sample_count]
        )
        ids += [stem for stem in expected if stem not in ids][:args.sample_count - len(ids)]
        if any(file_sha256(paths[stem]) != inventory[f"train/train_images/{paths[stem].name}"] for stem in ids):
            raise ValueError("Parity image differs from the official training manifest")
        return paths, ids, reference
    if args.fold != "test" or len(paths) != 180 or inventory != {p.name: file_sha256(p) for p in paths.values()}:
        raise ValueError("Expected the exact 180 official test JPEGs")
    return paths, sorted(paths), None


def predict(args):
    if args.output.exists() or args.receipt.exists() or args.output.resolve() == args.receipt.resolve():
        raise ValueError("Use two distinct new detector output paths")
    paths, ids, reference = input_paths(args)
    model, training, settings = initialize(args)
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(0.79)
    started = profile_device(device)
    predictions = {}
    for index, stem in enumerate(ids, 1):
        grayscale = load_image(paths[stem])
        image = np.repeat(grayscale[:, :, None], 3, axis=2)
        result, = model.predict(source=image, device=0, verbose=False, **settings)
        rows = encode_proposals(result)
        if reference is not None and rows != reference[stem]:
            raise ValueError(f"Portable detector differs from original development inference on {stem}")
        predictions[stem] = rows
        if index == 1 or index % 20 == 0:
            print(f"Detector {index}/{len(ids)}", flush=True)
    save_new_json(args.output, predictions)
    receipt = {**runtime_receipt(device, started), **training["identity"],
               "adapter_source_sha256": file_sha256(Path(__file__)), "checkpoint_sha256": training["checkpoint_sha256"],
               "prediction_sha256": file_sha256(args.output), "inventory_sha256": file_sha256(args.inventory),
               "fold": args.fold, "observations": len(predictions), "confidence_floor": settings["conf"],
               "nms_iou": settings["iou"], "native_shape": list(SHAPE), "proposals": sum(map(len, predictions.values()))}
    if reference is not None:
        receipt.update(reference_sha256=file_sha256(args.reference), exact_reference_parity=True,
                       observation_ids=ids)
    save_new_json(args.receipt, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "inventory", "config", "split", "training-receipt", "training-source",
                 "checkpoint", "settings-dir", "output", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--fold", choices=("dev", "test"), default="test")
    parser.add_argument("--reference", type=Path, help="Original development cache for exact subset parity")
    parser.add_argument("--sample-count", type=int, default=4)
    print(json.dumps(predict(parser.parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()
