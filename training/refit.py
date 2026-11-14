"""Rebuild the fixed 707-disk filament detector and semantic checkpoints."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import psutil
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train-files.sha256.json"
CONFIG = HERE / "config.json"
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import detector
from filament.data import annotation_path, file_sha256, image_paths, load_image, read_annotations, save_new_json
from filament.format import SHAPE
from filament.infer import profile_device, runtime_receipt
from filament.model import FilamentUNet
from filament.train import prepare_observation, train_epoch

scratch, native = detector.scratch, detector.native
scratch.CONFIG = REPO / "artifacts/detector-config.json"
native.CONFIG = REPO / "artifacts/detector-native-config.json"
GIB = 2**30


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def train_inventory(data):
    paths = image_paths(data / "train/train_images")
    manifest = read(TRAIN)
    if len(paths) != 707 or {p.relative_to(data).as_posix() for p in paths.values()} != {
            n for n in manifest if n.startswith("train/train_images/")}:
        raise ValueError("All and only 707 official fit JPEGs are required")
    return dict(sorted(paths.items())), manifest


def view_index(data):
    document = read(annotation_path(data))
    views, labels = defaultdict(list), defaultdict(list)
    for row in document["images"]:
        stem = Path(row["file_name"]).stem
        if (row["height"], row["width"]) != SHAPE or row["id"].split("-", 1)[1] != stem:
            raise ValueError("Bad official annotator view")
        views[stem].append(row["id"])
    for row in document["annotations"]:
        polygon, = row["segmentation"]
        if row["iscrowd"] or len(polygon) < 6 or len(polygon) % 2 or any(
                not math.isfinite(v) or not 0 <= v <= 2048 for v in polygon):
            raise ValueError("Unsupported official polygon")
        labels[row["image_id"]].append(polygon)
    all_views = [v for disk in views.values() for v in disk]
    if len(all_views) != 1154 or len(set(all_views)) != 1154 or set(labels) - set(all_views):
        raise ValueError("Official annotator inventory changed")
    if len(document["annotations"]) != 8199:
        raise ValueError("Official annotation count changed")
    return views, labels


def create_labels(path, polygons):
    with path.open("x", encoding="ascii", newline="\n") as stream:
        for polygon in polygons:
            stream.write("0 " + " ".join(f"{v:.10f}" for v in polygon) + "\n")


def fit_semantic(data, folder, device):
    cfg = read(REPO / "artifacts/semantic-config.json")
    torch.set_num_threads(min(6, cfg["workers"]))
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    observations = read_annotations(data)
    ids = sorted(observations)
    model = FilamentUNet(cfg["channels"]).to(device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=19 * 295, eta_min=cfg["learning_rate"] * .08)
    report = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        records = dict(zip(ids, pool.map(prepare_observation, (observations[x] for x in ids)), strict=True))
        preload = time.perf_counter() - start
        if len(records) != 707 or len(records) * 5 % 12 != 7:
            raise ValueError("Exact semantic sample scope changed")
        for epoch in range(1, 20):
            loss, batches, seconds = train_epoch(model, opt, sched, records, cfg, device, epoch, pool)
            if not math.isfinite(loss):
                raise ValueError("Nonfinite exact semantic loss")
            report.append({"epoch": epoch, "batches": batches, "mean_loss": loss, "seconds": seconds})
            print(json.dumps(report[-1]), flush=True)
    checkpoint = folder / "model.pt"
    if sum(row["batches"] for row in report) != 5605 or len(report) != 19:
        raise ValueError("Exact semantic final schedule changed")
    with checkpoint.open("xb") as handle:
        torch.save(model.state_dict(), handle)
    return {"checkpoint": checkpoint, "observed_batches": sum(r["batches"] for r in report),
            "optimizer_updates": sum(r["batches"] for r in report), "epoch_history": report,
            "validated_samples": 0, "validator_calls": 0, "best_checkpoint": False,
            "preload_seconds": preload,
            "effective_settings": cfg, "physical_training_disks": len(ids)}


def prepare(data, top, materialize):
    paths, inventory = train_inventory(data)
    views, labels = view_index(data)
    annotation_key = annotation_path(data).relative_to(data).as_posix()
    expected_input = {annotation_key, *(path.relative_to(data).as_posix() for path in paths.values())}
    if set(inventory) != expected_input or file_sha256(annotation_path(data)) != inventory[annotation_key]:
        raise ValueError("Exact competition train annotation/inventory differs")
    if set(paths) != set(views):
        raise ValueError("Official training images and views disagree")
    for component in ("scratch", "native"):
        for kind in ("images", "labels"):
            (top / component / kind / "train").mkdir(parents=True)
    cv2.setNumThreads(1)
    seed = scratch.read(scratch.CONFIG)["view_seed"]

    def collect(stem):
        path, source_views = paths[stem], sorted(views[stem])
        digest = sha256(f"{seed} {stem}".encode("ascii")).digest()
        view = source_views[int.from_bytes(digest[:8], "big") % len(source_views)]
        jpeg = file_sha256(path)
        if jpeg != inventory[path.relative_to(data).as_posix()]:
            raise ValueError(f"Official JPEG changed: {stem}")
        target = top / "scratch/images/train" / path.name
        if materialize:
            os.link(path, target)
        source_label = top / "scratch/labels/train" / f"{stem}.txt"
        create_labels(source_label, (np.asarray(poly) / 2048 for poly in labels[view]))
        polygons = [np.fromstring(row[2:], sep=" ", dtype=np.float64) * 2048
                    for row in source_label.read_text(encoding="ascii").splitlines()]
        caches = [[None] for _ in polygons]
        gray, tiles = load_image(path), {}
        for y, x in native.TILES:
            name = f"{stem}-{y:04d}-{x:04d}"
            label = top / "native/labels/train" / f"{name}.txt"
            sample = gray[y:y + 1024, x:x + 1024]
            ok, encoded = cv2.imencode(".png", sample, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise ValueError(f"Could not encode native tile {name}")
            png = encoded.tobytes()
            if materialize:
                (top / "native/images/train" / f"{name}.png").write_bytes(png)
            clipped = []
            for polygon, cache in zip(polygons, caches, strict=True):
                parts, _ = native.clipping(polygon, y, x, cache)
                clipped.extend(parts)
            create_labels(label, clipped)
            tiles[name] = {"pixels_sha256": sha256(png).hexdigest(), "label_sha256": file_sha256(label),
                           "instances": len(clipped)}
        return {"detector_view": view, "all_views": source_views, "detector_polygons": len(polygons),
                "semantic_annotation_count": sum(len(labels[v]) for v in source_views),
                "official_jpeg_sha256": jpeg, "full_label_sha256": file_sha256(source_label),
                "native_tiles": tiles}

    with ThreadPoolExecutor(max_workers=3) as workers:
        records = dict(zip(paths, workers.map(collect, paths), strict=True))
    if (sum(row["detector_polygons"] for row in records.values()) != 5191
            or sum(len(row["native_tiles"]) for row in records.values()) != 6363
            or sum(not tile["instances"] for row in records.values() for tile in row["native_tiles"].values()) != 1013):
        raise ValueError("Exact detector dataset counts changed")
    for component in ("scratch", "native"):
        stage = top / component
        with (stage / "data.yaml").open("x", encoding="utf-8", newline="\n") as stream:
            yaml.safe_dump({"path": stage.as_posix(), "train": "images/train", "val": "images/train",
                            "names": {0: "filament"}}, stream)
    return records


def recipe():
    release = read(REPO / "artifacts/release.json")
    for name, wanted in release["packages"].items():
        if version(name) != wanted:
            raise ValueError(f"Training environment differs for {name}")
    for name in ("artifacts/detector-config.json", "artifacts/detector-native-config.json",
                 "artifacts/semantic-config.json", "scripts/yolo11s_scratch.py", "scripts/yolo11s_native_tiles.py"):
        if file_sha256(REPO / name) != release["artifact_sha256"][name]:
            raise ValueError(f"Base release file changed: {name}")
    budgets = read(CONFIG)["budgets"]
    stages = {}
    for name, size in (("scratch", 707), ("native", 6363)):
        settings = read(scratch.CONFIG)["training"]
        if name == "native":
            settings.update(read(native.CONFIG)["training"])
        settings["val"] = False
        budget = budgets[name]
        args = (size, settings["epochs"], settings.get("warmup_epochs", 3.0))
        if detector.steps_expected(*args) != (budget["batches_per_epoch"], budget["optimizer_updates"]):
            raise ValueError(f"Mismatch in the {name} LR/accumulation schedule")
        stages[name] = settings
    stages["semantic"] = read(REPO / "artifacts/semantic-config.json")
    assert (stages["semantic"]["train_batch"], stages["semantic"]["epochs"],
            stages["semantic"]["tiles_per_observation"]) == (12, 19, 5)
    return release, budgets, stages


def run(data, output, dry_run):
    if output.exists():
        raise FileExistsError("Training output must be unused")
    if not dry_run:
        if not torch.cuda.is_available() or psutil.virtual_memory().available < 12 * GIB:
            raise RuntimeError("CUDA and at least 12 GiB of available host RAM are needed")
        if output.anchor.lower() != data.anchor.lower():
            raise ValueError("Put output on the data volume so original training JPEGs can be hardlinked")
        if shutil.disk_usage(data).free < 5 * GIB:
            raise ValueError("At least 5 GiB of free space is needed to generate the native tiles")
    _, budgets, settings = recipe()
    output.mkdir(parents=True)
    records = prepare(data, output / "data", materialize=not dry_run)
    save_new_json(output / "dataset.json", {"membership": records, "budgets": budgets, "settings": settings,
                  "train_only": True, "train_inventory_sha256": file_sha256(TRAIN), "annotation_views": 1154,
                  "semantic_annotations": 8199, "detector_polygons": 5191, "native_tiles": 6363,
                  "training_disks": 707, "dry_run": dry_run})
    print(f"Verified {len(records)} train disks and 6363 native tiles", flush=True)
    if dry_run:
        return
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "1"
    env["CAMPAIGN_CPU_WORKERS"] = "6"
    for component in ("scratch", "native", "semantic"):
        subprocess.run([sys.executable, "-B", __file__, "--data", str(data), "--output", str(output),
                        "--component", component], cwd=REPO, env=env, check=True)


def fit_component(data, output, component):
    dataset = read(output / "dataset.json")
    release, budgets, settings = recipe()
    if dataset["dry_run"] or dataset["budgets"] != budgets or dataset["settings"] != settings:
        raise ValueError("Actual fit must follow the full prepared dataset and the fixed recipe")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_per_process_memory_fraction(min(.75, 12 * GIB / torch.cuda.get_device_properties(device).total_memory),
                                               device.index)
    stage = output / component
    stage.mkdir()
    begin = profile_device(device)
    parent = None
    if component == "semantic":
        data_item = fit_semantic(data, stage, device)
    else:
        if component == "native":
            prior = read(output / "scratch/training.json")
            parent = Path(prior["checkpoint"])
            if file_sha256(parent) != prior["checkpoint_sha256"]:
                raise ValueError("Own final scratch parent changed")
        data_item = detector.fit_detector(component, output / "data" / component / "data.yaml", stage, parent,
                                          budgets[component], device, time.perf_counter)
        architecture = file_sha256(data_item.pop("architecture"))
        if architecture != release["model_yaml_sha256"]:
            raise ValueError("Installed architecture differs from the release")
        data_item["model_yaml_sha256"] = architecture
    checkpoint = data_item.pop("checkpoint")
    data_item.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                     parent_checkpoint_sha256=file_sha256(parent) if parent else None,
                     **runtime_receipt(device, begin))
    if data_item["effective_settings"] != settings[component]:
        raise ValueError(f"Effective settings changed for {component}")
    save_new_json(stage / "training.json", data_item)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Extracted official competition dataset root")
    parser.add_argument("--output", required=True, type=Path, help="An unused same-volume training output")
    parser.add_argument("--dry-run", action="store_true", help="Verify official data, labels, pixels and schedules on CPU")
    parser.add_argument("--component", choices=("scratch", "native", "semantic"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    if args.component:
        if args.dry_run:
            parser.error("A single fit stage cannot run as a dry run")
        fit_component(data, output, args.component)
        return
    run(data, output, args.dry_run)


if __name__ == "__main__":
    main()
