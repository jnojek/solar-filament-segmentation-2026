"""Rebuild the selected native detector and three inner-OOF prompt donors."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import psutil
import shutil
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train-files.sha256.json"
COHORT = HERE / "fitting-ids.json"
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

import detector
from filament import box_prompt_r18 as refiner
from filament.data import annotation_path, file_sha256, image_paths, load_image, save_new_json
from filament.detector_native_predict import native_proposals
from filament.format import SHAPE

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


def membership():
    plan = read(COHORT)
    train = sorted(plan["train"])
    groups = plan["group_by_id"]
    seen = set()
    if len(train) != 707 or len(set(train)) != 707 or set(train) != set(groups):
        raise ValueError("Exact 707 fitting IDs required")
    times = {s: datetime.strptime(s[:14], "%Y%m%d%H%M%S") for s in train}
    for index, fold in enumerate(plan["inner_folds"]):
        donor, held = set(fold["train"]), set(fold["oof"])
        if (fold["fold"] != index or donor & held or seen & held
                or donor | held | set(fold["purged"]) != set(train)
                or {groups[s] for s in donor} & {groups[s] for s in held}
                or min(abs((times[a]-times[b]).total_seconds()) for a in donor for b in held) < 48*3600):
            raise ValueError("An inner donor has an ineligible training or inference observation")
        seen.update(held)
    if seen != set(train):
        raise ValueError("Inner OOF coverage differs from fitting scope")
    return plan


def create_dataset(data, output, materialize):
    original, inventory = train_inventory(data)
    annotation = annotation_path(data)
    if file_sha256(annotation) != inventory[annotation.relative_to(data).as_posix()]:
        raise ValueError("Exact official annotations changed")
    plan = membership()
    stems = plan["train"]
    files = output / "pixels"
    rows = {}
    for part in ("scratch", "native"):
        for kind in ("images", "labels"):
            (files / part / kind / "train").mkdir(parents=True)
    cv2.setNumThreads(1)
    views, polygons = view_index(data)
    seed = read(scratch.CONFIG)["view_seed"]

    def prepare(stem):
        path = original[stem]
        jpg_hash = file_sha256(path)
        if jpg_hash != inventory[path.relative_to(data).as_posix()]:
            raise ValueError("One supplied training JPEG differs")
        source_views = sorted(views[stem])
        digest = hashlib.sha256(f"{seed} {stem}".encode("ascii")).digest()
        name = source_views[int.from_bytes(digest[:8], "big") % len(source_views)]
        full = files / "scratch/labels/train" / f"{stem}.txt"
        create_labels(full, (np.asarray(item) / 2048 for item in polygons[name]))
        if materialize:
            os.link(path, files / "scratch/images/train" / path.name)
        source = [np.fromstring(line[2:], sep=" ", dtype=np.float64) * 2048
                  for line in full.read_text(encoding="ascii").splitlines()]
        caches = [[None] for _ in source]
        gray = load_image(path)
        tiles = {}
        for top, left in native.TILES:
            key = f"{stem}-{top:04d}-{left:04d}"
            tile = gray[top:top+1024,left:left+1024]
            success, compressed = cv2.imencode(".png", tile, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not success:
                raise ValueError(f"Native compression failed for {key}")
            pixels = compressed.tobytes()
            label = files / "native/labels/train" / f"{key}.txt"
            clipped = []
            for contour, cache in zip(source, caches, strict=True):
                parts, _ = native.clipping(contour, top, left, cache)
                clipped.extend(parts)
            create_labels(label, clipped)
            if materialize:
                (files / "native/images/train" / f"{key}.png").write_bytes(pixels)
            tiles[key] = {"pixels_sha256": hashlib.sha256(pixels).hexdigest(),
                          "label_sha256": file_sha256(label), "instances": len(clipped)}
        return {"detector_view": name, "all_views": source_views, "detector_polygons": len(source),
                "semantic_annotation_count": sum(len(polygons[v]) for v in source_views),
                "official_jpeg_sha256": jpg_hash, "full_label_sha256": file_sha256(full), "native_tiles": tiles}

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = dict(zip(stems, pool.map(prepare, stems), strict=True))
    if materialize:
        training = {"control": stems, **{f"donor{d['fold']}": d["train"] for d in plan["inner_folds"]}}
        for kind, ids in training.items():
            for component in ("scratch", "native"):
                path = output / kind / "data" / component
                for sub in ("images", "labels"):
                    (path / sub / "train").mkdir(parents=True)
                for stem in ids:
                    chunks = (stem,) if component == "scratch" else rows[stem]["native_tiles"]
                    extension = "jpeg" if component == "scratch" else "png"
                    for key in chunks:
                        for directory, suffix in (("images", extension), ("labels", "txt")):
                            relative = Path(directory) / "train" / f"{key}.{suffix}"
                            os.link(files / component / relative, path / relative)
                with (path / "data.yaml").open("x", encoding="utf-8", newline="\n") as out:
                    yaml.safe_dump({"path": path.as_posix(), "train": "images/train", "val": "images/train",
                                    "names": {0: "filament"}}, out)
    return rows


def fit_detector_group(data, output, group, component):
    plan = membership()
    train = plan["train"] if group == "control" else plan["inner_folds"][int(group[-1])]["train"]
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_per_process_memory_fraction(min(.75,12*GIB/torch.cuda.get_device_properties(device).total_memory),device.index)
    directory = output / group
    stage = directory / component
    count = len(train) * (1 if component == "scratch" else 9)
    epochs = 36 if component == "scratch" else 8
    batches, updates = detector.steps_expected(count, epochs, 3.0 if component == "scratch" else .5)
    budget = {"epochs": epochs, "batches_per_epoch": batches, "optimizer_updates": updates, "tail": count % 4 or 4}
    stage.mkdir()
    donor = None
    if component == "native":
        prior = read(directory / "scratch/training.json")
        donor = Path(prior["checkpoint"])
        if file_sha256(donor) != prior["checkpoint_sha256"]:
            raise ValueError("Own scratch EMA was changed")
    receipt = detector.fit_detector(component,directory/"data"/component/"data.yaml",stage,donor,budget,
                                    device,False,time.perf_counter)
    weight = receipt.pop("checkpoint")
    architecture = file_sha256(receipt.pop("architecture"))
    if architecture != read(REPO / "artifacts/release.json")["model_yaml_sha256"]:
        raise ValueError("Installed architecture differs")
    save_new_json(stage/"training.json",{**receipt,"checkpoint":str(weight),"checkpoint_sha256":file_sha256(weight),
                  "parent_sha256":file_sha256(donor) if donor else None,"training_disks":len(train),
                  "model_yaml_sha256":architecture})


def infer_oof(data, output, group):
    fold = membership()["inner_folds"][int(group[-1])]
    target = output / group
    previous = read(target / "native/training.json")
    weights = Path(previous["checkpoint"])
    if (file_sha256(weights) != previous["checkpoint_sha256"]
            or previous["training_disks"] != len(fold["train"])):
        raise ValueError("Wrong inner donor checkpoint")
    with patch.object(scratch,"RUN",target/"inference"):
        YOLO, yaml_path = scratch.setup()
    if file_sha256(yaml_path) != previous["model_yaml_sha256"]:
        raise ValueError("Different YOLO configuration")
    model = YOLO(str(weights), task="segment")
    original, _ = train_inventory(data)
    cfg = read(native.CONFIG)
    options = read(scratch.CONFIG)["prediction"]
    rows = {stem:native_proposals(model,load_image(original[stem]),cfg,options,native) for stem in fold["oof"]}
    save_new_json(target/"predictions-oof.json",rows)
    save_new_json(target/"predictions-oof-receipt.json",{"fit_body":group,"checkpoint_sha256":file_sha256(weights),
                "held_count":len(rows),"sha256":file_sha256(target/"predictions-oof.json")})
def fit_refiner(data, output, asset):
    original, _ = train_inventory(data)
    stems = membership()["train"]
    stem_set = set(stems)
    from filament.data import polygon_rle

    official = read(annotation_path(data))
    by_view = {row["id"]:Path(row["file_name"]).stem for row in official["images"]
               if Path(row["file_name"]).stem in stem_set}
    labels = {view: [] for view in by_view}
    views = defaultdict(dict)
    for row in official["annotations"]:
        if row["image_id"] in labels:
            labels[row["image_id"]].append(polygon_rle(row["segmentation"], SHAPE))
    for view, stem in sorted(by_view.items()):
        views[stem][view] = labels[view]
    prompts = {}
    for fold in membership()["inner_folds"]:
        donor = output / f"donor{fold['fold']}"
        oof = donor/"predictions-oof.json"
        rows = read(oof)
        receipt = read(donor/"predictions-oof-receipt.json")
        if (receipt["sha256"] != file_sha256(oof) or set(rows) != set(fold["oof"])
                or set(rows) & set(prompts)
                or any("box_xyxy" not in row for predictions in rows.values() for row in predictions)):
            raise ValueError("Wrong or duplicated inner-held proposals")
        prompts.update(rows)
    plans = {}
    for stem in stems:
        positives, negatives = refiner.sample_plans(prompts[stem], views[stem], threshold=.10)
        plans[stem] = {"positive": [vars(x) for x in positives], "negative": [vars(x) for x in negatives]}
    if len(by_view) != 1154 or sum(len(row["positive"]) for row in plans.values()) != 8199:
        raise ValueError("Exact fit 707 plans differ")
    run = output/"refiner"
    run.mkdir()
    save_new_json(run/"crop-plans.json",plans)
    cfg = read(HERE/"refiner-config.json")
    torch.manual_seed(refiner.SEED)
    device = torch.device("cuda",torch.cuda.current_device())
    torch.cuda.set_per_process_memory_fraction(min(.75,12*GIB/torch.cuda.get_device_properties(device).total_memory),device.index)
    torch.backends.cudnn.benchmark = False
    model = refiner.open_refiner(asset).to(device,memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda",init_scale=1024)
    iterations = 0
    examples = 0
    history = []

    def prep(item):
        stem, plan_dict, seed = item
        plan = refiner.Crop(**plan_dict)
        target = None if plan.view is None else views[stem][plan.view][plan.truth_index]
        return refiner.make_training_example(images[stem],plan,target,np.random.default_rng(seed))

    with ThreadPoolExecutor(max_workers=6) as pool:
        images = dict(zip(stems,pool.map(load_image,(original[stem] for stem in stems)),strict=True))
        for epoch in range(cfg["epochs"]):
            items = schedule(plans,views,epoch)
            losses = []
            for start in range(0,len(items),cfg["batch"]):
                mini = items[start:start+cfg["batch"]]
                task = ((stem,plan,104+epoch*100000+start+j) for j,(stem,plan) in enumerate(mini))
                feature,target = zip(*pool.map(prep,task))
                x = torch.from_numpy(np.stack(feature)).to(device,memory_format=torch.channels_last)
                y = torch.from_numpy(np.stack(target)).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda"):
                    loss = refiner.bce_dice(model(x),y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad)):
                    raise ValueError("Nonfinite refiner loss or gradient")
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
                iterations += 1
                examples += len(mini)
            scheduler.step()
            row = {"epoch": epoch+1,"cumulative_batches":iterations,"cumulative_samples":examples,"mean_loss":float(np.mean(losses))}
            history.append(row)
            print(json.dumps(row),flush=True)
    weight = run/"final.pt"
    with weight.open("xb") as file:
        torch.save(model.state_dict(),file)
    save_new_json(run/"training.json",{"epochs":len(history),"batches":iterations,"samples":examples,
        "history":history,"checkpoint_sha256":file_sha256(weight),"generic_initialization_sha256":file_sha256(asset),
        "fit_disks":len(stems),"crop_plans_sha256":file_sha256(run/"crop-plans.json")})


def schedule(records, views, epoch):
    import random

    rng = random.Random(refiner.SEED+epoch)
    items = []
    for stem in sorted(records):
        view = rng.choice(tuple(views[stem]))
        items.extend((stem,x) for x in records[stem]["positive"] if x["view"] == view)
        items.extend((stem,x) for x in records[stem]["negative"])
    rng.shuffle(items)
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--resnet18-imagenet",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--internal",nargs="+",help=argparse.SUPPRESS)
    args = parser.parse_args()
    data, output = args.data.resolve(),args.output.resolve()
    if args.internal:
        op,*rest = args.internal
        if op == "detector":
            fit_detector_group(data,output,*rest)
        elif op == "oof":
            infer_oof(data,output,*rest)
        elif op == "refiner":
            fit_refiner(data,output,args.resnet18_imagenet.resolve())
        else:
            parser.error("Unknown internal stage")
        return
    if output.exists():
        raise FileExistsError(output)
    if not args.dry_run:
        if args.resnet18_imagenet is None or not torch.cuda.is_available() or psutil.virtual_memory().available < 12*GIB:
            parser.error("Full fit needs approved local ImageNet donor, CUDA and at least 12 GiB free host memory")
        if file_sha256(args.resnet18_imagenet) != refiner.ASSET_SHA:
            raise ValueError("Exact audited TorchVision V1 donor required")
        if shutil.disk_usage(data).free < 5 * GIB:
            raise ValueError("At least 5 GiB of free space required")
        if data.anchor.lower() != output.anchor.lower():
            raise ValueError("Use the same volume for image hardlinks")
    output.mkdir(parents=True)
    records = create_dataset(data,output,not args.dry_run)
    save_new_json(output/"dataset.json",{"fit_disks":len(records),"membership":records,"dry_run":args.dry_run,
                                       "train_inventory_sha256":file_sha256(TRAIN),"fitting_ids_sha256":file_sha256(COHORT)})
    if args.dry_run:
        return
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"):
        env[name] = "1"
    env["CAMPAIGN_CPU_WORKERS"] = "6"
    command = [sys.executable,"-B",__file__,"--data",str(data),"--output",str(output),
               "--resnet18-imagenet",str(args.resnet18_imagenet.resolve()),"--internal"]
    for group in ("control","donor0","donor1","donor2"):
        for component in ("scratch","native"):
            subprocess.run(command+["detector",group,component],env=env,check=True)
        if group != "control":
            subprocess.run(command+["oof",group],env=env,check=True)
    subprocess.run(command+["refiner"],env=env,check=True)


if __name__ == "__main__":
    main()
