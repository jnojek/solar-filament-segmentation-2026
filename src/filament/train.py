"""Train one compact baseline on native crops and select it on development PQ."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
from statistics import mean
import time

import numpy as np
from pycocotools import mask as coco
import torch

from .data import annotation_path, file_sha256, load_image, read_annotations, save_new_json
from .infer import instance_rles, label_instances, label_rle, normalization, normalize
from .infer import probability_map, profile_device, runtime_receipt
from .metrics import aggregate, evaluate_rles
from .model import FilamentUNet, filament_loss


def prepare_observation(observation):
    image = load_image(observation.path)
    # Give each physical JPEG equal weight. The mean of each independent
    # annotator's foreground union supplies a soft target for disputed pixels.
    target = np.zeros(image.shape, dtype=np.uint8)
    anchors = []
    for rles in observation.views.values():
        if not rles:
            continue
        target += coco.decode(coco.merge(list(rles)))
        for rle in rles:
            x, y, width, height = coco.toBbox(rle)
            anchors.append((y + height / 2, x + width / 2, max(width, height)))
    return image, target, len(observation.views), normalization(image), anchors


def sample_tile(record, tile, positive_fraction, seed):
    image, target, views, (center, scale), anchors = record
    rng = np.random.default_rng(seed)
    height, width = image.shape
    if anchors and rng.random() < positive_fraction:
        y, x, size = anchors[rng.integers(len(anchors))]
        jitter = max(16, min(tile // 3, int(size / 2)))
        row = int(np.clip(y - tile / 2 + rng.integers(-jitter, jitter + 1), 0, height - tile))
        col = int(np.clip(x - tile / 2 + rng.integers(-jitter, jitter + 1), 0, width - tile))
    else:
        row = rng.integers(0, height - tile + 1)
        col = rng.integers(0, width - tile + 1)
    crop = np.s_[row:row + tile, col:col + tile]
    inputs = normalize(image[crop], center, scale)
    labels = target[crop].astype(np.float32) / views
    turns = rng.integers(4)
    inputs, labels = np.rot90(inputs, turns), np.rot90(labels, turns)
    if rng.random() < 0.5:
        inputs, labels = inputs[:, ::-1], labels[:, ::-1]
    inputs = inputs * np.float32(rng.uniform(0.85, 1.15)) + np.float32(rng.uniform(-0.10, 0.10))
    return inputs, labels


def train_epoch(model, optimizer, scheduler, records, config, device, epoch, pool, limit=None):
    model.train()
    ids = list(records) * config["tiles_per_observation"]
    rng = random.Random(config["seed"] + epoch)
    rng.shuffle(ids)
    size = config["train_batch"]
    steps = min(limit, (len(ids) + size - 1) // size) if limit else (len(ids) + size - 1) // size
    losses = []
    started = time.perf_counter()

    def make_batch(batch_ids, seeds):
        pairs = list(pool.map(
            lambda pair: sample_tile(records[pair[0]], config["train_tile"],
                                     config["positive_tile_fraction"], pair[1]),
            zip(batch_ids, seeds),
        ))
        inputs, targets = (np.stack(values)[:, None] for values in zip(*pairs))
        return torch.from_numpy(inputs), torch.from_numpy(targets)

    next_batch = None
    for index in range(steps):
        if next_batch is None:
            batch = ids[index * size:(index + 1) * size]
            next_batch = make_batch(batch, [rng.getrandbits(64) for _ in batch])
        inputs, targets = next_batch
        inputs = inputs.to(device, memory_format=torch.channels_last, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = filament_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        next_batch = None
        if index + 1 < steps:
            batch = ids[(index + 1) * size:(index + 2) * size]
            next_batch = make_batch(batch, [rng.getrandbits(64) for _ in batch])
        losses.append(loss.detach())
    return float(torch.stack(losses).mean()), steps, time.perf_counter() - started


def summarize(metrics):
    return {
        "pq": metrics.pq,
        "iou_sum": metrics.iou_sum,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "splits": metrics.splits,
        "merges": metrics.merges,
        "mean_overlapping_pair_iou": mean(metrics.ious) if metrics.ious else None,
        "mean_overlapping_pair_dice": mean(metrics.dices) if metrics.dices else None,
    }


def score_probability(probability, observation, threshold, min_area):
    predictions = instance_rles(probability, threshold, min_area)
    return aggregate(evaluate_rles(list(truth), predictions) for truth in observation.views.values())


def screen(model, device, config, observations, ids):
    model.eval()
    results = {(threshold, area): [] for threshold in config["threshold_grid"]
               for area in config["min_area_grid"]}
    for image_id in ids:
        observation = observations[image_id]
        probability = probability_map(model, load_image(observation.path), config, device)
        for threshold in config["threshold_grid"]:
            labels, keep = label_instances(probability, threshold, min(config["min_area_grid"]))
            encoded = []
            for label, bounds in keep:
                rle = label_rle(labels, label, bounds)
                encoded.append((int(coco.area(rle)), rle))
            for area in config["min_area_grid"]:
                predictions = [rle for pixels, rle in encoded if pixels >= area]
                results[threshold, area].extend(
                    evaluate_rles(list(truth), predictions) for truth in observation.views.values()
                )
    ranked = [
        {"threshold": threshold, "min_area": area, **summarize(aggregate(values))}
        for (threshold, area), values in results.items()
    ]
    return sorted(ranked, key=lambda row: (-row["pq"], row["merges"], row["fp"], row["min_area"]))


def run_training(data_root, split_path, config_path, output, device_name, profile_steps=None):
    device = torch.device(device_name)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if file_sha256(annotation_path(data_root)) != split["annotation_sha256"]:
        raise ValueError("Training annotation identity differs from the frozen split")
    output = Path(output)
    output.mkdir(parents=True)
    torch.set_num_threads(min(6, config["workers"]))
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")
        torch.cuda.set_per_process_memory_fraction(0.79, torch.cuda.current_device())
    started = profile_device(device)
    observations = read_annotations(data_root)
    ids = split["train"][:max(12, config["train_batch"])] if profile_steps else split["train"]
    model = FilamentUNet(config["channels"]).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    tiles = len(ids) * config["tiles_per_observation"]
    steps = (tiles + config["train_batch"] - 1) // config["train_batch"]
    epochs = 1 if profile_steps else config["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * steps, eta_min=config["learning_rate"] * 0.08,
    )
    screen_ids = random.Random(config["seed"] + 99).sample(
        split["dev"], min(config["screen_count"], len(split["dev"])),
    )
    history, best = [], None
    checkpoint = output / "model.pt"
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        preload_started = time.perf_counter()
        records = dict(zip(ids, pool.map(prepare_observation, (observations[image_id] for image_id in ids))))
        preload_seconds = time.perf_counter() - preload_started
        print(f"Loaded {len(ids)} physical training disks in {preload_seconds:.1f}s", flush=True)
        for epoch in range(1, epochs + 1):
            loss, batch_count, seconds = train_epoch(
                model, optimizer, scheduler, records, config, device, epoch, pool, profile_steps,
            )
            log = {"epoch": epoch, "loss": loss, "batches": batch_count, "train_seconds": round(seconds, 3)}
            if not profile_steps and (epoch == 1 or epoch % config["score_every"] == 0 or epoch == epochs):
                ranked = screen(model, device, config, observations, screen_ids)
                log["screen"] = ranked
                winner = ranked[0]
                if best is None or winner["pq"] > best["pq"]:
                    best = {"epoch": epoch, **winner}
                    temp_checkpoint = output / "model.partial"
                    torch.save(model.state_dict(), temp_checkpoint)
                    temp_checkpoint.replace(checkpoint)
            history.append(log)
            print(json.dumps({**log, "screen": log["screen"][0]} if "screen" in log else log), flush=True)
            with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log, allow_nan=False) + "\n")
    receipt = {
        **runtime_receipt(device, started),
        "preload_seconds": round(preload_seconds, 3),
        "physical_training_disks": len(ids),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "annotation_sha256": split["annotation_sha256"],
        "split_sha256": file_sha256(split_path),
        "config_sha256": file_sha256(config_path),
    }
    if profile_steps:
        sample = load_image(observations[screen_ids[0]].path)
        start = time.perf_counter()
        model.eval()
        probability = probability_map(model, sample, config, device)
        receipt["one_native_map_seconds"] = round(time.perf_counter() - start, 3)
        receipt["native_shape"] = list(probability.shape)
        receipt["tile_batch_seconds"] = history[0]["train_seconds"] / history[0]["batches"]
        receipt.update(runtime_receipt(device, started))
        save_new_json(output / "profile.json", receipt)
        print(json.dumps(receipt), flush=True)
        return
    del records
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    config.update(threshold=best["threshold"], min_area=best["min_area"])
    save_new_json(output / "baseline.json", config)
    save_new_json(output / "training.json", {
        **receipt, "best": best, "screen_ids": screen_ids, "checkpoint_sha256": file_sha256(checkpoint),
    })
