"""Load the competition-only COCO annotations and group physical observations."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
from pycocotools import mask as coco


OBSERVATION = re.compile(r"(?P<utc>\d{14})(?P<site>[A-Za-z])h")
DATASET_DIR = "MAGFiLO_1.0_Kaggle_2026"
ANNOTATIONS = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
NATIVE_SHAPE = (2048, 2048)


@dataclass(frozen=True)
class Observation:
    path: Path
    shape: tuple[int, int]
    views: dict[str, tuple[dict, ...]]


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def timestamp(image_id):
    match = OBSERVATION.fullmatch(image_id)
    if not match:
        raise ValueError(f"Invalid GONG image stem: {image_id}")
    return datetime.strptime(match["utc"], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def site(image_id):
    return OBSERVATION.fullmatch(image_id)["site"]


def annotation_path(data_root):
    return Path(data_root) / "train" / ANNOTATIONS


def image_paths(directory):
    paths = sorted(Path(directory).glob("*.jpeg"))
    if not paths:
        raise FileNotFoundError(f"No GONG JPEG files found in {directory}")
    result = {}
    for path in paths:
        timestamp(path.stem)
        if path.stem in result:
            raise ValueError(f"Duplicate image stem: {path.stem}")
        result[path.stem] = path
    return result


def load_image(path):
    with Image.open(path) as image:
        if (image.height, image.width) != NATIVE_SHAPE:
            raise ValueError(f"Unexpected image size {image.size} in {path}")
        return np.array(image.convert("L"), dtype=np.uint8)


def polygon_rle(polygons, shape):
    height, width = shape
    if isinstance(polygons, dict):
        if tuple(polygons["size"]) != shape:
            raise ValueError(f"RLE shape {polygons['size']} differs from {shape}")
        counts = polygons["counts"]
        return (coco.frPyObjects(polygons, height, width) if isinstance(counts, list)
                else {"size": [height, width], "counts": counts.encode("ascii")})
    if not polygons or not all(len(polygon) >= 6 and len(polygon) % 2 == 0 for polygon in polygons):
        raise ValueError("Invalid COCO polygon list")
    rles = coco.frPyObjects(polygons, height, width)
    return rles[0] if len(rles) == 1 else coco.merge(rles)


def read_annotations(data_root):
    data_root = Path(data_root)
    annotation_file = annotation_path(data_root)
    doc = json.loads(annotation_file.read_text(encoding="utf-8"))
    paths = image_paths(data_root / "train" / "train_images")
    by_view = {}
    for image in doc["images"]:
        view_id = str(image["id"])
        image_id = Path(image["file_name"]).stem
        shape = int(image["height"]), int(image["width"])
        if shape != NATIVE_SHAPE or image_id not in paths or view_id in by_view:
            raise ValueError(f"Invalid or repeated image record: {image}")
        if view_id.split("-", 1)[-1] != image_id:
            raise ValueError(f"Annotator ID {view_id} disagrees with filename {image_id}")
        by_view[view_id] = image_id, shape

    annotations = defaultdict(list)
    seen = set()
    for item in doc["annotations"]:
        view_id = str(item["image_id"])
        annotation_id = str(item["id"])
        if annotation_id in seen:
            raise ValueError(f"Duplicate annotation: {annotation_id}")
        seen.add(annotation_id)
        _, shape = by_view[view_id]
        rle = polygon_rle(item["segmentation"], shape)
        if coco.area(rle):
            annotations[view_id].append(rle)
        else:
            raise ValueError(f"Zero-area annotation: {annotation_id}")

    grouped = defaultdict(dict)
    for view_id, (image_id, _) in by_view.items():
        grouped[image_id][view_id] = tuple(annotations[view_id])
    if set(paths) != set(grouped):
        raise ValueError(f"JPEG stems without COCO image records: {sorted(set(paths) - set(grouped))}")
    return {
        image_id: Observation(paths[image_id], NATIVE_SHAPE, dict(sorted(grouped[image_id].items())))
        for image_id in sorted(grouped)
    }


def grouped_split(observations, seed, embargo_hours=48, group_days=27):
    """Split 27-day time blocks, then purge cross-fold neighbors from training."""
    if embargo_hours < 24 or group_days < 2:
        raise ValueError("The split requires full UTC days and at least a 24-hour embargo")
    times = {image_id: timestamp(image_id) for image_id in observations}
    blocks = defaultdict(list)
    for image_id, time in sorted(times.items()):
        blocks[(time.date().toordinal() - 1) // group_days].append(image_id)
    by_year = defaultdict(list)
    for block, ids in sorted(blocks.items()):
        by_year[times[ids[len(ids) // 2]].year].append(block)
    randomizer = random.Random(seed)
    folds = {"train": [], "dev": [], "confirm": []}
    for year_blocks in by_year.values():
        randomizer.shuffle(year_blocks)
        count = len(year_blocks)
        dev = max(1, round(count * 0.18)) if count >= 4 else 0
        confirm = max(1, round(count * 0.12)) if count >= 5 else 0
        for index, block in enumerate(year_blocks):
            fold = "dev" if index < dev else "confirm" if index < dev + confirm else "train"
            folds[fold].extend(blocks[block])

    margin = timedelta(hours=embargo_hours)
    purged = []

    def purge_near(source, targets):
        for image_id in source:
            if any(abs(times[image_id] - times[other]) < margin for other in targets):
                purged.append(image_id)
            else:
                yield image_id

    folds["dev"] = list(purge_near(folds["dev"], folds["confirm"]))
    folds["train"] = list(purge_near(folds["train"], folds["confirm"] + folds["dev"]))
    if not all(folds.values()):
        raise ValueError("Too few date groups to make nonempty train, dev and confirm folds")
    # A secondary site-held-out evaluation uses a separate, never trained-on group.
    # Prefer a smaller adequately represented site to retain training coverage.
    site_ids = defaultdict(list)
    for image_id in folds["train"]:
        site_ids[site(image_id)].append(image_id)
    candidate_sites = sorted(
        (name for name, ids in site_ids.items() if 12 <= len(ids) <= 0.20 * len(folds["train"])),
        key=lambda name: (len(site_ids[name]), name),
    )
    holdout_site = candidate_sites[0] if candidate_sites else None
    site_holdout = []
    if holdout_site:
        site_holdout = [image_id for ids in folds.values() for image_id in ids
                        if site(image_id) == holdout_site]
        for fold, ids in folds.items():
            folds[fold] = list(purge_near(
                (image_id for image_id in ids if site(image_id) != holdout_site),
                site_holdout,
            ))

    return {
        **{key: sorted(values) for key, values in folds.items()},
        "purged": sorted(purged),
        "site_holdout": sorted(site_holdout),
        "holdout_site": holdout_site,
        "seed": seed,
        "group_days": group_days,
        "embargo_hours": embargo_hours,
        "group_by_id": {image_id: block for block, ids in blocks.items() for image_id in ids},
    }


def save_new_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(content, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
