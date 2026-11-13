"""Audit a frozen split or competition CSV in an independent process."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from pycocotools import mask as coco_mask

from filament.format import validate_counts

HEADER = ["filament_id", "segmentation_rle"]
SHAPE = (2048, 2048)
FORM_CSV_BYTE_LIMIT = 10_000_000
IMAGE_ID = re.compile(r"^(?P<stamp>\d{14})(?P<site>[A-Za-z])h$")
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
SPLITS = ("train", "dev", "confirm", "purged", "site_holdout")


def file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def image_catalog(directory):
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    images = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not IMAGE_ID.fullmatch(path.stem):
            raise ValueError(f"Unrecognized image file name: {path}")
        if path.stem in images:
            raise ValueError(f"Duplicate image ID: {path.stem}")
        images[path.stem] = path
    if not images:
        raise ValueError(f"No images found: {directory}")
    return images


def cross_fold_similarity(active_ids, images, members):
    """Screen retained disks for copies that differ only in JPEG encoding or intensity."""
    thumb_size = 96
    coordinate = np.arange(thumb_size) - (thumb_size - 1) / 2
    disk = coordinate[:, None] ** 2 + coordinate[None, :] ** 2 <= (0.41 * thumb_size) ** 2
    rows = []
    for image_id in active_ids:
        with Image.open(images[image_id]) as source:
            source.draft("L", (256, 256))
            thumb = source.convert("L").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            smooth = thumb.filter(ImageFilter.GaussianBlur(2))
        detail = (np.asarray(thumb, dtype=np.float32) - np.asarray(smooth, dtype=np.float32))[disk]
        detail -= detail.mean()
        norm = np.linalg.norm(detail)
        if norm == 0:
            raise ValueError(f"Cannot fingerprint flat solar image {image_id}")
        rows.append(detail / norm)
    fingerprints = np.stack(rows)
    similarity = fingerprints @ fingerprints.T
    folds = np.asarray([members[image_id] for image_id in active_ids])
    cross_fold = np.triu(folds[:, None] != folds[None, :], k=1)
    similarity[~cross_fold] = -2
    row, column = np.unravel_index(similarity.argmax(), similarity.shape)
    flags = np.argwhere(similarity >= 0.9)
    return {
        "pairs_screened": int(cross_fold.sum()),
        "highest_correlation": float(similarity[row, column]),
        "highest_pair": [active_ids[row], active_ids[column]],
        "flag_correlation": 0.9,
        "flagged_pairs": [
            {"image_ids": [active_ids[i], active_ids[j]], "correlation": float(similarity[i, j])}
            for i, j in flags
        ],
    }


def verify_split(path, images, embargo_hours):
    split = json.loads(path.read_text(encoding="utf-8"))
    split.setdefault("site_holdout", [])
    groups = split.get("group_by_id")
    if groups is not None and set(groups) != set(images):
        raise ValueError("Group map does not cover exactly the training images")
    assigned_groups = defaultdict(set)
    members = {}
    stamps = {}
    sites_by_split = {}
    dates = defaultdict(set)
    hashes = defaultdict(list)
    for name in SPLITS:
        image_ids = split[name]
        sites = Counter()
        sites_by_split[name] = sites
        if name in {"train", "dev", "confirm"} and not image_ids:
            raise ValueError(f"Split {name} is empty")
        for image_id in image_ids:
            if image_id in members:
                raise ValueError(f"{image_id} occurs more than once in the split")
            members[image_id] = name
            if image_id not in images:
                raise ValueError(f"Split contains image ID with no corresponding input: {image_id}")
            match = IMAGE_ID.fullmatch(image_id)
            date = datetime.strptime(match["stamp"], "%Y%m%d%H%M%S")
            sites[match["site"]] += 1
            if name == "purged":
                continue
            stamps[image_id] = date
            dates[date.date()].add(name)
            hashes[file_sha256(images[image_id])].append(image_id)
            if groups is not None and name != "site_holdout":
                expected_group = (date.date().toordinal() - 1) // split["group_days"]
                if groups[image_id] != expected_group:
                    raise ValueError(f"Observation group ID disagrees with timestamp: {image_id}")
                assigned_groups[expected_group].add(name)
    if set(members) != set(images):
        missing = sorted(set(images) - set(members))
        raise ValueError(f"{len(missing)} images lack a split, including {missing[:5]}")
    crossed_dates = [day for day, folds in dates.items() if len(folds) > 1]
    if crossed_dates:
        raise ValueError(f"Observing dates cross splits: {crossed_dates[:5]}")
    crossed_groups = [group for group, folds in assigned_groups.items() if len(folds) > 1]
    if crossed_groups:
        raise ValueError(f"Observing blocks cross splits: {crossed_groups[:5]}")
    crossed_hashes = [
        image_ids for image_ids in hashes.values()
        if len({members[image_id] for image_id in image_ids}) > 1
    ]
    if crossed_hashes:
        raise ValueError(f"Identical image bytes cross splits: {crossed_hashes[:5]}")

    nearest = {}
    active_ids = sorted(stamps, key=stamps.__getitem__)
    embargo = timedelta(hours=embargo_hours)
    for i, left in enumerate(active_ids):
        for right in active_ids[i + 1 :]:
            if members[left] == members[right]:
                continue
            pair = "-".join(sorted((members[left], members[right])))
            delta = stamps[right] - stamps[left]
            if delta < embargo:
                raise ValueError(f"{left} and {right} are only {delta} apart across {pair}")
            if pair not in nearest or delta < nearest[pair][0]:
                nearest[pair] = (delta, left, right)

    train_sites = set(sites_by_split["train"])
    holdout_site = split.get("holdout_site")
    if split["site_holdout"] and set(sites_by_split["site_holdout"]) != {holdout_site}:
        raise ValueError("Site holdout includes images from a different site")
    seen_in_regular_folds = set().union(*(sites_by_split[name] for name in ("train", "dev", "confirm")))
    if holdout_site in seen_in_regular_folds:
        raise ValueError(f"Site holdout {holdout_site} is still present in a regular fold")
    return {
        "split_sha256": file_sha256(path),
        "counts": {name: len(split[name]) for name in SPLITS},
        "observing_dates": len(dates),
        "observing_blocks": len(assigned_groups) if groups is not None else None,
        "site_counts": sites_by_split,
        "holdout_site": holdout_site,
        "dev_sites_absent_from_train": sorted(set(sites_by_split["dev"]) - train_sites),
        "confirm_sites_absent_from_train": sorted(set(sites_by_split["confirm"]) - train_sites),
        "cross_fold_embargo_hours": embargo_hours,
        "nearest_cross_fold_pairs": {
            pair: {"hours": delta.total_seconds() / 3600, "image_ids": [left, right]}
            for pair, (delta, left, right) in sorted(nearest.items())
        },
        "identical_image_groups_within_folds": sum(len(group) > 1 for group in hashes.values()),
        "cross_fold_image_similarity": cross_fold_similarity(active_ids, images, members),
    }


def verify_submission(path, coverage_path, images, sample_path=None, inventory_path=None):
    size = path.stat().st_size
    if size > FORM_CSV_BYTE_LIMIT:
        raise ValueError(f"CSV is {size:,} bytes, above the {FORM_CSV_BYTE_LIMIT:,}-byte form budget")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if tuple(coverage["shape"]) != SHAPE:
        raise ValueError("Coverage manifest has the wrong mask shape")
    receipts = coverage["images"]
    missing = set(images) - set(receipts)
    unexpected = set(receipts) - set(images)
    if missing or unexpected:
        raise ValueError(f"Coverage ID mismatch. Missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]}")

    if inventory_path is not None:
        expected_files = json.loads(inventory_path.read_text(encoding="utf-8"))
        actual_files = {image_path.name: file_sha256(image_path) for image_path in images.values()}
        if actual_files != expected_files:
            raise ValueError("Test images do not match the verified official file manifest")

    for image_id, image_path in images.items():
        with Image.open(image_path) as image:
            if image.size != (SHAPE[1], SHAPE[0]):
                raise ValueError(f"{image_id}: source image has wrong size {image.size}")

    seen_ids = set()
    by_image = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, strict=True)
        header = next(reader, None)
        if header != HEADER:
            raise ValueError(f"Expected competition CSV header {HEADER}, found {header}")
        if sample_path is not None:
            with sample_path.open(newline="", encoding="utf-8-sig") as sample:
                sample_header = next(csv.reader(sample, strict=True), None)
            if header != sample_header:
                raise ValueError("Submission columns do not match the official sample")
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != 2:
                raise ValueError(f"Line {line_number} must have two CSV fields")
            filament_id, counts = fields
            image_id, separator, suffix = filament_id.partition("_")
            if not separator or not suffix.isascii() or not suffix.isdecimal() or int(suffix) < 1:
                raise ValueError(f"Line {line_number}: invalid local instance ID {filament_id!r}")
            if image_id not in images:
                raise ValueError(f"Line {line_number}: unknown image ID {image_id}")
            if filament_id in seen_ids or int(suffix) in by_image[image_id]:
                raise ValueError(f"Line {line_number}: duplicate instance ID {filament_id}")
            seen_ids.add(filament_id)
            by_image[image_id][int(suffix)] = counts

    empty_images = []
    foreground_pixels = 0
    for image_id, image_receipt in receipts.items():
        counts_by_index = by_image.get(image_id, {})
        expected = image_receipt["instances"]
        if not isinstance(expected, int) or expected < 0:
            raise ValueError(f"{image_id}: invalid instance count in coverage manifest")
        if sorted(counts_by_index) != list(range(1, expected + 1)):
            raise ValueError(f"{image_id}: CSV instance rows disagree with the coverage manifest")
        if len(image_receipt["mask_hashes"]) != expected:
            raise ValueError(f"{image_id}: coverage mask hash count mismatch")
        if expected == 0:
            empty_images.append(image_id)
            continue
        occupied = np.zeros(SHAPE, dtype=bool)
        for index, expected_hash in enumerate(image_receipt["mask_hashes"], 1):
            counts = counts_by_index[index]
            validate_counts(counts, SHAPE)
            rle = {"size": list(SHAPE), "counts": counts.encode("ascii")}
            decoded = coco_mask.decode(rle)
            if decoded.shape != SHAPE or not decoded.any():
                raise ValueError(f"{image_id}_{index}: expected a nonempty native-size mask")
            if coco_mask.encode(np.asfortranarray(decoded))["counts"] != rle["counts"]:
                raise ValueError(f"{image_id}_{index}: COCO re-encoding changed the run lengths")
            if hashlib.sha256(decoded.tobytes()).hexdigest() != expected_hash:
                raise ValueError(f"{image_id}_{index}: decoded mask placement differs from inference")
            binary = decoded.view(bool)
            if np.any(occupied & binary):
                raise ValueError(f"{image_id}_{index}: masks overlap within an instance prediction")
            occupied |= binary
        foreground_pixels += int(occupied.sum())

    return {
        "csv_sha256": file_sha256(path),
        "coverage_sha256": file_sha256(coverage_path),
        "csv_bytes": size,
        "form_limit_bytes": FORM_CSV_BYTE_LIMIT,
        "expected_image_count": len(images),
        "images_with_predictions": len(by_image),
        "empty_image_ids": sorted(empty_images),
        "instance_rows": len(seen_ids),
        "foreground_pixels": foreground_pixels,
        "mask_shape": SHAPE,
        "header_reference": str(sample_path) if sample_path else "organizer submission documentation",
        "test_manifest_sha256": file_sha256(inventory_path) if inventory_path else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("split")
    split.add_argument("--split", type=Path, required=True)
    split.add_argument("--images", type=Path, required=True)
    split.add_argument("--embargo-hours", type=float, default=48.0)
    split.add_argument("--output", type=Path)
    export = sub.add_parser("export")
    export.add_argument("--csv", type=Path, required=True)
    export.add_argument("--coverage", type=Path, required=True)
    export.add_argument("--images", type=Path, required=True)
    export.add_argument("--sample", type=Path)
    export.add_argument("--inventory", type=Path)
    export.add_argument("--output", type=Path)
    args = parser.parse_args()
    images = image_catalog(args.images)
    if args.command == "split":
        result = verify_split(args.split, images, args.embargo_hours)
    else:
        result = verify_submission(args.csv, args.coverage, images, args.sample, args.inventory)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
