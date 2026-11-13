"""Competition CSV and compressed COCO mask encoding."""

import csv
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from pycocotools import mask as coco_mask

SHAPE = (2048, 2048)
HEADER = ("filament_id", "segmentation_rle")


def encode_mask(mask: np.ndarray) -> str:
    """Encode one 2D binary mask in COCO column-major order."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size:
        raise ValueError("COCO encoding requires a 2D binary mask")
    if mask.dtype.kind in "iu":
        binary = mask.min() >= 0 and mask.max() <= 1
    else:
        binary = mask.dtype.kind == "b" or np.isin(mask, (0, 1)).all()
    if not binary:
        raise ValueError("COCO encoding requires a 2D binary mask")
    return coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))["counts"].decode("ascii")


def validate_counts(counts: str, shape: tuple[int, int] = SHAPE) -> None:
    """Check run lengths before passing a CSV value to the native decoder."""
    if not isinstance(counts, str) or not counts:
        raise ValueError("Expected a nonempty compressed COCO RLE string")
    height, width = shape
    pixel_count = height * width
    runs: list[int] = []
    total = value = shift = 0
    for char in counts:
        code = ord(char) - 48
        if not 0 <= code <= 63 or shift > 30:
            raise ValueError("Invalid compressed COCO RLE character or run length")
        value |= (code & 31) << shift
        shift += 5
        if code & 32:
            continue
        if code & 16:
            value -= 1 << shift
        if len(runs) > 2:
            value += runs[-2]
        if value < 0 or total + value > pixel_count:
            raise ValueError("Compressed COCO RLE runs exceed the mask dimensions")
        total += value
        runs.append(value)
        value = shift = 0
    if shift or total != pixel_count:
        raise ValueError("Compressed COCO RLE does not fill the requested mask")


def decode_mask(counts: str, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """Decode a compressed COCO RLE string into the requested 2D mask."""
    validate_counts(counts, shape)
    rle = {"size": list(shape), "counts": counts.encode("ascii")}
    return coco_mask.decode(rle)


def read_rows(path: str | Path):
    """Yield ID and RLE pairs while enforcing the competition CSV header."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, strict=True)
        if tuple(next(reader, ())) != HEADER:
            raise ValueError(f"{path} must begin with {','.join(HEADER)}")
        for line_number, fields in enumerate(reader, 2):
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number} must have exactly two columns")
            yield tuple(fields)


def write_submission(
    path: str | Path,
    predictions,
    *,
    coverage_path: str | Path | None = None,
) -> None:
    """Write one CSV row per mask and optionally record coverage, including empty images."""
    path = Path(path)
    coverage_path = Path(coverage_path) if coverage_path is not None else None
    if coverage_path is not None and coverage_path.resolve() == path.resolve():
        raise ValueError("CSV and coverage paths must differ")

    path.parent.mkdir(parents=True, exist_ok=True)
    image_receipts: dict[str, dict] = {}
    pending: list[tuple[Path, Path]] = []
    try:
        with NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as handle:
            pending.append((Path(handle.name), path))
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(HEADER)
            for image_id, masks in predictions:
                if not image_id or "_" in image_id or image_id.strip() != image_id:
                    raise ValueError(f"Invalid image ID: {image_id!r}")
                if image_id in image_receipts:
                    raise ValueError(f"Duplicate image ID: {image_id}")
                mask_hashes: list[str] = []
                for index, mask in enumerate(masks, 1):
                    mask = np.asarray(mask)
                    if mask.shape != SHAPE or not mask.any():
                        raise ValueError(f"{image_id}_{index} must be a nonempty {SHAPE} mask")
                    writer.writerow((f"{image_id}_{index}", encode_mask(mask)))
                    mask_hashes.append(hashlib.sha256(mask.astype(np.uint8, copy=False).tobytes()).hexdigest())
                image_receipts[image_id] = {"instances": len(mask_hashes), "mask_hashes": mask_hashes}
        if coverage_path is not None:
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(
                "w", encoding="utf-8", dir=coverage_path.parent, delete=False
            ) as handle:
                pending.append((Path(handle.name), coverage_path))
                json.dump({"shape": list(SHAPE), "images": image_receipts}, handle, indent=2)
                handle.write("\n")
        for source, destination in pending:
            source.replace(destination)
    finally:
        for source, _ in pending:
            source.unlink(missing_ok=True)
