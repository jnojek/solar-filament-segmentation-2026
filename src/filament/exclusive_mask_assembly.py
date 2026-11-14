"""Apply fixed native mask NMS and exclusive ownership in detector score order."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from pycocotools import mask as coco

from .detector_ownership import proposal_bounds
from .format import validate_counts


MASK_NMS_IOU = 0.50
TRIMMED_MIN_AREA = 150


@dataclass(frozen=True)
class Assembly:
    accepted: list
    suppressed_indices: tuple[int, ...]
    clipped_indices: tuple[int, ...]
    rejected_indices: tuple[int, ...]


def exclusive_masks(rows, shape=(2048, 2048)):
    """Keep untouched RLEs exact and remove only lower-priority contested pixels.

    Rows are (original confidence, original proposal index, compressed COCO RLE).
    NMS compares original masks before any clipping.
    """
    seen = set()
    prepared = []
    for score, index, rle in rows:
        if not isfinite(score) or not 0 <= score <= 1 or type(index) is not int or index < 0 or index in seen:
            raise ValueError("Expected valid original scores and unique proposal indices")
        counts = rle["counts"]
        if tuple(rle["size"]) != shape or not isinstance(counts, (str, bytes)):
            raise ValueError("Expected native compressed COCO RLE")
        raw = counts.decode("ascii") if isinstance(counts, bytes) else counts
        validate_counts(raw, shape)
        internal = {"size": list(shape), "counts": raw.encode("ascii")}
        if not coco.area(internal):
            raise ValueError("Only nonempty native masks may enter assembly")
        seen.add(index)
        prepared.append((score, index, rle, internal))
    prepared.sort(key=lambda row: (-row[0], row[1]))

    survivors, kept_rles, suppressed = [], [], []
    for row in prepared:
        if kept_rles and (coco.iou([row[3]], kept_rles, [0] * len(kept_rles)) >= MASK_NMS_IOU).any():
            suppressed.append(row[1])
            continue
        survivors.append(row)
        kept_rles.append(row[3])

    owned = np.zeros(shape, dtype=bool)
    accepted, clipped, rejected = [], [], []
    for score, index, original, internal in survivors:
        bounds = proposal_bounds(internal, shape, pad=0)
        mask = coco.decode(internal)[bounds].astype(bool)
        prior = owned[bounds]
        if np.any(mask & prior):
            mask &= ~prior
            if int(mask.sum()) < TRIMMED_MIN_AREA:
                rejected.append(index)
                continue
            native = np.zeros(shape, dtype=np.uint8, order="F")
            native[bounds] = mask
            result = coco.encode(native)
            if isinstance(original["counts"], str):
                result["counts"] = result["counts"].decode("ascii")
            clipped.append(index)
        else:
            result = original
        prior |= mask
        accepted.append((score, index, result))
    return Assembly(accepted, tuple(suppressed), tuple(clipped), tuple(rejected))
