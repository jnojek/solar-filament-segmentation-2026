"""Assign exclusive native masks to detector proposals in confidence order."""

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as coco
from scipy import ndimage as ndi

from .infer import label_rle


DILATION_PX = 12
MIN_AREA = 150
APPEND_RETAIN_FRACTION = 0.65
NEIGHBORS = np.ones((3, 3), dtype=bool)


@dataclass
class PreparedDetection:
    score: float
    bounds: tuple[slice, slice]
    original: np.ndarray
    area: int
    grown: np.ndarray | None = None


def proposal_bounds(rle, shape, pad):
    x, y, width, height = map(int, coco.toBbox(rle))
    return (
        slice(max(0, y - pad), min(shape[0], y + height + pad)),
        slice(max(0, x - pad), min(shape[1], x + width + pad)),
    )


def connected_claim(candidate, seed):
    """Keep one component, prioritizing seeded pixels before component size."""
    labels, count = ndi.label(candidate, structure=NEIGHBORS)
    if not count:
        return None
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    seed_sizes = np.bincount(labels[seed & candidate], minlength=count + 1)
    best = max(range(1, count + 1), key=lambda label: (seed_sizes[label], sizes[label]))
    if not seed_sizes[best] or sizes[best] < MIN_AREA:
        return None
    return labels == best


def prepare_detections(detections, shape):
    result = []
    for score, rle in sorted(detections, key=lambda pair: -pair[0]):
        bounds = proposal_bounds(rle, shape, DILATION_PX)
        original = coco.decode(rle)[bounds].astype(bool)
        result.append(PreparedDetection(score, bounds, original, int(original.sum())))
    return result


def fuse_detector(probability, detections, *, mode, confidence, semantic_threshold=.65, parent=()):
    """Return detector-only, append, or strictly detector-owned native RLEs.

    Detections are (score, native compressed COCO RLE) pairs. Strict mode
    claims thresholded semantic pixels near each detector mask, with a
    semantic connection to its original mask and exclusive ownership.
    """
    prepared = prepare_detections(((score, rle) for score, rle in detections if score >= confidence),
                                  probability.shape)
    return fuse_prepared(probability, prepared, mode=mode, confidence=confidence,
                         semantic_threshold=semantic_threshold, parent=parent)


def fuse_prepared(probability, detections, *, mode, confidence, semantic_threshold=.65, parent=()):
    """Reuse decoded crops when evaluating multiple calibrated operating points."""
    if mode not in {"detector", "append", "strict"}:
        raise ValueError(f"Unknown detector fusion mode: {mode}")
    if mode != "append" and parent:
        raise ValueError("Only append mode accepts existing instances")
    if probability.ndim != 2 or not 0 <= confidence <= 1 or not 0 < semantic_threshold < 1:
        raise ValueError("Expected one native probability plane and valid thresholds")
    result = list(parent)
    labels = np.zeros(probability.shape, dtype=np.int32)
    if parent:
        labels[coco.decode(coco.merge(list(parent))).astype(bool)] = -1
    strict = mode == "strict"
    for item in detections:
        if item.score < confidence:
            break
        crop = labels[item.bounds]
        candidate = item.original
        if strict:
            if item.grown is None:
                item.grown = ndi.distance_transform_edt(~item.original) <= DILATION_PX
            candidate = item.grown & (probability[item.bounds] >= semantic_threshold)
        claim = connected_claim(candidate & (crop == 0), item.original)
        if claim is None:
            continue
        if mode == "append" and claim.sum() < APPEND_RETAIN_FRACTION * item.area:
            continue
        owner = len(result) + 1
        crop[claim] = owner
        result.append(label_rle(labels, owner, item.bounds))
    return result
