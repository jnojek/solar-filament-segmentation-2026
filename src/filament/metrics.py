"""Local evaluation with the organizer notebook's annotator and PQ semantics."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from pycocotools import mask as coco_mask

from .format import SHAPE, encode_mask, validate_counts


@dataclass(frozen=True)
class Metrics:
    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    splits: int = 0
    merges: int = 0
    ious: tuple[float, ...] = ()
    dices: tuple[float, ...] = ()

    @property
    def pq(self) -> float:
        denominator = self.tp + 0.5 * (self.fp + self.fn)
        return self.iou_sum / denominator if denominator else 0.0


def aggregate(results: Iterable[Metrics]) -> Metrics:
    iou_sum = 0.0
    tp = fp = fn = splits = merges = 0
    ious: list[float] = []
    dices: list[float] = []
    for result in results:
        iou_sum += result.iou_sum
        tp += result.tp
        fp += result.fp
        fn += result.fn
        splits += result.splits
        merges += result.merges
        ious.extend(result.ious)
        dices.extend(result.dices)
    return Metrics(iou_sum, tp, fp, fn, splits, merges, tuple(ious), tuple(dices))


def evaluate_rles(gt_rles: list[dict], pred_rles: list[dict]) -> Metrics:
    """Score validated native pycocotools RLE dictionaries for one annotator-image."""
    if not gt_rles or not pred_rles:
        return Metrics(fp=len(pred_rles), fn=len(gt_rles))

    iou = coco_mask.iou(gt_rles, pred_rles, [0] * len(pred_rles))
    # The organizer divides float32 intersection and union tensors.
    threshold_iou = iou.astype(np.float32)
    hits = threshold_iou > 0.5
    overlaps = iou > 0
    return Metrics(
        iou_sum=float(threshold_iou[hits].sum(dtype=np.float64)),
        tp=int(hits.sum()),
        fp=int((~hits.any(axis=0)).sum()),
        fn=int((~hits.any(axis=1)).sum()),
        splits=int((overlaps.sum(axis=1) > 1).sum()),
        merges=int((overlaps.sum(axis=0) > 1).sum()),
        ious=tuple(threshold_iou[overlaps].tolist()),
        dices=tuple((2 * iou[overlaps] / (1 + iou[overlaps])).astype(np.float32).tolist()),
    )


def evaluate_instances(
    gt_masks: Iterable[np.ndarray], pred_masks: Iterable[np.ndarray]
) -> Metrics:
    """Score one annotator's instance masks against the prediction for that image."""
    masks = [list(gt_masks), list(pred_masks)]
    shapes = {np.asarray(mask).shape for stack in masks for mask in stack}
    if len(shapes) > 1:
        raise ValueError("All instance masks must use the same shape")
    rles = [
        [
            {"size": list(mask.shape), "counts": encode_mask(mask).encode("ascii")}
            for mask in map(np.asarray, stack)
        ]
        for stack in masks
    ]
    return evaluate_rles(*rles)


def score_submission(
    gt_rows: Iterable[tuple[str, str]],
    pred_rows: Iterable[tuple[str, str]],
    shape: tuple[int, int] = SHAPE,
) -> Metrics:
    """Score CSV rows using the organizer's truth-backed annotator-image groups."""
    by_truth_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_image: dict[str, list[dict]] = defaultdict(list)
    for filament_id, counts in gt_rows:
        truth_group = filament_id.split("_", 1)[0]
        annotator, separator, image_id = truth_group.partition("-")
        if not separator or not annotator or not image_id:
            raise ValueError(f"Invalid organizer truth ID {filament_id!r}")
        validate_counts(counts, shape)
        by_truth_group[annotator, image_id].append(
            {"size": list(shape), "counts": counts.encode("ascii")}
        )
    for filament_id, counts in pred_rows:
        image_id = filament_id.split("_", 1)[0]
        validate_counts(counts, shape)
        by_image[image_id].append({"size": list(shape), "counts": counts.encode("ascii")})
    return aggregate(
        evaluate_rles(truth, by_image.get(image_id, []))
        for (_, image_id), truth in by_truth_group.items()
    )
