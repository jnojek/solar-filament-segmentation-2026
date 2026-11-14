
# Portions of the crop geometry, prompt construction and inverse resize adapt
# Artem Shelamanov's filaments/segmenter.py (ArtDoctor/solar-kaggle-public,
# ea487495ab55aa375fcc36eeb47adc701e79f60b), GNU AGPL-3.0.
# Modified by James Nojek, 2026-11-14, for grouped OOF YOLO11s proposals,
# exact TorchVision initialization and native inference.
# SPDX-License-Identifier: AGPL-3.0-only
"""Source-aligned 512-pixel box conditioning for native H-alpha instances."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco
from scipy import ndimage
import torch
from torch.nn import functional as F

from .data import file_sha256
from .format import SHAPE

INPUT = 512
SEED = 104
MAX_BOXES = 40
MIN_AREA = 150
MASK_THRESHOLD = .40
ASSET_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


@dataclass(frozen=True)
class Crop:
    box: tuple
    window: tuple
    score: float
    proposal_index: int
    view: str | None = None
    truth_index: int | None = None


def open_refiner(asset: Path | None = None):
    import segmentation_models_pytorch as smp
    from torchvision.models import resnet18

    if smp.__version__ != "0.5.0":
        raise ValueError("Expected pinned segmentation_models_pytorch==0.5.0")
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=2, classes=1)
    if asset is not None:
        if asset.stat().st_size != 46_830_571 or file_sha256(asset) != ASSET_SHA:
            raise ValueError("Exact audited ImageNet donor differs")
        source = resnet18(weights=None)
        source.load_state_dict(torch.load(asset, map_location="cpu", weights_only=True), strict=True)
        state = {key: value for key, value in source.state_dict().items() if not key.startswith("fc.")}
        state["conv1.weight"] = state["conv1.weight"][:, :2].contiguous() * 1.5
        model.encoder.load_state_dict(state, strict=True)
    return model


def square_crop(box, shape=SHAPE):
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    half = side / 2 + max(32, .3 * side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    width, height = shape[1], shape[0]
    return (max(0, int(np.floor(cx - half))), max(0, int(np.floor(cy - half))),
            min(width, int(np.ceil(cx + half))), min(height, int(np.ceil(cy + half))))


def letterbox(array, nearest=False):
    height, width = array.shape
    scale = min(INPUT / height, INPUT / width)
    rh, rw = min(INPUT, max(1, round(height * scale))), min(INPUT, max(1, round(width * scale)))
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    resized = np.asarray(Image.fromarray(array).resize((rw, rh), resampling))
    top, left = (INPUT - rh) // 2, (INPUT - rw) // 2
    result = np.zeros((INPUT, INPUT), dtype=resized.dtype)
    result[top:top + rh, left:left + rw] = resized
    return result, (top, left, rh, rw, height, width)


def make_input(cut, local_box):
    prompt = np.zeros(cut.shape, np.float32)
    x0, y0 = np.floor(local_box[:2]).astype(int)
    x1, y1 = np.ceil(local_box[2:]).astype(int)
    prompt[max(0, y0):min(prompt.shape[0], y1), max(0, x0):min(prompt.shape[1], x1)] = 1
    gray, geometry = letterbox(cut)
    prompt, _ = letterbox(prompt, nearest=True)
    gray = (gray.astype(np.float32) / 255 - .449) / .226
    return np.stack((gray, prompt)), geometry


def restore(probability, g):
    top, left, rh, rw, height, width = g
    crop = probability[top:top + rh, left:left + rw].astype(np.float32)
    return np.asarray(Image.fromarray(crop).resize((width, height), Image.Resampling.BILINEAR))


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def rle_box(mask):
    x, y, w, h = map(float, coco.toBbox(mask))
    return x, y, x + w, y + h


def roi(mask, window):
    if isinstance(mask["counts"], str):
        mask = {**mask, "counts": mask["counts"].encode("ascii")}
    x0, y0, x1, y1 = window
    return coco.decode(mask)[y0:y1, x0:x1]


def source_rows(rows, threshold):
    keep = sorted(((i, row) for i, row in enumerate(rows) if row["score"] >= threshold),
                  key=lambda pair: (-pair[1]["score"], pair[0]))[:MAX_BOXES]
    for index, row in keep:
        if list(row["size"]) != list(SHAPE) or len(row["box_xyxy"]) != 4:
            raise ValueError("Native detector proposal box is missing or malformed")
        box = tuple(np.asarray(row["box_xyxy"], dtype=float).clip(0, SHAPE[0]))
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Detector produced invalid native crop coordinates")
        yield Crop(box, square_crop(box), float(row["score"]), index)


def sample_plans(rows, views, threshold=.10):
    """Choose actual OOF prompts, per-view instance targets and whole-crop hard negatives."""
    proposals = list(source_rows(rows, threshold))
    all_masks = [mask for group in views.values() for mask in group]
    if not all_masks:
        raise ValueError("Fit observation lost its supplied annotation masks")
    union = coco.decode(coco.merge(all_masks)).astype(bool)
    rectangles = {}
    for proposal in proposals:
        x0,y0,x1,y1 = np.r_[np.floor(proposal.box[:2]), np.ceil(proposal.box[2:])].astype(int)
        rectangles[proposal.proposal_index] = coco.frPyObjects(
            [[int(x0),int(y0),int(x1),int(y0),int(x1),int(y1),int(x0),int(y1)]], *SHAPE)[0]
    positives, negatives = [], []
    for name, masks in views.items():
        best = {}
        target_boxes = [rle_box(mask) for mask in masks]
        for proposal in proposals:
            choice = []
            for ix, truth_box in enumerate(target_boxes):
                overlap = intersection(proposal.box, truth_box)
                if not overlap or not coco.area(coco.merge((masks[ix], rectangles[proposal.proposal_index]), intersect=True)):
                    continue
                box_area = intersection(proposal.box, proposal.box)
                truth_area = intersection(truth_box, truth_box)
                choice.append((overlap / (box_area + truth_area - overlap), ix))
            if not choice:
                continue
            quality, ix = max(choice)
            box = target_boxes[ix]
            if quality < .3 or intersection(proposal.window, box) != intersection(box, box):
                continue
            order = (quality, proposal.score, -proposal.proposal_index)
            if ix not in best or order > best[ix][0]:
                best[ix] = order, proposal
        for ix, box in enumerate(target_boxes):
            p = best[ix][1] if ix in best else Crop(box, square_crop(box), 0.0, -1)
            positives.append(Crop(p.box, p.window, p.score, p.proposal_index, name, ix))
    for p in proposals:
        x0, y0, x1, y1 = p.window
        if not union[y0:y1, x0:x1].any():
            negatives.append(p)
            if len(negatives) == 2:
                break
    return positives, negatives


def make_training_example(image, plan, truth, rng):
    x0, y0, x1, y1 = plan.window
    cut = image[y0:y1, x0:x1]
    target = roi(truth, plan.window) if truth is not None else np.zeros(cut.shape, np.uint8)
    box = np.asarray(plan.box, dtype=float) - (x0, y0, x0, y0)
    w, h = box[2:] - box[:2]
    box += rng.uniform(-.25, .25, 4) * (w, h, w, h)
    box[::2] = np.clip(box[::2], 0, cut.shape[1])
    box[1::2] = np.clip(box[1::2], 0, cut.shape[0])
    x, _ = make_input(cut, box)
    y, _ = letterbox(target, nearest=True)
    if rng.random() < .5:
        x, y = x[..., ::-1], y[..., ::-1]
    if rng.random() < .5:
        x, y = x[:, ::-1, :], y[::-1]
    k = int(rng.integers(4))
    x, y = np.rot90(x, k, axes=(1, 2)).copy(), np.rot90(y, k).copy()
    x[0] = x[0] * rng.uniform(.85, 1.15) + rng.uniform(-.15, .15)
    return x, y[None].astype(np.float32)


def bce_dice(logits, target):
    logits = logits.float()
    p = logits.sigmoid()
    dice = (2 * (p * target).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + 1 - dice.mean()


def largest(probability, window, threshold=MASK_THRESHOLD):
    labels, count = ndimage.label(probability >= threshold, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return None
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    winner = int(counts.argmax())
    if counts[winner] < MIN_AREA:
        return None
    x0, y0, x1, y1 = window
    mask = np.zeros(SHAPE, np.uint8, order="F")
    mask[y0:y1, x0:x1] = labels == winner
    return coco.encode(mask)


@torch.inference_mode()
def predict_maps(model, image, rows, confidence=.20, batch=4, flip_modes=("identity", "flips4")):
    device = next(model.parameters()).device
    plans = list(source_rows(rows, confidence))
    for start in range(0, len(plans), batch):
        current = plans[start:start + batch]
        items = []
        for p in current:
            x0, y0, x1, y1 = p.window
            items.append(make_input(image[y0:y1, x0:x1], np.asarray(p.box) - (x0, y0, x0, y0)))
        x = torch.from_numpy(np.stack([item[0] for item in items])).to(device)
        p = model(x)[:, 0].sigmoid().float()
        modes = {}
        if "identity" in flip_modes:
            modes["identity"] = p.cpu().numpy().astype(np.float16)
        if "flips4" in flip_modes:
            for dims in ((-1,), (-2,), (-2, -1)):
                p += torch.flip(model(torch.flip(x, dims))[:, 0].sigmoid().float(), dims)
            modes["flips4"] = (p / 4).cpu().numpy().astype(np.float16)
        for mode, maps in modes.items():
            for plan, (_, geo), probability in zip(current, items, maps, strict=True):
                yield mode, plan, restore(probability, geo)


def refine(model, image, rows, confidence, threshold, flip_mode):
    results = []
    for _, plan, probability in predict_maps(model, image, rows, confidence, flip_modes=(flip_mode,)):
        mask = largest(probability, plan.window, threshold)
        if mask is not None:
            results.append({"score": plan.score, "size": mask["size"],
                            "counts": mask["counts"].decode("ascii"),
                            "proposal_index": plan.proposal_index})
    return results

