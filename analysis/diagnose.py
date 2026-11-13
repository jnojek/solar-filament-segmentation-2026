"""Attribute native-resolution instance errors on frozen development predictions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


# These diagnostics complement the organizer's any-overlap split and merge counts.
MIN_INTERSECTION = 8
FRAGMENT_TRUTH_FRACTION = 0.10
FRAGMENT_PREDICTION_FRACTION = 0.25
LIMB_RADIUS_FRACTION = 0.80
MAX_GAP_PX = 32.0
MAX_GAP_WIDTHS = 2.5
MAX_DIRECTION_DEG = 35.0


def overlap_tables(truth, prediction):
    truth_areas = np.asarray(coco_mask.area(truth), dtype=float)
    pred_areas = np.asarray(coco_mask.area(prediction), dtype=float)
    if not truth or not prediction:
        iou = np.zeros((len(truth), len(prediction)), dtype=np.float32)
    else:
        iou = np.asarray(coco_mask.iou(truth, prediction, [0] * len(prediction)))
    area_sum = truth_areas[:, None] + pred_areas[None, :]
    intersection = np.rint(iou * area_sum / (1 + iou)).astype(np.int32)
    hit = iou.astype(np.float32) > 0.50
    truth_share = intersection / truth_areas[:, None]
    pred_share = intersection / pred_areas[None, :]
    shared = intersection >= MIN_INTERSECTION
    fragments = shared & (truth_share >= FRAGMENT_TRUTH_FRACTION) & (pred_share >= FRAGMENT_PREDICTION_FRACTION)
    merges = shared & (truth_share >= FRAGMENT_PREDICTION_FRACTION) & (pred_share >= FRAGMENT_TRUTH_FRACTION)
    return iou, hit, fragments, merges


def fit_disk(image):
    """Fit a circle to radial limb gradients, with robust rejection of artifacts."""
    height, width = image.shape
    center = np.array([(height - 1) / 2, (width - 1) / 2])
    length = min(image.shape)
    radii = np.arange(0.36 * length, 0.48 * length, 2.0)
    angles = np.arange(180) * (2 * np.pi / 180)
    unit = np.stack([np.sin(angles), np.cos(angles)])
    ray = center[:, None, None] + unit[:, :, None] * radii[None, None, :]
    sample = ndi.map_coordinates(image, ray, order=1, output=np.float32)
    gradient = -np.diff(ndi.gaussian_filter1d(sample, sigma=1.0, axis=1), axis=1)
    strongest = gradient.argmax(axis=1)
    positions = center[:, None] + unit * (radii[strongest] + 1.0)
    initial = np.r_[center, np.median(radii[strongest])]

    def residual(circle):
        return np.linalg.norm(positions - circle[:2, None], axis=0) - circle[2]

    result = least_squares(residual, initial, loss="soft_l1", f_scale=4.0).x
    support = float(np.mean(np.abs(residual(result)) <= 0.01 * length))
    return (*result, support)


def bounds(rles, shape, pad):
    boxes = np.asarray(coco_mask.toBbox(rles), dtype=int)
    left, top = boxes[:, :2].min(axis=0)
    right, bottom = (boxes[:, :2] + boxes[:, 2:]).max(axis=0)
    return (slice(max(0, top - pad), min(shape[0], bottom + pad)),
            slice(max(0, left - pad), min(shape[1], right + pad)))


def describe_truth(rle, image, all_truth, disk):
    roi = bounds([rle], image.shape, pad=40)
    mask = coco_mask.decode(rle)[roi].astype(bool)
    spine = skeletonize(mask)
    if not spine.any():
        spine = mask
    width = float(2.0 * np.median(ndi.distance_transform_edt(mask)[spine]))
    distance = ndi.distance_transform_edt(~mask)
    flank = (distance >= max(3.0, 0.45 * width)) & (distance <= max(7.0, 1.2 * width))
    rows, cols = np.ogrid[roi[0].start:roi[0].stop, roi[1].start:roi[1].stop]
    cy, cx, radius, support = disk
    radial = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2) / radius
    flank &= (~all_truth[roi]) & (radial < 0.98)
    intensities = image[roi]
    contrast = None
    if flank.sum() >= 16 and support >= 0.65:
        background = intensities[flank].astype(float)
        median = np.median(background)
        scale = max(2.0, 1.4826 * np.median(np.abs(background - median)))
        contrast = float((median - np.median(intensities[spine])) / scale)
    return {
        "width_px": width,
        "contrast": contrast,
        "radial_fraction": float(np.median(radial[spine])) if support >= 0.65 else None,
    }


def union_iou(truth, fragments):
    combined = coco_mask.merge(fragments)
    return float(coco_mask.iou([truth], [combined], [0])[0, 0])


def fragment_geometry(rle, shape):
    roi = bounds([rle], shape, pad=2)
    mask = coco_mask.decode(rle)[roi].astype(bool)
    spine = skeletonize(mask)
    neighbors = ndi.convolve(spine.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
    endpoints = np.argwhere(spine & (neighbors == 2))
    points = np.argwhere(spine)
    distance = ndi.distance_transform_edt(mask)
    offset = np.array([roi[0].start, roi[1].start])
    geometry = []
    for endpoint in endpoints:
        separation = np.linalg.norm(points - endpoint, axis=1)
        local = points[(separation >= 3) & (separation <= 10)]
        if len(local) < 3:
            continue
        tangent = endpoint - local.mean(axis=0)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1.0:
            continue
        width = float(2 * np.median(distance[local[:, 0], local[:, 1]]))
        geometry.append((endpoint + offset, tangent / tangent_norm, width))
    return geometry


def short_supported_gap(left, right, probability, threshold):
    if not left or not right:
        return False
    right_points = np.stack([point for point, _, _ in right])
    right_tree = cKDTree(right_points)
    minimum_cosine = np.cos(np.deg2rad(MAX_DIRECTION_DEG))
    for start, outgoing, width in left:
        max_distance = MAX_GAP_PX + width
        for index in right_tree.query_ball_point(start, max_distance):
            stop, incoming, other_width = right[index]
            chord = stop - start
            length = np.linalg.norm(chord)
            if length < 2 or min(width, other_width) / max(width, other_width) < 0.5:
                continue
            unit = chord / length
            if np.dot(outgoing, unit) < minimum_cosine or np.dot(incoming, -unit) < minimum_cosine:
                continue
            path = np.rint(np.linspace(start, stop, int(np.ceil(length)) + 1)).astype(int)
            values = np.asarray(probability[path[1:-1, 0], path[1:-1, 1]], dtype=np.float32)
            gap = values[values < threshold]
            if not len(gap) or len(gap) > min(MAX_GAP_PX, MAX_GAP_WIDTHS * max(width, other_width)):
                continue
            if np.median(gap) >= 0.4 * threshold:
                return True
    return False


def truth_slices(records, cutoffs=None):
    if cutoffs is None:
        widths = [record["width_px"] for record in records]
        contrasts = [record["contrast"] for record in records if record["contrast"] is not None]
        cutoffs = {
            "thin_max_width_px": float(np.quantile(widths, 0.25)) if widths else None,
            "faint_max_contrast": float(np.quantile(contrasts, 0.25)) if contrasts else None,
            "limb_min_radius_fraction": LIMB_RADIUS_FRACTION,
        }
    sliced = defaultdict(Counter)
    for record in records:
        tags = ["all", f"site:{record['site']}", f"date:{record['date']}"]
        if cutoffs["thin_max_width_px"] is not None and record["width_px"] <= cutoffs["thin_max_width_px"]:
            tags.append("thin")
        contrast = record["contrast"]
        if contrast is not None and cutoffs["faint_max_contrast"] is not None and contrast <= cutoffs["faint_max_contrast"]:
            tags.append("faint")
        radial = record["radial_fraction"]
        if radial is not None and radial >= cutoffs["limb_min_radius_fraction"]:
            tags.append("near_limb")
        for tag in tags + (["&".join(tags[3:])] if len(tags) > 4 else []):
            sliced[tag].update({
                "truth_instances": 1,
                "matched": int(record["matched"]),
                "splits": int(record["split"]),
                "merge_affected": int(record["merge_affected"]),
                "complete_misses": int(record["complete_miss"]),
            })
    return cutoffs, dict(sorted(sliced.items()))


def add_example(examples, kind, stem, view, score, truth, prediction):
    if not truth and not prediction:
        return
    examples[kind].append((stem, str(view), float(score), truth, prediction))


def prediction_geometry(rle, shape, disk):
    crop = bounds([rle], shape, pad=0)
    mask = coco_mask.decode(rle)[crop].astype(bool)
    rows, cols = np.where(mask)
    cy, cx, radius, support = disk
    if support < 0.65:
        return "unmeasured", None
    radial = np.hypot(rows + crop[0].start - cy, cols + crop[1].start - cx) / radius
    median = float(np.median(radial))
    if median >= 1.0:
        return "off_disk", median
    if median >= 0.95:
        return "rim", median
    if median >= LIMB_RADIUS_FRACTION:
        return "near_limb", median
    return "interior", median


def render_examples(examples, observations, probability_dir, output, limit):
    from filament.data import load_image

    manifest = []
    for kind, cases in sorted(examples.items()):
        ranked = sorted(cases, key=lambda item: item[2])
        median_score = ranked[len(ranked) // 2][2]
        ranked.sort(key=lambda item: (
            abs(item[2] - median_score), hashlib.sha256(f"{item[0]}/{item[1]}".encode()).hexdigest()
        ))
        stems = set()
        for stem, view, score, truth, prediction in ranked:
            if stem in stems:
                continue
            stems.add(stem)
            obs = observations[stem]
            roi = bounds(truth + prediction, obs.shape, pad=24)
            gray = load_image(obs.path)[roi]
            probabilities = np.load(probability_dir / f"{stem}.npy", mmap_mode="r")[roi]
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            overlay = rgb.copy()
            for items, color in ((truth, (46, 229, 247)), (prediction, (255, 175, 40))):
                for rle in items:
                    mask = coco_mask.decode(rle)[roi].astype(bool)
                    outline = mask & ~ndi.binary_erosion(mask, iterations=2)
                    overlay[outline] = color
            probability = np.asarray(np.clip(probabilities, 0, 1) * 255, dtype=np.uint8)
            panels = [Image.fromarray(panel) for panel in (rgb, overlay, probability)]
            for index, panel in enumerate(panels):
                panel.thumbnail((448, 448), Image.Resampling.LANCZOS)
                if panel.width < 160 and panel.height < 300:
                    scale = min(3.0, 160.0 / panel.width, 448.0 / panel.height)
                    panels[index] = panel.resize(
                        (round(scale * panel.width), round(scale * panel.height)), Image.Resampling.NEAREST
                    )
            panel_width = max(154, panels[0].width + 6)
            canvas = Image.new("RGB", (3 * panel_width, panels[0].height + 34), "#10141a")
            draw = ImageDraw.Draw(canvas)
            for index, (panel, label) in enumerate(zip(panels, ("H-alpha", "Truth / Prediction", "P(filament)"))):
                left = index * panel_width
                canvas.paste(panel, (left + (panel_width - panel.width) // 2, 17))
                draw.text((left + 2, 2), label, fill="white")
            draw.text((2, panels[0].height + 19), f"{kind}  {stem}  view {view}", fill="white")
            name = f"{kind}-{len(stems):02d}.png"
            canvas.save(output / name)
            manifest.append({"kind": kind, "stem": stem, "view": view, "score": score, "file": name})
            if len(stems) >= limit:
                break
    return manifest


def run(args):
    from filament.data import load_image, read_annotations, site, timestamp
    from filament.infer import instance_rles
    from filament.metrics import evaluate_rles

    observations = read_annotations(args.data)
    groups = json.loads(args.split.read_text(encoding="utf-8"))[args.partition]
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    threshold, min_area = float(settings["threshold"]), int(settings["min_area"])
    totals = Counter()
    unsupported_by_site = defaultdict(Counter)
    records = []
    examples = defaultdict(list)
    for stem in groups:
        obs = observations[stem]
        image = load_image(obs.path)
        probability = np.load(args.probabilities / f"{stem}.npy", mmap_mode="r")
        if probability.shape != obs.shape:
            raise ValueError(f"{stem}: probability shape does not match native image")
        pred = list(instance_rles(probability, threshold, min_area))
        disk = fit_disk(image)
        capture_date = timestamp(stem).date().isoformat()
        observatory = site(stem)
        geometry = {}
        spatial_region = {}
        supported = {}
        totals["disk_fit_low_support"] += int(disk[3] < 0.65)
        for view, raw_truth in obs.views.items():
            truth = list(raw_truth)
            iou, hit, fragments, merges = overlap_tables(truth, pred)
            official = evaluate_rles(truth, pred)
            splits = fragments.sum(axis=1) >= 2
            merged = merges.sum(axis=0) >= 2
            totals.update({
                "views": 1, "truth_instances": len(truth), "pred_instances": len(pred),
                "official_tp": official.tp, "official_fp": official.fp, "official_fn": official.fn,
                "official_any_overlap_splits": official.splits,
                "official_any_overlap_merges": official.merges,
                "meaningful_splits": int(splits.sum()), "meaningful_merges": int(merged.sum()),
            })
            totals["official_iou_sum"] += official.iou_sum
            truth_union = (coco_mask.decode(coco_mask.merge(truth)).astype(bool)
                           if truth else np.zeros(obs.shape, dtype=bool))
            for gi, target in enumerate(truth):
                linked_ids = np.flatnonzero(fragments[gi]).tolist()
                linked = [pred[pi] for pi in linked_ids]
                best_iou = float(iou[gi].max(initial=0.0))
                record = describe_truth(target, image, truth_union, disk)
                record.update({"date": capture_date, "site": observatory, "matched": bool(hit[gi].any()),
                               "split": bool(splits[gi]), "merge_affected": bool((merges[gi] & merged).any()),
                               "complete_miss": not bool((iou[gi] > 0).any())})
                records.append(record)
                if splits[gi]:
                    oracle = union_iou(target, linked)
                    candidate_pairs = []
                    for index, left in enumerate(linked_ids):
                        for right in linked_ids[index + 1:]:
                            key = left, right
                            if key not in supported:
                                for pi in key:
                                    if pi not in geometry:
                                        geometry[pi] = fragment_geometry(pred[pi], obs.shape)
                                supported[key] = short_supported_gap(geometry[left], geometry[right], probability, threshold)
                            if supported[key]:
                                candidate_pairs.append(key)
                    any_recoverable = any(union_iou(target, [pred[left], pred[right]]) > 0.50
                                          for left, right in candidate_pairs)
                    totals["splits_with_short_supported_gap"] += int(bool(candidate_pairs))
                    totals["splits_pair_recovers_unmatched_truth"] += int(not record["matched"] and any_recoverable)
                    totals["splits_with_unmatched_truth"] += int(not record["matched"])
                    totals["splits_union_above_iou50"] += int(oracle > 0.50)
                    totals["splits_union_recovers_unmatched_truth"] += int(not record["matched"] and oracle > 0.50)
                    add_example(examples, "split", stem, view, oracle, [target], linked)
                    if candidate_pairs:
                        add_example(examples, "supported_split", stem, view, oracle, [target], linked)
                elif record["complete_miss"]:
                    totals["complete_misses"] += 1
                    add_example(examples, "miss", stem, view, record["width_px"], [target], [])
                elif record["matched"]:
                    add_example(examples, "match", stem, view, best_iou, [target], [pred[int(iou[gi].argmax())]])
                else:
                    totals["other_unmatched_truth"] += 1
                    add_example(examples, "low_iou", stem, view, best_iou, [target], [pred[int(iou[gi].argmax())]])
            for pi, mask in enumerate(pred):
                if merged[pi]:
                    linked = [truth[gi] for gi in np.flatnonzero(merges[:, pi])]
                    add_example(examples, "merge", stem, view, iou[:, pi].max(), linked, [mask])
                if not hit[:, pi].any():
                    if not (iou[:, pi] > 0).any():
                        totals["unsupported_fp_zero_overlap"] += 1
                        add_example(examples, "unsupported_fp", stem, view, coco_mask.area(mask), [], [mask])
                        if pi not in spatial_region:
                            spatial_region[pi] = prediction_geometry(mask, obs.shape, disk)
                        region, radial = spatial_region[pi]
                        unsupported_by_site[observatory][region] += 1
                        totals[f"unsupported_fp_{region}"] += 1
                        if radial is not None:
                            add_example(examples, f"unsupported_fp_{region}", stem, view, radial, [], [mask])
                    elif fragments[:, pi].any():
                        totals["fp_with_meaningful_fragment_overlap"] += 1
                    else:
                        totals["fp_with_other_overlap"] += 1

    denominator = totals["official_tp"] + 0.5 * (totals["official_fp"] + totals["official_fn"])
    totals["official_pq"] = totals["official_iou_sum"] / denominator if denominator else 0.0
    cutoffs = None
    if args.cutoffs_from:
        cutoffs = json.loads(args.cutoffs_from.read_text(encoding="utf-8"))["slice_cutoffs"]
    cutoffs, slices = truth_slices(records, cutoffs)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = render_examples(examples, observations, args.probabilities, args.output, args.examples)
    receipt = {
        "partition": args.partition, "observation_count": len(groups), "totals": totals,
        "baseline_settings": {"threshold": threshold, "min_area": min_area},
        "unsupported_fp_regions_by_site": dict(sorted(unsupported_by_site.items())),
        "slice_cutoffs": cutoffs, "truth_slice_diagnostics": slices, "examples": manifest,
    }
    (args.output / "diagnostics.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("dev", "confirm"), default="dev")
    parser.add_argument("--cutoffs-from", type=Path)
    parser.add_argument("--examples", type=int, default=2)
    args = parser.parse_args()
    if args.partition == "confirm" and args.cutoffs_from is None:
        parser.error("--partition confirm requires a frozen development receipt via --cutoffs-from")
    if args.examples < 1:
        parser.error("--examples must be positive")
    run(args)


if __name__ == "__main__":
    main()
