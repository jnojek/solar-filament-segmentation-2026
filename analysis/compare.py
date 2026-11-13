"""Compare frozen instance methods on the same native H-alpha probability maps."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import partial
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask
from scipy import ndimage as ndi

from diagnose import bounds, describe_truth, fit_disk, overlap_tables


def score(iou_sum, tp, fp, fn):
    denominator = tp + 0.5 * (fp + fn)
    return float(iou_sum / denominator) if denominator else 0.0


def summarize(counts):
    result = dict(sorted(counts.items()))
    result["pq"] = score(*(counts[key] for key in ("iou_sum", "tp", "fp", "fn")))
    return result


def measure(truth, prediction):
    from filament.metrics import evaluate_rles

    iou, hit, split_edges, merge_edges = overlap_tables(truth, prediction)
    official = evaluate_rles(truth, prediction)
    splits = split_edges.sum(axis=1) >= 2
    merge_sets = [(index, frozenset(np.flatnonzero(merge_edges[:, index])))
                  for index in np.flatnonzero(merge_edges.sum(axis=0) >= 2)]
    complete_miss = ~(iou > 0).any(axis=1)
    unsupported_fp = ~(iou > 0).any(axis=0)
    matched = hit.any(axis=1)
    stats = Counter({
        "annotator_views": 1, "truth_instances": len(truth), "pred_instances": len(prediction),
        "iou_sum": official.iou_sum, "tp": official.tp, "fp": official.fp, "fn": official.fn,
        "any_overlap_splits": official.splits, "any_overlap_merges": official.merges,
        "meaningful_splits": int(splits.sum()), "meaningful_merges": len(merge_sets),
        "complete_misses": int(complete_miss.sum()), "unsupported_fp": int(unsupported_fp.sum()),
    })
    if len(prediction):
        best_iou = iou.max(axis=1)
    else:
        best_iou = np.zeros(len(truth), dtype=float)
    return {
        "stats": stats, "iou": iou, "best_iou": best_iou, "matched": matched, "splits": splits,
        "merge_sets": merge_sets, "complete_miss": complete_miss, "unsupported_fp": unsupported_fp,
        "unmatched_prediction": ~hit.any(axis=0),
        "merge_affected": (merge_edges[:, [index for index, _ in merge_sets]].any(axis=1)),
    }


def compare_view(baseline, candidate):
    recovered = ~baseline["matched"] & candidate["matched"]
    lost = baseline["matched"] & ~candidate["matched"]
    new_misses = ~baseline["complete_miss"] & candidate["complete_miss"]
    baseline_sets = [indices for _, indices in baseline["merge_sets"]]
    new_merges = [(index, indices) for index, indices in candidate["merge_sets"]
                  if indices not in baseline_sets]
    return recovered, lost, new_misses, new_merges


def close_rles(result, truth_indices, prediction):
    indices = np.flatnonzero((result["iou"][list(truth_indices)] > 0).any(axis=0))
    return [prediction[index] for index in indices]


def example_cases(examples, stem, view, truth, predictions, measures, transitions):
    baseline, candidate = measures["baseline"], measures["candidate"]
    recovered, lost, _, merges = transitions
    for kind, chosen in (("recovered_match", recovered), ("lost_match", lost),
                         ("retained_match", baseline["matched"] & candidate["matched"]),
                         ("missed_by_both", baseline["complete_miss"] & candidate["complete_miss"])):
        for index in np.flatnonzero(chosen):
            truth_indices = (index,)
            delta = candidate["best_iou"][index] - baseline["best_iou"][index]
            examples[kind].append((float(delta), stem, view, [truth[index]], {
                name: close_rles(results, truth_indices, predictions[name])
                for name, results in measures.items()
            }))
    for index, truth_indices in merges:
        related = {name: close_rles(results, truth_indices, predictions[name]) for name, results in measures.items()}
        related["candidate"] = [predictions["candidate"][index]]
        examples["new_false_merge"].append((float(candidate["iou"][:, index].max()), stem, view,
                                             [truth[i] for i in sorted(truth_indices)], related))
    if baseline["unsupported_fp"].any():
        baseline_indices = np.flatnonzero(baseline["unsupported_fp"])
        isolated = [predictions["baseline"][i] for i in baseline_indices]
        overlap = (coco_mask.iou(isolated, predictions["candidate"], [0] * len(predictions["candidate"]))
                   if predictions["candidate"] else np.zeros((len(isolated), 0)))
        for index, rle in enumerate(isolated):
            if overlap[index].max(initial=0.0) < 0.10:
                related = {name: [] for name in measures}
                related["baseline"] = [rle]
                if "closing" in predictions and predictions["closing"]:
                    others = coco_mask.iou([rle], predictions["closing"], [0] * len(predictions["closing"]))[0]
                    related["closing"] = [predictions["closing"][i] for i in np.flatnonzero(others > 0)]
                examples["removed_unsupported_fp"].append((float(coco_mask.area(rle)), stem, view, [], related))
    if candidate["unsupported_fp"].any():
        candidate_indices = np.flatnonzero(candidate["unsupported_fp"])
        isolated = [predictions["candidate"][i] for i in candidate_indices]
        overlap = (coco_mask.iou(isolated, predictions["baseline"], [0] * len(predictions["baseline"]))
                   if predictions["baseline"] else np.zeros((len(isolated), 0)))
        for index, rle in enumerate(isolated):
            if overlap[index].max(initial=0.0) < 0.10:
                related = {name: [] for name in measures}
                related["candidate"] = [rle]
                examples["introduced_unsupported_fp"].append((float(coco_mask.area(rle)), stem, view, [], related))


def render_cases(examples, observations, output, limit):
    from filament.data import load_image

    files = []
    for kind, cases in sorted(examples.items()):
        center = np.median([case[0] for case in cases])
        cases.sort(key=lambda case: (abs(case[0] - center), hashlib.sha256(f"{case[1]}/{case[2]}".encode()).hexdigest()))
        shown = set()
        for value, stem, view, truth, arms in cases:
            if stem in shown:
                continue
            shown.add(stem)
            obs = observations[stem]
            crop = bounds(truth + [mask for masks in arms.values() for mask in masks], obs.shape, pad=28)
            gray = load_image(obs.path)[crop]
            base = np.repeat(gray[:, :, None], 3, axis=2)
            panels = [("H-alpha", Image.fromarray(base))]
            for name, prediction in arms.items():
                pixels = base.copy()
                for masks, color in ((truth, (46, 229, 247)), (prediction, (255, 175, 40))):
                    for rle in masks:
                        mask = coco_mask.decode(rle)[crop].astype(bool)
                        contour = mask & ~ndi.binary_erosion(mask)
                        pixels[contour] = color
                panels.append((name.replace("_", " ").title(), Image.fromarray(pixels)))
            for _, panel in panels:
                panel.thumbnail((430, 390), Image.Resampling.LANCZOS)
            width, height = panels[0][1].size
            canvas = Image.new("RGB", (len(panels) * (width + 8), height + 48), "#10141a")
            draw = ImageDraw.Draw(canvas)
            for index, (label, panel) in enumerate(panels):
                x = index * (width + 8) + 4
                draw.text((x, 4), label, fill="white")
                canvas.paste(panel, (x, 21))
            draw.text((4, height + 25), f"{kind.replace('_', ' ')} / {stem} / view {view}", fill="white")
            name = f"{kind}-{len(shown):02d}.png"
            canvas.save(output / name)
            files.append({"kind": kind, "image": stem, "view": str(view), "diagnostic": value, "file": name})
            if len(shown) >= limit:
                break
    return files


def slice_memberships(attributes, cutoffs):
    tags = []
    if attributes["width_px"] <= cutoffs["thin_max_width_px"]:
        tags.append("thin")
    if attributes["contrast"] is not None and attributes["contrast"] <= cutoffs["faint_max_contrast"]:
        tags.append("faint")
    if attributes["radial_fraction"] is not None and attributes["radial_fraction"] >= cutoffs["limb_min_radius_fraction"]:
        tags.append("near_limb")
    intersections = [f"{a}&{b}" for index, a in enumerate(tags) for b in tags[index + 1:]]
    if len(tags) == 3:
        intersections.append("&".join(tags))
    return tags + intersections


def bootstrap_delta(by_block, candidate, repetitions=2000):
    metrics = ("iou_sum", "tp", "fp", "fn")
    blocks = sorted(by_block)
    if len(blocks) < 2:
        return {"groups": len(blocks), "95pct_interval": None}
    arms = ("baseline", candidate)
    values = np.asarray([[[by_block[block][arm][key] for key in metrics] for block in blocks] for arm in arms])
    samples = np.random.default_rng(921731).integers(0, len(blocks), size=(repetitions, len(blocks)))
    sample_values = values[:, samples].sum(axis=2)
    denominator = sample_values[:, :, 1] + 0.5 * (sample_values[:, :, 2] + sample_values[:, :, 3])
    pq = np.divide(sample_values[:, :, 0], denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return {"groups": len(blocks), "repetitions": repetitions,
            "95pct_interval": np.quantile(pq[1] - pq[0], [0.025, 0.975]).tolist()}


def slice_summary(slices):
    result = {}
    for label, table in sorted(slices.items()):
        result[label] = {}
        for name, counts in table.items():
            result[label][name] = dict(counts)
            result[label][name]["recall_at_iou_over_50"] = (
                counts["matched"] / counts["truth_instances"] if counts["truth_instances"] else None
            )
            result[label][name]["mean_matched_iou"] = (
                counts["matched_iou_sum"] / counts["matched"] if counts["matched"] else None
            )
            result[label][name]["diagnostic_pq"] = score(
                counts["matched_iou_sum"], counts["matched"], counts["unmatched_predictions"],
                counts["truth_instances"] - counts["matched"],
            )
    return result


def run(args):
    from filament.challenger import candidate_rles
    from filament.data import load_image, read_annotations, site, timestamp
    from filament.infer import instance_rles

    observations = read_annotations(args.data)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    config = json.loads(args.candidate.read_text(encoding="utf-8"))
    cutoffs = json.loads(args.cutoffs_from.read_text(encoding="utf-8"))["slice_cutoffs"]
    threshold, min_area = float(settings["threshold"]), int(settings["min_area"])
    methods = {"baseline": lambda probability, image: instance_rles(probability, threshold, min_area)}
    if args.closing:
        from filament.challenger import closing_rles

        methods["closing"] = partial(closing_rles, threshold=threshold, min_area=min_area, config=config)
    methods["candidate"] = partial(candidate_rles, threshold=threshold, min_area=min_area, config=config)
    ids = split[args.partition]
    by_partition = defaultdict(Counter)
    by_site = defaultdict(lambda: defaultdict(Counter))
    by_date = defaultdict(lambda: defaultdict(Counter))
    by_block = defaultdict(lambda: defaultdict(Counter))
    slices = defaultdict(lambda: defaultdict(Counter))
    timings = Counter()
    examples = defaultdict(list)
    for stem in ids:
        obs = observations[stem]
        image = load_image(obs.path)
        probability = np.load(args.probabilities / f"{stem}.npy", mmap_mode="r")
        if probability.shape != obs.shape:
            raise ValueError(f"{stem}: probability shape does not match native image")
        capture_date = timestamp(stem).date()
        block = (capture_date.toordinal() - 1) // split["group_days"]
        disk = fit_disk(image)
        predictions = {}
        prediction_slices = {}
        prediction_unions = {}
        for name, method in methods.items():
            start = perf_counter()
            predictions[name] = list(method(probability, image))
            prediction_slices[name] = {}
            timings[name] += perf_counter() - start
            count = {"observation_count": 1, "low_support_disk_count": int(disk[3] < 0.65)}
            for table in (by_partition, by_site[site(stem)], by_date[capture_date.isoformat()], by_block[block]):
                table[name].update(count)
        for view, annotation in obs.views.items():
            truth = list(annotation)
            measures = {name: measure(truth, pred) for name, pred in predictions.items()}
            for name, result in measures.items():
                recovered, lost, new_misses, new_merges = compare_view(measures["baseline"], result)
                counts = result["stats"]
                counts.update({"recovered_truth_matches": int(recovered.sum()), "lost_truth_matches": int(lost.sum()),
                               "new_complete_misses": int(new_misses.sum()), "new_false_merges": len(new_merges)})
                for table in (by_partition, by_site[site(stem)], by_date[capture_date.isoformat()], by_block[block]):
                    table[name].update(counts)
                unmatched = np.flatnonzero(result["unmatched_prediction"])
                if len(unmatched) and name not in prediction_unions:
                    prediction_unions[name] = coco_mask.decode(coco_mask.merge(predictions[name])).astype(bool)
                for index in unmatched:
                    if index not in prediction_slices[name]:
                        attrs = describe_truth(predictions[name][index], image, prediction_unions[name], disk)
                        prediction_slices[name][index] = slice_memberships(attrs, cutoffs)
                    for label in prediction_slices[name][index]:
                        slices[label][name]["unmatched_predictions"] += 1
                if name == "candidate":
                    example_cases(examples, stem, view, truth, predictions, measures,
                                  (recovered, lost, new_misses, new_merges))
            if not truth:
                continue
            union = coco_mask.decode(coco_mask.merge(truth)).astype(bool)
            for index, rle in enumerate(truth):
                attrs = describe_truth(rle, image, union, disk)
                for label in slice_memberships(attrs, cutoffs):
                    for name, result in measures.items():
                        matched = bool(result["matched"][index])
                        iou = float(result["best_iou"][index])
                        slices[label][name].update({
                            "truth_instances": 1, "matched": int(matched),
                            "splits": int(result["splits"][index]), "merge_affected": int(result["merge_affected"][index]),
                            "complete_misses": int(result["complete_miss"][index]),
                            "matched_iou_sum": float(np.float32(iou)) if matched else 0.0,
                        })

    args.output.mkdir(parents=True, exist_ok=True)
    overlays = render_cases(examples, observations, args.output, args.examples)
    aggregate = {name: summarize(stats) for name, stats in by_partition.items()}
    receipt = {
        "partition": args.partition, "observation_count": len(ids), "holdout_site": split.get("holdout_site"),
        "slice_cutoffs": cutoffs, "baseline_settings": {"threshold": threshold, "min_area": min_area},
        "candidate_config": config, "postprocess_seconds": dict(timings), "metrics": aggregate,
        "paired_pq_delta": {name: aggregate[name]["pq"] - aggregate["baseline"]["pq"] for name in methods},
        "paired_group_bootstrap": {name: bootstrap_delta(by_block, name) for name in methods if name != "baseline"},
        "site": {key: {name: summarize(stats) for name, stats in table.items()} for key, table in sorted(by_site.items())},
        "date": {key: {name: summarize(stats) for name, stats in table.items()} for key, table in sorted(by_date.items())},
        "slices": slice_summary(slices),
        "slice_pq_definition": "Match full annotator views first. Matched pairs and missed truths use truth membership. Unmatched predictions use their own membership. Overlapping slices are separate diagnostics.",
        "examples": overlays,
    }
    (args.output / "comparison.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": aggregate, "paired_pq_delta": receipt["paired_pq_delta"],
                      "paired_group_bootstrap": receipt["paired_group_bootstrap"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cutoffs-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("dev", "confirm", "site_holdout"), default="dev")
    parser.add_argument("--closing", action="store_true")
    parser.add_argument("--examples", type=int, default=2)
    args = parser.parse_args()
    if args.examples < 1:
        parser.error("--examples must be positive")
    run(args)


if __name__ == "__main__":
    main()
