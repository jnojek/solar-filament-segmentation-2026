"""Evaluate a fixed verifier that can rescue small components from frozen native maps."""

import argparse
from collections import Counter, defaultdict
import json
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np
from pycocotools import mask as coco
from scipy.optimize import minimize
from scipy.special import expit

from compare import bootstrap_delta, measure, summarize
from filament.data import file_sha256, load_image, read_annotations, save_new_json, timestamp
from filament.small_proposal import FEATURES, LARGE_AREA, SMALL_AREA, THRESHOLD, decisions, proposals


KEEP_PROBABILITY = 0.50
RIDGE = 0.02


def truth_targets(views, large, small):
    """Use the fraction of annotation views with a new match for each proposal."""
    target = np.zeros(len(small), dtype=float)
    if not small:
        return target
    for view in views:
        truth = list(view)
        if truth:
            already_matched = measure(truth, large)["matched"]
            hits = coco.iou(truth, small, [0] * len(small)).astype(np.float32) > 0.5
            target += (hits & ~already_matched[:, None]).any(axis=0)
    return target / len(views)


def fit_verifier(features, targets):
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), 1e-6)
    standardized = np.clip((features - center) / scale, -5.0, 5.0)
    design = np.column_stack([np.ones(len(standardized)), standardized])

    def objective(weights):
        logits = design @ weights
        loss = np.mean(np.logaddexp(0.0, logits) - targets * logits) + 0.5 * RIDGE * np.square(weights[1:]).sum()
        gradient = design.T @ (expit(logits) - targets) / len(design)
        gradient[1:] += RIDGE * weights[1:]
        return loss, gradient

    result = minimize(objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True)
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Proposal verifier fit failed: {result.message}")
    return {"feature_names": list(FEATURES), "center": center.tolist(), "scale": scale.tolist(),
            "weights": result.x.tolist(), "keep_probability": KEEP_PROBABILITY, "ridge": RIDGE}


def native_maps(cache, ids):
    if {path.stem for path in cache.glob("*.npy")} != set(ids):
        raise ValueError(f"Native probability coverage does not exactly match {cache}")


def predict_disk(cache, stem, observation):
    probability = np.load(cache / f"{stem}.npy", mmap_mode="r", allow_pickle=False)
    if probability.shape != observation.shape or probability.dtype != np.float16:
        raise ValueError(f"Unexpected native probability shape or dtype for {stem}")
    return proposals(probability, load_image(observation.path))


def check_split(split):
    dev = set(split["dev"])
    confirm = set(split["confirm"])
    site = set(split["site_holdout"])
    model_train = set(split["train"])
    union = dev | confirm | site | model_train
    if len(union) != sum(map(len, (dev, confirm, site, model_train))):
        raise ValueError("Model training, verifier fitting, and evaluation share physical disks")
    groups = split["group_by_id"]
    if {groups[stem] for stem in dev} & {groups[stem] for stem in confirm}:
        raise ValueError("Verifier fit and temporal confirmation share observation groups")
    times = {stem: timestamp(stem) for stem in union}
    hours = min(abs(times[a] - times[b]).total_seconds() / 3600 for a in dev for b in confirm | site)
    if hours < split["embargo_hours"]:
        raise ValueError("Verifier fit and evaluation violate the time embargo")
    return hours


def score_fold(obs, ids, groups, cache, model):
    totals = defaultdict(Counter)
    blocks = defaultdict(lambda: defaultdict(Counter))
    physical = Counter()
    for stem in ids:
        large, small, features = predict_disk(cache, stem, obs[stem])
        accepted = [rle for rle, keep in zip(small, decisions(features, model)) if keep]
        physical.update({"baseline_instances": len(large), "small_proposals": len(small), "rescued_instances": len(accepted)})
        arms = {"baseline": large, "verifier": large + accepted, "all_small": large + small}
        for view in obs[stem].views.values():
            truth = list(view)
            scores = {name: measure(truth, masks) for name, masks in arms.items()}
            baseline = scores["baseline"]
            for name, result in scores.items():
                counts = result["stats"]
                counts["new_truth_matches"] = int((result["matched"] & ~baseline["matched"]).sum())
                totals[name].update(counts)
                blocks[groups[stem]][name].update(counts)
    metrics = {name: summarize(counts) for name, counts in totals.items()}
    base, verified = metrics["baseline"], metrics["verifier"]
    return {"observations": len(ids), "physical": dict(physical), "metrics": metrics,
            "delta_pq": verified["pq"] - base["pq"], "paired_group_bootstrap": bootstrap_delta(blocks, "verifier"),
            "additional_zero_overlap_fp": verified["unsupported_fp"] - base["unsupported_fp"],
            "added_meaningful_splits": verified["meaningful_splits"] - base["meaningful_splits"],
            "added_meaningful_merges": verified["meaningful_merges"] - base["meaningful_merges"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("data", "split", "dev-cache", "confirm-cache", "site-cache", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    nearest = check_split(split)
    paths = {"dev": args.dev_cache, "confirm": args.confirm_cache, "site_holdout": args.site_cache}
    for fold, cache in paths.items():
        native_maps(cache, split[fold])
    obs = read_annotations(args.data)
    fit_x, fit_y = [], []
    for stem in split["dev"]:
        large, small, features = predict_disk(paths["dev"], stem, obs[stem])
        fit_x.append(features)
        fit_y.append(truth_targets(tuple(obs[stem].views.values()), large, small))
    features = np.concatenate(fit_x)
    target = np.concatenate(fit_y)
    model = fit_verifier(features, target)
    args.output.mkdir(parents=True, exist_ok=False)
    save_new_json(args.output / "verifier.json", model)
    evaluations = {fold: score_fold(obs, split[fold], split["group_by_id"], paths[fold], model)
                   for fold in ("confirm", "site_holdout")}
    confirm, site = evaluations["confirm"], evaluations["site_holdout"]
    gates = {
        "confirmation_delta_at_least_0.005": confirm["delta_pq"] >= 0.005,
        "confirmation_at_least_five_new_matches": confirm["metrics"]["verifier"]["new_truth_matches"] >= 5,
        "confirmation_added_unsupported_under_one_percent": confirm["additional_zero_overlap_fp"] <= ceil(0.01 * confirm["metrics"]["baseline"]["truth_instances"]),
        "confirmation_added_merges_under_half_percent": confirm["added_meaningful_merges"] <= ceil(0.005 * confirm["metrics"]["baseline"]["truth_instances"]),
        "site_delta_above_minus_0.005": site["delta_pq"] > -0.005,
    }
    receipt = {
        "split_sha256": file_sha256(args.split), "verifier_sha256": file_sha256(args.output / "verifier.json"),
        "base_threshold": THRESHOLD, "base_min_area": LARGE_AREA, "proposal_min_area": SMALL_AREA,
        "verifier_fit_observations": len(split["dev"]), "verifier_fit_proposals": len(target),
        "fit_proposals_any_positive_view": int((target > 0).sum()), "fit_mean_target": float(target.mean()),
        "nearest_verifier_fit_to_evaluation_hours": nearest, "evaluation": evaluations, "gates": gates,
        "elapsed_seconds": perf_counter() - started,
        "evaluation_status": "Frozen baseline model never trained on verifier fit or evaluation images. Verifier fit on development only. Confirmation and Udaipur were previously exposed in other experiments.",
    }
    save_new_json(args.output / "comparison.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
