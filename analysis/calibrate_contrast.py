"""Select one contrast floor using development predictions and a fixed small grid."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from compare import compare_view, measure, summarize


CONTRAST_FLOORS = (1.8, 2.1, 2.4, 2.7)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    from filament.challenger import contrast_accepts, scored_components
    from filament.data import load_image, read_annotations
    from filament.infer import instance_rles

    observations = read_annotations(args.data)
    development = json.loads(args.split.read_text(encoding="utf-8"))["dev"]
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    gate = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    threshold, min_area = float(settings["threshold"]), int(settings["min_area"])
    if gate["partition"] != "dev" or gate["observation_count"] != len(development):
        raise ValueError("Provide the complete development diagnostic receipt")
    if gate["baseline_settings"] != {"threshold": threshold, "min_area": min_area}:
        raise ValueError("The development diagnostic receipt uses different baseline settings")
    if gate["totals"]["unsupported_fp_zero_overlap"] <= gate["totals"].get("splits_with_short_supported_gap", 0):
        raise ValueError("Development diagnostics do not favor the local contrast challenger")
    template = json.loads(args.config.read_text(encoding="utf-8"))
    variants = {floor: {**template, "contrast_floor": floor} for floor in CONTRAST_FLOORS}
    baseline_totals = Counter()
    totals = {floor: Counter() for floor in variants}
    for stem in development:
        obs = observations[stem]
        image = load_image(obs.path)
        probability = np.load(args.probabilities / f"{stem}.npy", mmap_mode="r")
        if probability.shape != obs.shape:
            raise ValueError(f"{stem}: probability shape does not match native image")
        baseline_prediction = list(instance_rles(probability, threshold, min_area))
        scored = scored_components(probability, image, threshold, min_area, template)
        if [rle for rle, _, _ in scored] != baseline_prediction:
            raise ValueError(f"{stem}: scored components differ from the frozen baseline masks")
        predictions = {
            floor: [rle for rle, contrast, radial in scored if contrast_accepts(contrast, radial, config)]
            for floor, config in variants.items()
        }
        for view in obs.views.values():
            truth = list(view)
            baseline = measure(truth, baseline_prediction)
            baseline_totals.update(baseline["stats"])
            for floor, prediction in predictions.items():
                candidate = measure(truth, prediction)
                recovered, lost, new_misses, new_merges = compare_view(baseline, candidate)
                totals[floor].update(candidate["stats"])
                totals[floor].update({"recovered_truth_matches": int(recovered.sum()),
                                      "lost_truth_matches": int(lost.sum()), "new_complete_misses": int(new_misses.sum()),
                                      "new_false_merges": len(new_merges)})
    baseline = summarize(baseline_totals)
    tolerance = math.ceil(0.005 * baseline["truth_instances"])
    comparisons = []
    for floor, counts in totals.items():
        summary = summarize(counts)
        delta = summary["pq"] - baseline["pq"]
        eligible = (delta > 0 and summary["unsupported_fp"] < baseline["unsupported_fp"]
                    and summary["new_complete_misses"] <= tolerance)
        comparisons.append({"floor": floor, "delta_pq": delta, "eligible": eligible, "metrics": summary})
    eligible = sorted((row for row in comparisons if row["eligible"]), key=lambda row: (-row["delta_pq"], row["floor"]))
    selected = eligible[0]["floor"] if eligible else None
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "frozen.json"
    if selected is not None:
        with frozen.open("x", encoding="utf-8") as handle:
            json.dump(variants[selected], handle, indent=2)
            handle.write("\n")
    result = {
        "partition": "dev", "observations": len(development), "baseline": baseline,
        "diagnostics_sha256": sha256(args.diagnostics), "split_sha256": sha256(args.split),
        "baseline_settings_sha256": sha256(args.settings), "config_template_sha256": sha256(args.config),
        "new_complete_misses_limit": tolerance, "floor_grid": list(CONTRAST_FLOORS),
        "scores": comparisons, "selected_floor": selected,
        "frozen_config_sha256": sha256(frozen) if selected is not None else None,
    }
    with (args.output / "calibration.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"selected_floor": selected, "baseline_pq": baseline["pq"], "scores": comparisons}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
