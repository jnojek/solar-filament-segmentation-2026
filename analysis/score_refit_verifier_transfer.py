"""Paired read-only scoring of the locked verifier on exact refit maps."""
import hashlib
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from pycocotools import mask as coco

sys.path.insert(0, "analysis")
from compare import bootstrap_delta, compare_view, measure, slice_memberships, summarize
from diagnose import describe_truth, fit_disk
from filament.data import file_sha256, load_image, read_annotations, save_new_json
from filament.infer import instance_rles
from filament.small_proposal import candidate_rles


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/small-proposal-verifier/refit-transfer-shared-blocks.json"))
    args = parser.parse_args()
    start = perf_counter()
    refit = Path("runs/higher-data-refit")
    original = Path("runs/baseline")
    directory = Path("runs/small-proposal-verifier")
    split_path = Path("runs/higher-data-refit-plan/split.json")
    split, previous_split = read(split_path), read("artifacts/split.json")
    config = read(refit / "frozen.json")
    comparison = read(directory / "comparison.json")
    verifier_path = directory / "verifier.json"
    verifier = read(verifier_path)
    assert (config["threshold"], config["min_area"]) == (.65, 700)
    assert file_sha256(verifier_path) == comparison["verifier_sha256"] == "13812ae80da15cec8dac945b29e92f4f893776e44ac18e47b91051a4057e001b"
    assert file_sha256(refit / "model/model.pt") == "f4f24f591a4386237ef04c278fe8e8e8a1761a0ef79576c1e7aab74fadca18ce"
    assert file_sha256(refit / "frozen.json") == "8f7b53fe4e1954201f517c0420623a570aaa9e9def4f02e9c9476d4e8e51c4ff"
    assert file_sha256(Path("artifacts/model.pt")) == "a4ab9442cbf9b5e3570d5eec04e626587280fe82efec2e43feaa5be7b4aa16e9"
    cutoffs = read(original / "diagnostics/diagnostics.json")["slice_cutoffs"]
    obs = read_annotations(Path("data/MAGFiLO_1.0_Kaggle_2026"))
    output = {}
    pooled = defaultdict(Counter)
    pooled_blocks = defaultdict(lambda: defaultdict(Counter))
    names = ("refit", "original_verifier", "combined")
    for fold, directories in {"confirm": (refit / "confirm-probs", original / "confirm_probs"),
                              "site_holdout": (refit / "site-probs", original / "site_probs")}.items():
        ids = split[fold]
        assert ids == previous_split[fold], fold
        assert all({file.stem for file in path.glob("*.npy")} == set(ids) for path in directories), fold
        refit_score_path = refit / f"{fold}.json"
        cache_score_path = refit / f"{fold}-cache-score.json"
        scores = read(cache_score_path)
        prior_scores = read(refit_score_path)
        measurements = {"ram_rss_bytes", "ram_peak_bytes", "elapsed_seconds"}
        assert {k: v for k, v in scores.items() if k not in measurements} == {
            k: v for k, v in prior_scores.items() if k not in measurements}, fold
        assert scores["split_sha256"] == file_sha256(split_path), fold
        assert scores["checkpoint_sha256"] == file_sha256(refit / "model/model.pt"), fold
        assert scores["config_sha256"] == file_sha256(refit / "frozen.json"), fold
        totals = defaultdict(Counter)
        blocks = defaultdict(lambda: defaultdict(Counter))
        slices = defaultdict(lambda: defaultdict(Counter))
        physical = Counter()
        digests = [hashlib.sha256(), hashlib.sha256()]
        for stem in ids:
            o = obs[stem]
            image = load_image(o.path)
            disk = fit_disk(image)
            maps = []
            for path, digest in zip(directories, digests):
                file = path / f"{stem}.npy"
                digest.update(f"{stem} {file_sha256(file)}\n".encode())
                plane = np.load(file, mmap_mode="r", allow_pickle=False)
                assert plane.shape == o.shape and plane.dtype == np.float16, stem
                maps.append(plane)
            base = list(instance_rles(maps[0], .65, 700))
            incumbent = candidate_rles(maps[1], image, .65, 700, verifier)
            treatment = candidate_rles(maps[0], image, .65, 700, verifier)
            assert treatment[:len(base)] == base, stem
            additions = treatment[len(base):]
            assert all(150 <= coco.area(mask) < 700 for mask in additions), stem
            assert not treatment or int(coco.area(coco.merge(treatment))) == sum(int(coco.area(mask)) for mask in treatment), stem
            physical.update({"refit_masks": len(base), "original_verifier_masks": len(incumbent),
                             "combined_masks": len(treatment), "added_masks": len(additions), "disks_with_additions": int(bool(additions))})
            block = split["group_by_id"][stem]
            for raw in o.views.values():
                truth = list(raw)
                measured = {name: measure(truth, masks) for name, masks in zip(names, (base, incumbent, treatment))}
                for name, result in measured.items():
                    stats = result["stats"]
                    new, lost, misses, merges = compare_view(measured["refit"], result)
                    stats.update({"new_matches_vs_refit": int(new.sum()), "lost_matches_vs_refit": int(lost.sum()),
                                  "new_misses_vs_refit": int(misses.sum()), "new_merges_vs_refit": len(merges)})
                    for table in (totals, pooled, blocks[block], pooled_blocks[block]):
                        table[name].update(stats)
                union = coco.decode(coco.merge(truth)).astype(bool)
                for index, rle in enumerate(truth):
                    attrs = describe_truth(rle, image, union, disk)
                    for tag in slice_memberships(attrs, cutoffs):
                        for name, result in measured.items():
                            slices[tag][name].update({"truth_instances": 1, "matched": int(result["matched"][index]),
                                                      "complete_misses": int(result["complete_miss"][index])})
        metrics = {name: summarize(counts) for name, counts in totals.items()}
        assert np.isclose(metrics["refit"]["pq"], scores["pq"], rtol=0, atol=1e-12), (fold, metrics["refit"]["pq"], scores["pq"])
        expected = comparison["evaluation"][fold]["metrics"]["verifier"]["pq"]
        assert np.isclose(metrics["original_verifier"]["pq"], expected, rtol=0, atol=1e-12), (fold, metrics["original_verifier"]["pq"], expected)
        output[fold] = verdict(metrics, blocks) | {"slices": slices, "physical": physical,
                                                  "map_digests": dict(zip(names[:2], (d.hexdigest() for d in digests)))}
        base, candidate = metrics["refit"], metrics["combined"]
        output[fold]["gates"].update({
            "new_matches": candidate["new_matches_vs_refit"] >= 5,
            "fewer_complete_misses": candidate["complete_misses"] < base["complete_misses"],
            "zero_overlap_fp_cost": candidate["unsupported_fp"] - base["unsupported_fp"] <= int(np.ceil(.01 * base["truth_instances"])),
            "no_new_meaningful_merges": candidate["new_merges_vs_refit"] == 0,
            "no_slice_matches_lost": all(slices[tag]["combined"]["matched"] >= slices[tag]["refit"]["matched"]
                                         for tag in ("thin", "faint", "near_limb")),
        })
    output["pooled"] = verdict({name: summarize(counts) for name, counts in pooled.items()}, pooled_blocks)
    output["pooled"]["gates"]["paired_lower_bound_positive"] = output["pooled"]["bootstrap"]["95pct_interval"][0] > 0
    output["all_refit_mask_rles_preserved_and_disjoint"] = True
    output["identity"] = {"split": file_sha256(split_path), "verifier": file_sha256(verifier_path),
                          "refit_config": file_sha256(refit / "frozen.json"), "refit_checkpoint": file_sha256(refit / "model/model.pt")}
    output["evaluation_status"] = "Exposed confirmation and Udaipur; frozen models and no transfer tuning."
    output["elapsed_seconds"] = perf_counter() - start
    save_new_json(args.output, output)
    for fold in ("confirm", "site_holdout", "pooled"):
        result = output[fold]
        print(fold, {name: {k: row[k] for k in ("pq", "tp", "fp", "fn", "unsupported_fp", "complete_misses", "meaningful_splits", "meaningful_merges")} for name, row in result["metrics"].items()})
        print("delta", result["delta"], result["bootstrap"], "gates", result["gates"])
        if "slices" in result:
            print("slices", {tag: result["slices"][tag] for tag in ("thin", "faint", "near_limb")}, "physical", result["physical"])


def verdict(metrics, blocks):
    stronger = max(("refit", "original_verifier"), key=lambda name: metrics[name]["pq"])
    delta = {name: metrics["combined"]["pq"] - metrics[name]["pq"] for name in ("refit", "original_verifier")}
    paired = {block: {"baseline": arms[stronger], "combined": arms["combined"]} for block, arms in blocks.items()}
    return {"metrics": metrics, "stronger_parent": stronger, "delta": delta,
            "bootstrap": bootstrap_delta(paired, "combined"), "gates": {"beat_stronger_parent_by_0.005": delta[stronger] >= .005}}


if __name__ == "__main__":
    main()
