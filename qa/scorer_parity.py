"""Compare the local implementation with the organizer notebook on CPU."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util

from filament.format import HEADER, SHAPE, read_rows
from filament.metrics import score_submission

OFFICIAL_FUNCTIONS = {
    "get_overlap_matrices",
    "fp_count_hit",
    "fn_count_hit",
    "rles_to_layers",
    "process_entry",
    "get_overlap_df",
    "get_pq_score",
}
OFFICIAL_SHA256 = "ae7c9445b4eee2b77b216df5c9772fdf92fda15611e98abd7ef5f02a4e380801"


def load_official_functions(path: Path, sha256: str):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256.lower():
        raise ValueError(f"Organizer notebook checksum changed: {path}")
    notebook = json.loads(raw)
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not any(f"def {name}(" in source for name in OFFICIAL_FUNCTIONS):
            continue
        definitions.extend(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in OFFICIAL_FUNCTIONS
        )
    if {node.name for node in definitions} != OFFICIAL_FUNCTIONS:
        raise ValueError("Official notebook is missing expected scorer functions")
    environment = {"np": np, "pd": pd, "torch": torch, "mask_util": mask_util}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), environment)
    return environment


def compare(official, gt_rows, pred_rows, expected=None):
    gt = pd.DataFrame(list(gt_rows), columns=HEADER, dtype=str)
    pred = pd.DataFrame(list(pred_rows), columns=HEADER, dtype=str)
    gt = gt.astype("string")
    pred = pred.astype("string")
    overlap = official["get_overlap_df"](gt, pred)
    official_pq = official["get_pq_score"](overlap)
    local = score_submission(gt.itertuples(index=False, name=None), pred.itertuples(index=False, name=None))
    if not math.isclose(local.pq, official_pq, rel_tol=1e-7, abs_tol=1e-9):
        raise AssertionError(f"PQ mismatch: local {local.pq}, organizer {official_pq}")

    tp = fp = fn = splits = merges = 0
    ious, dices = [], []
    for row in overlap.itertuples(index=False):
        hit = row.iou_matrix > 0.5
        intersects = row.iou_matrix > 0
        tp += int(hit.sum())
        fp += int((~hit.any(dim=0)).sum())
        fn += int((~hit.any(dim=1)).sum())
        splits += int((intersects.sum(dim=1) > 1).sum())
        merges += int((intersects.sum(dim=0) > 1).sum())
        ious.extend(row.iou_matrix[intersects].tolist())
        dices.extend(row.dice_matrix[intersects].tolist())
    if (local.tp, local.fp, local.fn, local.splits, local.merges) != (tp, fp, fn, splits, merges):
        raise AssertionError("Organizer TP, FP, FN or split/merge parity failed")
    np.testing.assert_allclose(local.ious, ious, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(local.dices, dices, rtol=1e-7, atol=1e-9)
    if expected is not None:
        exp_pq, exp_tp, exp_fp, exp_fn, exp_splits, exp_merges = expected
        if not math.isclose(official_pq, exp_pq, rel_tol=1e-7) or (
            tp, fp, fn, splits, merges
        ) != (exp_tp, exp_fp, exp_fn, exp_splits, exp_merges):
            raise AssertionError("Unexpected organizer result for the fixture")
    return {
        "local_pq": local.pq, "organizer_pq": official_pq, "tp": tp, "fp": fp, "fn": fn,
        "splits": splits, "merges": merges, "annotator_images": len(overlap),
        "truth_rows": len(gt), "prediction_rows": len(pred),
    }


def fixture_rows():
    first = np.zeros(SHAPE, dtype=np.uint8, order="F")
    first[23:27, 71:78] = 1
    second = np.zeros_like(first, order="F")
    second[27:31, 71:78] = 1
    thin = np.zeros_like(first, order="F")
    thin[28:29, 71:78] = 1
    disjoint = np.zeros_like(first, order="F")
    disjoint[121:123, 387:396] = 1

    def encoded(mask):
        return mask_util.encode(np.asfortranarray(mask))["counts"].decode("ascii")

    left, right, both, small, extra = map(encoded, (first, second, first | second, thin, disjoint))
    cases = {
        "perfect": ([("A-I_1", both)], [("I_1", both)], (1, 1, 0, 0, 0, 0)),
        "miss": ([("A-I_1", both)], [], (0, 0, 0, 1, 0, 0)),
        "split_at_strict_half": (
            [("A-I_1", both)], [("I_1", left), ("I_2", right)], (0, 0, 2, 1, 1, 0),
        ),
        "merge_at_strict_half": (
            [("A-I_1", left), ("A-I_2", right)], [("I_1", both)], (0, 0, 1, 2, 0, 1),
        ),
        "subthreshold_fragments_and_false_positive": (
            [("A-I_1", both)], [("I_1", left), ("I_2", small), ("I_3", extra)],
            (0, 0, 3, 1, 1, 0),
        ),
        "annotators_scored_separately": (
            [("A-I_1", left), ("B-I_1", right)], [("I_1", left)],
            (0.5, 1, 1, 1, 0, 0),
        ),
        "overlapping_gt_counts_both_hits": (
            [("A-I_1", left), ("A-I_2", encoded(first | thin))], [("I_1", left)],
            (0.9, 2, 0, 0, 0, 1),
        ),
        "unannotated_prediction_image_ignored": (
            [("A-I_1", left)], [("I_1", left), ("J_1", extra)],
            (1, 1, 0, 0, 0, 0),
        ),
    }
    return cases


def development_rows(data_root, split_path, probabilities, threshold, min_area, limit):
    from pycocotools.coco import COCO

    from filament.infer import instance_rles

    split = json.loads(split_path.read_text(encoding="utf-8"))
    available = [image_id for image_id in split["dev"] if (probabilities / f"{image_id}.npy").is_file()]
    if limit is None and len(available) != len(split["dev"]):
        raise ValueError(f"Development cache is incomplete: {len(available)}/{len(split['dev'])} maps")
    if limit is not None:
        available = available[:limit]
    if not available:
        raise ValueError(f"No development probability maps available in {probabilities}")
    annotations = data_root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = COCO(str(annotations))
    wanted = set(available)
    truth, predictions = [], []
    for view_id, image in coco.imgs.items():
        if Path(image["file_name"]).stem not in wanted:
            continue
        truth.extend(
            (f"{view_id}_{index}", coco.annToRLE(annotation)["counts"].decode("ascii"))
            for index, annotation in enumerate(coco.imgToAnns[view_id], 1)
        )
    for image_id in available:
        probability = np.load(probabilities / f"{image_id}.npy", allow_pickle=False)
        if probability.shape != SHAPE:
            raise ValueError(f"{image_id} probability map is not native size")
        predictions.extend(
            (f"{image_id}_{index}", rle["counts"].decode("ascii"))
            for index, rle in enumerate(instance_rles(probability, threshold, min_area), 1)
        )
    if {row[0].split("_", 1)[0].split("-", 1)[1] for row in truth} != wanted:
        raise ValueError("Independent COCO readback did not find truth for every audited image")
    return truth, predictions, available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--organizer", type=Path, default=Path(".local/official_scorer.ipynb"))
    parser.add_argument("--organizer-sha256", default=OFFICIAL_SHA256)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", type=Path, default=Path("runs/split.json"))
    parser.add_argument("--dev-probabilities", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-area", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.truth) != bool(args.predictions):
        parser.error("--truth and --predictions must be supplied together")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.dev_probabilities and (args.truth or any(
        value is None for value in (args.data_root, args.threshold, args.min_area)
    )):
        parser.error("--dev-probabilities needs --data-root, --threshold and --min-area in place of CSV inputs")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    official = load_official_functions(args.organizer, args.organizer_sha256)
    report = {
        "organizer_sha256": args.organizer_sha256,
        "synthetic": {
            name: compare(official, *case)
            for name, case in fixture_rows().items()
        },
    }
    if args.truth:
        report["real"] = compare(official, read_rows(args.truth), read_rows(args.predictions))
    if args.dev_probabilities:
        gt, pred, image_ids = development_rows(
            args.data_root, args.split, args.dev_probabilities, args.threshold, args.min_area, args.limit,
        )
        report["real"] = {
            **compare(official, gt, pred), "image_ids": image_ids,
            "threshold": args.threshold, "min_area": args.min_area,
        }
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
