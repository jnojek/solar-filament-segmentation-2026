"""Freeze identity for a reviewed local inference artifact without publishing it."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def local(path):
    absolute = (ROOT / path).resolve()
    if not absolute.is_relative_to(ROOT):
        raise ValueError(f"Expected a campaign-relative path: {path}")
    return absolute

def relative(path):
    return local(path).relative_to(ROOT).as_posix()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("candidate", "config", "checkpoint", "inference", "csv", "coverage", "qa", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--extra-inference-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    config = local(args.config)
    checkpoint = local(args.checkpoint)
    receipt = load(local(args.inference))
    qa = load(local(args.qa))
    test = local("artifacts/test-files.sha256.json")
    train = local("artifacts/train-files.sha256.json")
    split = local("artifacts/split.json")
    output = local(args.output)
    csv = local(args.csv)
    coverage = local(args.coverage)
    assert receipt["config_sha256"] == sha256(config)
    assert receipt["checkpoint_sha256"] == sha256(checkpoint)
    assert receipt["csv_sha256"] == sha256(csv)
    assert receipt["coverage_sha256"] == sha256(coverage)
    assert receipt["csv_bytes"] == csv.stat().st_size < 10_000_000
    assert receipt["observations"] == len(load(test)) == 180
    assert receipt["fits_final_form"] is True
    assert qa["csv_sha256"] == receipt["csv_sha256"]
    paths = [config, checkpoint, test, train, split, local("artifacts/data-identity.json")]
    paths += sorted(local("src").rglob("*.py"))
    paths += [local("qa/verify_entry.py"), local("requirements.txt")]
    result = {
        "candidate": args.candidate,
        "config": relative(args.config),
        "checkpoint": relative(args.checkpoint),
        "device": "cuda" if receipt["cuda_peak_allocated_bytes"] is not None else "cpu",
        "environment": {"python": sys.version.split()[0], "torch": receipt["torch"], "device": receipt["device"]},
        "extra_inference_args": args.extra_inference_args,
        "test_manifest": relative(test),
        "submission_sha256": sha256(csv),
        "submission_bytes": csv.stat().st_size,
        "coverage_sha256": sha256(coverage),
        "observations": receipt["observations"],
        "empty_observations": receipt["empty_observations"],
        "instances": receipt["instances"],
        "inference": {key: receipt[key] for key in ("elapsed_seconds", "ram_peak_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")},
        "artifact_sha256": {relative(path): sha256(path) for path in sorted(paths)},
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"{output}: {sha256(output)}")

if __name__ == "__main__":
    main()
