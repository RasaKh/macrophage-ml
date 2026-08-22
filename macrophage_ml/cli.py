"""Commands for the public-data audit, evaluation, and release check."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import PROCESSED, RAW, download, prepare, parse_metadata, task_cohort, digest
from .evaluate import run


def verify(path):
    manifest = json.loads((path / "run_manifest.json").read_text())
    if manifest["smoke"]:
        raise ValueError("A smoke run cannot be released")
    for filename, expected in manifest["files"].items():
        if digest(path / filename) != expected:
            raise ValueError(f"Changed or missing artifact: {filename}")
    frame = pd.read_csv(path / "predictions.csv.gz")
    task = json.loads((path / "task_definition.json").read_text())
    splits = json.loads((path / "splits.json").read_text())
    if task["task_hash"] != manifest["task_hash"] or len(frame.model_id.unique()) != len(manifest["procedures"]):
        raise ValueError("Run identity or model coverage mismatch")
    for fold in splits:
        if set(fold["train_donors"]) & set(fold["held_donors"]):
            raise ValueError("Donor overlap in stored split")
    for name in manifest["procedures"]:
        rows = frame[frame.model_id.eq(name)]
        if set(rows.sample_id) != set(task["sample_ids"]) or rows.sample_id.duplicated().any():
            raise ValueError(f"Incomplete held-out coverage for {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "audit", "prepare", "evaluate", "verify-release"):
        sub.add_parser(name)
    sub.choices["evaluate"].add_argument("--smoke", action="store_true")
    sub.choices["evaluate"].add_argument("--seed", type=int, default=42)
    sub.choices["verify-release"].add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download(), indent=2))
    elif args.command == "audit":
        download()
        samples, _ = parse_metadata()
        cohort, eligible, donors = task_cohort(samples)
        print(json.dumps({"series_samples": len(samples), "macrophages": int(samples.description.eq("Macrophage").sum()),
                          "eligible_before_matching": len(eligible), "matched_donors": donors,
                          "included_samples": len(cohort)}, indent=2))
    elif args.command == "prepare":
        task, audit = prepare()
        print(json.dumps(audit, indent=2))
    elif args.command == "evaluate":
        if not (PROCESSED / "task_definition.json").exists():
            prepare()
        path, metrics = run(seed=args.seed, quick=args.smoke)
        print(path)
        print(json.dumps({name: values["balanced_accuracy"] for name, values in metrics.items()}, indent=2))
    else:
        path = Path(__file__).resolve().parents[1] / "artifacts" / args.run_id
        print(json.dumps(verify(path), indent=2))


if __name__ == "__main__":
    main()
