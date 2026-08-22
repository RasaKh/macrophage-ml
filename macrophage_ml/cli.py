"""Commands for the public-data audit, preparation and evaluation."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import PROCESSED, RAW, download, prepare, parse_metadata, task_cohort
from .evaluate import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "audit", "prepare", "evaluate"):
        sub.add_parser(name)
    sub.choices["evaluate"].add_argument("--smoke", action="store_true")
    sub.choices["evaluate"].add_argument("--seed", type=int, default=42)
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


if __name__ == "__main__":
    main()
