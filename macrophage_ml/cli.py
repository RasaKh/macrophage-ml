"""Commands for the public-data audit and preparation."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import PROCESSED, RAW, download, prepare, parse_metadata, task_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "audit", "prepare"):
        sub.add_parser(name)
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


if __name__ == "__main__":
    main()
