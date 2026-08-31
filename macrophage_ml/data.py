"""Read the public GEO matrix and freeze a donor-matched analysis cohort."""

from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SOURCES = {
    "GSE46903_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE46nnn/GSE46903/matrix/GSE46903_series_matrix.txt.gz",
    "GPL6947.annot.gz": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6947/annot/GPL6947.annot.gz",
}
CONDITIONS = ("con", "IFNg", "IL4", "TNFa+PGE2+P3C")
PANEL = ("ABCA1", "ABCG1", "NR1H3", "TNF", "IL1B")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in SOURCES.items():
        path = RAW / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            request = Request(url, headers={"User-Agent": "macrophage-ml/0.1 (GEO public data audit)"})
            with urlopen(request, timeout=60) as response, open(temporary, "wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            temporary.replace(path)
        with gzip.open(path, "rb") as check:
            check.read(128)
        manifest[name] = {"url": url, "bytes": path.stat().st_size, "sha256": digest(path)}
    (RAW / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_metadata():
    path = RAW / "GSE46903_series_matrix.txt.gz"
    meta = {}
    table_start = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for number, line in enumerate(stream):
            if line.startswith("!series_matrix_table_begin"):
                table_start = number + 1
                break
            if line.startswith("!Sample_"):
                cells = next(csv.reader([line], delimiter="\t"))
                meta.setdefault(cells[0], []).append(cells[1:])
    if table_start is None:
        raise ValueError("GEO matrix table not found")
    ids = meta["!Sample_geo_accession"][0]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate GEO sample accessions")
    rows = {"sample_id": ids, "description": meta["!Sample_description"][0],
            "title": meta["!Sample_title"][0], "platform": meta["!Sample_platform_id"][0]}
    characteristics = meta["!Sample_characteristics_ch1"]
    names = ("differentiation", "stimulus", "time", "donor_id")
    for name, values in zip(names, characteristics, strict=True):
        rows[name] = [value.partition(": ")[2] for value in values]
    samples = pd.DataFrame(rows)
    for name in names:
        if samples[name].eq("").any():
            raise ValueError(f"Missing parsed {name} metadata")
    return samples, table_start


def task_cohort(samples):
    eligible = samples.loc[
        samples.description.eq("Macrophage") & samples.differentiation.eq("GM-CSF")
        & samples.time.eq("72h") & samples.stimulus.isin(CONDITIONS)
        & samples.platform.eq("GPL6947")].copy()
    counts = eligible.groupby(["donor_id", "stimulus"]).size().unstack(fill_value=0).reindex(columns=CONDITIONS, fill_value=0)
    matched = counts.index[counts.eq(1).all(axis=1)].tolist()
    cohort = eligible.loc[eligible.donor_id.isin(matched)].sort_values(["donor_id", "stimulus", "sample_id"]).reset_index(drop=True)
    if cohort.empty or cohort.donor_id.nunique() < 6:
        raise ValueError("Fewer than six donors with one profile per selected condition")
    return cohort, eligible, matched


def platform_symbols():
    symbols = {}
    with gzip.open(RAW / "GPL6947.annot.gz", "rt", errors="replace") as stream:
        for line in stream:
            if line.startswith("!platform_table_begin"):
                reader = csv.DictReader(stream, delimiter="\t")
                for row in reader:
                    if not row.get("ID") or row["ID"].startswith("!platform_table_end"):
                        break
                    symbol = row.get("Gene symbol", "").strip()
                    if re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", symbol):
                        symbols[row["ID"]] = symbol
                break
    if not symbols:
        raise ValueError("No unambiguous symbols in platform annotation")
    return symbols


def prepare():
    download()
    samples, header_line = parse_metadata()
    cohort, eligible, matched = task_cohort(samples)
    symbols = platform_symbols()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sample_ids = cohort.sample_id.tolist()
    matrix = pd.read_csv(RAW / "GSE46903_series_matrix.txt.gz", compression="gzip", sep="\t",
                         skiprows=header_line, usecols=["ID_REF", *sample_ids], low_memory=False,
                         on_bad_lines="skip")
    # The closing marker has no numeric expression and must not enter analysis.
    matrix = matrix.loc[matrix.ID_REF.isin(symbols)].copy()
    if matrix.ID_REF.duplicated().any():
        raise ValueError("Duplicate probe IDs in deposited matrix")
    matrix["gene"] = matrix.ID_REF.map(symbols)
    matrix = matrix.drop(columns="ID_REF")
    for sample_id in sample_ids:
        matrix[sample_id] = pd.to_numeric(matrix[sample_id], errors="coerce")
    genes = matrix.groupby("gene", sort=True).median(numeric_only=True).T.reindex(sample_ids)
    genes.index.name = "sample_id"
    available = [gene for gene in PANEL if gene in genes.columns and genes[gene].notna().any()]
    if not set(PANEL[:3]).issubset(available):
        raise ValueError(f"Core thesis genes unavailable: {set(PANEL[:3]) - set(available)}")
    cohort.to_csv(PROCESSED / "samples.csv", index=False)
    genes.to_pickle(PROCESSED / "genes.pkl")  # trusted local output, never user-uploaded data
    task = {
        "dataset": "GSE46903", "platform": "GPL6947", "cell_type": "Macrophage",
        "differentiation": "GM-CSF", "time": "72h", "conditions": list(CONDITIONS),
        "donors": matched, "sample_ids": sample_ids, "reference_panel": available,
        "selection": "exactly one sample per donor per condition",
        "n_samples": len(cohort), "n_donors": len(matched),
        "normalization": "Deposited log2 quantile-normalized matrix; upstream normalization may include evaluation samples",
        "source_hashes": {name: digest(RAW / name) for name in SOURCES},
        "annotation_rule": "discard ambiguous/missing symbols; median across probes per gene within a sample",
    }
    task["task_hash"] = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    (PROCESSED / "task_definition.json").write_text(json.dumps(task, indent=2) + "\n")
    audit = {
        "accessed_utc": datetime.now(timezone.utc).isoformat(), "all_series_samples": len(samples),
        "macrophage_samples": int(samples.description.eq("Macrophage").sum()),
        "eligible_before_matching": len(eligible), "eligible_donors": eligible.donor_id.nunique(),
        "included_samples": len(cohort), "included_donors": len(matched),
        "included_class_counts": {key: int(value) for key, value in cohort.stimulus.value_counts().items()},
        "mapped_genes": len(genes.columns), "reference_panel": available,
        "missing_expression_cells": int(genes.isna().sum().sum()),
        "excluded_samples": len(samples) - len(cohort), "source_hashes": task["source_hashes"],
    }
    (PROCESSED / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return task, audit
