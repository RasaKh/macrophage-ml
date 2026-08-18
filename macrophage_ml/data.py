"""Read the public GEO matrix and freeze a donor-matched analysis cohort."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

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
    # NOTE: this mapping leans on the order GEO keeps the characteristics
    # lines in -- if a clean re-run throws here the column order moved and the
    # cohort needs re-picking, not a silent re-index.
    names = ("differentiation", "stimulus", "time", "donor_id")
    for name, values in zip(names, characteristics, strict=True):
        rows[name] = [value.partition(": ")[2] for value in values]
    samples = pd.DataFrame(rows)
    for name in names:
        if samples[name].eq("").any():
            raise ValueError(f"Missing parsed {name} metadata")
    return samples, table_start
