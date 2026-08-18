"""Read the public GEO matrix and freeze a donor-matched analysis cohort."""

import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SOURCES = {
    "GSE46903_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE46nnn/GSE46903/matrix/GSE46903_series_matrix.txt.gz",
    "GPL6947.annot.gz": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6947/annot/GPL6947.annot.gz",
}


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
