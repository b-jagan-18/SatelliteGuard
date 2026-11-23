"""Download the OPSSAT-AD Zenodo record, verify checksums and write a data manifest.

Usage:
    python -m satelliteguard.data.download            # download missing files
    python -m satelliteguard.data.download --force    # re-download everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from satelliteguard import load_config

API_URL = "https://zenodo.org/api/records/{record}"


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_rows(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1  # minus header


def download(force: bool = False) -> dict:
    cfg = load_config()
    raw_dir: Path = cfg["paths"]["raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_id = cfg["data"]["zenodo_record"]

    resp = requests.get(API_URL.format(record=record_id), timeout=30)
    resp.raise_for_status()
    record = resp.json()

    files = []
    for f in record["files"]:
        name = f["key"]
        algo, expected = f["checksum"].split(":")
        target = raw_dir / name

        if force or not target.exists() or _hash(target, algo) != expected:
            print(f"Downloading {name} ({f['size'] / 1e6:.1f} MB)...")
            with requests.get(f["links"]["self"], stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(target, "wb") as out:
                    for chunk in r.iter_content(1 << 20):
                        out.write(chunk)
        else:
            print(f"OK (cached) {name}")

        actual = _hash(target, algo)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: {actual} != {expected}")

        entry = {
            "file": name,
            "size_bytes": target.stat().st_size,
            "md5": actual if algo == "md5" else _hash(target, "md5"),
            "sha256": _hash(target, "sha256"),
        }
        if name.endswith(".csv"):
            entry["rows"] = _count_rows(target)
        files.append(entry)

    manifest = {
        "source": f"https://zenodo.org/records/{record_id}",
        "doi": record.get("doi"),
        "license": (record["metadata"].get("license") or {}).get("id"),
        "record_updated": record.get("updated"),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": sorted(files, key=lambda e: e["file"]),
    }
    manifest_path: Path = cfg["paths"]["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download all files")
    download(force=parser.parse_args().force)
