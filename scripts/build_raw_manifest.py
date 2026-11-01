#!/usr/bin/env python
"""Rebuild the raw-data manifest (relative path, size, SHA-256).

Read-only with respect to the dataset: opens every file for reading and never
writes into ``data/raw/`` except for the manifest itself.

Usage:  python scripts/build_raw_manifest.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complete_journey.paths import RAW_DATASET_DIR, RAW_DIR, RAW_MANIFEST  # noqa: E402

CHUNK = 1024 * 1024
MANIFEST_FIELDS = ["relative_path", "size_bytes", "sha256"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_files(dataset_dir: Path) -> list[Path]:
    """Return the exact raw inventory represented by the manifest."""
    return sorted(
        path for path in dataset_dir.rglob("*") if path.is_file() and path.name != ".DS_Store"
    )


def build(dataset_dir: Path = RAW_DATASET_DIR, manifest: Path = RAW_MANIFEST) -> int:
    if not dataset_dir.is_dir():
        raise SystemExit(f"raw dataset not found at {dataset_dir}")

    files = dataset_files(dataset_dir)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_FIELDS)
        for path in files:
            rel = path.relative_to(RAW_DIR)
            writer.writerow([rel.as_posix(), path.stat().st_size, sha256(path)])

    total = sum(p.stat().st_size for p in files)
    print(f"wrote {manifest.relative_to(RAW_DIR.parents[1])}: {len(files)} files, {total:,} bytes")
    return 0


def verify(
    manifest: Path = RAW_MANIFEST,
    dataset_dir: Path = RAW_DATASET_DIR,
    raw_dir: Path = RAW_DIR,
) -> int:
    """Verify exact raw inventory, sizes, and hashes against the manifest."""
    failures = 0
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_FIELDS:
            print(f"INVALID  manifest header: {reader.fieldnames!r}")
            return 1
        rows = list(reader)

    manifest_paths = [row["relative_path"] for row in rows]
    invalid_paths = [
        path for path in manifest_paths if Path(path).is_absolute() or ".." in Path(path).parts
    ]
    duplicate_paths = sorted({path for path in manifest_paths if manifest_paths.count(path) > 1})
    if invalid_paths:
        print(f"INVALID  manifest path(s): {sorted(invalid_paths)}")
        failures += len(invalid_paths)
    if duplicate_paths:
        print(f"DUPLICATE manifest path(s): {duplicate_paths}")
        failures += len(duplicate_paths)

    actual_paths = {path.relative_to(raw_dir).as_posix() for path in dataset_files(dataset_dir)}
    listed_paths = set(manifest_paths)
    for relative_path in sorted(listed_paths - actual_paths):
        print(f"MISSING  {relative_path}")
        failures += 1
    for relative_path in sorted(actual_paths - listed_paths):
        print(f"UNLISTED {relative_path}")
        failures += 1

    if not invalid_paths and not duplicate_paths:
        for row in rows:
            target = raw_dir / row["relative_path"]
            if not target.is_file():
                continue
            size = target.stat().st_size
            digest = sha256(target)
            if size != int(row["size_bytes"]) or digest != row["sha256"]:
                print(f"MISMATCH {row['relative_path']}")
                failures += 1
            else:
                print(f"ok       {row['relative_path']}")
    if failures:
        print(f"FAILED: {failures} file(s) differ from the manifest")
        return 1
    print(f"all files and inventory match the manifest ({dataset_dir.name})")
    return 0


if __name__ == "__main__":
    if "--verify" in sys.argv:
        raise SystemExit(verify())
    raise SystemExit(build())
