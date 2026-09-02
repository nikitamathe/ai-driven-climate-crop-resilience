#!/usr/bin/env python3
"""Compute and optionally assert SHA-256 checksums for raw datasets (E08 CI).

Usage:
    python scripts/check_data_checksums.py            # print checksums (info)
    python scripts/check_data_checksums.py --assert   # fail if changed vs manifest

The manifest (``data/raw/manifest.json``) records the expected SHA-256 of every
tracked raw input. In CI (GitHub Actions) we run with ``--assert`` so that any
accidental modification of raw datasets fails the pipeline loudly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import RAW_DIR

# Relative (to RAW_DIR) list of raw inputs that must never change.
WATCHED_FILES = [
    "crop_yield/Custom_Crops_yield_Historical_Dataset.csv",
    "nasa_power/nasa_power_updated.csv",
]


def _manifest_path() -> Path:
    """Return the manifest path, derived from the current RAW_DIR at call time.

    Deriving it dynamically (rather than binding a module constant) means both
    the CLI and tests that monkeypatch ``RAW_DIR`` write to and read from the
    same logical location.
    """
    return RAW_DIR / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_checksums() -> dict:
    checksums = {}
    for rel in WATCHED_FILES:
        path = RAW_DIR / rel
        if not path.exists():
            checksums[rel] = None
        else:
            checksums[rel] = sha256(path)
    return checksums


def write_manifest() -> dict:
    checksums = compute_checksums()
    with open(_manifest_path(), "w", encoding="utf-8") as fh:
        json.dump({"files": checksums}, fh, indent=2, sort_keys=True)
    return checksums


def load_manifest() -> dict | None:
    manifest = _manifest_path()
    if not manifest.exists():
        return None
    with open(manifest, "r", encoding="utf-8") as fh:
        return json.load(fh).get("files", {})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assert", dest="do_assert", action="store_true",
        help="Fail if any watched raw file differs from the recorded manifest.",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Write a fresh manifest.json from the current file checksums.",
    )
    args = parser.parse_args(argv)

    if args.write:
        checksums = write_manifest()
        print(f"[checksum] wrote {_manifest_path()}")
        for rel, c in checksums.items():
            print(f"  {rel}: {c}")
        return 0

    current = compute_checksums()

    if not args.do_assert:
        for rel, c in current.items():
            print(f"{rel}: {c}")
        return 0

    recorded = load_manifest()
    if recorded is None:
        print("[checksum] no manifest.json found; run with --write first")
        return 1

    failures = []
    for rel in WATCHED_FILES:
        cur, rec = current.get(rel), recorded.get(rel)
        if cur is None:
            failures.append(f"{rel}: MISSING on disk")
        elif rec is None:
            failures.append(f"{rel}: not recorded in manifest")
        elif cur != rec:
            failures.append(f"{rel}: checksum mismatch")
    if failures:
        print("[checksum] FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[checksum] OK: all watched raw files match manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
