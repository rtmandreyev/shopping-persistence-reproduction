#!/usr/bin/env python
"""Publish/check exactly F12/T8/T9 from the frozen Phase 7 Run C archive (no estimation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complete_journey.reporting.phase7_publication import (  # noqa: E402
    PublicationError,
    check,
    publish,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="rebuild and compare; write nothing")
    modes.add_argument("--publish", action="store_true", help="write the allowlisted final set")
    args = parser.parse_args(argv)
    try:
        hashes = check() if args.check else publish()
    except (PublicationError, ValueError, FileNotFoundError) as error:
        print(f"Phase 7 publication STOP: {error}", file=sys.stderr)
        return 1
    for path, digest in hashes.items():
        print(path, digest)
    print(
        "Phase 7 publication PASS: exactly F12/T8/T9 from the frozen Run C archive; no estimation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
