#!/usr/bin/env python
"""Publish/check exactly F11/T6/T7 from accepted Phase 5 evidence under P5-002."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complete_journey.choice.publication import check_publication, publish  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--check",
        action="store_true",
        help="verify publication by deterministic archive-only reproduction",
    )
    modes.add_argument(
        "--publish",
        action="store_true",
        help="publish exact researcher-approved final set; no estimation",
    )
    args = parser.parse_args(argv)
    try:
        report = check_publication() if args.check else publish()
    except (ValueError, FileNotFoundError) as error:
        print(f"Phase 5 publication STOP: {error}", file=sys.stderr)
        return 1
    print(
        "Phase 5 publication PASS: exactly F11/T6/T7; P5-002 baseline focal scalars only; Phase 5 ACCEPTED AND CLOSED (P5-003)"
    )
    for path, digest in sorted(report["artifacts"].items()):
        print(path, digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
