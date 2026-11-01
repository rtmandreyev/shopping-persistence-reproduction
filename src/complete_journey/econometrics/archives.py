"""Validate immutable accepted generating states without rewriting their provenance."""

from __future__ import annotations

import json
import subprocess

from complete_journey.choice.final_contracts import (
    FINAL_IGNORE_APPEND as PHASE5_FINAL_IGNORE_APPEND,
)
from complete_journey.econometrics.artifacts import OUTPUTS, sha256, source_identity
from complete_journey.econometrics.final_contract import (
    P4_003_APPEND,
    PHASE5_PREPARATION_IGNORE_APPEND,
    REVIEWED_COMMIT,
    approved_ignore_bytes,
    phase4_ledger_prefix,
)
from complete_journey.econometrics.samples import require
from complete_journey.paths import PROJECT_ROOT

ACCEPTED_PREFLIGHT_PROVENANCE = "25d7003716cc10a45e86f74dcce0c7f14ef904f0691d46e5a57d1015f00c1c37"
ACCEPTED_D_HASHES = (
    "d503fc5a52f7859547ac2cd6b1e58bdc81a2830b380b02c739d434832ec2e6f5",
    "60e49fe0b506adfaf97e6599a567ec863d317b7f4ffb28e7e3ac390f3e17945a",
    "20aef606a17f9ca5e11eeb2782a35433722df24ad2883721d8cb64283184fb78",
    "1b320f5c8ffebb0649d259a21e4ca57d814dab8884d3e03bff6814a06bfe7f5c",
)
CONTROL_PLANE = frozenset(
    (
        "scripts/run_panel.py",
        "src/complete_journey/econometrics/artifacts.py",
        "src/complete_journey/econometrics/core_artifacts.py",
        "src/complete_journey/econometrics/sensitivity_artifacts.py",
    )
)
# Exact authorized dependency-file revisions (P7-003 adds scikit-learn). Every package
# locked at the accepted generating state keeps its exact version, and every previously
# declared requirement is unchanged (pinned by tests). Any other change still fails.
AUTHORIZED_DEPENDENCY_REVISIONS = {
    "pyproject.toml": "9c5b346f74b17e3d30d7f1fbdf71e061aa17c6ab180966338e6e324024491059",
    "uv.lock": "9797b69ca0735437b3680dfdb3375c8b09c95b9b190f243c8135f9fba4cc2e8d",
}
PUBLICATION_ONLY_ADDITIONS = frozenset(
    f"src/complete_journey/econometrics/{name}.py"
    for name in ("archives", "final_contract", "publication")
)


def historical_bytes(root, relative, revision=REVIEWED_COMMIT):
    return subprocess.check_output(["git", "show", revision + ":" + relative], cwd=root)


def accepted_source(root=PROJECT_ROOT, revision=REVIEWED_COMMIT):
    """Algorithms stay fixed; decision/policy changes must equal exact authorization."""
    source = json.loads(historical_bytes(root, OUTPUTS["provenance"], revision))["source"]
    for relative, digest in source["files"].items():
        if relative in CONTROL_PLANE:
            continue
        path = root / relative
        require(path.is_file(), f"frozen generating dependency missing: {relative}")
        if sha256(path) == digest:
            continue
        old = historical_bytes(root, relative, revision)
        if relative == "docs/DECISIONS.md":
            require(
                phase4_ledger_prefix(path.read_bytes()).startswith(old + P4_003_APPEND.encode()),
                "historical decisions changed",
            )
        elif relative == ".gitignore":
            require(
                path.read_bytes()
                in (
                    approved_ignore_bytes(old),
                    approved_ignore_bytes(old) + PHASE5_PREPARATION_IGNORE_APPEND.encode(),
                    approved_ignore_bytes(old)
                    + PHASE5_PREPARATION_IGNORE_APPEND.encode()
                    + PHASE5_FINAL_IGNORE_APPEND.encode(),
                ),
                "unauthorized ignore-policy change",
            )
        elif AUTHORIZED_DEPENDENCY_REVISIONS.get(relative) == sha256(path):
            continue
        else:
            require(False, f"frozen generating dependency changed: {relative}")
    reviewed_paths = set(
        json.loads(historical_bytes(root, OUTPUTS["provenance"]))["source"]["files"]
    )
    current_paths = set(source_identity(root)["files"])
    require(
        reviewed_paths <= current_paths
        and current_paths - reviewed_paths <= PUBLICATION_ONLY_ADDITIONS,
        "unauthorized generating-source path addition/removal",
    )
    return source


def verify_history(root=PROJECT_ROOT):
    """All ten accepted B/C/D files, including audit provenance, remain immutable."""
    from complete_journey.econometrics.artifacts import RUN_C_PATHS, RUN_D_PATHS
    from complete_journey.econometrics.core_artifacts import ACCEPTED_CORE_BYTES, ACCEPTED_PREFLIGHT

    accepted_source(root)
    expected = {
        OUTPUTS["summary"]: ACCEPTED_PREFLIGHT["summary"],
        OUTPUTS["contract"]: ACCEPTED_PREFLIGHT["contract"],
        OUTPUTS["provenance"]: ACCEPTED_PREFLIGHT_PROVENANCE,
        **dict(zip(RUN_C_PATHS, ACCEPTED_CORE_BYTES, strict=True)),
        **dict(zip(RUN_D_PATHS, ACCEPTED_D_HASHES, strict=True)),
    }
    for relative, digest in expected.items():
        require((root / relative).is_file(), f"accepted historical artifact missing: {relative}")
        require(
            sha256(root / relative) == digest, f"accepted historical artifact changed: {relative}"
        )
    return expected
