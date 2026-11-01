"""Exploratory vs final output handling and publication safety.

The frozen Phase 3 output policy (``docs/phases/PHASE_03_DESCRIPTIVE_ANALYTICS.md``,
section 11) is:

```
outputs/figures/descriptive/exploratory/   untracked (local only)
outputs/figures/descriptive/final/         tracked only by explicit allowlist
outputs/tables/descriptive/exploratory/    untracked (local only)
outputs/tables/descriptive/final/          tracked only by explicit allowlist
```

This module encodes that policy as executable checks rather than prose:

* an exploratory artifact must be Git-ignored, and that is asserted;
* a final artifact must **not** be automatically trackable — it becomes trackable
  only after a human adds one explicit per-path allowlist entry to ``.gitignore``,
  and there is deliberately no wildcard over ``outputs/*/descriptive/**``;
* a tracked/public artifact must never carry a customer identifier, so a
  ``household_key`` column is rejected before anything is written;
* the journey label -> ``household_key`` mapping stays local and untracked.

Git-ignore status is read from the real ``.gitignore`` via ``git check-ignore``
rather than re-implemented, so the check cannot drift from the policy the
repository actually enforces.
"""

from __future__ import annotations

import csv
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from complete_journey.descriptive.contracts import (
    DEFAULT_FIGURE_DPI,
    EXPLORATORY_FIGURE_FORMAT,
    EXPLORATORY_TABLE_FORMAT,
)
from complete_journey.paths import PROJECT_ROOT, WAREHOUSE_DIR

ArtifactKind = Literal["figure", "table"]
Scope = Literal["exploratory", "final"]

#: Root of every Phase 3 output.
DESCRIPTIVE_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"

#: Column names that must never appear in a tracked/public artifact.
IDENTIFIER_COLUMNS: Final[frozenset[str]] = frozenset({"household_key"})

#: Public journey labels (plan section 8). The mapping from these labels to
#: ``household_key`` is local and untracked.
JOURNEY_LABELS: Final[tuple[str, str, str, str]] = (
    "Household A",
    "Household B",
    "Household C",
    "Household D",
)

#: Local, untracked home for the journey mapping and other working state.
LOCAL_STATE_DIR: Final[Path] = WAREHOUSE_DIR / "descriptive"


class ArtifactPolicyError(RuntimeError):
    """Raised when an output would violate the frozen publication policy."""


def output_dir(kind: ArtifactKind, scope: Scope) -> Path:
    """Return the directory for a Phase 3 output."""
    if kind not in ("figure", "table"):
        raise ArtifactPolicyError(f"unknown artifact kind {kind!r}")
    if scope not in ("exploratory", "final"):
        raise ArtifactPolicyError(f"unknown artifact scope {scope!r}")
    return DESCRIPTIVE_OUTPUT_DIR / f"{kind}s" / "descriptive" / scope


def artifact_path(kind: ArtifactKind, scope: Scope, filename: str) -> Path:
    """Return the full path of a Phase 3 output artifact."""
    if not filename or Path(filename).name != filename:
        raise ArtifactPolicyError(f"artifact filename must be a bare name, got {filename!r}")
    return output_dir(kind, scope) / filename


def exploratory_path(kind: ArtifactKind, filename: str) -> Path:
    return artifact_path(kind, "exploratory", filename)


def final_path(kind: ArtifactKind, filename: str) -> Path:
    return artifact_path(kind, "final", filename)


def git_ignore_status(path: Path) -> tuple[bool, str | None]:
    """Return ``(ignored, matching_rule)`` for a path, per the repository rules.

    Works for paths that do not exist yet, which is what makes it usable *before*
    anything is written.
    """
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:  # pragma: no cover - defensive
        raise ArtifactPolicyError(f"{path} is outside the repository") from exc
    result = subprocess.run(
        ["git", "check-ignore", "-v", "--", relative],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        # ``git check-ignore -v`` prints ``<source>:<linenum>:<pattern>`` and exits 0
        # even when the last matching pattern is a negation (an allowlist entry), so
        # the pattern itself decides the answer, not the exit status.
        fields = result.stdout.strip().split("\t")
        parts = fields[0].split(":", 2)
        rule = parts[2] if len(parts) == 3 else fields[0]
        if rule.startswith("!"):
            # The path is explicitly re-included: Git does not ignore it.
            return False, None
        return True, rule
    if result.returncode == 1:
        return False, None
    raise ArtifactPolicyError(f"git check-ignore failed for {relative}: {result.stderr.strip()}")


def assert_exploratory_untracked(kind: ArtifactKind, filename: str) -> None:
    """Assert an exploratory artifact is Git-ignored and can never be committed."""
    path = exploratory_path(kind, filename)
    ignored, _ = git_ignore_status(path)
    if not ignored:
        raise ArtifactPolicyError(
            f"exploratory artifact is trackable: {path.relative_to(PROJECT_ROOT)}. "
            "Exploratory outputs must stay local/untracked."
        )


def assert_final_not_auto_trackable(kind: ArtifactKind, filename: str) -> None:
    """Assert a final artifact is not trackable until explicitly allowlisted."""
    path = final_path(kind, filename)
    ignored, rule = git_ignore_status(path)
    if not ignored:
        raise ArtifactPolicyError(
            f"final artifact is automatically trackable: {path.relative_to(PROJECT_ROOT)} "
            f"(matched rule {rule!r}). Final artifacts require an explicit per-path "
            "allowlist entry, added after that artifact is reviewed."
        )


def assert_no_identifier_columns(columns: Iterable[str]) -> None:
    """Reject a customer identifier column before anything is written."""
    offending = sorted(set(columns) & IDENTIFIER_COLUMNS)
    if offending:
        raise ArtifactPolicyError(
            f"identifier column(s) not permitted in a Phase 3 artifact: {offending}"
        )


@dataclass(frozen=True)
class PublicationStatus:
    """Where an artifact stands relative to the publication policy."""

    path: str
    scope: Scope
    git_ignored: bool
    matching_rule: str | None
    allowlisted: bool

    @property
    def publishable_now(self) -> bool:
        """True only for a final artifact that a human has explicitly allowlisted."""
        return self.scope == "final" and not self.git_ignored


def publication_status(kind: ArtifactKind, scope: Scope, filename: str) -> PublicationStatus:
    """Describe an artifact's publication state without changing anything."""
    path = artifact_path(kind, scope, filename)
    ignored, rule = git_ignore_status(path)
    return PublicationStatus(
        path=path.relative_to(PROJECT_ROOT).as_posix(),
        scope=scope,
        git_ignored=ignored,
        matching_rule=rule,
        allowlisted=not ignored,
    )


def journey_mapping_path() -> Path:
    """Local, untracked path for the journey label -> household_key mapping."""
    return LOCAL_STATE_DIR / "journey_mapping.csv"


def assert_local_only(path: Path) -> None:
    """Assert a path is Git-ignored (used for the journey mapping and scratch state)."""
    ignored, _ = git_ignore_status(path)
    if not ignored:
        raise ArtifactPolicyError(
            f"{path.relative_to(PROJECT_ROOT)} is trackable but must stay local/untracked"
        )


def write_table_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    scope: Scope | None = None,
) -> Path:
    """Write a CSV table into an approved descriptive directory.

    Enforces: the target lies inside a Phase 3 descriptive directory, the column set
    carries no customer identifier, exploratory targets are Git-ignored, and final
    targets are not auto-trackable. Publishing (``git add``) stays a human act.

    Callers holding a pandas frame can pass ``frame.columns`` and
    ``frame.itertuples(index=False, name=None)``.
    """
    resolved_scope = scope or _scope_of(path)
    assert_no_identifier_columns(columns)
    _assert_suffix(path, EXPLORATORY_TABLE_FORMAT)
    if resolved_scope == "exploratory":
        assert_exploratory_untracked(_kind_of(path), path.name)
    else:
        assert_final_not_auto_trackable(_kind_of(path), path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        # LF line endings, matching every other tracked CSV in this repository.
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(list(columns))
        writer.writerows(rows)
    return path


def write_figure_png(fig: Any, path: Path, *, dpi: int = DEFAULT_FIGURE_DPI) -> Path:
    """Write a matplotlib figure as PNG into an approved descriptive directory.

    matplotlib is imported by the caller (or the figure is passed in), so the
    infrastructure and its checks do not require a plotting backend.
    """
    scope = _scope_of(path)
    _assert_suffix(path, EXPLORATORY_FIGURE_FORMAT)
    if scope == "exploratory":
        assert_exploratory_untracked(_kind_of(path), path.name)
    else:
        assert_final_not_auto_trackable(_kind_of(path), path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=EXPLORATORY_FIGURE_FORMAT, dpi=dpi, bbox_inches="tight")
    return path


def _assert_suffix(path: Path, expected_format: str) -> None:
    """Enforce the frozen Phase 3 file format for a given artifact kind."""
    expected = f".{expected_format}"
    if path.suffix.lower() != expected:
        raise ArtifactPolicyError(
            f"Phase 3 artifacts of this kind are written as {expected}; got {path.name!r}"
        )


def _relative_to_outputs(path: Path) -> tuple[str, ...]:
    """Return path parts below ``outputs/`` or raise if the path is elsewhere."""
    resolved = path.resolve()
    root = (PROJECT_ROOT / "outputs").resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactPolicyError(
            f"{path} is not inside outputs/; Phase 3 artifacts live only under "
            "outputs/{figures,tables}/descriptive/{exploratory,final}"
        ) from exc
    return relative.parts


def _scope_of(path: Path) -> Scope:
    parts = _relative_to_outputs(path)
    if "descriptive" not in parts:
        raise ArtifactPolicyError(f"{path} is not inside a descriptive output directory")
    if "final" in parts:
        return "final"
    if "exploratory" in parts:
        return "exploratory"
    raise ArtifactPolicyError(
        f"{path} is not inside outputs/{{figures,tables}}/descriptive/{{exploratory,final}}"
    )


def _kind_of(path: Path) -> ArtifactKind:
    parts = _relative_to_outputs(path)
    if "figures" in parts:
        return "figure"
    if "tables" in parts:
        return "table"
    raise ArtifactPolicyError(f"{path} is neither a figure nor a table path")
