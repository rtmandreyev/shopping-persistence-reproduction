"""Check results, statuses and the machine-readable reproduction report."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PASS, FAIL, SKIP, WARN = "PASS", "FAIL", "SKIPPED-BY-DESIGN", "WARN"


@dataclass
class Result:
    check: str
    status: str
    detail: str = ""
    seconds: float = 0.0
    data: dict = field(default_factory=dict)


class Report:
    def __init__(self, mode: str, build: Path):
        self.mode, self.build = mode, build
        self.results: list[Result] = []
        self.t0 = time.time()

    def add(self, check: str, status: str, detail: str = "", seconds: float = 0.0, **data):
        r = Result(check, status, detail, round(seconds, 2), data)
        self.results.append(r)
        mark = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP", WARN: "WARN"}[status]
        print(f"[{mark}] {check}" + (f": {detail}" if detail else ""), flush=True)
        return r

    def run(self, check: str, fn, *args, **kw):
        """Run ``fn`` which returns (status, detail[, data]); any exception is a FAIL."""
        t = time.time()
        try:
            out = fn(*args, **kw)
        except Exception as exc:  # a check that crashes is a failure, never a silent skip
            return self.add(check, FAIL, f"{type(exc).__name__}: {exc}", time.time() - t)
        status, detail, *rest = out
        return self.add(check, status, detail, time.time() - t, **(rest[0] if rest else {}))

    @property
    def failed(self) -> bool:
        return any(r.status == FAIL for r in self.results)

    def write(self) -> Path:
        self.build.mkdir(parents=True, exist_ok=True)
        path = self.build / f"reproduction_report_{self.mode}.json"
        doc = {
            "mode": self.mode,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "seconds": round(time.time() - self.t0, 1),
            "summary": {s: sum(r.status == s for r in self.results) for s in (PASS, FAIL, SKIP, WARN)},
            "results": [asdict(r) for r in self.results],
        }
        path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
        return path
