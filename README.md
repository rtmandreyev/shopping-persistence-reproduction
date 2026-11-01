# Heterogeneity, Latent State or Purchase History?

*Known-Truth Limits on Attributing Persistence in a Household Purchase Panel*

Reproduction materials for: Artyom Andreyev (2026), "Heterogeneity, Latent State or Purchase
History? Known-Truth Limits on Attributing Persistence in a Household Purchase Panel." Working
paper, Arizona State University. The paper is [`paper/main.pdf`](paper/main.pdf), and its
source is `paper/main.tex`.

In dunnhumby's *Complete Journey* panel (2,500 households over 102 weeks at one grocery
retailer), a household that shopped at the retailer in one week is much more likely to shop
there the next. The paper asks how far standard tools can tell apart three sources of that
persistence: stable differences between households, circumstances that last several weeks, and
dependence on the household's own purchase history. It combines fixed-effects regressions, a
review of the causal designs the campaign records allow, an out-of-sample prediction benchmark
and a model comparison whose error rates are measured on simulated panels with known
mechanisms, including mechanisms whose response functions come from learned decision models.

## Reproduce the paper

With [uv](https://docs.astral.sh/uv/) installed:

```
./reproduce.sh
```

The first run installs Python 3.12 and the locked dependencies (`uv.lock`), which needs
internet access. The checks themselves make no network connection. The run ends with

```
summary: {'PASS': 9, 'FAIL': 0, 'SKIPPED-BY-DESIGN': 2, 'WARN': 0}
```

and a report in `build/reproduction_report_quick.json`. The two skipped checks need inputs
that are not distributed: the Complete Journey files and the stored learned-model responses.
If `tectonic` is not installed or has no cached TeX bundle, the paper check is an additional
`SKIPPED-BY-DESIGN` entry (here and in `--standard`), because reproduction never installs or
downloads TeX.
On a platform other than the original, the figure check may end in `WARN` (see Software).

## What the default run checks

1. Every file matches `SHA256SUMS`.
2. 99 numbers printed in the manuscript equal their values in the frozen aggregate results or
   in `expected/empirical_values.json`, formatted as printed.
3. All 53 in-family known truths of Section 6 (200 simulated panels each) are reclassified from
   their stored moments by an implementation of the adequacy rule written independently of the
   study's code, and every verdict count, family-verdict count and naive-rule count must equal
   the frozen record.
4. The real-panel comparison of Table D.1 is recomputed in the same way, including κ\*, the
   binding moments, the finer-grid analysis and the alternative observation rule.
5. The grid-resolution diagnostic of Appendix D.7 is recomputed.
6. The learned-model aggregates of Section 7 are checked against one another and against the
   in-family records.
7. The exact binomial interval for 0 of 200 panels is recomputed.
8. The three figures are redrawn from their hash-pinned inputs and compared with the published
   files.
9. The paper is compiled, if `tectonic` is available.

`./reproduce.sh --check` runs steps 1 to 7 only and ends with
`{'PASS': 7, 'FAIL': 0, 'SKIPPED-BY-DESIGN': 1, 'WARN': 0}`.

The default run does not read the Complete Journey files, does not repeat the in-family
production simulations or the learned-model simulations, and never queries a learned model.
The sections below explain what each of these needs. [`PROVENANCE.md`](PROVENANCE.md) records
when each part of the design was fixed and where the study departs from a fully prospective
design.

## Data

The Complete Journey data are not included ([`NOTICE.md`](NOTICE.md)). Download them from
dunnhumby's source-files page, <https://www.dunnhumby.com/source-files/>, under dunnhumby's
terms, and place the release folder at `data/raw/dunnhumby_The-Complete-Journey/`.
[`data/README.md`](data/README.md) lists the nine files and the folder layout.
`data/raw/manifest_raw.csv` gives the size and SHA-256 of each file in the version used for the
paper, and `--standard` checks them before using them.

```
./reproduce.sh --standard
./reproduce.sh --standard --data-dir /path/to/folder
```

The second form is for files kept elsewhere. A successful run ends with
`{'PASS': 12, 'FAIL': 0, 'SKIPPED-BY-DESIGN': 1, 'WARN': 0}`. It adds three checks: the raw
files match the manifest; 79 quantities of Sections 2 to 4 and Appendices A to C are recomputed
from the raw files by an implementation independent of the study's code; and the study's own
code rebuilds the processed data, whose two tables recorded at the time of the study must be
byte-identical. The processed tables are written to `build/workspace/data/processed/`, inside a
disposable copy of the package. They are household-level data derived from the release, so
keep `build/` private and delete it when done. Nothing under `data/` is modified.

## Reproduction modes

| Command | Needs | What it does |
|---|---|---|
| `./reproduce.sh` (or `--quick`) | nothing | the checks listed above |
| `./reproduce.sh --check` | nothing | the verification checks only, without redrawing figures or compiling |
| `./reproduce.sh --standard` | the Complete Journey files | `--quick` plus the raw-data reproduction |
| `./reproduce.sh --learned-model-simulations --archive DIR` | an authorized copy of the stored responses | reruns the Section 7 simulations and compares their result digests with the frozen values |
| `./reproduce.sh --known-truth-production` | nothing | explains why the in-family production run is not repeated |
| `./reproduce.sh --model-queries` | nothing | describes the original learned-model queries, which are not repeated |

`./reproduce.sh --help` lists all options. An option that the selected mode does not use, such
as `--data-dir` without `--standard`, is rejected as a usage error rather than ignored.

## Outputs and statuses

Apart from uv's environment (`.venv/`), every mode writes only under `build/` (or
`--build-dir DIR`) and reads no credentials. Each run ends with
`build/reproduction_report_<mode>.json`, which lists every check with one of four statuses:

- `PASS`: the check ran and its result agrees with the frozen record.
- `FAIL`: a value differs, an input that was supplied (or that `--standard` requires) is
  missing or does not match its recorded hash, or the check could not run.
- `SKIPPED-BY-DESIGN`: the check was not attempted, because it needs something that is not
  distributed (the data, the stored responses) or not installed (TeX), or because the mode
  leaves it out (`--check`).
- `WARN`: the check ran, but its result can only be exact on the original platform (the
  figures).

The exit status is 1 if any check fails, 2 for a usage error, and 0 otherwise.

## Reference runtimes

Measured on the machine used for the study (macOS, Apple silicon, 8 worker processes). Other
hardware will differ, and a heavily loaded machine can take considerably longer.

| Mode | Time |
|---|---|
| `--check` | under 1 minute |
| `--quick` | 0.5 to 2 minutes |
| `--standard` | 2 to 8 minutes |
| `--learned-model-simulations` | 2.3 hours for the direct run and 7.5 hours for the anchored run; `--small-scale` runs both at small scale in minutes |

The in-family production run of Section 6 is the expensive computation. It took 37.4 hours of
wall time on six worker processes, and its 3,948 jobs used 199.7 hours of processor time. It is
not repeated: its runner is bound to the frozen design and refuses to rerun a completed stage,
so a new run would be a new experiment. The default run instead reclassifies the stored
moments of every simulated panel (`artifacts/known_truth_panels.json.gz`).

## Learned-model responses

Section 7 uses three hosted learned decision models, queried in October 2026 at pinned
versions: `jev-1.13.0` and `jev-1.14.0` from TypeSafe, and `inception/mercury-decide-20260930`
from Inception through OpenRouter. The models were used as sources of out-of-family response functions, and every
query described a synthetic household. Hosted versions can change or be withdrawn, so the
queries cannot be guaranteed to give the same answers again, and reproduction never repeats
them.

The stored responses are not distributed, because the right to redistribute them has not been
confirmed ([`NOTICE.md`](NOTICE.md)). Without them, the Section 7 entries of Tables 4 and 5 are
checked against the shipped aggregates (`docs/phases/PHASE_09/final/final_summary.json`), and
the aggregates are checked against one another, but the simulations themselves cannot be rerun.
A holder of an authorized copy of the response archive can verify it against the recorded
hashes and rerun the simulations:

```
./reproduce.sh --learned-model-simulations --archive DIR [--checkpoint-stores DIR] [--small-scale]
```

The anchored run (Section 7.5) also reads the frozen in-family model tables, which are not
distributed either (`--checkpoint-stores`, a folder containing `phase8_final/` and
`phase8_stage1_p8005/`). Without them, that run is skipped. Both runs are checkpointed and
resume after an interruption.

## Repository structure

| Path | Contents |
|---|---|
| `reproduce.sh`, `reproduce.py` | the entry point |
| `repro/` | the reproduction checks, including the independent implementations of the adequacy rule (`adequacy.py`, `known_truth.py`, `real_panel.py`) and of the empirical results (`empirical.py`), and the manuscript-number check (`numbers.py`) |
| `expected/` | expected values and tolerances for the raw-data check and the diagnostic |
| `artifacts/known_truth_panels.json.gz` | the stored moments of every simulated in-family panel and of every family grid point |
| `src/complete_journey/`, `scripts/`, `sql/` | the study's code (`PROVENANCE.md` lists the few files added or changed after the results were frozen) |
| `docs/phases/`, `outputs/tables/` | frozen aggregate results and design snapshots |
| `paper/` | manuscript source, PDF and figures |
| `data/` | instructions and the file manifest for the raw data (no data) |
| `tests/` | tests of the reproduction checks |
| `PROVENANCE.md` | when each part of the design was fixed, departures from a prospective design, and hashes |
| `docs/DECISIONS.md` | the design records of the in-family experiment, which the frozen code verifies by hash |
| `SHA256SUMS` | checksums of every shipped file |

Directory and module names inside the study's code keep its internal stage labels. In
`docs/phases/`, `PHASE_07_RUN_C` holds the prediction benchmark and the real panel's dynamic
moments (Section 4), the `PHASE_08_*` folders the in-family known truths and the real-panel
comparison (Section 6 and Appendix D), and `PHASE_09` the learned-model truths (Section 7 and
Appendices E and F). The module `src/complete_journey/systemone/` is the learned-model code.
Its name follows TypeSafe's term for decision models that return probabilities.

## Software

- uv 0.13 or later. It installs Python 3.12 and the packages pinned in `uv.lock`, including
  some (such as JupyterLab) that the checks do not use, so even `--help` needs the first
  install. Without uv, create a Python 3.12 environment, install the versions listed in
  `uv.lock` and run `python reproduce.py`.
- Optional: [tectonic](https://tectonic-typesetting.github.io/) to compile the paper.
  Reproduction never downloads TeX resources, so tectonic needs its bundle in the local cache.
  To fetch it once, copy `paper/` to a new folder outside the repository and run
  `tectonic main.tex` there. `pdfinfo` (poppler), if present, is used to check the page count.
- Tests: `uv run --frozen --extra dev pytest tests`.
- Tested on macOS on Apple silicon, starting from a clean copy of the package. The code uses no
  macOS-specific features, but no other platform has been tested end to end. The figures are
  byte-identical only on the original platform; elsewhere the figure check reports `WARN` and
  keeps the redrawn files in `build/figures/` for visual comparison. Two other checks compare
  bytes or digests and have been confirmed only on the original platform: the processed tables
  in `--standard` and the result digests of `--learned-model-simulations`. A `FAIL` there on
  another platform may reflect numerical differences rather than an error.

## Citation

Andreyev, Artyom (2026). "Heterogeneity, Latent State or Purchase History? Known-Truth Limits
on Attributing Persistence in a Household Purchase Panel." Working paper, Arizona State
University. Machine-readable metadata are in [`CITATION.cff`](CITATION.cff).

## License

The original source code is under the MIT license ([`LICENSE`](LICENSE)). The Complete Journey
data, the learned-model outputs, the manuscript and the cited literature are not covered by it.
See [`NOTICE.md`](NOTICE.md).
