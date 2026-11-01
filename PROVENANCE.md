# Provenance

This note records what was fixed when, the departures from a fully prospective design that
bear on interpretation, and the hashes that tie the shipped files to the frozen study. The
paper discusses these points in Sections 5.4, 7.2, 7.5 and 8.1 and in Appendices D.4, D.7, E.6
and G.

## Frozen research version

The in-family results were frozen on 21 October 2026 and the learned-model results on
22 October 2026. The result records under `docs/phases/` and `outputs/tables/` are copied from
those frozen versions byte for byte, and so is the code under `src/`, `scripts/` and `sql/`,
with the exceptions below. Many of these files are pinned by hash in the design snapshots
(`docs/phases/*/design_snapshot.json`, which also record the commit, in the unpublished
development repository, at which each stage was fixed). Where verification depends on a file's
hash, the file is preserved byte for byte, comments included, even when a comment is out of
date; `scripts/launch_abm.py`, for example, is a development launcher kept only because the
in-family design snapshot pins it. Comments in the code are working notes written during
development; in them, terms such as "approved" and "promoted" refer to entries in the study's
decision records. The decision identifiers they cite are listed at the end of this note.

Code added or changed after the results were frozen, none of which changes a computation:

- added for the paper and this package: `scripts/paper_publication_figures.py` (draws the
  figures from the frozen records), `scripts/export_phase8_compact.py` (writes
  `artifacts/known_truth_panels.json.gz`) and `src/complete_journey/reporting/binding.py`
  (reports no binding moment when κ\* = 0, as in Table D.1);
- changed on 30 October 2026, after an independent audit: `scripts/phase9_final_artifacts.py`
  (a non-writing `--check` mode), four modules in `src/complete_journey/choice/` (a portable
  temporary-directory default in place of a machine-specific one) and
  `src/complete_journey/choice/final_contracts.py` (the hashes of those four modules and one
  comment).

## Design commitments

- **In-family experiment (Section 5).** The three model families, the calibration target, the
  five dynamic moments, the tolerance rule with reference value κ = 0.25 (and 0.15 and 0.35 as
  sensitivity), the verdict rules, the parameter grids, the known truths and the admissibility
  rule were recorded before any known-truth result or real-panel adequacy verdict was computed.
  The known-truth results were frozen (`docs/phases/PHASE_08_FINAL/rq2_freeze.json`) before the
  real-panel comparison was revealed once, at 2026-10-21T00:44Z
  (`docs/phases/PHASE_08_FINAL/real_reveal_ledger.json`).
- **Learned-model experiment (Section 7).** The criterion for an interpretable effect
  (paraphrase, placebo and repeat-noise thresholds) was fixed before any query was sent. The
  model versions, the household descriptions and their wording, the two-by-two information
  design and the stability screen were fixed before data collection began. The anchored design
  and its admissibility and degeneracy rules were fixed before any anchored truth was evaluated.

## Departures relevant to interpretation

1. **Timing of the in-family design.** The rules were fixed after the real panel's values of
   the calibration target and the five moments had been examined, and the tolerances are built
   from those moments (Section 5.4).
2. **Earlier designs.** Two earlier versions of the in-family design were abandoned after
   failing their own checks, before any known-truth result or real-panel adequacy verdict. The
   mixture specification was added at that stage, and simulated moments for every coarse grid
   point had been committed before the final design was fixed (Appendix D.4).
3. **Common calibration.** The candidate families were not recalibrated to each simulated
   panel, so the known-truth rates are conditional on the calibration to the real panel
   (Section 5.4).
4. **Anchoring.** The anchored construction of learned-model truths was adopted after every
   unanchored model-driven truth had failed the admissibility check (Section 7.5).
5. **Mercury's stability screen.** The screen stopped Mercury's data collection at its third
   check. It was replaced, for Mercury only, by a rule whose construction was chosen after the
   stop, and one element of that construction decided whether Mercury could resume. Mercury's
   results are therefore exploratory; the two Jev versions provide the cleanly screened
   learned-model evidence (Section 7.2, Appendix E.6). Some frozen records
   (`docs/phases/PHASE_09/final/*.md` and `final_summary.json`) call the original screen
   pre-registered, say it was shown to be miscalibrated, and describe its replacement as derived
   from pre-collection information only. The account in Section 7.2 and Appendix E.6 supersedes
   that wording: the miscalibration was suggested by a simulation, not established, and the
   replacement's inputs were pre-collection measurements but its construction was chosen after
   the stop, with both shifts known.
6. **Grid-resolution diagnostic.** The all-points diagnostic of Appendix D.7 was run after the
   results were frozen, on the stored panels and without new simulation. It is not part of the
   recorded design and changes no reported rate.
7. **Common shocks.** The adequacy tolerances use household-bootstrap standard errors, which
   ignore shocks common to many households, and the simulated households share no common weekly
   shocks (Section 8.1).

## Hashes

| Object | Where the hash is recorded |
|---|---|
| Every shipped file | `SHA256SUMS` |
| Raw Complete Journey files (version used) | `data/raw/manifest_raw.csv` |
| Processed tables used by the prediction benchmark | `docs/phases/PHASE_07_RUN_C/test_ledger.json` (`data_sha256`) |
| Frozen in-family design | `docs/phases/PHASE_08_FINAL/design_snapshot.sha256` and `design_snapshot.amend-1.sha256` |
| Known-truth result freeze | `6ebe1552c68411447f736d2c3b8b2b29e71020d72b2d91d4553ddf947767d78e` (`docs/phases/PHASE_08_FINAL/rq2_freeze.json`) |
| Stored learned-model responses (not shipped) | tree hashes and success digests in `docs/phases/PHASE_09/production_archive_freeze.json` |
| Learned-model simulation results | `963e65e9172cd884361c39d49ec1b543dd529989d10c25b1ac6558b0854c7c69` (direct run) and `a38d979057649663bda47561568b29ada5a8fd9ecc9d1ed847d43ae0208fa5ac` (anchored run) |
| Figure inputs | `paper/figures/PUBLICATION_MANIFEST.json` |

## Decision identifiers in the code

The study's design was governed by numbered decision records, and the frozen code cites them
by identifier. Only P8-001 to P8-006 are included (`docs/DECISIONS.md`), byte for byte, because
the frozen code verifies their hashes before it reads the stored in-family results.

| Identifier | Subject |
|---|---|
| P1-009, P2-001, P2-003, P2-005, P3-004 | data audit, sample definitions and descriptive layer |
| P4-001 to P4-004 | fixed-effects design and the Driscoll–Kraay standard errors (Section 2) |
| P5-002, P5-003 | dry-spaghetti composition design (Appendix B) |
| P7-002, P7-003 | observation rule O, dynamic moments and prediction benchmark (Sections 2.3 and 4) |
| P8-001 to P8-003 | in-family experiment: families, moments, tolerances and adequacy rule |
| P8-004, P8-005 | the two abandoned designs and the mixture specification (Appendix D.4) |
| P8-006 | final in-family design: direct simulation at the empirical calibration |
| P8-007 | in-family results frozen |
| P9-001, P9-002 | learned-model design |
| P9-003 | request manifests, wording and analysis rules fixed before any query |
| P9-004 | limits on query expenditure (no scientific content) |
| P9-005 | stability-screen bounds |
| P9-006 | Mercury's amended stability screen (Appendix E.6) |
| P9-007 | anchored learned-model truths (Section 7.5) |
| P9-008 | learned-model results frozen |
| B-009 | learned models used only as experimental instruments, never as evidence about households |
