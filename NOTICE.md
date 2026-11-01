# Data, third-party material and rights

## What the license covers

The MIT license in `LICENSE` covers the original source code only: `src/`, `scripts/`, `sql/`,
`repro/`, `tests/`, `reproduce.py` and `reproduce.sh`. It grants no right in anything else in
this repository or used by it.

## The Complete Journey data: not included

The analysis uses dunnhumby's *The Complete Journey* release. The data are not redistributed
here, in raw or processed form, and the repository contains no household-level records. Obtain
the release from dunnhumby's source-files page (<https://www.dunnhumby.com/source-files/>)
under dunnhumby's terms; `data/README.md` explains where to put the files. As read on
30 October 2026, the download is offered for research, personal or non-commercial use, and
copying, transmitting or adapting dunnhumby material requires dunnhumby's prior written
permission. Check the current terms before use.

The aggregate tables, statistics and figures shipped here (under `docs/phases/`,
`outputs/tables/`, `expected/`, `artifacts/` and `paper/figures/`) are computed from those data
and contain no household identifiers. They are provided so that the paper's results can be
checked, and the MIT license does not apply to them. The `--standard` mode writes
household-level derived tables under `build/`; keep them private.

## Learned-model responses: not included

The experiment of Section 7 queried three hosted learned decision models in October 2026:
TypeSafe's Jev (`jev-1.13.0` and `jev-1.14.0`) and Inception's Mercury Decide
(`inception/mercury-decide-20260930`, through OpenRouter). Every query described a fully
synthetic household, and no data from the panel were sent. The stored responses are not
included, because the right to redistribute them has not been confirmed for any of the three
models. The repository contains only aggregate statistics derived from them, as printed in the
paper, and the hashes needed to verify an authorized copy
(`docs/phases/PHASE_09/production_archive_freeze.json`). That file omits billing, token-count
and latency fields. Model names are the providers' trademarks and are used descriptively.

## The manuscript

`paper/main.tex`, `paper/main.pdf` and the figures in `paper/figures/` are © 2026 Artyom
Andreyev. They are included so that the reported numbers can be checked against the text, and
they are not covered by the MIT license.

## Literature and software

No publisher PDFs, full texts or excerpts of the cited literature are included. Python packages
installed from `uv.lock` are under their own licenses.
