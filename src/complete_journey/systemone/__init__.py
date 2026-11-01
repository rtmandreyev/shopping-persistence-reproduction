"""Phase 9: learned System One decision policies as out-of-family known truths.

Authority: ``docs/DECISIONS.md`` P9-001 (Design D) as amended by P9-002.

Learned decision models are *artificial decision policies* over fully synthetic
household states. They are never consumers, people or evidence about human behaviour.
Scientific semantics live in provider-neutral code. Provider adapters are mechanical,
and every external call is locked behind ``systemone.authorize``. P9-001 and P9-002
authorize design and local, mock-backed implementation only.
"""
