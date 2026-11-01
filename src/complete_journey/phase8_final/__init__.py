"""Phase 8 final run under P8-006 (deadline-constrained direct plug-in completion).

This package is deliberately separate from ``complete_journey.abm``. The S1-P8005
snapshot hashes every ``abm/*.py`` and ``prediction/*.py`` file, so leaving them
byte-identical lets the final run re-verify that snapshot and import its direct S1A
work with full integrity checks.
"""
