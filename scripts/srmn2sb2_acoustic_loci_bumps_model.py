#!/usr/bin/env python3
"""Tiny Bumps CLI model for the SrMn2Sb2 acoustic-locus global search.

Bumps model arguments:

    srmn2sb2_acoustic_loci_bumps_model.py LOCUS_JSONL CASE

``CASE`` is ``all102`` or the explicitly predeclared ``center005``
common-center sensitivity.  Scientific endpoint validation is separate.
"""

from __future__ import annotations

import sys
from pathlib import Path

from bumps.fitproblem import FitProblem

MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    # Bumps executes model source from an isolated namespace after changing to
    # the model directory, but its loader does not prepend that directory to
    # sys.path in all 1.0.3 installations. Keep sibling imports explicit.
    sys.path.insert(0, str(MODEL_DIR))

from fit_srmn2sb2_acoustic_loci import PositionOnlyObjective, load_locus_pairs
from fit_srmn2sb2_acoustic_loci_bumps import BumpsPositionFitness


if len(sys.argv) != 3:
    raise SystemExit(
        "usage: srmn2sb2_acoustic_loci_bumps_model.py LOCUS_JSONL "
        "{all102,center005}"
    )

INPUT_PATH = Path(sys.argv[1]).resolve()
CASE = sys.argv[2]
if CASE not in {"all102", "center005"}:
    raise SystemExit(f"unsupported case {CASE!r}; use all102 or center005")

loaded = load_locus_pairs(
    INPUT_PATH,
    max_common_center_offset_rlu=None if CASE == "all102" else 0.05,
)
objective = PositionOnlyObjective(loaded.pairs, grid_size=129)
fitness = BumpsPositionFitness(objective)
problem = FitProblem(fitness, name=f"srmn2sb2_acoustic_{CASE}")
