#!/usr/bin/env python3
"""Dense-grid Bumps LM model for the SrMn2Sb2 center-offset postcheck.

This model is deliberately separate from the grid-129 DREAM model.  It is
used only to polish the best raw-chain point after every DREAM draw has been
re-ranked with the grid-513 hard position objective.  Final acceptance is a
separate grid-1025 validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from bumps.fitproblem import FitProblem


MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from fit_srmn2sb2_acoustic_loci import PositionOnlyObjective, load_locus_pairs
from fit_srmn2sb2_acoustic_loci_bumps import BumpsPositionFitness


if len(sys.argv) != 3 or sys.argv[2] != "center005":
    raise SystemExit(
        "usage: srmn2sb2_acoustic_loci_bumps_dense_lm_model.py "
        "LOCUS_JSONL center005"
    )

INPUT_PATH = Path(sys.argv[1]).resolve()
loaded = load_locus_pairs(INPUT_PATH, max_common_center_offset_rlu=0.05)
objective = PositionOnlyObjective(loaded.pairs, grid_size=513)
fitness = BumpsPositionFitness(objective)
problem = FitProblem(fitness, name="srmn2sb2_acoustic_center005_dense_lm")
