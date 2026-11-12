#!/usr/bin/env python
"""Build an alignment-peak planning report from a cell, without the ICSD database.

The `suggest-alignment-peaks` workflow ranks candidate reflections by structure factor, but it
needs the private ICSD SQLite. When that file is not on the machine running at the instrument --
it is 1.5 GB and licensed, so it often is not -- the alignment state machine loses its
a4-prediction and contradiction checks entirely, and those are the checks that catch a findPeak
that latched onto the wrong reflection.

This is the fallback: an in-plane candidate list computed directly from a cell you supply and
tabulated neutron scattering lengths. It is not a substitute for the real workflow. The report
carries a provenance block saying so, with status "candidate" and no reviewer, so it cannot be
mistaken downstream for the structure-factor-ranked version.

Only the cubic Pm-3m perovskite ABO3 structure factor is implemented, because that is the cell
the BT7 LSMO experiment indexes on. Other structures need their own `structure_factor`.

Example:
    python scripts/make_alignment_peak_report.py \
        --formula La0.7Sr0.3MnO3 --a 3.9005 --ef 14.7 --a4-max 110 \
        --out local_runs/align100_20260912/alignment_peak_report_pseudocubic.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Bound coherent scattering lengths, fm, from the NIST neutron scattering lengths table.
SCATTERING_LENGTHS_FM = {"La": 8.24, "Sr": 7.02, "Mn": -3.73, "O": 5.803}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="path to write the report to")
    parser.add_argument("--a", type=float, required=True, help="cubic lattice parameter, angstrom")
    parser.add_argument("--ef", type=float, default=14.7, help="fixed final energy, meV")
    parser.add_argument("--formula", default="La0.7Sr0.3MnO3")
    parser.add_argument("--x-sr", type=float, default=0.3, help="Sr fraction on the A site")
    parser.add_argument("--a4-max", type=float, default=110.0,
                        help="upper A4 soft limit; candidates past it are dropped")
    parser.add_argument("--hkl-max", type=int, default=3, help="largest |h| and |k| to enumerate")
    parser.add_argument("--measured-note", default=None,
                        help="optional free text recording what the cell actually measured as")
    return parser.parse_args()


def structure_factor(h: int, k: int, l: int, b_a: float) -> float:
    """Cubic Pm-3m perovskite ABO3: A at (0,0,0), B at (1/2,1/2,1/2), O at the three face centres."""
    def phase(*indices: int) -> float:
        return (-1.0) ** sum(indices)

    return (b_a
            + SCATTERING_LENGTHS_FM["Mn"] * phase(h, k, l)
            + SCATTERING_LENGTHS_FM["O"] * (phase(h, k) + phase(h, l) + phase(k, l)))


def main() -> None:
    args = parse_args()
    b_a = (1.0 - args.x_sr) * SCATTERING_LENGTHS_FM["La"] + args.x_sr * SCATTERING_LENGTHS_FM["Sr"]
    kf = math.sqrt(args.ef / 2.072)
    wavelength = 2.0 * math.pi / kf

    candidates = []
    for h in range(-args.hkl_max, args.hkl_max + 1):
        for k in range(-args.hkl_max, args.hkl_max + 1):
            l = 0  # the scattering plane is [1,0,0];[0,1,0], so in-plane reflections have l = 0
            if h == 0 and k == 0:
                continue
            d = args.a / math.sqrt(h * h + k * k)
            sin_theta = wavelength / (2.0 * d)
            if not 0.0 < sin_theta < 1.0:
                continue
            two_theta = 2.0 * math.degrees(math.asin(sin_theta))
            if two_theta > args.a4_max:
                continue
            f = structure_factor(h, k, l, b_a)
            candidates.append({"hkl": [h, k, l], "d_spacing": round(d, 5),
                               "two_theta_deg": round(two_theta, 4),
                               "q_inv_angstrom": round(2.0 * math.pi / d, 5),
                               "structure_factor_fm": round(f, 4),
                               "intensity_arb": round(f * f, 3)})

    candidates.sort(key=lambda c: (-c["intensity_arb"], c["two_theta_deg"]))
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        candidate["reason"] = ("nuclear structure factor for the pseudocubic perovskite cell; "
                               "|F|^2 = %.1f fm^2 at 2theta = %.3f deg"
                               % (candidate["intensity_arb"], candidate["two_theta_deg"]))

    report = {
        "formula": args.formula,
        "scattering_plane": "[1,0,0];[0,1,0]",
        "alignment_peaks": candidates,
        "selected_structure": {
            "lattice": {"a": args.a, "b": args.a, "c": args.a,
                        "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "space_group": "P m -3 m",
            "setting": "pseudocubic perovskite; the real 300 K structure of x = 0.3 LSMO is "
                       "rhombohedral R-3c, but this experiment indexes on the pseudocubic cell "
                       "and so does NICE's UB",
            "site_model": {"A": "La/Sr at (0,0,0)", "B": "Mn at (1/2,1/2,1/2)",
                           "X": "O at the three face centres"},
        },
        "provenance": {
            "generator": "scripts/make_alignment_peak_report.py -- hand-built from the cell, "
                         "not GSAS-II/ICSD",
            "why": "suggest-alignment-peaks requires the private ICSD SQLite, which was not on "
                   "the machine at the instrument. Replace this report with the workflow's "
                   "output once that database is available.",
            "scattering_lengths_fm": SCATTERING_LENGTHS_FM,
            "ef_mev": args.ef,
            "wavelength_angstrom": round(wavelength, 5),
            "a4_max_deg": args.a4_max,
            "lattice_source": args.measured_note or "supplied on the command line",
            "status": "candidate",
            "reviewed_by": None,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(candidates)} in-plane candidates)")
    for candidate in candidates[:8]:
        print("  rank %d  %-10s 2theta=%8.4f  d=%.5f  F=%8.3f fm  |F|^2=%9.1f"
              % (candidate["rank"], str(candidate["hkl"]), candidate["two_theta_deg"],
                 candidate["d_spacing"], candidate["structure_factor_fm"],
                 candidate["intensity_arb"]))


if __name__ == "__main__":
    main()
