#!/usr/bin/env python
"""Run the LSMO reference grid through the tasreduction pipeline.

The point is not to produce a final reduced dataset -- it is to find out which
corrections the shipped pipeline actually applies to a fixed-Ef constant-E grid,
and to test the ones it does apply against a physical invariant the data can
check on its own: detailed balance between the E<0 and E>0 rows.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from tasreduction.dataset import TripleAxisScan, BOLTZMANN
from tasreduction.instrument import TripleAxisInstrument

MANIFEST = Path(sys.argv[1])
MON_REF = 150000.0
TEMPERATURE = 300.0     # NICE writes nan; room temperature asserted, see handoff
EF = 14.7


def load_rows(manifest: Path):
    """Group counted points into one constant-E row per scan file."""
    rows: dict[float, list[dict]] = {}
    for line in manifest.read_text().splitlines()[1:]:
        r = line.split("\t")
        if r[16] != "count" or not r[13] or not r[14]:
            continue
        counts, monitor = float(r[13]), float(r[14])
        if monitor <= 0:
            continue
        E = round(float(r[5]), 3)
        rows.setdefault(E, []).append({
            "K": float(r[4]),
            "dq": abs(float(r[4]) - round(float(r[4]))),
            "counts": counts,
            "monitor": monitor,
            "time": float(r[15]) if r[15] else 0.0,
        })
    return {E: sorted(v, key=lambda p: p["K"]) for E, v in sorted(rows.items())}


def build_scan(E: float, pts: list[dict]) -> TripleAxisScan:
    axis = np.array([p["K"] for p in pts])
    counts = np.array([p["counts"] for p in pts])
    return TripleAxisScan(
        instrument=TripleAxisInstrument(),
        axis=axis,
        counts=counts,
        errors=np.sqrt(np.clip(counts, 1.0, None)),   # Poisson
        monitor=np.array([p["monitor"] for p in pts]),
        time=np.array([p["time"] for p in pts]),
        # The corrections read E and Temp off metadata, not off the scan object.
        metadata={"E": np.full(len(pts), E), "Temp": np.full(len(pts), TEMPERATURE)},
    )


def main() -> None:
    rows = load_rows(MANIFEST)
    print("loaded %d constant-E rows, %d points"
          % (len(rows), sum(len(v) for v in rows.values())))

    # --- stage 1: monitor normalization (the one correction that is wired up) --
    scans = {}
    for E, pts in rows.items():
        scan = build_scan(E, pts)
        scan.normalize_to_monitor(MON_REF)
        scans[E] = scan
    print("stage 1 monitor normalization -> treatments: %s"
          % scans[min(scans)].treatments)

    # --- stage 2: what happens if we ask for the other two corrections --------
    probe = build_scan(2.0, rows[2.0]).clone() if 2.0 in rows else None
    if probe is not None:
        probe.normalize_to_monitor(MON_REF)
        for label, fn in (
            ("monitor_correction (harmonics)", lambda s: s.monitor_correction([1.0])),
            ("resolution_correction", lambda s: s.resolution_correction(EF + 2.0, 3.35416)),
            ("detailed_balance", lambda s: s.detailed_balance()),
        ):
            trial = probe.clone()
            try:
                fn(trial)
                ratio = float(np.mean(trial.counts / probe.counts))
                print("stage 2 %-34s OK   mean scale = %.6f" % (label, ratio))
            except Exception as exc:                      # noqa: BLE001
                print("stage 2 %-34s FAIL %s: %s" % (label, type(exc).__name__, exc))

    # --- test: detailed balance between the +E and -E rows -------------------
    # S(-E) / S(+E) = exp(-E/kT) is a property of the sample, not the fit, so it
    # tests the whole normalization chain against something we did not put in.
    print("\ndetailed-balance cross-check at T = %.0f K (kT = %.2f meV)"
          % (TEMPERATURE, BOLTZMANN * TEMPERATURE))
    print("  E   dq range      I(-E)/I(+E)     expected    pull")
    for E in (1.0, 2.0):
        if E not in scans or -E not in scans:
            continue
        pos, neg = scans[E], scans[-E]
        for lo, hi in ((0.0, 0.40), (0.0, 0.12), (0.12, 0.40)):
            num = den = vnum = vden = 0.0
            for p, n in zip(rows[E], rows[-E]):
                dq = p["dq"]
                if not (lo <= dq < hi):
                    continue
                i_pos = p["counts"] / p["monitor"] * MON_REF
                i_neg = n["counts"] / n["monitor"] * MON_REF
                den += i_pos; num += i_neg
                vden += i_pos ** 2 / max(p["counts"], 1.0)
                vnum += i_neg ** 2 / max(n["counts"], 1.0)
            if den <= 0:
                continue
            ratio = num / den
            sigma = ratio * math.sqrt(vnum / num ** 2 + vden / den ** 2)
            expect = math.exp(-E / (BOLTZMANN * TEMPERATURE))
            print("  %+.0f  %.2f-%.2f   %.4f +/- %.4f    %.4f     %+.1f sigma"
                  % (E, lo, hi, ratio, sigma, expect, (ratio - expect) / sigma))

    # --- what the symmetrization does to the map ------------------------------
    print("\nscale applied by detailed_balance() = exp(-E/2kT), and the Bose")
    print("factor 1/(n+1) that gives chi'' instead -- they are not the same:")
    print("    E     exp(-E/2kT)   1/(n+1)    ratio")
    for E in (2.0, 6.0, 10.0, 14.0, 18.0, 22.0):
        beta = 1.0 / (BOLTZMANN * TEMPERATURE)
        sym = math.exp(-0.5 * beta * E)
        bose = 1.0 - math.exp(-beta * E)          # = 1/(n+1)
        print("  %+5.1f    %8.4f   %8.4f   %6.3f" % (E, sym, bose, bose / sym))


if __name__ == "__main__":
    main()
