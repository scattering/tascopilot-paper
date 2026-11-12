#!/usr/bin/env python
"""Stage-by-stage detailed-balance audit of the LSMO grid.

Monitor normalization alone does not survive the E -> -E test.  This walks the
candidate ki-dependent factors and reports how much of the gap each one closes,
so the missing correction is identified by the data rather than asserted.

RESOLVED 2026-09-12.  The answer is the DAVE BT-7 higher-order monitor
correction ``CF(Ei)``, which takes the cross-check from -6.2/-7.6 sigma to
-1.3/-0.2 sigma.  The ``tan(A2/2)`` stage below is kept only to show what it
was competing against; it is **not** a correction and must never be stacked on
CF.  See ``_A2_IS_SPURIOUS`` for why.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

BOLTZMANN = 0.086173324
MON_REF = 150000.0
TEMPERATURE = 300.0
EF = 14.7
D_MONO = 3.35416          # PG(002)
KT_MOD = 27.6             # thermal source, ~320 K -- assumption, see notes

_A2_IS_SPURIOUS = """
At fixed Ef the counts are

    counts  ~  Phi_sample(ki) * kf^3 cot(theta_A) / ki * S(Q,E)

and a thin 1/v incident monitor counts M ~ Phi_sample(ki) / ki, so

    counts / M  ~  kf^3 cot(theta_A) * S(Q,E)

with no ki dependence at all.  The monochromator band ki*cot(theta_M) lives in
Phi_sample and the monitor sees it identically, so it cancels: there is no
cot(theta_M) -- equivalently cot(A2/2) -- term left to divide out.  The only
ki-dependent residue is the monitor's own departure from 1/v, which is exactly
what CF(Ei) below measures.  The apparent improvement from tan(A2/2) is the
lambda/2 tilt wearing a different name; both raise the low-Ei points.
"""

# DAVE BT-7 higher-order monitor correction, PG(002) monochromator, fixed Ef.
# Read from the calibration registry rather than hard-coded here, so the
# coefficients and the window rules have exactly one home.  It is an empirical
# end-to-end factor and already contains the lambda/2 term -- do not stack a
# separate harmonic or cot() correction on top of it.
from tas_copilot_core.instrument_calibrations import (  # noqa: E402
    MONITOR_CORRECTION_KIND,
    find_instrument_calibration,
    monitor_correction_factor,
)

CF_RECORD = find_instrument_calibration(kind=MONITOR_CORRECTION_KIND, instrument="BT-7")
if CF_RECORD is None:
    raise SystemExit(
        "no %s record for BT-7 in the calibration registry; expected "
        "bt7-monitor-higher-order-correction-dave-pg002" % MONITOR_CORRECTION_KIND
    )
CF_EI_MIN, CF_EI_MAX = (float(v) for v in CF_RECORD.payload["valid_incident_energy_mev"])


def dave_cf(Ei: float) -> float:
    """Higher-order monitor correction factor at incident energy `Ei`.

    Above the calibrated window the registry supplies ``factor_above_window``
    (1.0); the polynomial reaches 1.013 at the top of the window, so that step
    is continuous to ~1%.  Below it the registry returns None -- DAVE refuses
    the correction there -- and so do we, rather than extrapolating.
    """
    factor = monitor_correction_factor(CF_RECORD, Ei)[0]
    if factor is None:
        raise ValueError("CF is not calibrated below %g meV (got %g)" % (CF_EI_MIN, Ei))
    return factor


def a2(Ei: float) -> float:
    """Monochromator take-off angle A2, in radians.

    A2 is the arm that actually moves at fixed Ef and the angle BT7 reports;
    the Bragg angle is A2/2.  Verified against expt 23816: this reproduces the
    logged A2 column to 4 decimals, and A1 == A2/2 exactly there.
    """
    lam = math.sqrt(81.80425 / Ei)
    return 2.0 * math.asin(lam / (2.0 * D_MONO))


def cot_half_a2(Ei: float) -> float:
    return 1.0 / math.tan(a2(Ei) / 2.0)


def harmonic_monitor_inflation(Ei: float, refl_ratio: float) -> float:
    """M_measured / M_lambda for an unfiltered incident beam.

    The monitor is 1/v, so a lambda/2 neutron is counted at half the efficiency
    of the fundamental; the source term is a Maxwellian evaluated at 4*Ei.
    Superseded by `dave_cf`, which measures this end-to-end instead of modelling
    it; kept because it shows the effect has the right sign and rough size.
    """
    phi = lambda E: E * math.exp(-E / KT_MOD)
    return 1.0 + 0.5 * refl_ratio * phi(4.0 * Ei) / phi(Ei)


def load_rows(manifest: Path):
    rows: dict[float, list[dict]] = {}
    for line in manifest.read_text().splitlines()[1:]:
        r = line.split("\t")
        if r[16] != "count" or not r[13] or not r[14]:
            continue
        counts, monitor = float(r[13]), float(r[14])
        if monitor <= 0:
            continue
        rows.setdefault(round(float(r[5]), 3), []).append({
            "K": float(r[4]),
            "dq": abs(float(r[4]) - round(float(r[4]))),
            "counts": counts,
            "monitor": monitor,
        })
    return {E: sorted(v, key=lambda p: p["K"]) for E, v in sorted(rows.items())}


def ratio_for(rows, E, lo, hi, scale):
    """Sum-weighted I(-E)/I(+E) over a dq window, with `scale(Ei)` applied."""
    s_pos = scale(EF + E)
    s_neg = scale(EF - E)
    num = den = vnum = vden = 0.0
    for p, n in zip(rows[E], rows[-E]):
        if not (lo <= p["dq"] < hi):
            continue
        i_pos = p["counts"] / p["monitor"] * MON_REF * s_pos
        i_neg = n["counts"] / n["monitor"] * MON_REF * s_neg
        den += i_pos; num += i_neg
        vden += i_pos ** 2 / max(p["counts"], 1.0)
        vnum += i_neg ** 2 / max(n["counts"], 1.0)
    if den <= 0:
        return None
    ratio = num / den
    return ratio, ratio * math.sqrt(vnum / num ** 2 + vden / den ** 2)


def main() -> None:
    rows = load_rows(Path(sys.argv[1]))

    stages = [
        ("monitor normalization only",        lambda Ei: 1.0),
        ("+ lambda/2 monitor model, R=0.15",  lambda Ei: harmonic_monitor_inflation(Ei, 0.15)),
        ("+ lambda/2 monitor model, R=0.30",  lambda Ei: harmonic_monitor_inflation(Ei, 0.30)),
        ("+ tan(A2/2)  [SPURIOUS, see note]", lambda Ei: math.tan(a2(Ei) / 2.0)),
        ("+ Ei * tan(A2/2)  [SPURIOUS]",      lambda Ei: Ei * math.tan(a2(Ei) / 2.0)),
        ("+ DAVE CF(Ei)  [the correction]",   dave_cf),
    ]

    print("Detailed balance: I(-E)/I(+E) must equal exp(-E/kT) at T = %.0f K." % TEMPERATURE)
    print("Restricted to dq >= 0.12 to stay out of the elastic line.\n")
    print("%-38s %14s %14s" % ("stage", "E = 1 (0.962)", "E = 2 (0.926)"))
    print("-" * 68)
    for label, scale in stages:
        cells = []
        for E in (1.0, 2.0):
            got = ratio_for(rows, E, 0.12, 0.40, scale)
            expect = math.exp(-E / (BOLTZMANN * TEMPERATURE))
            if got is None:
                cells.append("     --       ")
                continue
            ratio, sigma = got
            cells.append("%.3f (%+.1f s)" % (ratio, (ratio - expect) / sigma))
        print("%-38s %14s %14s" % (label, cells[0], cells[1]))

    print("\nSize of each candidate factor across the grid's Ei range:")
    print("   E     Ei        A2      cot(A2/2)   lam/2 infl   DAVE CF")
    for E in (-2.0, 0.0, 6.0, 12.0, 18.0, 19.0, 21.0, 22.0):
        Ei = EF + E
        flag = "" if Ei <= CF_EI_MAX else "  (CF uncalibrated, clamped)"
        print("  %+5.1f  %5.1f   %6.2f deg   %7.4f      %7.4f   %7.4f%s"
              % (E, Ei, math.degrees(a2(Ei)), cot_half_a2(Ei),
                 harmonic_monitor_inflation(Ei, 0.15), dave_cf(Ei), flag))

    lo, hi = cot_half_a2(EF - 2.0), cot_half_a2(EF + 22.0)
    h_lo = harmonic_monitor_inflation(EF - 2.0, 0.15)
    h_hi = harmonic_monitor_inflation(EF + 22.0, 0.15)
    print("\nAcross E = -2 .. +22 these tilt the intensity scale by:")
    print("  cot(A2/2):           %.1f%%  (spurious -- it cancels against the monitor)"
          % (100.0 * (hi / lo - 1.0)))
    print("  lambda/2 model:      %.1f%%  (R=0.15; scales ~linearly in R)"
          % (100.0 * (h_lo / h_hi - 1.0)))
    print("  DAVE CF:             %.1f%%  (measured end-to-end; this is the one to use)"
          % (100.0 * (dave_cf(EF - 2.0) / dave_cf(EF + 22.0) - 1.0)))
    print("\nCF record: %s (status %s), window %g-%g meV"
          % (CF_RECORD.id, CF_RECORD.status, CF_EI_MIN, CF_EI_MAX))
    print(_A2_IS_SPURIOUS)


if __name__ == "__main__":
    main()
