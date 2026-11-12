"""Magnetic form factors f(Q) for neutron intensity corrections.

Why this exists: pySpinW has no magnetic form factor. ``formfact`` appears zero
times in the package (checked against the local checkout, the legacy install,
and the Stampede3 reference used for the SrMn2Sb2 fits), and the Rust
cross-section ends at the bare perpendicular projection

    perp = I - qhat qhat^T ;  intensity = sum(Sab * perp)

with no |Q|-magnitude term at all. The only attenuation anywhere in that chain
was a placeholder in the legacy Python API:

    # Simplified implementation
    ff = np.exp(-0.1 * Q_mag**2)

which is ion-independent (identical for Mn2+, Fe3+, anything), and was applied
linearly to Sab rather than as f^2. Fit reports nevertheless recorded
"form_factor: pySpinW spinwave(..., formfact=True), Mn2+ labels", asserting a
correction that never happened.

Coefficients come from ``periodictable`` (Kienzle et al.), which carries the
standard <j0>, <j2>, <j4>, <j6> tables rather than a hand-copied subset.

Physics notes that matter for use:

- For a free ion, f(Q) is the Fourier transform of a smooth, positive, nodeless
  spin density normalised to f(0)=1, so it decreases monotonically. An upturn is
  not physical.
- The analytic <j0> = A exp(-a s^2) + B exp(-b s^2) + C exp(-c s^2) + D with
  s = Q/4pi is a least-squares fit. Outside its fitted range the residual
  constant D and the C term (whose c is NEGATIVE for several ions, e.g. -0.609
  for Mn2+) take over and the curve turns up. For Mn2+ that happens near
  10 1/A, where |f| < 0.02 and f^2 < 4e-4 - four orders below Q->0 and far
  below phonon background, which is why no measurement shows it. This module
  reports that range rather than pretending the tail is meaningful.
- <j2> IS intrinsically non-monotonic (zero at Q=0, peaking near 5 1/A) and
  enters the dipole approximation for ions with orbital moment. Mn2+ is 3d5,
  L=0, so <j0> alone is correct there.
"""

from __future__ import annotations

from typing import Any, Sequence

# Above this |Q| the analytic <j0> parameterisation stops being trustworthy for
# 3d ions: the fitted residual terms dominate and the curve can turn upward.
_ANALYTIC_VALID_Q_MAX = 10.0


def _require_periodictable() -> Any:
    try:
        import periodictable
    except ImportError as exc:  # pragma: no cover - exercised via the error path
        raise ImportError(
            "magnetic form factors need the 'periodictable' package "
            "(pip install periodictable, or install tas-copilot-core[materials])"
        ) from exc
    return periodictable


def magnetic_form_factor(
    ion: str,
    q_inv_angstrom: Sequence[float] | float,
    *,
    include_j2: bool = False,
    g_factor: float = 2.0,
) -> dict[str, Any]:
    """f(Q) and f^2(Q) for a magnetic ion.

    ``ion`` is written as element plus charge, e.g. ``"Mn2+"``. Returns the
    form factor, its square (the factor that multiplies intensity), and a
    validity report naming any Q beyond the parameterisation's usable range.

    ``include_j2`` adds the dipole-approximation orbital term
    ``f = <j0> + (1 - 2/g) <j2>``. Leave it off for spin-only ions such as
    Mn2+ (3d5, L=0), where <j2> carries no weight.
    """
    values = [float(q_inv_angstrom)] if isinstance(q_inv_angstrom, (int, float)) else [float(q) for q in q_inv_angstrom]
    if any(q < 0 for q in values):
        raise ValueError("|Q| must be non-negative")

    element_symbol, charge = _split_ion(ion)
    periodictable = _require_periodictable()
    try:
        element = getattr(periodictable, element_symbol)
    except AttributeError as exc:
        raise ValueError(f"unknown element in ion {ion!r}") from exc
    try:
        ion_obj = element.ion[charge]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"unknown charge state for ion {ion!r}") from exc
    table = getattr(ion_obj, "magnetic_ff", None)
    entry = None if table is None else table.get(charge)
    if entry is None:
        raise ValueError(f"no magnetic form factor tabulated for {ion!r}")

    import numpy as np

    q_array = np.asarray(values, dtype=float)
    f_values = np.asarray(entry.j0_Q(q_array), dtype=float)
    if include_j2:
        f_values = f_values + (1.0 - 2.0 / float(g_factor)) * np.asarray(entry.j2_Q(q_array), dtype=float)

    beyond = [q for q in values if q > _ANALYTIC_VALID_Q_MAX]
    warnings: list[str] = []
    if beyond:
        warnings.append(
            f"{len(beyond)} |Q| value(s) exceed {_ANALYTIC_VALID_Q_MAX} 1/A, where the analytic "
            "parameterisation's residual terms dominate and f(Q) can turn upward. That upturn is a "
            "fitting artifact, not physics: f^2 there is <1e-3 and far below background."
        )

    return {
        "ion": ion,
        "q_inv_angstrom": values,
        "form_factor": [float(v) for v in f_values],
        "form_factor_squared": [float(v * v) for v in f_values],
        "j0_coefficients": list(entry.j0),
        "includes_j2_orbital_term": bool(include_j2),
        "source": "periodictable magnetic_ff (<j0> analytic coefficients)",
        "valid_q_max_inv_angstrom": _ANALYTIC_VALID_Q_MAX,
        "monotonic_over_requested_q": bool(len(f_values) < 2 or all(np.diff(f_values) < 0)),
        "warnings": warnings,
    }


def _split_ion(ion: str) -> tuple[str, int]:
    """'Mn2+' -> ('Mn', 2). Also accepts 'Mn+2' and bare 'Mn' (neutral)."""
    text = str(ion).strip()
    if not text:
        raise ValueError("ion must be a non-empty string like 'Mn2+'")
    symbol = "".join(ch for ch in text if ch.isalpha())
    digits = "".join(ch for ch in text if ch.isdigit())
    if not symbol:
        raise ValueError(f"could not read an element symbol from {ion!r}")
    if not digits:
        return symbol, 0
    charge = int(digits)
    if "-" in text:
        charge = -charge
    return symbol, charge


def apply_form_factor(
    intensities: Sequence[float],
    q_inv_angstrom: Sequence[float],
    *,
    ion: str,
    include_j2: bool = False,
) -> dict[str, Any]:
    """Multiply model intensities by f^2(Q).

    Intensity carries f^2, not f: the form factor enters the cross-section
    amplitude once per matrix element and the cross-section is its modulus
    squared. The legacy placeholder multiplied Sab by f once, which is wrong
    even ignoring that its f was invented.
    """
    if len(intensities) != len(q_inv_angstrom):
        raise ValueError(
            f"intensities and |Q| must be the same length, got {len(intensities)} and {len(q_inv_angstrom)}"
        )
    report = magnetic_form_factor(ion, q_inv_angstrom, include_j2=include_j2)
    corrected = [float(i) * f2 for i, f2 in zip(intensities, report["form_factor_squared"])]
    return {
        "corrected_intensities": corrected,
        "form_factor_squared": report["form_factor_squared"],
        "ion": ion,
        "source": report["source"],
        "warnings": report["warnings"],
    }
