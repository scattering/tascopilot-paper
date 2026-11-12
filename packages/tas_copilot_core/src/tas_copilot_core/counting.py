"""Counting uncertainties through tasreduction's ``Measurement`` class.

Every detector count in TASCopilot that becomes a rate, a normalized
intensity, or a fit weight gets its uncertainty here, from the external
``tasreduction`` package (the reduction repo owns error propagation; this
module is the typed boundary and never re-implements it). Uncertainties are
one standard deviation in the Gaussian approximation to counting statistics,
``sqrt(N)`` with a floor of one count for empty bins, the convention
``tasreduction.translator`` uses when it reads a scan, and ``Measurement``
propagates them through scaling, addition and subtraction.

``tasreduction`` is an external plugin. When it is not installed the functions
raise :class:`CountingUnavailable` with the install hint instead of silently
computing something else.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TASREDUCTION_INSTALL_HINT = (
    "tasreduction is not installed; install William Ratcliff's tasreduction package "
    "(see README 'Reduction workflows') to propagate counting uncertainties."
)


class CountingUnavailable(RuntimeError):
    """Raised when tasreduction, which owns error propagation, is not importable."""


def tasreduction_available() -> bool:
    try:
        import tasreduction.uncertainty  # noqa: F401
    except Exception:
        return False
    return True


def _measurement_class() -> Any:
    try:
        from tasreduction.uncertainty import Measurement
    except Exception as exc:  # ImportError or a broken install
        raise CountingUnavailable(TASREDUCTION_INSTALL_HINT) from exc
    return Measurement


def counting_errors(counts: Any) -> np.ndarray:
    """One-sigma uncertainty of raw detector counts (Gaussian approximation,
    floor of one count), the same rule tasreduction applies on file read."""
    arr = np.asarray(counts, dtype=float)
    return np.sqrt(np.maximum(arr, 1.0))


def counting_measurement(counts: Any, errors: Any | None = None) -> Any:
    """Wrap raw counts as a ``tasreduction.uncertainty.Measurement``.

    ``errors`` defaults to :func:`counting_errors`; pass explicit errors for
    data that were already reduced."""
    Measurement = _measurement_class()
    arr = np.asarray(counts, dtype=float)
    err = counting_errors(arr) if errors is None else np.asarray(errors, dtype=float)
    return Measurement.from_counts_errors(arr, err)


def rate_with_uncertainty(counts: Any, denominator: Any, *, target: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Counts divided by time or monitor, rescaled to ``target`` (1.0 gives
    counts per second or per monitor count), with the uncertainty propagated
    by ``Measurement.scale_inplace``.

    The denominator is treated as exact; a zero or missing denominator leaves
    that point as NaN rather than inventing a rate."""
    measurement = counting_measurement(counts)
    denom = np.asarray(denominator, dtype=float)
    bad = ~np.isfinite(denom) | (denom <= 0)
    safe = np.where(bad, 1.0, denom)
    measurement.scale_inplace(float(target) / safe)
    values = np.where(bad, np.nan, measurement.values)
    errors = np.where(bad, np.nan, measurement.errors)
    return values, errors


def scaled_errors(errors: Any, factor: Any) -> np.ndarray:
    """Errors after multiplying the values by ``factor`` (a shift by a constant
    leaves them unchanged), via ``Measurement`` so the rule lives in one place."""
    Measurement = _measurement_class()
    err = np.asarray(errors, dtype=float)
    return Measurement.from_counts_errors(np.zeros_like(err), err).scaled(factor).errors


__all__ = [
    "CountingUnavailable",
    "TASREDUCTION_INSTALL_HINT",
    "counting_errors",
    "counting_measurement",
    "rate_with_uncertainty",
    "scaled_errors",
    "tasreduction_available",
]
