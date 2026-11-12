"""Shared, pure instrument-state physics helpers.

Wavelength naming discipline (three distinct meanings; never a bare
``lambda``/``wavelength`` field where these could be confused):

- ``nominal_incident_wavelength``: sqrt(MEV_ANGSTROM_SQUARED / Ei) from the
  Ei setpoint.
- ``selector_passband_center_wavelength``: constant / selector_speed from a
  ``velocity_selector_speed_to_passband_center`` calibration record.
- Harmonics are ``harmonic_order`` n of the nominal incident wavelength
  (wavelength / n). ``lambda_half`` is the established label for
  harmonic_order=2.
"""

from __future__ import annotations

import math
from typing import Any

from tas_copilot_core.psd_calibration import MEV_ANGSTROM_SQUARED


LAMBDA_HALF_HARMONIC_ORDER = 2
SELECTOR_TUNED_MISMATCH_FRACTION = 0.02
SELECTOR_OFF_SPEED_THRESHOLD = 1.0


def nominal_incident_wavelength_from_ei_mev(ei_mev: float) -> float:
    if ei_mev <= 0:
        raise ValueError("ei_mev must be positive")
    return math.sqrt(MEV_ANGSTROM_SQUARED / ei_mev)


def harmonic_suppression_state(
    filter_state: Any,
    selector_speed: Any,
    ei_mev: Any,
    calibration_record: Any = None,
) -> dict[str, Any]:
    """Classify harmonic (lambda_half = harmonic_order 2) suppression state.

    Pure function, no I/O. ``calibration_record`` is an
    InstrumentCalibrationRecord (or equivalent dict) of kind
    ``velocity_selector_speed_to_passband_center``; pass None when no
    calibration is available.

    Verdicts:
    - ``suppressed_by_filter``: harmonic filter translation is IN.
    - ``selector_tuned_passband_width_unknown``: filter OUT, selector passband
      center matches the nominal incident wavelength within 2%, but the
      passband width is unknown or the calibration is still a candidate, so
      suppression of harmonic_order 2 cannot be claimed (candidate language).
    - ``selector_tuned_suppressed``: only when the calibration is accepted AND
      a passband width is present and excludes harmonic_order 2.
    - ``unsuppressed_candidate``: filter OUT and the selector is off or its
      passband center is far from the nominal incident wavelength.
    - ``unknown``: missing inputs (filter state unknown, Ei missing, selector
      speed missing, or no usable calibration while the selector is running).
    """

    rationale: list[str] = []
    normalized_filter = _normalize_filter_state(filter_state)
    speed = _to_float(selector_speed)
    ei = _to_float(ei_mev)
    calibration = _calibration_fields(calibration_record)
    calibration_status = calibration["status"]

    nominal_incident_wavelength = None
    if ei is not None and ei > 0:
        nominal_incident_wavelength = math.sqrt(MEV_ANGSTROM_SQUARED / ei)
    elif ei_mev is not None:
        rationale.append("ei_mev is not a positive number; nominal_incident_wavelength unavailable")

    selector_passband_center_wavelength = None
    if speed is not None and speed > SELECTOR_OFF_SPEED_THRESHOLD and calibration["constant"] is not None:
        selector_passband_center_wavelength = calibration["constant"] / speed

    passband_center_mismatch_fraction = None
    if selector_passband_center_wavelength is not None and nominal_incident_wavelength:
        passband_center_mismatch_fraction = abs(selector_passband_center_wavelength - nominal_incident_wavelength) / nominal_incident_wavelength

    verdict = "unknown"
    if normalized_filter == "in":
        verdict = "suppressed_by_filter"
        rationale.append("harmonic filter translation is IN; harmonic_order 2 (lambda_half) is suppressed by the filter")
    elif normalized_filter == "unknown":
        rationale.append("harmonic filter translation state is unknown")
    elif speed is None:
        rationale.append("filter is OUT and velocity selector speed is not recorded; suppression state unknown")
    elif speed <= SELECTOR_OFF_SPEED_THRESHOLD:
        verdict = "unsuppressed_candidate"
        rationale.append(f"filter is OUT and velocity selector speed {speed:g} indicates the selector is off; harmonic_order 2 is not suppressed")
    elif nominal_incident_wavelength is None:
        rationale.append("filter is OUT and the selector is running, but Ei is missing so the passband center cannot be compared to the nominal incident wavelength")
    elif calibration["constant"] is None:
        rationale.append(
            "filter is OUT and the selector is running, but no velocity_selector_speed_to_passband_center calibration is available; selector_passband_center_wavelength unknown"
        )
    elif passband_center_mismatch_fraction is not None and passband_center_mismatch_fraction >= SELECTOR_TUNED_MISMATCH_FRACTION:
        verdict = "unsuppressed_candidate"
        rationale.append(
            f"filter is OUT and selector_passband_center_wavelength {selector_passband_center_wavelength:.4f} A is "
            f"{passband_center_mismatch_fraction:.1%} from nominal_incident_wavelength {nominal_incident_wavelength:.4f} A (>= 2%); selector is not tuned to the nominal beam"
        )
    else:
        rationale.append(
            f"filter is OUT and selector_passband_center_wavelength {selector_passband_center_wavelength:.4f} A matches "
            f"nominal_incident_wavelength {nominal_incident_wavelength:.4f} A within {passband_center_mismatch_fraction:.2%} (< 2%)"
        )
        width = calibration["passband_width_fwhm_fraction"]
        if calibration_status == "accepted" and width is not None and _width_excludes_harmonic(width, harmonic_order=LAMBDA_HALF_HARMONIC_ORDER):
            verdict = "selector_tuned_suppressed"
            rationale.append(
                f"accepted calibration provides passband_width_fwhm_fraction={width:g}, which excludes harmonic_order {LAMBDA_HALF_HARMONIC_ORDER} (lambda_half) from the passband"
            )
        else:
            verdict = "selector_tuned_passband_width_unknown"
            if calibration_status != "accepted":
                rationale.append(
                    f"calibration status is {calibration_status!r} (not accepted): the selector appears tuned as a candidate observation, but suppression of harmonic_order 2 cannot be claimed"
                )
            if width is None:
                rationale.append("passband_width_fwhm_fraction is null: the passband width is unknown, so harmonic_order 2 rejection cannot be verified")

    return {
        "filter_state": normalized_filter,
        "nominal_incident_wavelength": round(nominal_incident_wavelength, 6) if nominal_incident_wavelength is not None else None,
        "selector_passband_center_wavelength": round(selector_passband_center_wavelength, 6) if selector_passband_center_wavelength is not None else None,
        "passband_center_mismatch_fraction": round(passband_center_mismatch_fraction, 6) if passband_center_mismatch_fraction is not None else None,
        "velocity_selector_speed": speed,
        "harmonic_order": LAMBDA_HALF_HARMONIC_ORDER,
        "verdict": verdict,
        "rationale": rationale,
        "calibration_status": calibration_status,
        "calibration_id": calibration["id"],
    }


def _width_excludes_harmonic(passband_width_fwhm_fraction: float, *, harmonic_order: int) -> bool:
    """True when harmonic_order n of the nominal incident wavelength lies outside the passband FWHM.

    The harmonic wavelength is nominal/n, a fractional offset of (1 - 1/n)
    from the passband center; it is excluded when that offset exceeds half the
    FWHM fraction.
    """

    harmonic_offset_fraction = 1.0 - 1.0 / float(harmonic_order)
    return harmonic_offset_fraction > 0.5 * float(passband_width_fwhm_fraction)


def _normalize_filter_state(value: Any) -> str:
    text = str(value).strip().upper() if value is not None else ""
    if text == "IN":
        return "in"
    if text == "OUT":
        return "out"
    return "unknown"


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _calibration_fields(calibration_record: Any) -> dict[str, Any]:
    if calibration_record is None:
        return {"status": "missing", "constant": None, "passband_width_fwhm_fraction": None, "id": None}
    if isinstance(calibration_record, dict):
        payload = calibration_record.get("payload") or {}
        status = str(calibration_record.get("status") or "candidate")
        record_id = calibration_record.get("id")
        kind = calibration_record.get("kind")
    else:
        payload = getattr(calibration_record, "payload", None) or {}
        status = str(getattr(calibration_record, "status", None) or "candidate")
        record_id = getattr(calibration_record, "id", None)
        kind = getattr(calibration_record, "kind", None)
    if kind is not None and str(kind) != "velocity_selector_speed_to_passband_center":
        return {"status": "missing", "constant": None, "passband_width_fwhm_fraction": None, "id": None}
    constant = _to_float(payload.get("constant_velsel_units_angstrom"))
    if constant is not None and constant <= 0:
        constant = None
    width = _to_float(payload.get("passband_width_fwhm_fraction"))
    return {"status": status, "constant": constant, "passband_width_fwhm_fraction": width, "id": record_id}
