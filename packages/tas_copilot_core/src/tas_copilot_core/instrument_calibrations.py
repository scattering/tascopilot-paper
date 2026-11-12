"""General instrument-calibration registry (parallel to the PSD registry).

This registry stores typed instrument-state calibrations (velocity-selector
speed-to-passband constants now; 3He cell polarization and PSD channel-to-A4
kinds later) as reviewable YAML records. Loading fails closed: unknown kinds
or malformed payloads are validation errors, and consumers must not use
records that do not validate.

Naming discipline for wavelength-like quantities (do not use a bare
``lambda``/``wavelength`` field where these could be confused):

- ``nominal_incident_wavelength``: from the Ei setpoint,
  sqrt(MEV_ANGSTROM_SQUARED / Ei).
- ``selector_passband_center_wavelength``: from the velocity-selector speed
  via a calibration record of kind ``velocity_selector_speed_to_passband_center``.
- Harmonics are expressed as ``harmonic_order`` n of the nominal incident
  wavelength; ``lambda_half`` is kept as an established label defined as
  harmonic_order=2.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_contracts import InstrumentCalibrationRecord


ALLOWED_STATUSES = {"candidate", "accepted", "rejected", "superseded"}

VELOCITY_SELECTOR_KIND = "velocity_selector_speed_to_passband_center"
MONITOR_CORRECTION_KIND = "monitor_higher_order_correction_polynomial"

# Kind-specific payload validators. Add new kinds here (for example
# "he3_cell_polarization" or "psd_channel_to_a4") with their own payload
# checks; records whose kind is absent from this map fail validation.
_KIND_PAYLOAD_VALIDATORS: dict[str, Any] = {}


def default_instrument_calibrations_registry_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "instrument_calibrations" / "registry.yaml"


def load_instrument_calibrations_registry(path: str | Path | None = None) -> list[InstrumentCalibrationRecord]:
    """Load and validate the registry. Fails closed.

    Raises ValueError if any record has an unknown kind, malformed payload,
    invalid status, or malformed scope. Returns [] for a missing file.
    """

    registry_path = Path(path).expanduser() if path else default_instrument_calibrations_registry_path()
    if not registry_path.exists():
        return []
    records, errors = _load_and_collect_errors(registry_path)
    if errors:
        raise ValueError(f"instrument calibration registry failed validation ({registry_path}): " + "; ".join(errors))
    return records


def validate_instrument_calibrations_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path).expanduser() if path else default_instrument_calibrations_registry_path()
    if not registry_path.exists():
        return {
            "status": "missing_input",
            "registry_path": str(registry_path),
            "record_count": 0,
            "status_counts": {},
            "errors": [f"registry file does not exist: {registry_path}"],
            "warnings": [],
        }
    records, errors = _load_and_collect_errors(registry_path)
    warnings: list[str] = []
    for record in records:
        if record.status == "candidate" and record.reviewed_by:
            warnings.append(f"{record.id}: candidate record has reviewed_by set; promote to accepted or clear reviewed_by")
    return {
        "status": "error" if errors else "ok",
        "registry_path": str(registry_path),
        "record_count": len(records),
        "accepted_count": sum(1 for record in records if record.status == "accepted"),
        "status_counts": _status_counts(records),
        "errors": errors,
        "warnings": warnings,
    }


def instrument_calibrations_report(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path).expanduser() if path else default_instrument_calibrations_registry_path()
    validation = validate_instrument_calibrations_registry(registry_path)
    records: list[InstrumentCalibrationRecord] = []
    if validation["status"] == "ok":
        records = load_instrument_calibrations_registry(registry_path)
    return {
        "status": validation["status"],
        "registry_path": str(registry_path),
        "record_count": validation["record_count"],
        "accepted_count": validation.get("accepted_count", 0),
        "status_counts": validation["status_counts"],
        "records": [record.model_dump(mode="json") for record in records],
        "safe_conclusion": (
            "Registry failed validation; no instrument calibration records may be consumed (fail-closed)."
            if validation["status"] != "ok"
            else "Candidate records may only feed candidate-language evidence; accepted records require reviewed_by."
        ),
        "errors": validation["errors"],
        "warnings": validation["warnings"],
    }


def find_instrument_calibration(
    *,
    kind: str,
    instrument: str | None = None,
    on_date: str | None = None,
    experiment_id: str | None = None,
    path: str | Path | None = None,
) -> InstrumentCalibrationRecord | None:
    """Return the best matching usable record (accepted preferred over candidate).

    Rejected/superseded records never match. Returns None when the registry is
    missing or fails validation (fail-closed) or when no record matches.
    """

    try:
        records = load_instrument_calibrations_registry(path)
    except ValueError:
        return None
    usable = [
        record
        for record in records
        if record.kind == kind
        and record.status in {"accepted", "candidate"}
        and (instrument is None or record.instrument.lower() == instrument.lower())
        and _scope_matches(record.scope, on_date=on_date, experiment_id=experiment_id)
    ]
    usable.sort(key=lambda record: 0 if record.status == "accepted" else 1)
    return usable[0] if usable else None


def _load_and_collect_errors(registry_path: Path) -> tuple[list[InstrumentCalibrationRecord], list[str]]:
    errors: list[str] = []
    try:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return [], [f"registry is not valid YAML: {exc}"]
    raw_records = payload.get("instrument_calibrations")
    if raw_records is None:
        return [], ["registry must contain a top-level 'instrument_calibrations' list"]
    records: list[InstrumentCalibrationRecord] = []
    ids: set[str] = set()
    for index, item in enumerate(list(raw_records or [])):
        label = f"instrument_calibrations[{index}]"
        try:
            record = InstrumentCalibrationRecord(**item)
        except Exception as exc:
            errors.append(f"{label}: does not match the record schema: {exc}")
            continue
        label = record.id or label
        if record.id in ids:
            errors.append(f"duplicate id: {record.id}")
        ids.add(record.id)
        errors.extend(f"{label}: {message}" for message in _validate_record(record))
        records.append(record)
    return records, errors


def _validate_record(record: InstrumentCalibrationRecord) -> list[str]:
    errors: list[str] = []
    if not record.id:
        errors.append("id is required")
    if not record.instrument:
        errors.append("instrument is required")
    if record.status not in ALLOWED_STATUSES:
        errors.append(f"invalid status {record.status!r}; allowed: {sorted(ALLOWED_STATUSES)}")
    if record.status == "accepted" and not record.reviewed_by:
        errors.append("accepted calibration requires reviewed_by")
    validator = _KIND_PAYLOAD_VALIDATORS.get(record.kind)
    if validator is None:
        errors.append(f"unknown kind {record.kind!r}; known kinds: {sorted(_KIND_PAYLOAD_VALIDATORS)}")
    else:
        errors.extend(validator(record.payload or {}))
    errors.extend(_validate_scope(record.scope or {}))
    return errors


def _validate_scope(scope: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    valid_from = _parse_date_only(scope.get("valid_from"))
    if scope.get("valid_from") is not None and valid_from is None:
        errors.append(f"scope.valid_from is not an ISO date: {scope.get('valid_from')!r}")
    valid_to = _parse_date_only(scope.get("valid_to"))
    if scope.get("valid_to") is not None and valid_to is None:
        errors.append(f"scope.valid_to is not an ISO date: {scope.get('valid_to')!r}")
    if valid_from and valid_to and valid_to < valid_from:
        errors.append("scope.valid_to precedes scope.valid_from")
    experiment_ids = scope.get("experiment_ids")
    if experiment_ids is not None and not isinstance(experiment_ids, list):
        errors.append("scope.experiment_ids must be a list of experiment ids when present")
    return errors


def _validate_velocity_selector_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be a mapping"]
    constant = payload.get("constant_velsel_units_angstrom")
    if constant is None:
        errors.append("payload.constant_velsel_units_angstrom is required")
    elif not isinstance(constant, (int, float)) or isinstance(constant, bool) or float(constant) <= 0:
        errors.append(f"payload.constant_velsel_units_angstrom must be a positive number, got {constant!r}")
    if "passband_width_fwhm_fraction" not in payload:
        errors.append("payload.passband_width_fwhm_fraction is required (null when unknown)")
    else:
        width = payload.get("passband_width_fwhm_fraction")
        if width is not None and (not isinstance(width, (int, float)) or isinstance(width, bool) or not 0 < float(width) < 1):
            errors.append(f"payload.passband_width_fwhm_fraction must be null or a fraction in (0, 1), got {width!r}")
    known_keys = {"constant_velsel_units_angstrom", "passband_width_fwhm_fraction"}
    unknown = sorted(set(payload) - known_keys)
    if unknown:
        errors.append(f"payload has unknown field(s) {unknown}; known fields: {sorted(known_keys)}")
    return errors


_KIND_PAYLOAD_VALIDATORS[VELOCITY_SELECTOR_KIND] = _validate_velocity_selector_payload


def _validate_monitor_correction_payload(payload: dict[str, Any]) -> list[str]:
    """``monitor_higher_order_correction_polynomial``: the DAVE-style monitor
    correction factor CF(Ei) = sum_k c_k Ei^k that multiplies counts-per-monitor
    on a fixed-Ef instrument whose monitor sits in the unfiltered incident beam.
    ``coefficients`` are in ascending powers of the incident energy in meV."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be a mapping"]
    if payload.get("variable") != "incident_energy_mev":
        errors.append("payload.variable must be 'incident_energy_mev'")
    coefficients = payload.get("coefficients")
    if not isinstance(coefficients, list) or not coefficients or any(isinstance(c, bool) or not isinstance(c, (int, float)) for c in coefficients):
        errors.append("payload.coefficients must be a non-empty list of numbers (ascending powers of Ei)")
    window = payload.get("valid_incident_energy_mev")
    if not (isinstance(window, list) and len(window) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in window) and window[0] < window[1]):
        errors.append("payload.valid_incident_energy_mev must be [low, high] in meV with low < high")
    above = payload.get("factor_above_window")
    if above is not None and (isinstance(above, bool) or not isinstance(above, (int, float)) or float(above) <= 0):
        errors.append("payload.factor_above_window must be null or a positive number")
    if payload.get("operation") != "multiply_counts_per_monitor":
        errors.append("payload.operation must be 'multiply_counts_per_monitor'")
    conditions = payload.get("applies_when")
    if not isinstance(conditions, dict) or conditions.get("fixed_energy") != "Ef" or not isinstance(conditions.get("monochromator_dspacing_angstrom"), (int, float)):
        errors.append("payload.applies_when must give fixed_energy: Ef and monochromator_dspacing_angstrom")
    known_keys = {"variable", "coefficients", "valid_incident_energy_mev", "factor_above_window", "operation", "applies_when"}
    unknown = sorted(set(payload) - known_keys)
    if unknown:
        errors.append(f"payload has unknown field(s) {unknown}; known fields: {sorted(known_keys)}")
    return errors


_KIND_PAYLOAD_VALIDATORS[MONITOR_CORRECTION_KIND] = _validate_monitor_correction_payload


def monitor_correction_factor(record: InstrumentCalibrationRecord, incident_energy_mev: Any) -> list[float | None]:
    """Evaluate CF(Ei) for each incident energy with the record's window rules:
    inside the window the polynomial; above it ``factor_above_window`` (DAVE
    uses 1.0); below it ``None`` (DAVE refuses the correction there)."""
    if record.kind != MONITOR_CORRECTION_KIND:
        raise ValueError(f"record {record.id!r} has kind {record.kind!r}, not {MONITOR_CORRECTION_KIND!r}")
    payload = record.payload or {}
    coefficients = [float(c) for c in payload["coefficients"]]
    low, high = (float(v) for v in payload["valid_incident_energy_mev"])
    above = payload.get("factor_above_window")
    factors: list[float | None] = []
    for value in (incident_energy_mev if isinstance(incident_energy_mev, (list, tuple)) else [incident_energy_mev]):
        ei = float(value)
        if ei < low:
            factors.append(None)
        elif ei > high:
            factors.append(None if above is None else float(above))
        else:
            factors.append(sum(c * ei**k for k, c in enumerate(coefficients)))
    return factors


def monitor_correction_coefficients_in_energy_transfer(record: InstrumentCalibrationRecord, fixed_final_energy_mev: float) -> list[float]:
    """Re-expand CF(Ei) = CF(Ef + E) as a polynomial in the energy transfer E
    (ascending powers), so tasreduction's ``TripleAxisScan.monitor_correction``
    (which evaluates its coefficients in E) can apply the record at fixed Ef.
    The window rules are not encoded in the coefficients: only use the result
    where Ef + E lies inside ``valid_incident_energy_mev``."""
    from math import comb

    coefficients = [float(c) for c in (record.payload or {})["coefficients"]]
    ef = float(fixed_final_energy_mev)
    shifted = [0.0] * len(coefficients)
    for k, c in enumerate(coefficients):
        for j in range(k + 1):
            shifted[j] += c * comb(k, j) * ef ** (k - j)
    return shifted


def _scope_matches(scope: dict[str, Any], *, on_date: str | None, experiment_id: str | None) -> bool:
    scope = scope or {}
    if experiment_id is not None:
        experiment_ids = scope.get("experiment_ids")
        if isinstance(experiment_ids, list) and experiment_ids and str(experiment_id) not in {str(item) for item in experiment_ids}:
            return False
    if on_date is not None:
        parsed = _parse_date_only(on_date)
        valid_from = _parse_date_only(scope.get("valid_from"))
        valid_to = _parse_date_only(scope.get("valid_to"))
        if parsed and valid_from and parsed < valid_from:
            return False
        if parsed and valid_to and parsed > valid_to:
            return False
    return True


def _parse_date_only(value: Any) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _status_counts(records: list[InstrumentCalibrationRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    return dict(sorted(counts.items()))
