from __future__ import annotations

from math import sqrt

from tas_copilot_contracts import ProvenanceRecord, Scan


def _same(left, right) -> bool | None:
    if left is None or right is None:
        return None
    return left == right


def _q_distance(left, right) -> float | None:
    if left is None or right is None:
        return None
    return sqrt((left.h - right.h) ** 2 + (left.k - right.k) ** 2 + (left.l - right.l) ** 2)


def compare_scans(reference: Scan, candidate: Scan) -> dict:
    matched: list[str] = []
    mismatched: list[str] = []
    missing: list[str] = []

    dimensions = {
        "instrument": _same(reference.instrument, candidate.instrument),
        "scan_type": _same(reference.scan_type, candidate.scan_type),
        "physics_target": _same(reference.physics_target, candidate.physics_target),
    }
    for name, value in dimensions.items():
        if value is None:
            missing.append(name)
        elif value:
            matched.append(name)
        else:
            mismatched.append(name)

    q_distance = _q_distance(reference.q_center, candidate.q_center)
    if q_distance is None:
        missing.append("q_center")
    elif q_distance <= 0.15:
        matched.append("q_center")
    else:
        mismatched.append("q_center")

    if reference.instrument_config and candidate.instrument_config:
        for field in ["fixed_energy_mode", "monochromator", "analyzer", "collimation"]:
            if getattr(reference.instrument_config, field) == getattr(candidate.instrument_config, field):
                matched.append(field)
            else:
                mismatched.append(field)
    else:
        missing.append("instrument_config")

    denom = len(matched) + len(mismatched) + len(missing)
    score = len(matched) / denom if denom else 0.0
    return {
        "similarity_score": score,
        "matched_dimensions": matched,
        "mismatched_dimensions": mismatched,
        "missing_dimensions": missing,
        "provenance": [ProvenanceRecord(source_type="scan_similarity", method="deterministic_dimension_match").model_dump(mode="json")],
    }
