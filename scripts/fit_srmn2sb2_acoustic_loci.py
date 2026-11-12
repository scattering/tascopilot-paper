#!/usr/bin/env python3
"""Fit the SrMn2Sb2 acoustic dispersion to fixed-energy peak positions only.

This script deliberately does *not* fit intensities, infer an optical branch, or
reuse the legacy ``scan_peak_fits.json`` branch assignment.  It consumes the
reviewed full-locus JSONL artifact, predicts the left/right coordinate roots of
the gapless McNally-style acoustic dispersion, and fits ``SJ1``, ``J2/J1``, and
``SJc`` jointly to H and L scans.

The exchange convention is calibrated to McNally et al., Phys. Rev. B 91,
180407(R) (2015), Eq. (2).  ``SJ`` products are the fitted quantities, so the
factor ``2S`` in that equation becomes a factor two multiplying expressions in
``SJ``.  Equivalently, the post-PR110 high-level pySpinW Hamiltonian requires
representative exchange tensors ``2*SJ/S`` for this convention.  Anisotropy is
fixed to zero here because this is an acoustic-only, gap-agnostic fit.

For each fixed-energy scan, the observed left/right peak coordinates are kept
as a correlated pair and reduced to their half-separation with uncertainty from
the full 2x2 covariance.  The common-center displacement is reported but not
fitted because resolution focusing can shift both peaks in the same direction.
The nominal energy is fixed by default.  An optional shared energy displacement
can be profiled as a sensitivity check; it is not a full TAS convolution.

Outputs include the best fit, raw and dimensionless Jacobian singular values,
right-singular parameter combinations, and one-dimensional profile-objective
diagnostics.  Every included pair must have two exact coordinate roots; a
parameter point that fails this all-pairs acoustic constraint is infeasible and
is never assigned an experimental chi-square.
Generated real-data outputs belong in an ignored local directory such as
``local_runs/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from scipy.optimize import brentq, minimize, minimize_scalar
except ImportError as exc:  # pragma: no cover - exercised only without fit extras
    raise SystemExit(
        "scipy is required for the position-only SrMn2Sb2 dispersion fit"
    ) from exc


PARAMETER_NAMES = ("sj1_mev", "j2_over_j1", "sjc_mev")
DEFAULT_START = np.array([7.9, 0.165, 0.51], dtype=float)
DEFAULT_LOWER = np.array([0.25, -0.10, 0.0], dtype=float)
DEFAULT_UPPER = np.array([30.0, 0.60, 8.0], dtype=float)
EXPECTED_BRANCH_LABEL = "unassigned_symmetric_pair"
MODEL_ID = "srmn2sb2_gapless_acoustic_z1_3_z2_6_zc_3_mcnally_sj_units"
HARD_INFEASIBLE_RESIDUAL = 1.0e6
ROOT_CONSTRAINT_TOLERANCE_MEV = 5.0e-7
SEARCH_NO_ROOT_SCALE_MEV = 1.0


class LocusSchemaError(ValueError):
    """Raised when a locus artifact cannot support a defensible fit."""


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LocusSchemaError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise LocusSchemaError(f"{field} must be finite, got {value!r}")
    return result


def _optional_positive_float(value: Any, field: str) -> float | None:
    """Parse a nullable positive scale without inventing a replacement."""

    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none"}):
        return None
    result = _finite_float(value, field)
    if result <= 0.0:
        raise LocusSchemaError(f"{field} must be positive when present, got {value!r}")
    return result


def _vector3(value: Any, field: str) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise LocusSchemaError(f"{field} must be a JSON three-vector") from exc
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise LocusSchemaError(f"{field} must be a finite three-vector")
    return array


def _matrix2(value: Any, field: str) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise LocusSchemaError(f"{field} must be a JSON 2x2 matrix") from exc
    array = np.asarray(value, dtype=float)
    if array.shape != (2, 2) or not np.all(np.isfinite(array)):
        raise LocusSchemaError(f"{field} must be a finite 2x2 matrix")
    if not np.allclose(array, array.T, rtol=1e-8, atol=1e-12):
        raise LocusSchemaError(f"{field} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(array)
    if np.min(eigenvalues) <= 0.0:
        raise LocusSchemaError(
            f"{field} must be positive definite; eigenvalues={eigenvalues.tolist()}"
        )
    return array


def _json_object(value: Any, field: str) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise LocusSchemaError(f"{field} must be a JSON object") from exc
    if not isinstance(value, dict):
        raise LocusSchemaError(f"{field} must be an object")
    return dict(value)


def _bool_field(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if value in (0, 1):
        return bool(value)
    raise LocusSchemaError(f"{field} must be boolean, got {value!r}")


def _optional_bool_field(value: Any, field: str) -> bool | None:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none"}):
        return None
    return _bool_field(value, field)


def _coordinate_index(axis: str) -> int:
    if axis == "H":
        return 0
    if axis == "L":
        return 2
    raise LocusSchemaError(f"axis must be 'H' or 'L', got {axis!r}")


@dataclass(frozen=True)
class LocusPoint:
    point_id: str
    pair_id: str
    scan_id: str
    source_path: str
    axis: str
    side: str
    energy_mev: float
    energy_sigma_mev: float | None
    energy_sigma_source: str
    fixed_ef_mev: float | None
    hkl: tuple[float, float, float]
    coordinate_rlu: float
    coordinate_offset_rlu: float
    coordinate_sigma_rlu: float
    zone_center_hkl: tuple[float, float, float]
    scan_path_hkl_start: tuple[float, float, float]
    scan_path_hkl_end: tuple[float, float, float]
    pair_coordinate_covariance_rlu2: tuple[tuple[float, float], tuple[float, float]]
    resolution_classification: str
    resolution_scenario_summaries: Mapping[str, Any]


@dataclass(frozen=True)
class LocusPair:
    pair_id: str
    scan_id: str
    source_path: str
    axis: str
    energy_mev: float
    energy_sigma_mev: float | None
    energy_sigma_source: str
    fixed_ef_mev: float | None
    observed_coordinates_rlu: tuple[float, float]  # left, right
    covariance_rlu2: tuple[tuple[float, float], tuple[float, float]]
    zone_center_hkl: tuple[float, float, float]
    scan_path_hkl_start: tuple[float, float, float]
    scan_path_hkl_end: tuple[float, float, float]
    point_ids: tuple[str, str]
    resolution_classification: str
    resolution_scenario_summaries: Mapping[str, Any]

    @property
    def coordinate_index(self) -> int:
        return _coordinate_index(self.axis)

    @property
    def zone_coordinate(self) -> float:
        return self.zone_center_hkl[self.coordinate_index]

    @property
    def path_coordinate_bounds(self) -> tuple[float, float]:
        i = self.coordinate_index
        values = (self.scan_path_hkl_start[i], self.scan_path_hkl_end[i])
        return min(values), max(values)

    def hkl_at(self, coordinate: float) -> np.ndarray:
        """Interpolate the explicit scan path at an absolute scan coordinate."""

        start = np.asarray(self.scan_path_hkl_start, dtype=float)
        end = np.asarray(self.scan_path_hkl_end, dtype=float)
        i = self.coordinate_index
        denominator = end[i] - start[i]
        if abs(denominator) < 1e-12:
            raise LocusSchemaError(
                f"pair {self.pair_id}: scan path does not vary along {self.axis}"
            )
        fraction = (float(coordinate) - start[i]) / denominator
        return start + fraction * (end - start)


@dataclass(frozen=True)
class LoadedLoci:
    pairs: tuple[LocusPair, ...]
    total_rows: int
    excluded_rows: int
    exclusion_counts: Mapping[str, int]


class McNallyAcousticBackend:
    """Gapless Sr z1=3, z2=6, zc=3 energy in McNally-style SJ units."""

    model_id = MODEL_ID

    @staticmethod
    def radicand(hkl: np.ndarray, parameters: Sequence[float]) -> np.ndarray:
        """Return the quantity under the square root before the leading factor 2.

        In the explicit Sr nearest-shell model, the three opposite-sublattice
        J1 and three Jc representative bonds share the honeycomb in-plane
        structure factor.  The six same-sublattice J2 neighbors give two times
        the sum of three cosines.  The zc=3 fact comes from that Sr bond list,
        not from McNally's separate printed mean-field coordination expression.
        """

        points = np.asarray(hkl, dtype=float)
        if points.shape[-1] != 3:
            raise ValueError("hkl must end in a length-three reciprocal coordinate")
        sj1, ratio, sjc = np.asarray(parameters, dtype=float)
        sj2 = sj1 * ratio
        h, k, ell = np.moveaxis(points, -1, 0)
        honeycomb = 1.0 + np.exp(2j * np.pi * k) + np.exp(-2j * np.pi * h)
        opposite = honeycomb * (sj1 + sjc * np.exp(2j * np.pi * ell))
        same = 2.0 * sj2 * (
            np.cos(2 * np.pi * h)
            + np.cos(2 * np.pi * k)
            + np.cos(2 * np.pi * (h + k))
        )
        diagonal = 3.0 * (sj1 + sjc) - 6.0 * sj2 + same
        return np.asarray(diagonal**2 - np.abs(opposite) ** 2, dtype=float)

    @classmethod
    def energy(cls, hkl: np.ndarray, parameters: Sequence[float]) -> np.ndarray:
        """Return the acoustic energy in meV.

        Small negative values from roundoff at a Goldstone point are clipped.
        Larger negative radicands are also clipped for optimizer continuity and
        are rejected separately by the stability residual.
        """

        radicand = cls.radicand(hkl, parameters)
        return 2.0 * np.sqrt(np.maximum(radicand, 0.0))

    @staticmethod
    def exact_exchange_swap(parameters: Sequence[float]) -> np.ndarray:
        """Return the exactly energy-equivalent J1/Jc label-swap partner.

        Eq. (2) depends on the opposite-sublattice coefficients symmetrically:
        ``|f(H,K) * (SJ1 + SJc exp(i 2pi L))|^2`` and their sum in the diagonal.
        Holding ``SJ2=SJ1*(J2/J1)`` fixed while exchanging SJ1 and SJc leaves
        every acoustic energy invariant.
        """

        sj1, ratio, sjc = np.asarray(parameters, dtype=float)
        if sjc <= 0.0:
            raise ValueError("exact exchange swap is undefined for SJc <= 0")
        sj2 = sj1 * ratio
        return np.array([sjc, sj2 / sjc, sj1], dtype=float)

    @staticmethod
    def identifiable_exchange_invariants(parameters: Sequence[float]) -> dict[str, float]:
        sj1, ratio, sjc = np.asarray(parameters, dtype=float)
        return {
            "sj1_plus_sjc_mev": float(sj1 + sjc),
            "sj1_times_sjc_mev2": float(sj1 * sjc),
            "sj2_mev": float(sj1 * ratio),
        }


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LocusSchemaError(
                        f"{path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise LocusSchemaError(
                        f"{path}:{line_number}: each JSONL row must be an object"
                    )
                records.append(record)
        return records
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise LocusSchemaError("input must be a .jsonl, .ndjson, or .csv locus artifact")


def _parse_included_point(
    record: Mapping[str, Any],
    *,
    resolution_scenario_name: str | None = None,
) -> LocusPoint:
    required = {
        "point_id",
        "pair_id",
        "scan_id",
        "source_path",
        "axis",
        "side",
        "energy_mev",
        "energy_sigma_mev",
        "h",
        "k",
        "l",
        "coordinate_rlu",
        "coordinate_offset_rlu",
        "coordinate_sigma_rlu",
        "zone_center_hkl",
        "scan_path_hkl_start",
        "scan_path_hkl_end",
        "pair_coordinate_covariance_rlu2",
        "branch_assignment",
    }
    missing = sorted(name for name in required if name not in record)
    if missing:
        raise LocusSchemaError(f"included locus row is missing fields: {missing}")

    branch = str(record["branch_assignment"])
    if branch != EXPECTED_BRANCH_LABEL:
        raise LocusSchemaError(
            "included locus rows must remain unassigned symmetric pairs; "
            f"got branch_assignment={branch!r}"
        )
    axis = str(record["axis"]).upper()
    coordinate_index = _coordinate_index(axis)
    side = str(record["side"]).lower()
    if side not in {"left", "right"}:
        raise LocusSchemaError(f"side must be left or right, got {side!r}")
    resolution_classification = str(
        record.get("resolution_classification", "legacy_resolved_true")
    )
    resolution_scenario_summaries = _json_object(
        record.get("resolution_scenario_summaries"),
        "resolution_scenario_summaries",
    )
    fixed_ef_mev = _optional_positive_float(
        record.get("fixed_ef_mev"), "fixed_ef_mev"
    )
    if resolution_scenario_name is None:
        sigma_e = _optional_positive_float(
            record.get("energy_sigma_mev"), "energy_sigma_mev"
        )
        energy_sigma_source = "producer_preferred_resolution_scenario"
    else:
        scenario = resolution_scenario_summaries.get(resolution_scenario_name)
        if not isinstance(scenario, dict):
            raise LocusSchemaError(
                f"resolution scenario {resolution_scenario_name!r} is missing"
            )
        sigma_e = _optional_positive_float(
            scenario.get("energy_resolution_sigma_mev"),
            (
                "resolution_scenario_summaries."
                f"{resolution_scenario_name}.energy_resolution_sigma_mev"
            ),
        )
        energy_sigma_source = f"resolution_scenario:{resolution_scenario_name}"

    hkl_array = np.array(
        [
            _finite_float(record["h"], "h"),
            _finite_float(record["k"], "k"),
            _finite_float(record["l"], "l"),
        ],
        dtype=float,
    )
    center = _vector3(record["zone_center_hkl"], "zone_center_hkl")
    path_start = _vector3(record["scan_path_hkl_start"], "scan_path_hkl_start")
    path_end = _vector3(record["scan_path_hkl_end"], "scan_path_hkl_end")
    covariance = _matrix2(
        record["pair_coordinate_covariance_rlu2"],
        "pair_coordinate_covariance_rlu2",
    )
    coordinate = _finite_float(record["coordinate_rlu"], "coordinate_rlu")
    offset = _finite_float(record["coordinate_offset_rlu"], "coordinate_offset_rlu")
    sigma_q = _finite_float(record["coordinate_sigma_rlu"], "coordinate_sigma_rlu")
    if sigma_q <= 0.0:
        raise LocusSchemaError("coordinate_sigma_rlu must be positive")
    if not math.isclose(coordinate, hkl_array[coordinate_index], abs_tol=2e-5):
        raise LocusSchemaError(
            f"point {record['point_id']}: coordinate_rlu disagrees with hkl"
        )
    if not math.isclose(offset, coordinate - center[coordinate_index], abs_tol=2e-5):
        raise LocusSchemaError(
            f"point {record['point_id']}: coordinate_offset_rlu disagrees with zone center"
        )
    if side == "left" and not coordinate < center[coordinate_index]:
        raise LocusSchemaError(f"point {record['point_id']}: left locus is not left of center")
    if side == "right" and not coordinate > center[coordinate_index]:
        raise LocusSchemaError(f"point {record['point_id']}: right locus is not right of center")

    probe = LocusPair(
        pair_id=str(record["pair_id"]),
        scan_id=str(record["scan_id"]),
        source_path=str(record["source_path"]),
        axis=axis,
        energy_mev=_finite_float(record["energy_mev"], "energy_mev"),
        energy_sigma_mev=sigma_e,
        energy_sigma_source=energy_sigma_source,
        fixed_ef_mev=fixed_ef_mev,
        observed_coordinates_rlu=(coordinate, coordinate),
        covariance_rlu2=tuple(tuple(float(v) for v in row) for row in covariance),
        zone_center_hkl=tuple(float(v) for v in center),
        scan_path_hkl_start=tuple(float(v) for v in path_start),
        scan_path_hkl_end=tuple(float(v) for v in path_end),
        point_ids=(str(record["point_id"]), str(record["point_id"])),
        resolution_classification=resolution_classification,
        resolution_scenario_summaries=resolution_scenario_summaries,
    )
    lower, upper = probe.path_coordinate_bounds
    if not lower - 2e-5 <= coordinate <= upper + 2e-5:
        raise LocusSchemaError(f"point {record['point_id']}: coordinate lies outside scan path")
    expected_hkl = probe.hkl_at(coordinate)
    # BT-7 files can contain tiny motor/rounding drift in nominally fixed H/K/L.
    # The explicit path is authoritative for root prediction; require only that
    # the observed point lies close to it, never silently repair a material drift.
    if not np.allclose(hkl_array, expected_hkl, rtol=0.0, atol=1e-3):
        raise LocusSchemaError(
            f"point {record['point_id']}: hkl is inconsistent with explicit scan path"
        )
    if not lower < probe.zone_coordinate < upper:
        raise LocusSchemaError(f"pair {record['pair_id']}: zone center lies outside scan path")

    return LocusPoint(
        point_id=str(record["point_id"]),
        pair_id=str(record["pair_id"]),
        scan_id=str(record["scan_id"]),
        source_path=str(record["source_path"]),
        axis=axis,
        side=side,
        energy_mev=probe.energy_mev,
        energy_sigma_mev=sigma_e,
        energy_sigma_source=energy_sigma_source,
        fixed_ef_mev=fixed_ef_mev,
        hkl=tuple(float(v) for v in hkl_array),
        coordinate_rlu=coordinate,
        coordinate_offset_rlu=offset,
        coordinate_sigma_rlu=sigma_q,
        zone_center_hkl=tuple(float(v) for v in center),
        scan_path_hkl_start=tuple(float(v) for v in path_start),
        scan_path_hkl_end=tuple(float(v) for v in path_end),
        pair_coordinate_covariance_rlu2=tuple(
            tuple(float(v) for v in row) for row in covariance
        ),
        resolution_classification=resolution_classification,
        resolution_scenario_summaries=resolution_scenario_summaries,
    )


def load_locus_pairs(
    path: Path,
    *,
    resolution_scenario_filter: str | None = None,
    max_common_center_offset_rlu: float | None = None,
) -> LoadedLoci:
    """Load producer-included, uncensored, non-unresolved two-sided candidates.

    ``resolved=null`` is retained: it means the available instrument metadata
    cannot support a hard resolution classification.  ``resolved=false`` is
    excluded.  Resolution-scenario sensitivity belongs upstream or in an
    explicit downstream mask, never in a silent default.
    """

    records = _read_records(path)
    if not records:
        raise LocusSchemaError(f"no locus records found in {path}")
    included: list[LocusPoint] = []
    exclusion_counts: dict[str, int] = {}
    seen_point_ids: set[str] = set()
    for record in records:
        point_id = str(record.get("point_id", ""))
        if not point_id:
            raise LocusSchemaError("every locus row requires a non-empty point_id")
        if point_id in seen_point_ids:
            raise LocusSchemaError(f"duplicate point_id {point_id!r}")
        seen_point_ids.add(point_id)

        reasons = []
        if not _bool_field(record.get("include", False), "include"):
            reasons.append("include_false")
        resolved = _optional_bool_field(record.get("resolved"), "resolved")
        if resolved is False:
            reasons.append("unresolved")
        if _bool_field(record.get("censored", False), "censored"):
            reasons.append("censored")
        if resolution_scenario_filter and not reasons:
            summaries = _json_object(
                record.get("resolution_scenario_summaries"),
                "resolution_scenario_summaries",
            )
            scenario = summaries.get(resolution_scenario_filter)
            if not isinstance(scenario, dict):
                reasons.append("resolution_scenario_missing")
            elif scenario.get("resolved_by_scenario") is not True:
                reasons.append("not_resolved_in_requested_scenario")
        if reasons:
            for reason in reasons:
                exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            continue
        included.append(
            _parse_included_point(
                record,
                resolution_scenario_name=resolution_scenario_filter,
            )
        )

    groups: dict[str, list[LocusPoint]] = {}
    for point in included:
        groups.setdefault(point.pair_id, []).append(point)
    pairs: list[LocusPair] = []
    for pair_id, points in sorted(groups.items()):
        if len(points) != 2 or {point.side for point in points} != {"left", "right"}:
            raise LocusSchemaError(
                f"included pair {pair_id!r} must contain exactly one left and one right locus"
            )
        by_side = {point.side: point for point in points}
        left, right = by_side["left"], by_side["right"]
        fields = (
            "scan_id",
            "source_path",
            "axis",
            "energy_mev",
            "energy_sigma_mev",
            "energy_sigma_source",
            "fixed_ef_mev",
            "zone_center_hkl",
            "scan_path_hkl_start",
            "scan_path_hkl_end",
            "pair_coordinate_covariance_rlu2",
            "resolution_classification",
            "resolution_scenario_summaries",
        )
        for field in fields:
            left_value, right_value = getattr(left, field), getattr(right, field)
            if isinstance(left_value, float):
                equal = math.isclose(left_value, right_value, rel_tol=1e-10, abs_tol=1e-12)
            else:
                equal = left_value == right_value
            if not equal:
                raise LocusSchemaError(f"pair {pair_id!r} disagrees on {field}")
        covariance = np.asarray(left.pair_coordinate_covariance_rlu2, dtype=float)
        expected_sigmas = np.sqrt(np.diag(covariance))
        observed_sigmas = np.array(
            [left.coordinate_sigma_rlu, right.coordinate_sigma_rlu], dtype=float
        )
        if not np.allclose(expected_sigmas, observed_sigmas, rtol=2e-3, atol=1e-10):
            raise LocusSchemaError(
                f"pair {pair_id!r}: covariance diagonal disagrees with coordinate sigma"
            )
        pairs.append(
            LocusPair(
                pair_id=pair_id,
                scan_id=left.scan_id,
                source_path=left.source_path,
                axis=left.axis,
                energy_mev=left.energy_mev,
                energy_sigma_mev=left.energy_sigma_mev,
                energy_sigma_source=left.energy_sigma_source,
                fixed_ef_mev=left.fixed_ef_mev,
                observed_coordinates_rlu=(left.coordinate_rlu, right.coordinate_rlu),
                covariance_rlu2=left.pair_coordinate_covariance_rlu2,
                zone_center_hkl=left.zone_center_hkl,
                scan_path_hkl_start=left.scan_path_hkl_start,
                scan_path_hkl_end=left.scan_path_hkl_end,
                point_ids=(left.point_id, right.point_id),
                resolution_classification=left.resolution_classification,
                resolution_scenario_summaries=left.resolution_scenario_summaries,
            )
        )

    if max_common_center_offset_rlu is not None:
        if (
            not math.isfinite(max_common_center_offset_rlu)
            or max_common_center_offset_rlu <= 0.0
        ):
            raise LocusSchemaError(
                "max_common_center_offset_rlu must be finite and positive"
            )
        retained_pairs = []
        excluded_pair_count = 0
        for pair in pairs:
            common_offset = abs(
                0.5 * sum(pair.observed_coordinates_rlu) - pair.zone_coordinate
            )
            if common_offset <= max_common_center_offset_rlu:
                retained_pairs.append(pair)
            else:
                excluded_pair_count += 1
        if excluded_pair_count:
            exclusion_counts["common_center_offset_sensitivity_mask"] = (
                2 * excluded_pair_count
            )
        pairs = retained_pairs

    if not pairs:
        raise LocusSchemaError("no complete, producer-included, uncensored locus pairs remain")
    axes = {pair.axis for pair in pairs}
    if axes != {"H", "L"}:
        raise LocusSchemaError(
            f"joint fit requires both H and L locus pairs; retained axes={sorted(axes)}"
        )
    return LoadedLoci(
        pairs=tuple(pairs),
        total_rows=len(records),
        excluded_rows=len(records) - 2 * len(pairs),
        exclusion_counts=dict(sorted(exclusion_counts.items())),
    )


@dataclass(frozen=True)
class RootPrediction:
    coordinate_rlu: float
    energy_mismatch_mev: float
    exact_root: bool


def _predict_root(
    backend: McNallyAcousticBackend,
    pair: LocusPair,
    side: str,
    target_energy_mev: float,
    parameters: Sequence[float],
    *,
    grid_size: int,
    root_energy_tolerance_mev: float,
) -> RootPrediction:
    lower, upper = pair.path_coordinate_bounds
    center = pair.zone_coordinate
    if side == "left":
        interval = (lower, center)
    elif side == "right":
        interval = (center, upper)
    else:  # pragma: no cover - internal invariant
        raise ValueError(side)
    if interval[1] - interval[0] <= 1e-10:
        raise LocusSchemaError(f"pair {pair.pair_id}: empty {side} scan interval")

    coordinates = np.linspace(interval[0], interval[1], max(65, int(grid_size)))
    hkl = np.vstack([pair.hkl_at(value) for value in coordinates])
    differences = backend.energy(hkl, parameters) - float(target_energy_mev)
    roots: list[float] = []
    for index in range(len(coordinates) - 1):
        left_f, right_f = differences[index], differences[index + 1]
        if abs(left_f) <= root_energy_tolerance_mev:
            roots.append(float(coordinates[index]))
        if left_f * right_f < 0.0:
            root = brentq(
                lambda value: float(
                    backend.energy(pair.hkl_at(value)[None, :], parameters)[0]
                    - target_energy_mev
                ),
                float(coordinates[index]),
                float(coordinates[index + 1]),
                xtol=1e-12,
                rtol=1e-12,
            )
            roots.append(float(root))
    if abs(differences[-1]) <= root_energy_tolerance_mev:
        roots.append(float(coordinates[-1]))

    # A root at a band extremum only touches zero and has no sign change.  Refine
    # only the best few grid-local minima of |E-target|.  In an unstable trial
    # model the clipped dispersion can contain long flat zero regions; refining
    # every grid point in such a region makes profile calculations pathological.
    absolute = np.abs(differences)
    candidate_indices = [] if roots else [
        index
        for index in range(1, len(coordinates) - 1)
        if absolute[index] <= absolute[index - 1]
        and absolute[index] <= absolute[index + 1]
    ]
    candidate_indices = sorted(
        candidate_indices,
        key=lambda index: (
            absolute[index],
            abs(coordinates[index] - center),
        ),
    )[:3]
    refined: list[tuple[float, float]] = []
    for index in candidate_indices:
        solution = minimize_scalar(
            lambda value: float(
                backend.energy(pair.hkl_at(value)[None, :], parameters)[0]
                - target_energy_mev
            )
            ** 2,
            bounds=(float(coordinates[index - 1]), float(coordinates[index + 1])),
            method="bounded",
            options={"xatol": 1e-9},
        )
        coordinate = float(solution.x)
        mismatch = float(
            backend.energy(pair.hkl_at(coordinate)[None, :], parameters)[0]
            - target_energy_mev
        )
        refined.append((coordinate, mismatch))
        if abs(mismatch) <= root_energy_tolerance_mev:
            roots.append(coordinate)

    if roots:
        # Follow the acoustic root continuously outward from the zone center.
        # Do not select a model root using the observed locus.
        coordinate = min(roots, key=lambda value: abs(value - center))
        mismatch = float(
            backend.energy(pair.hkl_at(coordinate)[None, :], parameters)[0]
            - target_energy_mev
        )
        return RootPrediction(coordinate, mismatch, True)

    # Outside the current model's band there is no real root.  Return the closest
    # point in energy and retain its mismatch as a finite optimizer residual.
    fallback = [
        (float(coordinates[0]), float(differences[0])),
        (float(coordinates[-1]), float(differences[-1])),
        *refined,
    ]
    coordinate, mismatch = min(
        fallback,
        key=lambda item: (abs(item[1]), abs(item[0] - center)),
    )
    return RootPrediction(coordinate, mismatch, False)


class PositionOnlyObjective:
    """Coordinate-space objective for correlated fixed-energy locus pairs.

    The coordinate half-separation is the sole statistical residual.  A
    parameter point without two exact acoustic roots for every included scan
    lies outside the tested all-pairs acoustic hypothesis.  Such points are
    rejected as infeasible and never assigned an experimental chi-square.
    """

    def __init__(
        self,
        pairs: Sequence[LocusPair],
        *,
        grid_size: int = 257,
        profile_scan_energy: bool = False,
        energy_shift_sigma_limit: float = 4.0,
        root_energy_tolerance_mev: float = 1e-6,
    ):
        self.pairs = tuple(pairs)
        self.backend = McNallyAcousticBackend()
        self.grid_size = int(grid_size)
        self.profile_scan_energy = bool(profile_scan_energy)
        self.energy_shift_sigma_limit = float(energy_shift_sigma_limit)
        self.root_energy_tolerance_mev = float(root_energy_tolerance_mev)
        if self.profile_scan_energy:
            missing_sigma = [pair.pair_id for pair in self.pairs if pair.energy_sigma_mev is None]
            non_scenario_sigma = [
                pair.pair_id
                for pair in self.pairs
                if not pair.energy_sigma_source.startswith("resolution_scenario:")
            ]
            if missing_sigma:
                raise LocusSchemaError(
                    "--profile-scan-energy requires a finite scenario-specific energy "
                    f"resolution sigma for every pair; missing for {missing_sigma[:5]}"
                )
            if non_scenario_sigma:
                raise LocusSchemaError(
                    "--profile-scan-energy requires --resolution-scenario-filter so its "
                    "energy widths have an explicit configuration provenance"
                )
        grid = np.linspace(0.0, 1.0, 17)
        self._stability_hkl = np.array(
            [(h, k, ell) for h in grid for k in grid for ell in grid], dtype=float
        )
        # Vectorized side-wise path grids make the nonlinear band-support
        # constraints cheap enough for SLSQP.  A small positive energy margin
        # absorbs the finite path grid error; final acceptance still calls the
        # exact root finder on every pair.
        support_grid_size = max(513, self.grid_size)
        support_paths = []
        for pair in self.pairs:
            lower, upper = pair.path_coordinate_bounds
            center = pair.zone_coordinate
            support_paths.append(
                [
                    np.vstack(
                        [
                            pair.hkl_at(value)
                            for value in np.linspace(lower, center, support_grid_size)
                        ]
                    ),
                    np.vstack(
                        [
                            pair.hkl_at(value)
                            for value in np.linspace(center, upper, support_grid_size)
                        ]
                    ),
                ]
            )
        self._support_hkl = np.asarray(support_paths, dtype=float)

    def _pair_components(
        self,
        pair: LocusPair,
        parameters: Sequence[float],
        energy_shift_mev: float,
    ) -> tuple[np.ndarray, tuple[RootPrediction, RootPrediction]]:
        target = pair.energy_mev + float(energy_shift_mev)
        observed = np.asarray(pair.observed_coordinates_rlu, dtype=float)
        predictions = tuple(
            _predict_root(
                self.backend,
                pair,
                side,
                target,
                parameters,
                grid_size=self.grid_size,
                root_energy_tolerance_mev=self.root_energy_tolerance_mev,
            )
            for index, side in enumerate(("left", "right"))
        )
        predicted_q = np.array([item.coordinate_rlu for item in predictions], dtype=float)
        half_vector = np.array([-0.5, 0.5], dtype=float)
        covariance = np.asarray(pair.covariance_rlu2, dtype=float)
        half_sigma = math.sqrt(float(half_vector @ covariance @ half_vector))
        observed_half = 0.5 * (observed[1] - observed[0])
        predicted_half = 0.5 * (predicted_q[1] - predicted_q[0])
        exact_pair = all(prediction.exact_root for prediction in predictions)
        # The fallback coordinate makes a continuous extension for constrained
        # optimizer trial evaluations. residuals() still rejects the complete
        # parameter point if exact_pair is false, so this is never reported as
        # an experimental residual.
        position_residual = np.array(
            [(predicted_half - observed_half) / half_sigma]
        )
        shift_residual = (
            # Constructor validation proves this optional scale is non-null and
            # comes from the explicitly requested resolution scenario.
            np.array([energy_shift_mev / float(pair.energy_sigma_mev)])
            if self.profile_scan_energy
            else np.empty(0, dtype=float)
        )
        return np.concatenate([position_residual, shift_residual]), predictions

    def _profile_pair(
        self, pair: LocusPair, parameters: Sequence[float]
    ) -> tuple[np.ndarray, float, tuple[RootPrediction, RootPrediction]]:
        if not self.profile_scan_energy:
            components, predictions = self._pair_components(pair, parameters, 0.0)
            return components, 0.0, predictions
        sigma_e = float(pair.energy_sigma_mev)
        limit = self.energy_shift_sigma_limit * sigma_e

        def scalar_objective(shift: float) -> float:
            components, predictions = self._pair_components(pair, parameters, shift)
            if not all(prediction.exact_root for prediction in predictions):
                return math.inf
            return float(np.dot(components, components))

        solution = minimize_scalar(
            scalar_objective,
            bounds=(-limit, limit),
            method="bounded",
            options={"xatol": max(1e-8, sigma_e * 1e-6)},
        )
        candidates = [float(solution.x), 0.0, -limit, limit]
        feasible_candidates = [
            candidate for candidate in candidates if math.isfinite(scalar_objective(candidate))
        ]
        shift = (
            min(feasible_candidates, key=scalar_objective)
            if feasible_candidates
            else 0.0
        )
        components, predictions = self._pair_components(pair, parameters, shift)
        return components, shift, predictions

    def stability_residual(self, parameters: Sequence[float]) -> float:
        minimum = float(np.min(self.backend.radicand(self._stability_hkl, parameters)))
        # radicand units are meV^2 before the leading factor four in E^2.
        return max(0.0, -minimum - 1e-9) / 0.25

    def _support_side_maxima(self, parameters: Sequence[float]) -> np.ndarray:
        """Grid maxima with a local quadratic refinement in energy."""

        energies = np.asarray(self.backend.energy(self._support_hkl, parameters), dtype=float)
        flat = energies.reshape(-1, energies.shape[-1])
        maxima = np.max(flat, axis=1)
        indices = np.argmax(flat, axis=1)
        for row_index, peak_index in enumerate(indices):
            if peak_index <= 0 or peak_index >= flat.shape[1] - 1:
                continue
            left, center, right = flat[row_index, peak_index - 1 : peak_index + 2]
            curvature = left - 2.0 * center + right
            if curvature < 0.0:
                offset = 0.5 * (left - right) / curvature
                if abs(offset) <= 1.0:
                    maxima[row_index] = center - 0.25 * (left - right) * offset
        return maxima.reshape(len(self.pairs), 2)

    def feasibility_margins(self, parameters: Sequence[float]) -> np.ndarray:
        """Continuous SLSQP inequalities for side-wise acoustic band support.

        Dynamic stability is handled by start/end rejection.  It is not placed
        in SLSQP because the exact Goldstone point makes the minimum radicand an
        identically active zero constraint with a singular numerical gradient.
        """

        physical = np.asarray(parameters, dtype=float)
        if physical.shape != (3,) or not np.all(np.isfinite(physical)):
            return np.full(2 * len(self.pairs), -1.0e6, dtype=float)
        side_maxima = self._support_side_maxima(physical)
        targets = np.array(
            [
                pair.energy_mev + self._profile_pair(pair, physical)[1]
                if self.profile_scan_energy
                else pair.energy_mev
                for pair in self.pairs
            ],
            dtype=float,
        )
        root_margins = (
            side_maxima - targets[:, None] + ROOT_CONSTRAINT_TOLERANCE_MEV
        )
        return root_margins.ravel()

    def extended_sum_squares(self, parameters: Sequence[float]) -> float:
        """Continuous optimizer extension; equals reported objective when feasible."""

        physical = np.asarray(parameters, dtype=float)
        if physical.shape != (3,) or not np.all(np.isfinite(physical)):
            return math.inf
        evaluated = [self._profile_pair(pair, physical) for pair in self.pairs]
        components = [item[0] for item in evaluated]
        value = float(np.dot(np.concatenate(components), np.concatenate(components)))
        # This assists restoration from finite-difference trial steps outside
        # the feasible set. SLSQP separately enforces the margins >= 0, and the
        # penalty is absent from every accepted/reported objective.
        violations = np.minimum(self.feasibility_margins(physical), 0.0)
        return value + 100.0 * float(np.dot(violations, violations))

    def search_residuals(self, parameters: Sequence[float]) -> np.ndarray:
        """Smooth continuation residual used only to locate feasible endpoints.

        The first component per pair is the continuously extended coordinate
        residual.  The second is one shared worst-side energy mismatch if the
        scan lacks two roots, divided by a fixed numerical search scale.  This
        vector is never reported as chi-square or used to rank infeasible
        endpoints; accepted endpoints have every mismatch component equal zero.
        """

        physical = np.asarray(parameters, dtype=float)
        if physical.shape != (3,) or not np.all(np.isfinite(physical)):
            per_pair = 3 if self.profile_scan_energy else 2
            return np.full(per_pair * len(self.pairs) + 1, 1.0e6, dtype=float)
        evaluated = [self._profile_pair(pair, physical) for pair in self.pairs]
        result = []
        for components, _, predictions in evaluated:
            result.extend(components.tolist())
            missing = [
                prediction.energy_mismatch_mev
                for prediction in predictions
                if not prediction.exact_root
            ]
            worst = max(missing, key=abs) if missing else 0.0
            result.append(worst / SEARCH_NO_ROOT_SCALE_MEV)
        result.append(self.stability_residual(physical))
        return np.asarray(result, dtype=float)

    def root_feasible(self, parameters: Sequence[float]) -> bool:
        """Whether every included pair has two exact roots under the treatment."""

        physical = np.asarray(parameters, dtype=float)
        if physical.shape != (3,) or not np.all(np.isfinite(physical)):
            return False
        if self.stability_residual(physical) > 0.0:
            return False
        for pair in self.pairs:
            _, _, predictions = self._profile_pair(pair, physical)
            if not all(prediction.exact_root for prediction in predictions):
                return False
        return True

    def residuals(self, parameters: Sequence[float]) -> np.ndarray:
        physical = np.asarray(parameters, dtype=float)
        per_pair = 2 if self.profile_scan_energy else 1
        if physical.shape != (3,) or not np.all(np.isfinite(physical)):
            return np.full(per_pair * len(self.pairs) + 1, 1e6, dtype=float)
        instability = self.stability_residual(physical)
        if instability > 0.0:
            return np.full(
                per_pair * len(self.pairs) + 1, 1e3 + instability, dtype=float
            )
        evaluated = [self._profile_pair(pair, physical) for pair in self.pairs]
        if any(
            not all(prediction.exact_root for prediction in predictions)
            for _, _, predictions in evaluated
        ):
            # Purely numerical hard wall. Admissible starts/endpoints are
            # prefiltered, and this value is never reported as chi-square.
            return np.full(
                per_pair * len(self.pairs) + 1,
                HARD_INFEASIBLE_RESIDUAL,
                dtype=float,
            )
        components = [item[0] for item in evaluated]
        return np.concatenate([*components, [0.0]])

    def predictions(self, parameters: Sequence[float]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for pair in self.pairs:
            components, shift, predictions = self._profile_pair(pair, parameters)
            exact_pair = all(prediction.exact_root for prediction in predictions)
            position_chi2 = float(components[0] ** 2) if exact_pair else None
            energy_shift_prior = (
                float(components[1] ** 2)
                if self.profile_scan_energy and exact_pair
                else (0.0 if exact_pair else None)
            )
            result.append(
                {
                    "pair_id": pair.pair_id,
                    "scan_id": pair.scan_id,
                    "axis": pair.axis,
                    "energy_mev": pair.energy_mev,
                    "profiled_energy_shift_mev": shift,
                    "observed_coordinates_rlu": list(pair.observed_coordinates_rlu),
                    "predicted_coordinates_rlu": [
                        prediction.coordinate_rlu for prediction in predictions
                    ],
                    "observed_half_separation_rlu": 0.5
                    * (
                        pair.observed_coordinates_rlu[1]
                        - pair.observed_coordinates_rlu[0]
                    ),
                    "predicted_half_separation_rlu": 0.5
                    * (
                        predictions[1].coordinate_rlu
                        - predictions[0].coordinate_rlu
                    ),
                    "observed_common_center_offset_rlu": 0.5
                    * sum(pair.observed_coordinates_rlu)
                    - pair.zone_coordinate,
                    "exact_roots": [prediction.exact_root for prediction in predictions],
                    "energy_mismatches_mev": [
                        prediction.energy_mismatch_mev for prediction in predictions
                    ],
                    "has_two_exact_roots": exact_pair,
                    "position_chi2_contribution": position_chi2,
                    "energy_shift_prior_contribution": energy_shift_prior,
                    "objective_contribution": (
                        float(np.dot(components, components)) if exact_pair else None
                    ),
                }
            )
        return result


def _fit_once(
    objective: PositionOnlyObjective,
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_nfev: int,
) -> Any:
    return minimize(
        objective.extended_sum_squares,
        np.clip(np.asarray(start, dtype=float), lower + 1e-10, upper - 1e-10),
        method="Nelder-Mead",
        bounds=list(zip(lower, upper)),
        options={
            "maxiter": int(max_nfev),
            "maxfev": int(max_nfev) * 3,
            "xatol": 1.0e-9,
            "fatol": 1.0e-8,
            "adaptive": True,
            "disp": False,
        },
    )


def _objective_point_admissible(
    objective: Any, parameters: Sequence[float]
) -> bool:
    """Apply stability and, when available, all-roots feasibility gates."""

    if objective.stability_residual(parameters) > 0.0:
        return False
    root_feasible = getattr(objective, "root_feasible", None)
    return bool(root_feasible(parameters)) if root_feasible is not None else True


def _solution_objective_value(result: Any) -> float:
    value = np.asarray(result.fun, dtype=float)
    return float(value) if value.ndim == 0 else float(np.dot(value, value))


def fit_multistart(
    objective: PositionOnlyObjective,
    start: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    multistart: int = 6,
    seed: int = 20260812,
    max_nfev: int = 800,
) -> Any:
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    requested_start = np.asarray(start, dtype=float)
    starts = (
        [requested_start]
        if objective.stability_residual(requested_start) == 0.0
        else []
    )
    rng = np.random.default_rng(seed)
    attempts = 0
    while len(starts) < max(1, int(multistart)) and attempts < 100 * max(1, multistart):
        candidate = rng.uniform(lower_array, upper_array)
        attempts += 1
        if objective.stability_residual(candidate) == 0.0:
            starts.append(candidate)
    if not starts:
        raise RuntimeError(
            "no dynamically stable multistart point lies within the bounds"
        )
    solutions = [
        _fit_once(
            objective,
            candidate,
            lower_array,
            upper_array,
            max_nfev=max_nfev,
        )
        for candidate in starts
    ]
    valid = [
        result
        for result in solutions
        if result.success
        and np.all(np.isfinite(result.fun))
        and _objective_point_admissible(objective, result.x)
    ]
    if not valid:
        messages = [str(result.message) for result in solutions]
        raise RuntimeError(
            "no stable all-pairs-root-feasible multistart fit converged "
            f"successfully: {messages}"
        )
    chosen = min(valid, key=_solution_objective_value)
    chosen.multistart_attempted = len(starts)
    chosen.multistart_converged_stable = len(valid)
    return chosen


def finite_difference_jacobian(
    objective: PositionOnlyObjective,
    parameters: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> np.ndarray:
    point = np.asarray(parameters, dtype=float)
    lower_array, upper_array = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if not _objective_point_admissible(objective, point):
        raise RuntimeError("Jacobian point is outside the all-pairs-root-feasible domain")
    center_residual = objective.residuals(point)
    columns = []
    for index, value in enumerate(point):
        initial_step = max(abs(value) * 2e-4, [2e-4, 2e-5, 2e-5][index])
        derivative = None
        for shrink in range(9):
            step = initial_step * (0.5**shrink)
            left = point.copy()
            right = point.copy()
            left[index] = max(lower_array[index], value - step)
            right[index] = min(upper_array[index], value + step)
            left_ok = (
                left[index] < value and _objective_point_admissible(objective, left)
            )
            right_ok = (
                right[index] > value and _objective_point_admissible(objective, right)
            )
            if left_ok and right_ok:
                derivative = (
                    objective.residuals(right) - objective.residuals(left)
                ) / (right[index] - left[index])
                break
            if right_ok:
                derivative = (
                    objective.residuals(right) - center_residual
                ) / (right[index] - value)
                break
            if left_ok:
                derivative = (
                    center_residual - objective.residuals(left)
                ) / (value - left[index])
                break
        if derivative is None:
            raise RuntimeError(
                f"cannot finite-difference {PARAMETER_NAMES[index]} within the "
                "all-pairs-root-feasible domain"
            )
        columns.append(derivative)
    return np.column_stack(columns)


def jacobian_diagnostics(
    objective: PositionOnlyObjective,
    parameters: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict[str, Any]:
    point = np.asarray(parameters, dtype=float)
    jacobian = finite_difference_jacobian(objective, point, lower, upper)
    raw_u, raw_s, raw_vt = np.linalg.svd(jacobian, full_matrices=False)
    del raw_u
    scales = np.maximum(np.abs(point), np.array([1.0, 0.05, 0.1]))
    scaled_jacobian = jacobian @ np.diag(scales)
    scaled_u, scaled_s, scaled_vt = np.linalg.svd(scaled_jacobian, full_matrices=False)
    del scaled_u
    condition = math.inf if scaled_s[-1] <= 0.0 else float(scaled_s[0] / scaled_s[-1])
    rank_tolerance = np.finfo(float).eps * max(jacobian.shape) * scaled_s[0]
    rank = int(np.sum(scaled_s > rank_tolerance))
    jtj = jacobian.T @ jacobian
    covariance = np.linalg.pinv(jtj)
    return {
        "parameter_order": list(PARAMETER_NAMES),
        "raw_singular_values": raw_s.tolist(),
        "raw_right_singular_vectors": raw_vt.tolist(),
        "dimensionless_parameter_scales": scales.tolist(),
        "dimensionless_singular_values": scaled_s.tolist(),
        "dimensionless_right_singular_vectors": scaled_vt.tolist(),
        "dimensionless_condition_number": condition,
        "numerical_rank": rank,
        "unscaled_inverse_jtj": covariance.tolist(),
    }


def profile_likelihoods(
    objective: PositionOnlyObjective,
    best: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    points: int = 17,
    multistart: int = 2,
    max_nfev: int = 500,
) -> dict[str, list[dict[str, Any]]]:
    if points < 3:
        return {}
    best_array = np.asarray(best, dtype=float)
    lower_array, upper_array = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    best_objective = float(
        np.dot(objective.residuals(best_array), objective.residuals(best_array))
    )
    profiles: dict[str, list[dict[str, Any]]] = {}
    for fixed_index, name in enumerate(PARAMETER_NAMES):
        # Quadratic spacing retains the full declared bounds while resolving the
        # neighborhood of the optimum/flat valley much better than a uniform
        # grid over intentionally broad bounds.
        coordinate = np.linspace(-1.0, 1.0, points)
        grid = np.where(
            coordinate < 0.0,
            best_array[fixed_index]
            - (best_array[fixed_index] - lower_array[fixed_index]) * coordinate**2,
            best_array[fixed_index]
            + (upper_array[fixed_index] - best_array[fixed_index]) * coordinate**2,
        )
        grid = np.unique(np.append(grid, best_array[fixed_index]))
        free_indices = [index for index in range(3) if index != fixed_index]
        free_lower = lower_array[free_indices]
        free_upper = upper_array[free_indices]
        rows: list[dict[str, Any]] = []
        free_start = best_array[free_indices].copy()
        # An independent deterministic seed targets the low-SJ1/high-ratio/high-
        # SJc part of the broad prior box.  It is generated from bounds rather
        # than hard-coding fitted values and catches the disconnected basin seen
        # in this model.  Additional templates cover the low-ratio basin.
        fraction_templates = (
            np.array([0.03, 0.70, 0.55], dtype=float),
            np.array([0.10, 0.25, 0.16], dtype=float),
            np.array([0.26, 0.66, 0.065], dtype=float),
        )
        template_points = [
            lower_array + fraction * (upper_array - lower_array)
            for fraction in fraction_templates
        ]
        for fixed_value in sorted(grid, key=lambda value: abs(value - best_array[fixed_index])):
            def residual_free(free_values: np.ndarray) -> np.ndarray:
                candidate = best_array.copy()
                candidate[fixed_index] = fixed_value
                candidate[free_indices] = free_values
                return candidate

            def scalar_free(free_values: np.ndarray) -> float:
                return objective.extended_sum_squares(residual_free(free_values))

            candidate_starts = [free_start]
            for template in template_points[: max(0, int(multistart) - 1)]:
                candidate_starts.append(template[free_indices])
            solutions = []
            for candidate_start in candidate_starts:
                full_start = best_array.copy()
                full_start[fixed_index] = fixed_value
                full_start[free_indices] = candidate_start
                if objective.stability_residual(full_start) > 0.0:
                    continue
                solutions.append(
                    minimize(
                        scalar_free,
                        np.clip(candidate_start, free_lower + 1e-10, free_upper - 1e-10),
                        method="Nelder-Mead",
                        bounds=list(zip(free_lower, free_upper)),
                        options={
                            "maxiter": int(max_nfev),
                            "maxfev": int(max_nfev) * 3,
                            "xatol": 2.0e-8,
                            "fatol": 1.0e-6,
                            "adaptive": True,
                            "disp": False,
                        },
                    )
                )
            valid = []
            for candidate_solution in solutions:
                candidate_parameters = best_array.copy()
                candidate_parameters[fixed_index] = fixed_value
                candidate_parameters[free_indices] = candidate_solution.x
                if (
                    candidate_solution.success
                    and np.all(np.isfinite(candidate_solution.fun))
                    and _objective_point_admissible(
                        objective, candidate_parameters
                    )
                ):
                    valid.append(candidate_solution)
            if not valid:
                rows.append(
                    {
                        "fixed_value": float(fixed_value),
                        "objective_sum_squares": None,
                        "signed_delta_objective": None,
                        "optimized_parameters": None,
                        "success": False,
                        "message": (
                            "no stable all-pairs-root-feasible profile start "
                            "converged successfully"
                        ),
                        "attempted_starts": len(candidate_starts),
                        "converged_stable_starts": 0,
                    }
                )
                continue
            solution = min(
                valid, key=_solution_objective_value
            )
            free_start = solution.x.copy()
            candidate = best_array.copy()
            candidate[fixed_index] = fixed_value
            candidate[free_indices] = solution.x
            objective_value = _solution_objective_value(solution)
            rows.append(
                {
                    "fixed_value": float(fixed_value),
                    "objective_sum_squares": objective_value,
                    # Preserve the sign: a negative value proves the main
                    # multistart fit missed a lower basin and must never be
                    # hidden by clipping to zero.
                    "signed_delta_objective": objective_value - best_objective,
                    "optimized_parameters": dict(zip(PARAMETER_NAMES, candidate.tolist())),
                    "success": bool(solution.success),
                    "message": str(solution.message),
                    "attempted_starts": len(candidate_starts),
                    "converged_stable_starts": len(valid),
                }
            )
        profiles[name] = sorted(rows, key=lambda row: row["fixed_value"])
    return profiles


def coverage_diagnostics(
    objective: PositionOnlyObjective, parameters: Sequence[float]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for axis in ("H", "L"):
        pairs = [pair for pair in objective.pairs if pair.axis == axis]
        observed_max = max(pair.energy_mev for pair in pairs)
        path_maxima = []
        for pair in pairs:
            coordinates = np.linspace(*pair.path_coordinate_bounds, 2001)
            hkl = np.vstack([pair.hkl_at(value) for value in coordinates])
            path_maxima.append(float(np.max(objective.backend.energy(hkl, parameters))))
        model_band_top = max(path_maxima)
        result[axis] = {
            "pair_count": len(pairs),
            "observed_max_energy_mev": observed_max,
            "model_path_band_top_mev": model_band_top,
            "observed_max_to_model_band_top_fraction": (
                observed_max / model_band_top if model_band_top > 0.0 else math.nan
            ),
            "reaches_band_top_within_20_percent": bool(
                model_band_top > 0.0 and observed_max >= 0.8 * model_band_top
            ),
        }
    return result


def repeat_scatter_diagnostics(pairs: Sequence[LocusPair]) -> dict[str, Any]:
    """Compare repeated nominal loci to their local fitted-position errors."""

    groups: dict[tuple[Any, ...], list[LocusPair]] = {}
    for pair in pairs:
        key = (
            pair.axis,
            *(round(value, 6) for value in pair.zone_center_hkl),
            round(pair.energy_mev, 6),
        )
        groups.setdefault(key, []).append(pair)
    rows = []
    repeated_pair_count = 0
    for key, repeated in groups.items():
        if len(repeated) < 2:
            continue
        repeated_pair_count += len(repeated)
        half_separations = np.array(
            [
                0.5
                * (
                    pair.observed_coordinates_rlu[1]
                    - pair.observed_coordinates_rlu[0]
                )
                for pair in repeated
            ],
            dtype=float,
        )
        half_sigmas = []
        vector = np.array([-0.5, 0.5], dtype=float)
        for pair in repeated:
            covariance = np.asarray(pair.covariance_rlu2, dtype=float)
            half_sigmas.append(math.sqrt(float(vector @ covariance @ vector)))
        scatter = float(np.std(half_separations, ddof=1))
        median_sigma = float(np.median(half_sigmas))
        rows.append(
            {
                "axis": key[0],
                "zone_center_hkl": list(key[1:4]),
                "energy_mev": key[4],
                "repeat_count": len(repeated),
                "scan_ids": [pair.scan_id for pair in repeated],
                "fixed_ef_mev_values": sorted(
                    {
                        pair.fixed_ef_mev
                        for pair in repeated
                        if pair.fixed_ef_mev is not None
                    }
                ),
                "half_separations_rlu": half_separations.tolist(),
                "sample_scatter_rlu": scatter,
                "median_local_half_separation_sigma_rlu": median_sigma,
                "scatter_over_median_local_sigma": scatter / median_sigma,
            }
        )
    rows.sort(key=lambda row: row["scatter_over_median_local_sigma"], reverse=True)
    maximum_ratio = max(
        (row["scatter_over_median_local_sigma"] for row in rows), default=0.0
    )
    return {
        "repeated_nominal_group_count": len(rows),
        "pair_count_in_repeated_groups": repeated_pair_count,
        "maximum_scatter_over_median_local_sigma": maximum_ratio,
        "repeat_systematics_detected": maximum_ratio > 2.0,
        "largest_scatter_groups": rows[:10],
        "interpretation": (
            "repeat scatter above local peak-center covariance indicates unmodeled "
            "configuration/resolution/repeat systematics; treating all repeats as "
            "independent position-only Gaussian observations understates uncertainty"
        ),
    }


def run_fit(
    loaded: LoadedLoci,
    *,
    start: Sequence[float] = DEFAULT_START,
    lower: Sequence[float] = DEFAULT_LOWER,
    upper: Sequence[float] = DEFAULT_UPPER,
    multistart: int = 6,
    seed: int = 20260812,
    max_nfev: int = 800,
    profile_points: int = 17,
    profile_multistart: int = 2,
    profile_max_nfev: int = 500,
    grid_size: int = 257,
    min_band_top_fraction: float = 0.80,
    profile_scan_energy: bool = False,
) -> dict[str, Any]:
    objective = PositionOnlyObjective(
        loaded.pairs,
        grid_size=grid_size,
        profile_scan_energy=profile_scan_energy,
    )
    solution = fit_multistart(
        objective,
        start,
        lower,
        upper,
        multistart=multistart,
        seed=seed,
        max_nfev=max_nfev,
    )
    best = np.asarray(solution.x, dtype=float)
    residual = objective.residuals(best)
    objective_value = float(np.dot(residual, residual))
    coverage = coverage_diagnostics(objective, best)
    insufficient = [
        axis
        for axis, item in coverage.items()
        if item["observed_max_to_model_band_top_fraction"] < min_band_top_fraction
    ]
    if insufficient:
        values = {
            axis: coverage[axis]["observed_max_to_model_band_top_fraction"]
            for axis in insufficient
        }
        raise LocusSchemaError(
            "locus artifact does not reach the requested fraction of the fitted band top "
            f"for axes {values}; lower --min-band-top-fraction only with an explicit reason"
        )

    diagnostics = jacobian_diagnostics(objective, best, lower, upper)
    profiles = profile_likelihoods(
        objective,
        best,
        lower,
        upper,
        points=profile_points,
        multistart=profile_multistart,
        max_nfev=profile_max_nfev,
    )
    signed_profile_deltas = [
        float(row["signed_delta_objective"])
        for rows in profiles.values()
        for row in rows
        if row["signed_delta_objective"] is not None
    ]
    minimum_profile_delta = min(signed_profile_deltas, default=0.0)
    profile_found_lower_than_main = minimum_profile_delta < -1e-4
    pair_predictions = objective.predictions(best)
    pairs_without_two_exact_roots = sum(
        not row["has_two_exact_roots"] for row in pair_predictions
    )
    if pairs_without_two_exact_roots:
        raise RuntimeError(
            "optimizer returned a point outside the all-pairs-root-feasible domain"
        )
    number_physical_parameters = 3
    statistical_position_observation_count = (
        len(loaded.pairs) - pairs_without_two_exact_roots
    )
    statistical_position_dof = (
        statistical_position_observation_count - number_physical_parameters
    )
    statistical_position_chi2 = float(
        sum(row["position_chi2_contribution"] for row in pair_predictions)
    )
    energy_shift_prior = float(
        sum(row["energy_shift_prior_contribution"] for row in pair_predictions)
    )
    reduced_position_chi2 = (
        statistical_position_chi2 / statistical_position_dof
        if statistical_position_dof > 0
        else None
    )
    repeat_diagnostics = repeat_scatter_diagnostics(loaded.pairs)
    centered_predictions = [
        row
        for row in pair_predictions
        if abs(row["observed_common_center_offset_rlu"]) <= 0.05
    ]
    common_center_sensitivity = {
        "absolute_common_center_offset_threshold_rlu": 0.05,
        "retained_pair_count": len(centered_predictions),
        "excluded_pair_count": len(pair_predictions) - len(centered_predictions),
        "position_chi2_at_full_fit_parameters": float(
            sum(row["position_chi2_contribution"] for row in centered_predictions)
        ),
        "note": "diagnostic subset evaluation only; parameters were not refit",
    }
    mcnally_parameters = np.array([7.9, 0.165, 0.51], dtype=float)
    mcnally_feasible = _objective_point_admissible(objective, mcnally_parameters)
    mcnally_objective = None
    if mcnally_feasible:
        mcnally_residual = objective.residuals(mcnally_parameters)
        mcnally_objective = float(np.dot(mcnally_residual, mcnally_residual))
    swap_defined = bool(best[2] > 0.0)
    swap_partner = (
        objective.backend.exact_exchange_swap(best) if swap_defined else None
    )
    swap_objective = None
    swap_inside_bounds = False
    if swap_partner is not None:
        swap_residual = objective.residuals(swap_partner)
        swap_objective = float(np.dot(swap_residual, swap_residual))
        swap_inside_bounds = bool(
            np.all(swap_partner >= np.asarray(lower, dtype=float))
            and np.all(swap_partner <= np.asarray(upper, dtype=float))
        )
    return {
        "schema_version": "2.0",
        "analysis_type": "position_only_acoustic_dispersion_fit",
        "model": {
            "id": MODEL_ID,
            "equation": "E=2*sqrt((3*(SJ1+SJc)-6*SJ2+J2(Q))^2-|J1c(Q)|^2)",
            "exchange_input_bridge": "post-PR110 high-level pySpinW representative bonds use 2*SJ/S",
            "topology_provenance": {
                "source": "explicit SrMn2Sb2 nearest-shell representative bond list",
                "representative_bond_counts": {"J1": 3, "J2": 6, "Jc": 3},
                "independent_energy_check": "post-PR110 high-level pySpinW model agrees with the analytic acoustic energies at locked Q points",
                "paper_caveat": "McNally's printed mean-field expression contains 2*SJc; it does not independently establish this Sr zc=3 topology or its parameter normalization",
            },
            "anisotropy": "fixed_zero_gapless_acoustic",
            "intensities_used": False,
            "optical_assignment_used": False,
            "branch_hypothesis": "all retained unassigned symmetric pairs are tested against the gapless acoustic branch",
            "position_observable": "pair_half_separation_with_full_covariance",
            "scan_energy_treatment": (
                "profiled_shared_shift" if profile_scan_energy else "fixed_nominal_energy"
            ),
            "primary_energy_resolution_weighting": (
                "explicit_requested_resolution_scenario"
                if profile_scan_energy
                else "none"
            ),
            "missing_root_treatment": "hard_all_pairs_two_exact_roots_constraint",
        },
        "input_summary": {
            "total_rows": loaded.total_rows,
            "excluded_rows": loaded.excluded_rows,
            "exclusion_counts": dict(loaded.exclusion_counts),
            "included_pair_count": len(loaded.pairs),
            "included_point_count": 2 * len(loaded.pairs),
            "included_pairs_by_resolution_classification": dict(
                sorted(
                    (
                        classification,
                        sum(
                            pair.resolution_classification == classification
                            for pair in loaded.pairs
                        ),
                    )
                    for classification in {
                        pair.resolution_classification for pair in loaded.pairs
                    }
                )
            ),
        },
        "fit": {
            "parameters": dict(zip(PARAMETER_NAMES, best.tolist())),
            "objective_sum_squares": objective_value,
            "objective_contains_nonstatistical_feasibility_penalty": False,
            "statistical_position_chi2": statistical_position_chi2,
            "statistical_position_observation_count": statistical_position_observation_count,
            "statistical_position_dof": statistical_position_dof,
            "statistical_position_chi2_per_dof": reduced_position_chi2,
            "energy_shift_prior_sum_squares": energy_shift_prior,
            "optimizer_residual_vector_length": int(len(residual)),
            "success": bool(solution.success),
            "optimizer_success_only": True,
            "multistart_attempted": int(solution.multistart_attempted),
            "multistart_converged_stable": int(
                solution.multistart_converged_stable
            ),
            "pairs_without_two_exact_roots": pairs_without_two_exact_roots,
            "all_included_pairs_have_two_exact_roots": True,
            "acoustic_hypothesis_adequate": bool(
                reduced_position_chi2 is not None
                and reduced_position_chi2 <= 2.0
                and pairs_without_two_exact_roots == 0
            ),
            "message": str(solution.message),
            "active_mask": [
                -1
                if math.isclose(value, lo, rel_tol=0.0, abs_tol=1e-7)
                else (
                    1
                    if math.isclose(value, hi, rel_tol=0.0, abs_tol=1e-7)
                    else 0
                )
                for value, lo, hi in zip(best, lower, upper)
            ],
            "bounds": {
                name: [float(lo), float(hi)]
                for name, lo, hi in zip(PARAMETER_NAMES, lower, upper)
            },
        },
        "coverage": coverage,
        "data_systematics": {
            "repeat_scatter": repeat_diagnostics,
            "common_center_offset_sensitivity": common_center_sensitivity,
        },
        "reference_evaluations": {
            "mcnally_camn2sb2_published_sj": {
                "parameters": dict(zip(PARAMETER_NAMES, mcnally_parameters.tolist())),
                "all_included_pairs_have_two_exact_roots": mcnally_feasible,
                "objective_sum_squares": mcnally_objective,
                "signed_delta_objective_from_best": (
                    mcnally_objective - objective_value
                    if mcnally_objective is not None
                    else None
                ),
                "note": (
                    "not assigned an objective because at least one included scan "
                    "has no two-sided acoustic root"
                    if not mcnally_feasible
                    else "all included scan pairs are root-feasible"
                ),
            }
        },
        "exact_nonidentifiability": {
            "transformation": "(SJ1, J2/J1, SJc) -> (SJc, SJ1*(J2/J1)/SJc, SJ1)",
            "reason": "This zc=3 Sr bond-topology realization is invariant when SJ1 and SJc are exchanged while SJ2 is held fixed",
            "identifiable_invariant_combinations": objective.backend.identifiable_exchange_invariants(best),
            "swap_partner_defined": swap_defined,
            "swap_partner_parameters": (
                dict(zip(PARAMETER_NAMES, swap_partner.tolist()))
                if swap_partner is not None
                else None
            ),
            "swap_partner_inside_fit_bounds": swap_inside_bounds,
            "swap_partner_objective_sum_squares": swap_objective,
            "signed_objective_difference_from_reported_solution": (
                swap_objective - objective_value
                if swap_objective is not None
                else None
            ),
            "undefined_reason": (
                "the reported SJc is zero, so the ratio-coordinate form of the "
                "swap partner is singular"
                if not swap_defined
                else None
            ),
            "conclusion": "Acoustic energies alone cannot label the two opposite-sublattice coefficients as in-plane SJ1 versus interlayer SJc.",
        },
        "identifiability": diagnostics,
        "profile_likelihood": profiles,
        "profile_likelihood_method": {
            "type": "deterministic_multistart_continuation",
            "starts_per_fixed_value": int(profile_multistart),
            "global_minimum_guaranteed": False,
            "calibrated_likelihood": False,
            "profiled_quantity": (
                "position chi-square plus explicit scenario energy-shift prior"
                if profile_scan_energy
                else "position chi-square"
            ),
            "domain_constraint": "every included scan pair must have two exact acoustic roots",
            "minimum_signed_delta_objective": minimum_profile_delta,
            "profile_found_lower_than_main_fit": profile_found_lower_than_main,
            "warning": "nonsmooth root switching and disconnected basins can still hide a lower profiled minimum",
        },
        "pair_predictions": pair_predictions,
        "limitations": [
            "Position-only fit; measured intensities are not read.",
            "The fit tests the explicit hypothesis that all retained unassigned symmetric pairs follow the gapless acoustic branch; it does not infer branch identities.",
            (
                "A profiled one-dimensional energy displacement is not a full TAS resolution convolution."
                if profile_scan_energy
                else "Nominal scan energies are fixed; TAS energy resolution is not convolved."
            ),
            "The analytic backend uses McNally-style SJ units on the explicit Sr topology z1=3, z2=6, zc=3; it is not a claim that the paper uniquely specifies zc=3.",
            "The zc=3 topology comes from the explicit SrMn2Sb2 nearest-shell bond list and its independent post-PR110 pySpinW energy check, not from the paper's mean-field coordination expression, which prints 2*SJc and is conventionally ambiguous here.",
            "The energy-only model has an exact SJ1/SJc label-swap symmetry at fixed SJ2; individual SJ1, J2/J1, and SJc labels are not globally identifiable without an external constraint.",
            "Parameter and profile points lacking two exact roots for any included pair are infeasible and receive no reported objective value.",
            "Repeated nominal scans show scatter beyond local peak-center covariance, often across configurations; the independent-Gaussian position chi-square understates systematic uncertainty.",
            "The common-center offset sensitivity is evaluated at the full-fit parameters and is not a resolution-convolved refit.",
            "Optimizer convergence is not model acceptance; signed profile-objective differences are diagnostics, not calibrated confidence intervals when the acoustic hypothesis has global misfit.",
        ],
    }


def _write_outputs(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    profile_path = output_dir / "profile_likelihood.csv"
    with profile_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "parameter",
                "fixed_value",
                "objective_sum_squares",
                "signed_delta_objective",
                *PARAMETER_NAMES,
                "success",
                "attempted_starts",
                "converged_stable_starts",
            ],
        )
        writer.writeheader()
        for parameter, rows in report["profile_likelihood"].items():
            for row in rows:
                writer.writerow(
                    {
                        "parameter": parameter,
                        "fixed_value": row["fixed_value"],
                        "objective_sum_squares": row["objective_sum_squares"],
                        "signed_delta_objective": row["signed_delta_objective"],
                        **(row["optimized_parameters"] or {}),
                        "success": row["success"],
                        "attempted_starts": row["attempted_starts"],
                        "converged_stable_starts": row[
                            "converged_stable_starts"
                        ],
                    }
                )
    prediction_path = output_dir / "pair_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "pair_id",
            "scan_id",
            "axis",
            "energy_mev",
            "profiled_energy_shift_mev",
            "observed_left_rlu",
            "observed_right_rlu",
            "predicted_left_rlu",
            "predicted_right_rlu",
            "observed_half_separation_rlu",
            "predicted_half_separation_rlu",
            "observed_common_center_offset_rlu",
            "left_exact_root",
            "right_exact_root",
            "position_chi2_contribution",
            "energy_shift_prior_contribution",
            "objective_contribution",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in report["pair_predictions"]:
            writer.writerow(
                {
                    "pair_id": row["pair_id"],
                    "scan_id": row["scan_id"],
                    "axis": row["axis"],
                    "energy_mev": row["energy_mev"],
                    "profiled_energy_shift_mev": row["profiled_energy_shift_mev"],
                    "observed_left_rlu": row["observed_coordinates_rlu"][0],
                    "observed_right_rlu": row["observed_coordinates_rlu"][1],
                    "predicted_left_rlu": row["predicted_coordinates_rlu"][0],
                    "predicted_right_rlu": row["predicted_coordinates_rlu"][1],
                    "observed_half_separation_rlu": row["observed_half_separation_rlu"],
                    "predicted_half_separation_rlu": row["predicted_half_separation_rlu"],
                    "observed_common_center_offset_rlu": row["observed_common_center_offset_rlu"],
                    "left_exact_root": row["exact_roots"][0],
                    "right_exact_root": row["exact_roots"][1],
                    "position_chi2_contribution": row[
                        "position_chi2_contribution"
                    ],
                    "energy_shift_prior_contribution": row[
                        "energy_shift_prior_contribution"
                    ],
                    "objective_contribution": row["objective_contribution"],
                }
            )


def _triple(values: Iterable[str], name: str) -> np.ndarray:
    result = np.asarray([float(value) for value in values], dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError(f"{name} requires exactly three finite values")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="full-locus JSONL/CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", nargs=3, default=DEFAULT_START.tolist(), metavar=("SJ1", "RATIO", "SJC"))
    parser.add_argument("--lower", nargs=3, default=DEFAULT_LOWER.tolist(), metavar=("SJ1", "RATIO", "SJC"))
    parser.add_argument("--upper", nargs=3, default=DEFAULT_UPPER.tolist(), metavar=("SJ1", "RATIO", "SJC"))
    parser.add_argument("--multistart", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--max-nfev", type=int, default=800)
    parser.add_argument("--profile-points", type=int, default=17)
    parser.add_argument("--profile-max-nfev", type=int, default=500)
    parser.add_argument("--profile-multistart", type=int, default=2)
    parser.add_argument("--root-grid-size", type=int, default=257)
    parser.add_argument("--min-band-top-fraction", type=float, default=0.80)
    parser.add_argument(
        "--resolution-scenario-filter",
        help=(
            "optional sensitivity mask: retain only pairs with "
            "resolved_by_scenario=true for this named scenario"
        ),
    )
    parser.add_argument(
        "--max-common-center-offset-rlu",
        type=float,
        help=(
            "explicit sensitivity mask excluding pairs whose fitted common "
            "center differs from the nominal zone center by more than this value"
        ),
    )
    parser.add_argument(
        "--profile-scan-energy",
        action="store_true",
        help=(
            "profile one shared energy shift per pair as a sensitivity check; "
            "requires --resolution-scenario-filter"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = _triple(args.start, "--start")
    lower = _triple(args.lower, "--lower")
    upper = _triple(args.upper, "--upper")
    if np.any(lower >= upper):
        raise SystemExit("each --lower bound must be smaller than --upper")
    if np.any(start < lower) or np.any(start > upper):
        raise SystemExit("--start must lie inside the supplied bounds")
    if not 0.0 < args.min_band_top_fraction <= 1.1:
        raise SystemExit("--min-band-top-fraction must be in (0, 1.1]")
    if args.profile_points not in {0} and args.profile_points < 3:
        raise SystemExit("--profile-points must be 0 (skip) or at least 3")
    if args.profile_multistart < 1:
        raise SystemExit("--profile-multistart must be at least 1")
    if args.profile_scan_energy and not args.resolution_scenario_filter:
        raise SystemExit(
            "--profile-scan-energy requires --resolution-scenario-filter"
        )
    loaded = load_locus_pairs(
        args.input,
        resolution_scenario_filter=args.resolution_scenario_filter,
        max_common_center_offset_rlu=args.max_common_center_offset_rlu,
    )
    report = run_fit(
        loaded,
        start=start,
        lower=lower,
        upper=upper,
        multistart=args.multistart,
        seed=args.seed,
        max_nfev=args.max_nfev,
        profile_points=args.profile_points,
        profile_multistart=args.profile_multistart,
        profile_max_nfev=args.profile_max_nfev,
        grid_size=args.root_grid_size,
        min_band_top_fraction=args.min_band_top_fraction,
        profile_scan_energy=args.profile_scan_energy,
    )
    report = {
        "input_path": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "resolution_scenario_filter": args.resolution_scenario_filter,
        "max_common_center_offset_rlu": args.max_common_center_offset_rlu,
        **report,
    }
    _write_outputs(args.output_dir, report)
    fit = report["fit"]
    print(
        json.dumps(
            {
                "parameters": fit["parameters"],
                "objective_sum_squares": fit["objective_sum_squares"],
                "statistical_position_chi2": fit["statistical_position_chi2"],
                "pairs_without_two_exact_roots": fit[
                    "pairs_without_two_exact_roots"
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {args.output_dir / 'fit_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
