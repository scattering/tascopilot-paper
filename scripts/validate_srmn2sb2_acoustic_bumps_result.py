#!/usr/bin/env python3
"""Fail-closed comparison of Bumps DREAM and LM SrMn2Sb2 endpoints.

The Bumps objective contains numerical band-support continuation terms, so its
printed ``chisq`` is not an experimental statistic.  This validator reloads
both independent ``.par`` files, applies the exact all-pairs two-root gate,
computes position chi-square only for feasible endpoints, and retains LM only
when it genuinely improves on a feasible DREAM endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fit_srmn2sb2_acoustic_loci import (
    MODEL_ID,
    PARAMETER_NAMES,
    PositionOnlyObjective,
    load_locus_pairs,
    repeat_scatter_diagnostics,
)
from fit_srmn2sb2_acoustic_loci_bumps import (
    BUMPS_PARAMETER_NAMES,
    DEFAULT_BUMPS_LOWER,
    DEFAULT_BUMPS_UPPER,
    DEFAULT_FEASIBILITY_WEIGHT,
    continuation_residuals,
    exact_dynamic_stability,
    exact_optimizer_swap,
    optimizer_to_core,
    search_objective_provenance,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameter_dict(parameters: Sequence[float], names: Sequence[str]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, parameters)}


def read_bumps_parameters(path: Path) -> np.ndarray:
    """Read symmetric optimizer coordinates by name from a Bumps ``.par``."""

    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2 or fields[0] not in BUMPS_PARAMETER_NAMES:
            raise ValueError(f"unexpected Bumps parameter line {line!r}: {path}")
        if fields[0] in values:
            raise ValueError(f"duplicate Bumps parameter {fields[0]!r}: {path}")
        values[fields[0]] = float(fields[1])
    missing = [name for name in BUMPS_PARAMETER_NAMES if name not in values]
    if missing:
        raise ValueError(f"Bumps parameter file lacks {missing}: {path}")
    parameters = np.asarray([values[name] for name in BUMPS_PARAMETER_NAMES])
    if not np.all(np.isfinite(parameters)):
        raise ValueError(f"Bumps parameter file contains non-finite values: {path}")
    return parameters


def _inside_default_bounds(parameters: Sequence[float]) -> bool:
    point = np.asarray(parameters, dtype=float)
    return bool(
        point.shape == (3,)
        and np.all(point >= DEFAULT_BUMPS_LOWER - 1e-12)
        and np.all(point <= DEFAULT_BUMPS_UPPER + 1e-12)
    )


def validate_endpoint(
    objective: PositionOnlyObjective,
    *,
    parameter_path: Path,
    stage: str,
    feasibility_weight: float = DEFAULT_FEASIBILITY_WEIGHT,
) -> dict[str, Any]:
    """Validate one endpoint without assigning chi-square when roots are absent."""

    optimizer_parameters = read_bumps_parameters(parameter_path)
    core_parameters = optimizer_to_core(optimizer_parameters)
    predictions = objective.predictions(core_parameters)
    missing = [
        {
            "pair_id": row["pair_id"],
            "scan_id": row["scan_id"],
            "axis": row["axis"],
            "energy_mev": row["energy_mev"],
            "energy_mismatches_mev": row["energy_mismatches_mev"],
            "common_center_offset_rlu": row["observed_common_center_offset_rlu"],
        }
        for row in predictions
        if not row["has_two_exact_roots"]
    ]
    within_bounds = _inside_default_bounds(optimizer_parameters)
    analytic_stable = exact_dynamic_stability(optimizer_parameters)
    stable = bool(
        analytic_stable and objective.stability_residual(core_parameters) == 0.0
    )
    feasible = bool(
        within_bounds
        and stable
        and not missing
        and objective.root_feasible(core_parameters)
    )
    search_residual = continuation_residuals(
        objective,
        optimizer_parameters,
        feasibility_weight=feasibility_weight,
    )
    position_chi2 = None
    position_dof = None
    position_chi2_per_dof = None
    if feasible:
        position_chi2 = float(
            sum(float(row["position_chi2_contribution"]) for row in predictions)
        )
        position_dof = len(objective.pairs) - 3
        position_chi2_per_dof = (
            position_chi2 / position_dof if position_dof > 0 else None
        )
    return {
        "stage": stage,
        "status": (
            "accepted_root_feasible_endpoint"
            if feasible
            else "rejected_no_root_or_out_of_domain_endpoint"
        ),
        "parameter_path": str(parameter_path.resolve()),
        "parameter_sha256": _sha256(parameter_path),
        "optimizer_parameters": _parameter_dict(
            optimizer_parameters, BUMPS_PARAMETER_NAMES
        ),
        "core_parameters": _parameter_dict(core_parameters, PARAMETER_NAMES),
        "inside_declared_symmetric_bounds": within_bounds,
        "dynamically_stable": stable,
        "exact_analytic_dynamic_stability": analytic_stable,
        "dynamic_stability_condition": "SJ2 <= (SJ_a+SJ_b)/6 when SJ2>=0; negative SJ2 stable",
        "root_feasible": feasible,
        "missing_root_pair_count": len(missing),
        "missing_root_pairs": missing,
        "minimum_continuous_feasibility_margin_mev": float(
            np.min(objective.feasibility_margins(core_parameters))
        ),
        "search_sum_squares_not_experimental_chi2": float(
            np.dot(search_residual, search_residual)
        ),
        "statistical_position_chi2": position_chi2,
        "statistical_position_dof": position_dof,
        "statistical_position_chi2_per_dof": position_chi2_per_dof,
        "pair_predictions": predictions if feasible else None,
    }


def compare_endpoints(
    *,
    input_path: Path,
    dream_parameter_path: Path,
    lm_parameter_path: Path | None,
    case: str,
    root_grid_size: int,
    feasibility_weight: float = DEFAULT_FEASIBILITY_WEIGHT,
) -> dict[str, Any]:
    """Validate DREAM/LM independently and retain the better feasible stage."""

    if case not in {"all102", "center005"}:
        raise ValueError(f"unsupported case {case!r}")
    center_limit = None if case == "all102" else 0.05
    loaded = load_locus_pairs(
        input_path,
        max_common_center_offset_rlu=center_limit,
    )
    objective = PositionOnlyObjective(loaded.pairs, grid_size=root_grid_size)
    stages = [
        validate_endpoint(
            objective,
            parameter_path=dream_parameter_path,
            stage="dream",
            feasibility_weight=feasibility_weight,
        )
    ]
    if lm_parameter_path is not None:
        stages.append(
            validate_endpoint(
                objective,
                parameter_path=lm_parameter_path,
                stage="lm",
                feasibility_weight=feasibility_weight,
            )
        )
    feasible = [row for row in stages if row["root_feasible"]]
    accepted_stage = feasible[0] if feasible else None
    lm_improvement_tolerance = None
    if len(feasible) == 2:
        dream = next(row for row in feasible if row["stage"] == "dream")
        lm = next(row for row in feasible if row["stage"] == "lm")
        dream_chi2 = float(dream["statistical_position_chi2"])
        lm_chi2 = float(lm["statistical_position_chi2"])
        lm_improvement_tolerance = max(1.0e-8, 1.0e-10 * abs(dream_chi2))
        accepted_stage = (
            lm if lm_chi2 < dream_chi2 - lm_improvement_tolerance else dream
        )
    accepted_fit = None
    if accepted_stage is not None:
        optimizer_parameters = np.asarray(
            [
                accepted_stage["optimizer_parameters"][name]
                for name in BUMPS_PARAMETER_NAMES
            ],
            dtype=float,
        )
        core_parameters = optimizer_to_core(optimizer_parameters)
        swap_optimizer = exact_optimizer_swap(optimizer_parameters)
        swap_core = optimizer_to_core(swap_optimizer)
        swap_feasible = objective.root_feasible(swap_core)
        swap_chi2 = None
        if swap_feasible:
            swap_residual = objective.residuals(swap_core)
            swap_chi2 = float(np.dot(swap_residual, swap_residual))
        position_chi2 = float(accepted_stage["statistical_position_chi2"])
        position_chi2_per_dof = accepted_stage["statistical_position_chi2_per_dof"]
        accepted_fit = {
            "stage": accepted_stage["stage"],
            "optimizer_parameters": accepted_stage["optimizer_parameters"],
            "core_parameters": accepted_stage["core_parameters"],
            "statistical_position_chi2": position_chi2,
            "statistical_position_dof": accepted_stage["statistical_position_dof"],
            "statistical_position_chi2_per_dof": position_chi2_per_dof,
            "all_included_pairs_have_two_exact_roots": True,
            "pairs_without_two_exact_roots": 0,
            "objective_contains_nonstatistical_search_penalty": False,
            "acoustic_hypothesis_adequate": bool(
                position_chi2_per_dof is not None and position_chi2_per_dof <= 2.0
            ),
            "pair_predictions": accepted_stage["pair_predictions"],
            "exact_nonidentifiability": {
                "optimizer_transformation": "(SJ_a,SJ2,SJ_b)->(SJ_b,SJ2,SJ_a)",
                "core_transformation": (
                    "(SJ1,J2/J1,SJc)->(SJc,SJ1*(J2/J1)/SJc,SJ1)"
                ),
                "swap_partner_optimizer_parameters": _parameter_dict(
                    swap_optimizer, BUMPS_PARAMETER_NAMES
                ),
                "swap_partner_core_parameters": _parameter_dict(
                    swap_core, PARAMETER_NAMES
                ),
                "swap_partner_inside_declared_bounds": _inside_default_bounds(
                    swap_optimizer
                ),
                "swap_partner_root_feasible": swap_feasible,
                "swap_partner_position_chi2": swap_chi2,
                "signed_position_chi2_difference": (
                    swap_chi2 - position_chi2 if swap_chi2 is not None else None
                ),
                "identifiable_invariant_combinations": (
                    objective.backend.identifiable_exchange_invariants(core_parameters)
                ),
                "scientific_conclusion": (
                    "acoustic energies alone cannot label either member as the "
                    "in-plane SJ1 versus interlayer SJc exchange"
                ),
            },
        }
    report = {
        "schema_version": "srmn2sb2.acoustic_bumps_dream_lm.v1",
        "status": (
            "accepted_hard_root_feasible_endpoint"
            if accepted_fit is not None
            else "no_hard_root_feasible_endpoint"
        ),
        "analysis_type": "position_only_acoustic_dispersion_bumps_dream_lm",
        "case": case,
        "model_id": MODEL_ID,
        "input_path": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "input_summary": {
            "total_rows": loaded.total_rows,
            "excluded_rows": loaded.excluded_rows,
            "included_pair_count": len(loaded.pairs),
            "included_point_count": 2 * len(loaded.pairs),
            "exclusion_counts": dict(loaded.exclusion_counts),
        },
        "root_grid_size": root_grid_size,
        "observable": {
            "type": "correlated_pair_half_separation_at_fixed_nominal_energy",
            "intensities_used": False,
            "optical_assignment_used": False,
            "energy_resolution_used_in_primary_fit": False,
            "hard_final_gate": "two exact acoustic roots for every included pair",
        },
        "optimizer_coordinates": {
            "names": list(BUMPS_PARAMETER_NAMES),
            "lower": DEFAULT_BUMPS_LOWER.tolist(),
            "upper": DEFAULT_BUMPS_UPPER.tolist(),
            "bounds_are_exact_swap_symmetric": True,
        },
        "search_objective": search_objective_provenance(feasibility_weight),
        "stage_endpoints": {row["stage"]: row for row in stages},
        "lm_retained_over_dream": bool(
            accepted_fit is not None and accepted_fit["stage"] == "lm"
        ),
        "lm_improvement_tolerance_position_chi2": lm_improvement_tolerance,
        "accepted_fit": accepted_fit,
        "data_systematics": {
            "repeat_scatter": repeat_scatter_diagnostics(loaded.pairs),
        },
        "scientific_outcome": (
            "No DREAM or LM endpoint passed the all-pairs two-root gate; no "
            "experimental fit statistic or exchange estimate is reported."
            if accepted_fit is None
            else "A hard-root-feasible endpoint was found; model adequacy is "
            "separate from optimizer success and exchange labels remain exactly ambiguous."
        ),
        "limitations": [
            "DREAM is a global search trace, not a calibrated posterior, because the search residual contains nonstatistical feasibility terms.",
            "LM is retained only if hard-feasible and better in pure position chi-square; Bumps optimizer success is not scientific acceptance.",
            "No LM covariance is reported because the root objective is nonsmooth at band edges and has an exact disconnected exchange-label symmetry.",
            "All retained unassigned pairs are tested as acoustic; branch identity is not inferred.",
            "Nominal scan energies are fixed and TAS resolution is not convolved.",
            "Repeat/configuration scatter can exceed local peak-center covariance, so position chi-square is not a complete uncertainty model.",
        ],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dream-par", type=Path, required=True)
    parser.add_argument("--lm-par", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--selector-report", type=Path, required=True)
    parser.add_argument("--case", choices=("all102", "center005"), required=True)
    parser.add_argument("--root-grid-size", type=int, default=129)
    parser.add_argument(
        "--feasibility-weight", type=float, default=DEFAULT_FEASIBILITY_WEIGHT
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare_endpoints(
        input_path=args.input,
        dream_parameter_path=args.dream_par,
        lm_parameter_path=args.lm_par,
        case=args.case,
        root_grid_size=args.root_grid_size,
        feasibility_weight=args.feasibility_weight,
    )
    source_paths = {
        "validator": Path(__file__).resolve(),
        "core": Path(__file__).resolve().with_name(
            "fit_srmn2sb2_acoustic_loci.py"
        ),
        "bumps_bridge": Path(__file__).resolve().with_name(
            "fit_srmn2sb2_acoustic_loci_bumps.py"
        ),
        "bumps_model": args.model.resolve(),
        "dream_selector": Path(__file__).resolve().with_name(
            "select_srmn2sb2_acoustic_bumps_dream_candidate.py"
        ),
        "selector_report": args.selector_report.resolve(),
    }
    missing_sources = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing_sources:
        raise SystemExit(f"missing provenance source files: {missing_sources}")
    report["provenance"] = {
        "argv": list(sys.argv if argv is None else [str(Path(__file__).resolve()), *argv]),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "environment_whitelist": {
            name: os.environ.get(name)
            for name in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_CPUS_PER_TASK",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "NUMBA_DISABLE_JIT",
                "PYTHONHASHSEED",
                "PYTHONNOUSERSITE",
                "MPLBACKEND",
            )
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["accepted_fit"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
