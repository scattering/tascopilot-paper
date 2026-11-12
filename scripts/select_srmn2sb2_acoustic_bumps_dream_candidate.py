#!/usr/bin/env python3
"""Select the best hard-feasible candidate from a raw Bumps DREAM chain.

Every finite raw draw is reevaluated with the denser hard-gate objective.  The
selected point is the globally smallest pure position chi-square among the
hard-feasible chain draws, with a deterministic stored-index tie break. Raw
``state.chains()`` is used;
``state.draw()`` is intentionally avoided because Bumps 1.0.3 may filter chains
marked as outliers even after a run requested ``--outliers=none``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from bumps.cli import load_model
from bumps.dream.state import load_state

from fit_srmn2sb2_acoustic_loci import PositionOnlyObjective, load_locus_pairs
from fit_srmn2sb2_acoustic_loci_bumps import (
    BUMPS_PARAMETER_NAMES,
    DEFAULT_BUMPS_LOWER,
    DEFAULT_BUMPS_UPPER,
    continuation_residuals,
    exact_dynamic_stability,
    optimizer_to_core,
)
from validate_srmn2sb2_acoustic_bumps_result import read_bumps_parameters


_WORKER_OBJECTIVE: PositionOnlyObjective | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_objective(input_path: str, case: str, grid_size: int) -> PositionOnlyObjective:
    loaded = load_locus_pairs(
        Path(input_path),
        max_common_center_offset_rlu=None if case == "all102" else 0.05,
    )
    return PositionOnlyObjective(loaded.pairs, grid_size=grid_size)


def _init_worker(input_path: str, case: str, grid_size: int) -> None:
    global _WORKER_OBJECTIVE
    _WORKER_OBJECTIVE = _load_objective(input_path, case, grid_size)


def _hard_evaluate(parameters: Sequence[float]) -> tuple[bool, float]:
    assert _WORKER_OBJECTIVE is not None
    point = np.asarray(parameters, dtype=float)
    if (
        point.shape != (3,)
        or not np.all(np.isfinite(point))
        or np.any(point < DEFAULT_BUMPS_LOWER - 1e-12)
        or np.any(point > DEFAULT_BUMPS_UPPER + 1e-12)
        or not exact_dynamic_stability(point)
    ):
        return False, float("inf")
    core_point = optimizer_to_core(point)
    if not _WORKER_OBJECTIVE.root_feasible(core_point):
        return False, float("inf")
    residual = _WORKER_OBJECTIVE.residuals(core_point)
    chi2 = float(np.dot(residual, residual))
    return (
        (True, chi2)
        if np.all(np.isfinite(residual)) and np.isfinite(chi2)
        else (False, float("inf"))
    )


def _canonical_par(path: Path, parameters: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{name} {float(value):.17g}\n"
            for name, value in zip(BUMPS_PARAMETER_NAMES, parameters)
        ),
        encoding="utf-8",
    )


def _deduplicate(candidates: Iterable[tuple[str, float, np.ndarray]]):
    seen: set[tuple[float, ...]] = set()
    for source, logp, point in candidates:
        key = tuple(np.round(point, 13).tolist())
        if key not in seen:
            seen.add(key)
            yield source, logp, point


def select_candidate(
    *,
    input_path: Path,
    model_path: Path,
    dream_state_base: Path,
    dream_parameter_path: Path,
    case: str,
    output_parameter_path: Path,
    root_grid_size: int = 257,
    parallel: int = 1,
) -> dict[str, Any]:
    if case not in {"all102", "center005"}:
        raise ValueError(f"unsupported case {case!r}")
    if parallel < 1:
        raise ValueError("parallel must be at least one")
    # Never allow a prior selector result to survive a current fail-closed run.
    output_parameter_path.unlink(missing_ok=True)

    # Bind state columns to the live model definition. Legacy DREAM text state
    # reloads with placeholder labels P0..Pn, so those placeholders are accepted
    # only after the model itself proves the exact expected label order.
    problem = load_model(str(model_path), [str(input_path), case])
    model_labels = tuple(problem.labels())
    if model_labels != BUMPS_PARAMETER_NAMES:
        raise ValueError(
            f"model parameter order {model_labels} does not match {BUMPS_PARAMETER_NAMES}"
        )
    state = load_state(str(dream_state_base), report=0)
    _, raw_points, raw_logp = state.chains()
    points = np.asarray(raw_points, dtype=float).reshape(-1, raw_points.shape[-1])
    logp = np.asarray(raw_logp, dtype=float).reshape(-1)
    if points.shape != (len(logp), len(BUMPS_PARAMETER_NAMES)):
        raise ValueError(
            f"DREAM state has incompatible point/logp shapes {points.shape}/{logp.shape}"
        )
    state_labels = tuple(state.labels)
    expected_placeholders = tuple(f"P{i}" for i in range(len(BUMPS_PARAMETER_NAMES)))
    if state_labels not in {BUMPS_PARAMETER_NAMES, expected_placeholders}:
        raise ValueError(f"unexpected DREAM state labels {state_labels}")

    finite_indices = np.flatnonzero(np.isfinite(logp) & np.all(np.isfinite(points), axis=1))
    best_point, best_logp = state.best()
    par_point = read_bumps_parameters(dream_parameter_path)
    best_point = np.asarray(best_point, dtype=float)
    if (
        best_point.shape != (3,)
        or not np.all(np.isfinite(best_point))
        or not np.isfinite(best_logp)
    ):
        raise ValueError("DREAM state.best point/logp must be finite and three-dimensional")
    if not np.allclose(par_point, best_point, rtol=1e-11, atol=1e-12):
        raise ValueError(
            "DREAM final .par does not equal state.best(); refusing a stale or "
            "mismatched state/parameter handoff"
        )
    problem.setp(par_point)
    par_logp = -float(problem.nllf())
    if not np.isfinite(par_logp):
        raise ValueError("current model evaluation of the DREAM final .par is non-finite")
    logp_tolerance = max(1.0e-6, 1.0e-10 * abs(float(best_logp)))
    if abs(par_logp - float(best_logp)) > logp_tolerance:
        raise ValueError(
            "DREAM state.best logp disagrees with the current model/input "
            f"evaluation ({best_logp} versus {par_logp})"
        )
    candidates = list(
        _deduplicate(
            [
                ("state_best", float(best_logp), np.asarray(best_point, dtype=float)),
                ("final_par", par_logp, par_point),
                *[
                    (f"raw_chain_{int(index)}", float(logp[index]), points[index])
                    for index in finite_indices
                ],
            ]
        )
    )

    evaluations: list[tuple[bool, float]]
    if parallel == 1:
        _init_worker(str(input_path), case, root_grid_size)
        evaluations = [_hard_evaluate(row[2]) for row in candidates]
    else:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=parallel,
            initializer=_init_worker,
            initargs=(str(input_path), case, root_grid_size),
        ) as pool:
            evaluations = pool.map(_hard_evaluate, [row[2] for row in candidates])
    feasible_indices = [
        index for index, (feasible, _) in enumerate(evaluations) if feasible
    ]
    selected_index = (
        min(feasible_indices, key=lambda index: (evaluations[index][1], index))
        if feasible_indices
        else None
    )
    selected = candidates[selected_index] if selected_index is not None else None

    selected_point = selected[2] if selected is not None else None
    selected_position_chi2 = None
    selected_search_sum_squares = None
    selected_stored_minus_two_logp = None
    selected_metric_difference = None
    if selected_point is not None:
        _canonical_par(output_parameter_path, selected_point)
        selected_core = optimizer_to_core(selected_point)
        selected_position_residual = _WORKER_OBJECTIVE.residuals(selected_core) if parallel == 1 else None
        if selected_position_residual is None:
            audit_objective = _load_objective(str(input_path), case, root_grid_size)
            selected_position_residual = audit_objective.residuals(selected_core)
        else:
            audit_objective = _WORKER_OBJECTIVE
        selected_position_chi2 = evaluations[selected_index][1]
        selected_search_residual = continuation_residuals(
            audit_objective, selected_point
        )
        selected_search_sum_squares = float(
            np.dot(selected_search_residual, selected_search_residual)
        )
        selected_stored_minus_two_logp = -2.0 * float(selected[1])
        selected_metric_difference = (
            selected_search_sum_squares - selected_stored_minus_two_logp
        )
    state_files = [
        Path(str(dream_state_base) + suffix)
        for suffix in ("-chain.mc.gz", "-point.mc.gz", "-stats.mc.gz")
    ]
    return {
        "schema_version": "srmn2sb2.acoustic_dream_chain_selector.v1",
        "status": "hard_feasible_candidate_selected" if selected is not None else "no_hard_feasible_chain_candidate",
        "case": case,
        "root_grid_size": root_grid_size,
        "parallel_workers": parallel,
        "raw_chain_point_count": len(points),
        "candidate_count": len(candidates),
        "hard_gate_candidates_checked": len(candidates),
        "hard_feasible_candidate_count": len(feasible_indices),
        "selection_rule": "global minimum dense-grid hard-feasible pure position chi2, ties by deterministic candidate index",
        "dynamic_stability_gate": "exact SJ2<=(SJ_a+SJ_b)/6 for SJ2>=0; negative SJ2 stable; plus core grid diagnostic",
        "state_labels": list(state_labels),
        "state_label_binding": (
            "exact_saved_labels" if state_labels == BUMPS_PARAMETER_NAMES else "legacy_Pn_placeholders_bound_by_verified_model_labels"
        ),
        "verified_model_labels": list(model_labels),
        "selected_source": selected[0] if selected is not None else None,
        "selected_logp": float(selected[1]) if selected is not None else None,
        "selected_stored_minus_two_logp": selected_stored_minus_two_logp,
        "selected_recomputed_search_sum_squares": selected_search_sum_squares,
        "selected_hard_position_chi2": selected_position_chi2,
        "stored_grid129_minus_two_logp_vs_dense_selector_grid_metric_diagnostic": {
            "stored_minus_two_logp": selected_stored_minus_two_logp,
            "recomputed_selector_grid_search_sum_squares": selected_search_sum_squares,
            "recomputed_minus_stored": selected_metric_difference,
            "selector_grid_size": root_grid_size,
            "interpretation": (
                "diagnostic only; the metrics may differ when a point is root-feasible "
                "on the denser selector grid but was penalized on the DREAM grid129"
            ),
        },
        "stored_best_current_model_logp_tolerance": logp_tolerance,
        "selected_optimizer_parameters": (
            {name: float(value) for name, value in zip(BUMPS_PARAMETER_NAMES, selected_point)}
            if selected_point is not None else None
        ),
        "selected_core_parameters": (
            {name: float(value) for name, value in zip(("sj1_mev", "j2_over_j1", "sjc_mev"), optimizer_to_core(selected_point))}
            if selected_point is not None else None
        ),
        "output_parameter_path": str(output_parameter_path.resolve()) if selected is not None else None,
        "input_sha256": _sha256(input_path),
        "model_sha256": _sha256(model_path),
        "dream_parameter_sha256": _sha256(dream_parameter_path),
        "dream_state_sha256": {
            path.name: _sha256(path) for path in state_files if path.is_file()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dream-state-base", type=Path, required=True)
    parser.add_argument("--dream-par", type=Path, required=True)
    parser.add_argument("--case", choices=("all102", "center005"), required=True)
    parser.add_argument("--root-grid-size", type=int, default=257)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--output-par", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = select_candidate(
        input_path=args.input,
        model_path=args.model,
        dream_state_base=args.dream_state_base,
        dream_parameter_path=args.dream_par,
        case=args.case,
        output_parameter_path=args.output_par,
        root_grid_size=args.root_grid_size,
        parallel=args.parallel,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["selected_optimizer_parameters"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
