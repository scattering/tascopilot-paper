"""Notebook-free tasAI policy adapter example.

This adapter demonstrates the same `propose_next_point(payload) -> dict`
protocol as the adaptive NICE workflow. It is a lightweight bridge for tasAI
policy ideas such as uncertainty sampling, motion-aware scoring, and forecasting
without importing tasAI's MCMC/bumps stack into the core environment.
"""

from __future__ import annotations

from typing import Any


def propose_next_point(payload: dict[str, Any]) -> dict[str, Any]:
    """Select a candidate from uncertainty, information-gain, and motion hints.

    Candidate metadata may include:
    - `posterior_variance`
    - `expected_information_gain`
    - `estimated_move_time`
    - `forecast_step`

    Model spec options:
    - `eta`: exponent on information gain, matching the HH information-rate
      spirit. Defaults to 0.7.
    - `motion_penalty_weight`: extra penalty on move time. Defaults to 1.0.
    - `default_count_time`: returned count time when no candidate count hint is
      supplied.
    """

    candidates = payload.get("candidates") or []
    if not candidates:
        return {"point_id": "no-candidate", "score": 0.0, "reason": "missing_candidates"}
    model_spec = payload.get("model_spec") or {}
    eta = float(model_spec.get("eta") or 0.7)
    motion_penalty_weight = float(model_spec.get("motion_penalty_weight") or 1.0)
    default_count_time = float(model_spec.get("default_count_time") or 60.0)
    best: dict[str, Any] | None = None
    best_score = float("-inf")
    best_components: dict[str, float] = {}
    for index, candidate in enumerate(candidates):
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        variance = float(candidate.get("posterior_variance") or metadata.get("posterior_variance") or 0.0)
        info_gain = float(candidate.get("expected_information_gain") or metadata.get("expected_information_gain") or variance)
        move_time = float(candidate.get("estimated_move_time") or metadata.get("estimated_move_time") or 0.0)
        count_time = float(candidate.get("count_time") or metadata.get("count_time") or default_count_time)
        forecast_bonus = 1.0 / (1.0 + float(candidate.get("forecast_step") or metadata.get("forecast_step") or 0.0))
        numerator = max(info_gain, 0.0) ** eta + max(variance, 0.0)
        denominator = max(count_time + motion_penalty_weight * move_time, 1.0)
        score = forecast_bonus * numerator / denominator - index * 1e-12
        if score > best_score:
            best = candidate
            best_score = score
            best_components = {
                "posterior_variance": variance,
                "expected_information_gain": info_gain,
                "estimated_move_time": move_time,
                "count_time": count_time,
                "forecast_bonus": forecast_bonus,
            }
    assert best is not None
    return {
        "point_id": str(best.get("point_id") or "tasai-candidate"),
        "nodes": best.get("nodes"),
        "score": float(best_score),
        "reason": "tasai_uncertainty_information_rate",
        "count_time": best_components["count_time"],
        "metadata": {"tasai_components": best_components},
    }
