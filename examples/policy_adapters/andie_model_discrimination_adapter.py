"""Notebook-free ANDiE policy adapter example.

This module is intentionally small and dependency-light. It demonstrates the
`propose_next_point(payload) -> dict` boundary used by
`workflow tasai-adaptive-nice-sim --policy-backend external_andie_adapter`.

The adapter does not execute notebooks and does not copy ANDiE internals into
tascopilot. Real ANDiE/MCMC/bumps policies should live in an external runtime
or TACC handoff and expose the same JSON-compatible function.
"""

from __future__ import annotations

from typing import Any


def propose_next_point(payload: dict[str, Any]) -> dict[str, Any]:
    """Select the candidate where active model hypotheses disagree most.

    Expected candidate fields:
    - `point_id`
    - `nodes` or h/k/l/e aliases
    - `model_predictions`, either a list of intensities or a mapping from
      model name to predicted intensity.

    Optional payload/model_spec fields:
    - `model_weights`, either a list matching prediction order or a mapping
      from model name to prior weight.
    - `default_count_time`

    Returns a JSON-compatible proposal dict.
    """

    candidates = payload.get("candidates") or []
    if not candidates:
        return {"point_id": "no-candidate", "score": 0.0, "reason": "missing_candidates"}
    model_spec = payload.get("model_spec") or {}
    best: dict[str, Any] | None = None
    best_score = float("-inf")
    best_components: dict[str, Any] = {}
    for index, candidate in enumerate(candidates):
        predictions = _candidate_predictions(candidate)
        if predictions:
            values, names = _prediction_values(predictions)
            weights = _prediction_weights(model_spec.get("model_weights"), names, len(values))
            mean = sum(weight * value for weight, value in zip(weights, values))
            weighted_variance = sum(weight * (value - mean) ** 2 for weight, value in zip(weights, values))
            active_span = max(values) - min(values)
            score = weighted_variance
            reason = "andie_model_discrimination_weighted_variance"
            components = {
                "weighted_mean": mean,
                "weighted_variance": weighted_variance,
                "active_prediction_span": active_span,
                "model_count": len(values),
                "model_names": names,
            }
        else:
            score = float(candidate.get("score") or 0.0) - index * 1e-9
            reason = "missing_model_predictions_fallback_score"
            components = {"fallback_score": score}
        if score > best_score:
            best = candidate
            best_score = score
            best_components = {"reason": reason, **components}
    assert best is not None
    return {
        "point_id": str(best.get("point_id") or "andie-candidate"),
        "nodes": best.get("nodes"),
        "score": float(best_score),
        "reason": best_components["reason"],
        "count_time": float(model_spec.get("default_count_time") or 60.0),
        "metadata": {"andie_components": best_components},
    }


def _candidate_predictions(candidate: dict[str, Any]) -> Any:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    return candidate.get("model_predictions") or metadata.get("model_predictions")


def _prediction_values(predictions: Any) -> tuple[list[float], list[str]]:
    if isinstance(predictions, dict):
        names = [str(key) for key in predictions]
        return [float(predictions[key]) for key in predictions], names
    values = [float(value) for value in predictions]
    return values, [f"model_{index}" for index in range(len(values))]


def _prediction_weights(raw_weights: Any, names: list[str], count: int) -> list[float]:
    if count == 0:
        return []
    if isinstance(raw_weights, dict):
        weights = [float(raw_weights.get(name, 0.0)) for name in names]
    elif raw_weights:
        weights = [float(value) for value in raw_weights][:count]
    else:
        weights = [1.0] * count
    if len(weights) < count:
        weights.extend([0.0] * (count - len(weights)))
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0.0:
        return [1.0 / count] * count
    return [max(0.0, weight) / total for weight in weights]
