"""pySpinW count adapter protocol example.

Real pySpinW/SpinW work should run through an external adapter, container, or
TACC job. This file only demonstrates the JSON-compatible
`simulate_counts(payload) -> dict` boundary used by
`workflow tasai-adaptive-nice-sim --count-backend external_pyspinw_adapter`.
"""

from __future__ import annotations

from typing import Any


def simulate_counts(payload: dict[str, Any]) -> dict[str, Any]:
    point = payload.get("point") or {}
    nodes = point.get("nodes") or {}
    model_spec = payload.get("model_spec") or {}
    count_time = float(payload.get("count_time") or 1.0)
    background = float(model_spec.get("background") or 20.0)
    scale = float(model_spec.get("count_scale") or 100.0)
    h = float(nodes.get("Q.H", 0.0))
    k = float(nodes.get("Q.K", 0.0))
    l = float(nodes.get("Q.L", 0.0))
    energy = float(nodes.get("et.deltaE", 0.0))
    toy_intensity = max(0.0, 1.0 - (h * h + k * k + l * l) - 0.01 * abs(energy))
    rate = background + scale * toy_intensity
    return {
        "expected_rate_hz": rate,
        "expected_counts": rate * count_time,
        "metadata": {
            "adapter_kind": "protocol_example",
            "not_real_pyspinw": True,
        },
    }
