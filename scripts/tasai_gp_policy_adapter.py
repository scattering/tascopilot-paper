"""Three-phase tasAI policy adapter: raster -> GP -> physics-informed.

tascopilot calls this over the JSON boundary in
``tasai_adaptive_nice_sim._external_tasai_policy_proposal``; the phase is chosen
from how many real observations have come back, and in every phase the final
pick is tasAI's own information-rate rule
(``tasai.adapters.tascopilot.propose_next_point``), which divides expected
information gain by cost. All this adapter does is decide what
``expected_information_gain`` means at the current stage of the experiment.

  phase 1  raster    n < gp_min_observations
            distance to the nearest measured point, so the first points spread
            out over the box instead of clustering.

  phase 2  gp        gp_min_observations <= n < physics_min_observations
            tasAI's LogGaussianProcess over the measured rates; gain is the
            upper confidence bound mean + kappa*std, so bright uncertain
            regions beat dark uncertain ones.

  phase 3  physics   n >= physics_min_observations
            competing ferromagnetic-magnon hypotheses are fitted to everything
            measured so far and weighted by BIC; gain is the weighted variance
            of their predicted intensities (tasAI's ANDiEAcquisition), so the
            next point is wherever the surviving models most disagree. A small
            GP term is kept so the policy still explores once they converge.

Use with::

  tas-copilot workflow tasai-adaptive-nice-sim --policy-backend external_tasai_adapter \
      --tasai-policy-adapter scripts/tasai_gp_policy_adapter.py ...

Requires the tasAI checkout on PYTHONPATH (TASAI_PATH env var or
~/tasai_nist_clean). scikit-learn gives the real GP kernel; without it tasAI
falls back to a distance-weighted estimate. scipy is needed for phase 3, which
degrades to phase 2 if it is missing.

Observations arrive as AdaptiveObservation rows with ``observed_counts``
(detector counts from the live NICE trace) and ``count_time``; they are
converted to rates so counts taken at different presets are comparable.
This file belongs in tasai/adapters/ once it has been exercised on the
instrument; it lives here so the test does not modify the tasAI tree.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_TASAI = Path(os.environ.get("TASAI_PATH") or "~/tasai_nist_clean").expanduser()
if str(_TASAI) not in sys.path and _TASAI.exists():
    sys.path.insert(0, str(_TASAI))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tasai.adapters.tascopilot import propose_next_point as _tasai_rule  # noqa: E402
from tasai.core.gaussian_process import LogGaussianProcess  # noqa: E402

try:                                             # phase 3 is optional
    from tasai.core.acquisition import ANDiEAcquisition
    from tasai_fm_magnon import FMMagnonModel, fit_hypotheses
    _PHYSICS_AVAILABLE = True
    _PHYSICS_IMPORT_ERROR = None
except Exception as exc:                         # noqa: BLE001
    _PHYSICS_AVAILABLE = False
    _PHYSICS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

NODE_ORDER = ("Q.H", "Q.K", "Q.L", "et.deltaE")


def _vector(nodes: dict[str, Any]) -> np.ndarray:
    return np.array([float(nodes.get(name, 0.0)) for name in NODE_ORDER], dtype=float)


def _observed(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Measured rows as rates, so different count presets stay comparable."""
    rows = []
    for obs in observations:
        counts = obs.get("observed_counts")
        if counts is None:
            continue
        t = float(obs.get("count_time_actual") or obs.get("count_time") or 1.0)
        if t <= 0:
            t = 1.0
        nodes = obs.get("nodes") or {}
        rows.append({
            "x": _vector(nodes),
            "rate": float(counts) / t,
            "sigma": math.sqrt(max(float(counts), 1.0)) / t,
            "source": obs.get("count_source"),
        })
    return rows


# ------------------------------------------------------------------- phase 1
def _phase_raster(candidates, xs, span, observed):
    if observed:
        # farthest from everything measured so far
        dist = np.array([np.min(np.linalg.norm((x - np.array([o["x"] for o in observed]))
                                               / span, axis=1)) for x in xs])
    else:
        # nothing measured yet: start at the centre of the box. Gain is squared
        # below, which would turn a negative distance back into "farthest", so
        # flip it into a non-negative "closeness" here instead.
        nrm = np.linalg.norm((xs - (xs.max(axis=0) + xs.min(axis=0)) / 2) / span, axis=1)
        dist = nrm.max() - nrm
    for c, d in zip(candidates, dist):
        meta = c.setdefault("metadata", {})
        meta["posterior_variance"] = float(d ** 2)
        meta["expected_information_gain"] = float(d ** 2)
        meta["surrogate"] = "space_filling"
    return candidates, {"surrogate": "space_filling"}


# ------------------------------------------------------------------- phase 2
def _fit_gp(observed, span, model_spec):
    length_scales = model_spec.get("gp_length_scales") or (span / 4).tolist()
    gp = LogGaussianProcess(length_scales=np.array(length_scales, dtype=float),
                            background=float(model_spec.get("gp_background") or 0.5),
                            n_dims=4)
    for o in observed:
        gp.add_observation(o["x"], o["rate"], o["sigma"])
    try:
        gp.fit()
    except Exception:                            # noqa: BLE001
        pass
    return gp


def _gp_predict(gp, xs):
    means, stds = [], []
    for x in xs:
        try:
            m, s = gp.predict(x)
        except Exception:                        # noqa: BLE001
            m, s = 0.0, 1.0
        means.append(float(m))
        stds.append(float(s))
    return np.array(means), np.array(stds)


def _phase_gp(candidates, xs, observed, span, model_spec):
    gp = _fit_gp(observed, span, model_spec)
    means, stds = _gp_predict(gp, xs)
    kappa = float(model_spec.get("ucb_kappa") or 2.0)
    for c, m, s in zip(candidates, means, stds):
        meta = c.setdefault("metadata", {})
        meta["posterior_variance"] = float(s ** 2)
        meta["predicted_rate"] = float(m)
        # information gain proxy: upper confidence bound on the rate, so bright
        # uncertain regions are preferred over dark uncertain ones
        meta["expected_information_gain"] = float(max(m, 0.0) + kappa * s)
        meta["surrogate"] = "tasai_log_gp"
    return candidates, {"surrogate": "tasai_log_gp"}


# ------------------------------------------------------------------- phase 3
def _phase_physics(candidates, xs, observed, span, model_spec):
    """Weight FM hypotheses by BIC, then chase where they disagree."""
    rows = [(float(o["x"][0]), float(o["x"][1]), float(o["x"][2]), float(o["x"][3]),
             o["rate"], o["sigma"]) for o in observed]
    kinds = tuple(model_spec.get("fm_hypotheses") or FMMagnonModel.KINDS)
    models, weights, diagnostics = fit_hypotheses(
        rows,
        lattice_a=float(model_spec.get("lattice_a") or 3.9005),
        T=float(model_spec.get("temperature_k") or 300.0),
        S=float(model_spec.get("spin") or 1.85),
        kinds=kinds,
    )

    acq = ANDiEAcquisition(models, prior_weights=list(weights))
    points = [SimpleNamespace(h=float(x[0]), k=float(x[1]), l=float(x[2]), E=float(x[3]))
              for x in xs]
    results = acq.evaluate(points)
    disc = np.array([float(r.score) for r in results])

    # Normalise so the blend below is scale-free, then keep a small GP term:
    # once the hypotheses converge their predictions stop disagreeing and the
    # discrimination score goes flat, and the policy should still explore.
    dmax = float(disc.max())
    disc_n = disc / dmax if dmax > 0 else np.zeros_like(disc)
    gp = _fit_gp(observed, span, model_spec)
    means, stds = _gp_predict(gp, xs)
    smax = float(stds.max())
    stds_n = stds / smax if smax > 0 else np.zeros_like(stds)
    w_explore = float(model_spec.get("physics_explore_weight") or 0.15)

    gains = (1.0 - w_explore) * disc_n + w_explore * stds_n
    for c, g, s, m, d in zip(candidates, gains, stds, means, disc):
        meta = c.setdefault("metadata", {})
        meta["expected_information_gain"] = float(g)
        meta["posterior_variance"] = float(s ** 2)
        meta["predicted_rate"] = float(m)
        meta["model_discrimination"] = float(d)
        meta["surrogate"] = "tasai_fm_andie"

    best = int(np.argmax(weights))
    info = {
        "surrogate": "tasai_fm_andie",
        "hypotheses": diagnostics,
        "leading_hypothesis": diagnostics[best]["kind"],
        "leading_weight": float(weights[best]),
        "leading_description": diagnostics[best]["description"],
        "explore_weight": w_explore,
        "discriminating": bool(dmax > 0),
    }
    return candidates, info


# ---------------------------------------------------------------------- entry
def propose_next_point(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = list(payload.get("candidates") or [])
    if not candidates:
        return _tasai_rule(payload)
    model_spec = dict(payload.get("model_spec") or {})
    observed = _observed(list(payload.get("observations") or []))
    xs = np.array([_vector(c.get("nodes") or {}) for c in candidates])
    span = np.maximum(xs.max(axis=0) - xs.min(axis=0), 1e-6)

    gp_min = int(model_spec.get("gp_min_observations") or 2)
    physics_min = int(model_spec.get("physics_min_observations") or 0)
    n = len(observed)

    phase_note = None
    if n < gp_min:
        phase, (candidates, info) = "raster", _phase_raster(candidates, xs, span, observed)
    elif not physics_min or n < physics_min:
        phase, (candidates, info) = "gp", _phase_gp(candidates, xs, observed, span, model_spec)
    elif not _PHYSICS_AVAILABLE:
        phase, (candidates, info) = "gp", _phase_gp(candidates, xs, observed, span, model_spec)
        phase_note = f"physics phase unavailable ({_PHYSICS_IMPORT_ERROR}); stayed on the GP"
    else:
        try:
            candidates, info = _phase_physics(candidates, xs, observed, span, model_spec)
            phase = "physics"
        except Exception as exc:                 # noqa: BLE001
            candidates, info = _phase_gp(candidates, xs, observed, span, model_spec)
            phase = "gp"
            phase_note = f"physics phase failed ({type(exc).__name__}: {exc}); fell back to the GP"

    proposal = _tasai_rule({**payload, "candidates": candidates})
    meta = proposal.setdefault("metadata", {})
    meta.update(info)
    meta["phase"] = phase
    meta["observations_used"] = n
    meta["phase_thresholds"] = {"gp_min_observations": gp_min,
                                "physics_min_observations": physics_min}
    if phase_note:
        meta["phase_note"] = phase_note
    return proposal


__all__ = ["propose_next_point"]
