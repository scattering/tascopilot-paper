#!/usr/bin/env python
"""Competing ferromagnetic-magnon hypotheses for the physics-informed phase.

LSMO at 300 K is well below Tc (~370 K) and is a simple collinear ferromagnet,
so the excitation is a single damped magnon branch. That makes an analytic FM
form the right model here rather than a Sunny/SpinW calculation: there is no
multi-sublattice structure for a linear-spin-wave engine to resolve, and the
interesting question is not "which Hamiltonian" but "how does the single branch
soften away from the zone centre".

That question is what the summer-school data actually poses. Fitting E = D q^2
window-by-window to the 2017 300 K scans gives

    dq (rlu)   0.10   0.15   0.20   0.25   0.30
    D (meVA^2)  127    110     98     85     74

which is near-linear in dq, not the constant a quadratic dispersion requires.
So the hypotheses below differ precisely where that matters -- at large q --
and agree near the zone centre, which is exactly the structure a
model-discrimination acquisition can exploit.

Hypotheses
----------
``quadratic``    E_q = D q^2
                 ideal Heisenberg FM in the small-q limit; the textbook answer,
                 and the one the literature D = 135 meV A^2 refers to.
``quartic``      E_q = D q^2 (1 - beta q^2)
                 leading zone-boundary correction; the usual phenomenological
                 way to describe a softening branch.
``heisenberg_nn``E_q = 2 z J S [1 - gamma(q)],  gamma = (1/3) sum cos(2 pi zeta_i)
                 nearest-neighbour Heisenberg on the pseudo-cubic lattice, which
                 softens on its own with no extra free parameter.

Intensity uses a damped harmonic oscillator times the Bose factor -- the
standard lineshape for a magnon at a temperature this close to Tc, and one that
behaves correctly as E -> 0 (detailed balance) instead of diverging::

    I(Q,E) = A f^2(Q) [n(E)+1] * 4 Gamma E E_q / [(E^2 - E_q^2)^2 + 4 Gamma^2 E^2] + B

Each model owns an amplitude, a width and a flat background, so a hypothesis is
never rejected merely for having the wrong overall scale.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

_TASAI = Path(os.environ.get("TASAI_PATH") or "~/tasai_nist_clean").expanduser()
if str(_TASAI) not in sys.path and _TASAI.exists():
    sys.path.insert(0, str(_TASAI))

from tasai.physics.base import Parameter, PhysicsModel  # noqa: E402

K_B = 0.0861733           # meV / K

# <j0> analytic approximation, Mn3+ (International Tables for Crystallography C).
_MN3_J0 = (0.4198, 14.2829, 0.6054, 5.4689, 0.9241, -0.0088, -0.9498)


def mn_form_factor(Q_invA: float) -> float:
    """<j0>(Q) for Mn3+; Q is |Q| in 1/A."""
    A, a, B, b, C, c, D = _MN3_J0
    s2 = (Q_invA / (4.0 * math.pi)) ** 2
    return A * math.exp(-a * s2) + B * math.exp(-b * s2) + C * math.exp(-c * s2) + D


def bose_plus_one(E: float, T: float) -> float:
    """n(E) + 1, finite and correct on both sides of E = 0."""
    if T <= 0:
        return 1.0 if E > 0 else 0.0
    x = E / (K_B * T)
    if abs(x) < 1e-8:
        return K_B * T / max(abs(E), 1e-8) if E else 1.0
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 - math.exp(-x))


class FMMagnonModel(PhysicsModel):
    """One ferromagnetic-magnon hypothesis: dispersion kind + DHO lineshape."""

    KINDS = ("quadratic", "quartic", "heisenberg_nn")

    def __init__(self, kind: str = "quadratic", *, lattice_a: float = 3.9005,
                 T: float = 300.0, z: int = 6, S: float = 1.85):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"unknown dispersion kind {kind!r}")
        self.kind = kind
        self.a = float(lattice_a)
        self.T = float(T)
        self.z = int(z)
        self.S = float(S)

        p = [
            Parameter("amplitude", 1.0, (1e-6, 1e9), units="counts/s",
                      description="overall scale, absorbs detector efficiency"),
            Parameter("gamma", 2.0, (0.05, 40.0), units="meV",
                      description="DHO damping (half width)"),
            Parameter("background", 0.5, (0.0, 1e6), units="counts/s"),
        ]
        if kind in ("quadratic", "quartic"):
            p.append(Parameter("D", 120.0, (5.0, 600.0), units="meV A^2",
                               description="spin-wave stiffness"))
        if kind == "quartic":
            p.append(Parameter("beta", 0.05, (0.0, 2.0), units="A^2",
                               description="zone-boundary softening"))
        if kind == "heisenberg_nn":
            p.append(Parameter("JS", 5.0, (0.1, 100.0), units="meV",
                               description="J*S for the nearest-neighbour bond"))
        self._parameters = p
        self._param_index = {q.name: i for i, q in enumerate(p)}

    # -------------------------------------------------------------- dispersion
    def _reduced(self, h, k, l):
        """Reduced wavevector from the nearest zone centre: (zeta vector, |q| 1/A)."""
        zeta = np.array([v - round(v) for v in (h, k, l)], dtype=float)
        q = 2.0 * math.pi * float(np.linalg.norm(zeta)) / self.a
        return zeta, q

    def dispersion(self, h, k, l) -> float:
        zeta, q = self._reduced(h, k, l)
        v = self.get_values()
        if self.kind == "quadratic":
            return v["D"] * q * q
        if self.kind == "quartic":
            return v["D"] * q * q * max(1.0 - v["beta"] * q * q, 0.0)
        gamma_q = float(np.mean(np.cos(2.0 * math.pi * zeta)))
        return 2.0 * self.z * v["JS"] * (1.0 - gamma_q)

    # --------------------------------------------------------------- intensity
    def compute_intensity(self, h: float, k: float, l: float, E: float, **kw) -> float:
        v = self.get_values()
        Eq = self.dispersion(h, k, l)
        G = v["gamma"]
        if Eq <= 0.0:
            return v["background"]
        denom = (E * E - Eq * Eq) ** 2 + 4.0 * G * G * E * E
        if denom <= 0.0:
            return v["background"]
        dho = 4.0 * G * E * Eq / denom
        Q = (2.0 * math.pi / self.a) * math.sqrt(h * h + k * k + l * l)
        f2 = mn_form_factor(Q) ** 2
        return v["amplitude"] * f2 * bose_plus_one(E, self.T) * dho + v["background"]

    def description(self) -> str:
        v = self.get_values()
        if self.kind == "quadratic":
            core = f"E = {v['D']:.1f} q^2"
        elif self.kind == "quartic":
            core = f"E = {v['D']:.1f} q^2 (1 - {v['beta']:.3f} q^2)"
        else:
            core = f"E = 2*{self.z}*{v['JS']:.3f}*[1 - gamma(q)]  (D_eff = {self.stiffness():.1f})"
        return f"FM magnon [{self.kind}]: {core}, Gamma = {v['gamma']:.2f} meV"

    def stiffness(self) -> float:
        """Small-q stiffness D in meV A^2, comparable across all three kinds."""
        v = self.get_values()
        if self.kind in ("quadratic", "quartic"):
            return v["D"]
        return 2.0 * v["JS"] * self.a * self.a


# ------------------------------------------------------------------ fitting
def _residuals(theta, model, names, pts, rates, sigmas):
    """scipy passes the parameter vector first; keep that order."""
    for name, val in zip(names, theta):
        model.set_parameter(name, float(val))
    pred = np.array([model.compute_intensity(*p) for p in pts])
    return (pred - rates) / sigmas


def fit_model(model, pts, rates, sigmas):
    """Least-squares fit in place. Returns (chi2, n_free, error_or_None)."""
    from scipy.optimize import least_squares

    free = model.free_parameters
    names = [p.name for p in free]
    x0 = np.array([p.value for p in free], dtype=float)
    lo = np.array([p.bounds[0] for p in free], dtype=float)
    hi = np.array([p.bounds[1] for p in free], dtype=float)
    # a sane starting amplitude beats a generic 1.0 by a long way
    if "amplitude" in names and len(rates):
        model.set_parameter("background", float(np.percentile(rates, 10)))
        model.set_parameter("amplitude", 1.0)
        pred = np.array([model.compute_intensity(*p) for p in pts])
        scale = float(np.max(rates)) / max(float(np.max(pred)), 1e-9)
        model.set_parameter("amplitude", min(max(scale, 1e-6), 1e9))
        x0 = np.array([p.value for p in model.free_parameters], dtype=float)
    try:
        res = least_squares(_residuals, np.clip(x0, lo, hi),
                            bounds=(lo, hi), max_nfev=2000,
                            args=(model, names, pts, rates, sigmas))
        chi2 = float(2.0 * res.cost)
        err = None
    except Exception as exc:              # noqa: BLE001
        # Report it: a silently reset fit looks like a converged one whose
        # parameters happen to equal the defaults, which is very hard to spot.
        for name, val in zip(names, x0):
            model.set_parameter(name, float(val))
        chi2 = float(np.sum(_residuals(x0, model, names, pts, rates, sigmas) ** 2))
        err = f"{type(exc).__name__}: {exc}"
    return chi2, len(free), err


def fit_hypotheses(observations, *, lattice_a=3.9005, T=300.0, S=1.85,
                   kinds=FMMagnonModel.KINDS, prior=None):
    """Fit every hypothesis to the observations and weight them by BIC.

    ``observations`` are (h, k, l, E, rate, sigma) tuples. Returns
    ``(models, weights, diagnostics)``; weights sum to 1.
    """
    pts = [(o[0], o[1], o[2], o[3]) for o in observations]
    rates = np.array([o[4] for o in observations], dtype=float)
    sigmas = np.array([max(o[5], 1e-6) for o in observations], dtype=float)
    n = len(rates)

    models, chi2s, ks, errs = [], [], [], []
    for kind in kinds:
        m = FMMagnonModel(kind, lattice_a=lattice_a, T=T, S=S)
        chi2, nfree, err = fit_model(m, pts, rates, sigmas)
        models.append(m)
        chi2s.append(chi2)
        ks.append(nfree)
        errs.append(err)

    # BIC = chi2 + k ln n; posterior weight ~ prior * exp(-BIC/2)
    bics = np.array([c + k * math.log(max(n, 2)) for c, k in zip(chi2s, ks)])
    rel = -(bics - bics.min()) / 2.0
    w = np.exp(np.clip(rel, -700, 0))
    if prior is not None:
        w = w * np.array(prior, dtype=float)
    total = float(w.sum())
    weights = (w / total) if total > 0 else np.ones(len(models)) / len(models)

    diagnostics = [
        {"kind": m.kind, "chi2": float(c), "chi2_per_dof": float(c / max(n - k, 1)),
         "n_free": int(k), "bic": float(b), "weight": float(wi),
         "stiffness_meVA2": float(m.stiffness()),
         "params": {p.name: float(p.value) for p in m.parameters},
         "fit_error": e,
         "description": m.description()}
        for m, c, k, b, wi, e in zip(models, chi2s, ks, bics, weights, errs)
    ]
    return models, weights, diagnostics


__all__ = ["FMMagnonModel", "fit_hypotheses", "fit_model",
           "mn_form_factor", "bose_plus_one"]
