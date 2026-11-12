#!/usr/bin/env python3
"""Export per-point data/model tables at the final SrMn2Sb2 LM endpoint.

Plotting/export-only adapter: loads the ordinary Bumps FitProblem exactly as
render_srmn2sb2_bumps_model_plot.py does, applies the complete .par, and writes
(1) every scan's measured points with the exact QMC64 prediction,
(2) the QMC resolution sample cloud behind every measured point,
(3) linear spin-wave modes along two reciprocal-space paths at the endpoint,
(4) a provenance json.  Nothing is optimized or altered.
"""
from __future__ import annotations
import argparse, hashlib, json, csv
from pathlib import Path
import numpy as np

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("model", "data", "backgrounds", "pars", "core"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", default="center005")
    ap.add_argument("--normalization", default="per_scan")
    ap.add_argument("--qmc", type=int, default=64)
    a = ap.parse_args()
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("srcore", a.core)
    core = importlib.util.module_from_spec(spec); sys.modules["srcore"] = core; spec.loader.exec_module(core)
    from bumps.cli import load_best, load_model
    opts = [str(a.data.resolve()), str(a.backgrounds.resolve()), a.seed, a.normalization,
            "single_y", "legacy", str(a.qmc), "full", "official_processpool"]
    problem = load_model(str(a.model.resolve()), model_options=opts)
    load_best(problem, a.pars.resolve())
    fit = problem.fitness
    obj = fit.objective
    physics = (float(fit.sj1.value), float(fit.j2_ratio.value), float(fit.sjc.value), float(fit.d_easy.value))
    scales = {k: float(v.value) for k, v in fit.scale_parameters.items()}
    nllf = float(problem.nllf()); chisq = float(problem.chisq())
    preds = obj.predict_groups(*physics, scales)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "scan_predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "zone", "zone_key", "coord_label", "coord", "E_meV", "H", "K", "L",
                    "y", "sigma", "prediction", "signal", "background", "scale", "sigma_E_conditional"])
        for g, p in zip(obj.prepared.groups, preds):
            key = g.filename if obj.normalization_policy == "per_scan" else g.zone_key
            hkl_pts = np.zeros((len(g.y), 3))
            sig_e = np.zeros(len(g.y))
            cloud = obj.prepared.all_sample_hkl[g.sample_slice]
            for i in range(len(g.y)):
                m = g.sample_to_point == i
                hkl_pts[i] = cloud[m].mean(axis=0); sig_e[i] = g.sample_sigma_e[m][0]
            for i in range(len(g.y)):
                w.writerow([g.filename, g.zone, g.zone_key, g.scan_coordinate_label, g.scan_coordinate[i],
                            g.energy_transfer_mev[i], *hkl_pts[i], g.y[i], g.sigma[i], p.prediction[i],
                            p.signal[i], p.fixed_background, scales[key], sig_e[i]])
    with (a.out / "resolution_samples.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["filename", "point", "H", "K", "L", "E_meV"])
        for g in obj.prepared.groups:
            cloud = obj.prepared.all_sample_hkl[g.sample_slice]
            for j in range(len(g.sample_e)):
                w.writerow([g.filename, int(g.sample_to_point[j]), *cloud[j], g.sample_e[j]])
    model = core.build_official_srmn2sb2_model(*physics, 90.0)
    paths = {"HH2": np.column_stack([np.linspace(-0.5, 1.5, 401)]*2 + [np.full(401, 2.0)]),
             "HH3": np.column_stack([np.linspace(-0.5, 1.5, 401)]*2 + [np.full(401, 3.0)]),
             "00L": np.column_stack([np.zeros(401), np.zeros(401), np.linspace(1.0, 3.0, 401)])}
    with (a.out / "dispersion_paths.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "H", "K", "L", "mode", "E_meV", "Sperp"])
        for name, q in paths.items():
            e, s = core.python_sperp(model, q, scheduling_policy="official_processpool")
            e, s = core.physical_positive_modes(e, s)
            for i in range(len(q)):
                for m in range(e.shape[1]):
                    if e[i, m] > 0:
                        w.writerow([name, *q[i], m, e[i, m], s[i, m]])
    meta = {"export_only": True, "nllf": nllf, "reduced_chi_squared": chisq, "npoints": obj.prepared.npoints,
            "physics": dict(zip(["SJ1_meV", "J2_over_J1", "SJc_meV", "D_easy_plane_meV"], physics)),
            "scales": scales, "model_options": opts,
            "sha256": {k: sha(getattr(a, k)) for k in ("model", "data", "backgrounds", "pars", "core")},
            "resolution": {k: v for k, v in obj.provenance().items() if k != "backend"}}
    (a.out / "export_provenance.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({"nllf": nllf, "chisq": chisq, "npoints": obj.prepared.npoints}))

if __name__ == "__main__":
    main()
