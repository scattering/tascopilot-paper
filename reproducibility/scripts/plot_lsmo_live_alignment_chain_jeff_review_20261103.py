"""Figure: the live (0 1 0) alignment chain on BT-7 exp 202608/14, 2026-09-12.
Panels: rapidScan hunt (a3 parked 40 deg off), findPeak zoom, theta-2theta, a3, lower tilt (wide), a3 again.
Data: saved rates and one-sigma uncertainties in S2, with public-file lineage."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from figure_inputs import plot_args, read_json

args = plot_args("lsmo_live_alignment_chain_jeff_review_20261103.pdf")
OUT = args.out
snapshot = read_json(args.data_dir / "figure_inputs/lsmo_alignment.json")
A4_PRED = 35.2023; A3_PRED = 107.6017  # NICE prediction for (0 1 0) at Ef = 14.7 meV, a = 3.9005 A
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9.5,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "pdf.fonttype": 42})

def gauss(x, a, c, w, b): return a * np.exp(-0.5 * ((x - c) / w) ** 2) + b
def fit(x, y):
    p0 = [y.max() - y.min(), x[np.argmax(y)], (x.max() - x.min()) / 6, y.min()]
    try:
        p, _ = curve_fit(gauss, x, y, p0=p0, sigma=np.sqrt(np.maximum(y, 1)), maxfev=20000)
        return p[1], 2.3548 * abs(p[2]), p
    except Exception:
        return float("nan"), float("nan"), None

panels = [
    ("rapid_sampleTheta.softPosition863.dat.bt7", "A3", "(a) rapidScan hunt: a3 parked 40° off\n80° sweep in 0.25 s bins"),
    ("align010_fp_A3_868.dat.bt7", "A3", "(b) findPeak zoom on the\nrapid-scan estimate"),
    ("align010_theta_two_theta_A4_877.dat.bt7", "A4", "(c) theta–two-theta\n(a4 scanned, a3 coupled)"),
    ("align010_a3_A3_878.dat.bt7", "A3", "(d) a3 rocking, 4.0° window"),
    ("align010_lower_tilt_wide_sampleLTilt_879.dat.bt7", "sampleLTilt", "(e) lower tilt, 14° window"),
    ("align010_a3_again_A3_880.dat.bt7", "A3", "(f) a3 rocking again\nafter the tilt"),
]
# Two columns keep text near its nominal size when printed at manuscript width.
fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.4), layout="constrained")
fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.06, wspace=0.08, hspace=0.08)
results = []
for ax, (name, axis, title) in zip(axes.flat, panels):
    r = next(row for row in snapshot["scans"] if row["filename"] == name)
    x, rate, rate_err = (np.asarray(r[k], float) for k in ("x", "rate", "rate_error"))
    ax.errorbar(x, rate, yerr=rate_err, fmt="o-", ms=3, lw=1, color="#2c3e50", ecolor="#7f8c8d", elinewidth=0.8, capsize=0)
    c, w, p = fit(x, rate)
    if p is not None and axis != "sampleLTilt" or (p is not None and axis == "sampleLTilt"):
        xx = np.linspace(x.min(), x.max(), 300); ax.plot(xx, gauss(xx, *p), "-", color="#c0392b", lw=1.2, alpha=.8)
    if name.startswith("rapid"):
        ax.set_yscale("log"); ax.axvline(A3_PRED, color="grey", ls="--", lw=1); ax.text(A3_PRED, ax.get_ylim()[1]*0.5, " predicted", color="grey", fontsize=8.5)
        ax.axvline(147.60, color="k", ls=":", lw=1); ax.text(147.6, ax.get_ylim()[1]*0.5, "parked ", color="k", fontsize=8.5, ha="right")
    else:
        pred = A3_PRED if axis == "A3" else (A4_PRED if axis == "A4" else None)
        if pred is not None: ax.axvline(pred, color="grey", ls="--", lw=1)
        ax.set_title(title + f"\ncenter {c:.3f}°, FWHM {w:.2f}°", fontsize=9.5, loc="left")
    if name.startswith("rapid"): ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlabel({"A3": "a3 (deg)", "A4": "a4 (deg)", "sampleLTilt": "lower tilt (deg)"}[axis]); ax.set_ylabel("counts / s"); ax.grid(alpha=.3)
    results.append((name, axis, c, w, r["max_counts"], len(x), r["median_time"]))
fig.suptitle("Live fine alignment of LSMO (0 1 0) on BT-7, 2026-09-12\n"
             "NICE findPeak as the primitive; TASCopilot as the orchestrator", fontsize=10.5)
fig.savefig(OUT); fig.savefig(OUT.with_suffix(".png"), dpi=200)
for row in results: print(row)
print(OUT)
