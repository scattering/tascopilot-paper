"""Plot the saved SrMn2Sb2 fit and resolution; no data reduction or refit."""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import numpy as np
import pandas as pd

from figure_inputs import plot_args


MODEL_COLOR = "#0072b2"
PROFILES = (
    "SMS_HH_Q112_e8_ef28_841931",
    "SMS_HH_Q003_e10_841019",
    "SMS_HH_Q002_e2ef14o7m_841224",
    "L_Q002_120K_ef28_e5_841959",
    "L_Q002_120K_ef28_e9_841964",
    "L_Q002_120K_ef28_e10_841965",
)


def scan_label(filename):
    ef = "28" if "ef28" in filename else "14.7"
    energy = re.search(r"_e(\d+)", filename).group(1)
    if filename.startswith("L_Q002"):
        path = r"$(0\,0\,L)$"
    elif "Q003" in filename:
        path = r"$(H\,H\,3)$"
    else:
        path = r"$(H\,H\,2)$"
    return f"{path}\n$\\Delta E = {energy}$ meV, $E_f = {ef}$ meV"


def resolution_projection(samples, sigma_energy):
    """Recover the projected covariance from the exported conditional samples."""
    covariance = np.cov(samples, rowvar=False)
    # Exported energies are E[E | Q]; add E[Var(E | Q)] for the full width.
    covariance[1, 1] += sigma_energy ** 2
    return covariance


def draw_exchange_model(ax):
    ax.axis("off")
    ax.set_title("(a) Exchange model", loc="left")
    plan = ax.inset_axes([0.0, 0.27, 0.65, 0.68])
    plan.set_aspect("equal")
    plan.axis("off")
    a1, a2 = np.array([1.0, 0.0]), np.array([-0.5, np.sqrt(3) / 2])
    mn1 = {(i, j): (i + 1/3) * a1 + (j + 2/3) * a2
           for i in range(-1, 1) for j in range(-1, 2)}
    mn2 = {(i, j): (i + 2/3) * a1 + (j + 1/3) * a2
           for i in range(-1, 1) for j in range(-1, 2)}
    for (i, j), p in mn1.items():
        for di, dj in ((0, 0), (0, 1), (-1, 0)):
            q = mn2.get((i + di, j + dj))
            if q is not None:
                plan.plot([p[0], q[0]], [p[1], q[1]], color=MODEL_COLOR, lw=1.5)
    p = mn1[(-1, 0)]
    for di, dj in ((0, 1), (1, 0), (1, 1)):
        q = mn1.get((-1 + di, dj))
        if q is not None:
            plan.plot([p[0], q[0]], [p[1], q[1]], color="#009e73", lw=1.3, ls="--")
    for sites, fill in ((mn1, "black"), (mn2, "white")):
        for p in sites.values():
            plan.plot(*p, "o", color="black", mfc=fill, ms=5, zorder=3)
    for p, sign in ((mn1[(-1, 0)], 1), (mn2[(-1, 0)], -1)):
        plan.annotate("", xy=p + sign * np.array([0.22, 0]),
                      xytext=p - sign * np.array([0.22, 0]),
                      arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2), zorder=4)

    side = ax.inset_axes([0.72, 0.27, 0.28, 0.68])
    side.axis("off")
    for shift in (0, 1):
        for x in (0, 1):
            side.plot([x, 0.5], [0.62 + shift, 0.38 + shift], color=MODEL_COLOR, lw=1.5)
        side.plot([0, 1], [0.62 + shift] * 2, "o", color="black", ms=5)
        side.plot(0.5, 0.38 + shift, "o", color="black", mfc="white", ms=5)
    for x in (0, 1):
        side.plot([x, 0.5], [0.62, 1.38], color="#cc79a7", lw=1.5, zorder=1)
    side.annotate("$c$", xy=(1.28, 1.9), xytext=(1.28, 0.4), ha="center",
                  arrowprops=dict(arrowstyle="->", color="black"))
    side.set_xlim(-0.2, 1.5)
    side.set_ylim(0.15, 2.05)
    ax.legend([Line2D([], [], color=MODEL_COLOR, lw=1.5),
               Line2D([], [], color="#009e73", ls="--", lw=1.3),
               Line2D([], [], color="#cc79a7", lw=1.5)],
              ["$J_1$", "$J_2$", "$J_c$"], ncol=3, loc="lower center",
              frameon=False, handlelength=1.6, columnspacing=1.4)


def draw_dispersion(ax, pred, samples, modes, path, xkey, filename, target, xlim):
    for _, group in modes[modes["path"] == path].groupby("mode"):
        group = group.sort_values(xkey)
        ax.plot(group[xkey], group["E_meV"], color="black", lw=1.1)
    points = pred[pred["filename"] == filename].reset_index(drop=True)
    point_index = int((points["coord"] - target).abs().idxmin())
    point = points.iloc[point_index]
    cloud = samples[(samples["filename"] == filename) & (samples["point"] == point_index)]
    covariance = resolution_projection(cloud[[xkey, "E_meV"]].to_numpy(),
                                       point["sigma_E_conditional"])
    values, vectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    ax.add_patch(Ellipse((point["coord"], point["E_meV"]),
                         4 * np.sqrt(values[1]), 4 * np.sqrt(values[0]), angle=angle,
                         facecolor=MODEL_COLOR, edgecolor=MODEL_COLOR, alpha=0.25, lw=1.2))
    ax.plot([points["coord"].min(), points["coord"].max()], [point["E_meV"]] * 2,
            color="0.45", ls="--", lw=1.0)
    ax.plot(point["coord"], point["E_meV"], "+", color=MODEL_COLOR, ms=7, mew=1.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 20)
    ax.set_xlabel(f"${xkey}$ (r.l.u.)")
    ax.set_ylabel(r"$\Delta E$ (meV)")
    path_label = r"$(H\,H\,2)$" if path == "HH2" else r"$(0\,0\,L)$"
    ef = "14.7" if path == "HH2" else "28"
    ax.set_title(f"({'b' if path == 'HH2' else 'c'}) {path_label}, $E_f = {ef}$ meV", loc="left")


def build_figure(pred, samples, modes):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.titlecolor": "black",
                         "axes.labelsize": 10, "legend.fontsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig = plt.figure(figsize=(10.6, 8.8))
    grid = fig.add_gridspec(3, 3, hspace=0.70, wspace=0.38,
                           left=0.075, right=0.985, top=0.96, bottom=0.10)
    draw_exchange_model(fig.add_subplot(grid[0, 0]))
    for col, config in enumerate((
        ("HH2", "H", "SMS_HH_Q002_e10ef14o7m_841228", 0.0875, (-0.35, 0.35)),
        ("00L", "L", "L_Q002_120K_ef28_e10_841965", 2.125, (1.4, 2.6)),
    ), start=1):
        draw_dispersion(fig.add_subplot(grid[0, col]), pred, samples, modes, *config)
    fig.legend([Line2D([], [], color="black", lw=1.1),
                Line2D([], [], color="0.45", ls="--", lw=1.0),
                Patch(facecolor=MODEL_COLOR, alpha=0.25)],
               ["Spin-wave modes", "10 meV scan path", r"Resolution ($2\sigma$)"],
               ncol=3, frameon=False, loc="center", bbox_to_anchor=(0.675, 0.696),
               columnspacing=1.1, handlelength=1.6)

    for index, filename in enumerate(PROFILES):
        ax = fig.add_subplot(grid[1 + index // 3, index % 3])
        group = pred[pred["filename"] == filename].sort_values("coord")
        ax.errorbar(group["coord"], group["y"] * 1e3, yerr=group["sigma"] * 1e3,
                    fmt="o", ms=3.1, color="black", ecolor="0.55", lw=0.8, zorder=3)
        ax.plot(group["coord"], group["prediction"] * 1e3, color=MODEL_COLOR, lw=1.5)
        ax.axhline(group["background"].iloc[0] * 1e3, color="0.45", ls=":", lw=1)
        ax.set_xlabel("$L$ (r.l.u.)" if filename.startswith("L_") else "$H = K$ (r.l.u.)")
        if index % 3 == 0:
            ax.set_ylabel(r"Intensity (counts / $10^3$ monitor)")
        ax.set_title(f"({'defghi'[index]}) {scan_label(filename)}", loc="left")
        ax.margins(y=0.12)
    fig.legend([Line2D([], [], color="black", marker="o", ls="", ms=4),
                Line2D([], [], color=MODEL_COLOR, lw=1.5),
                Line2D([], [], color="0.45", ls=":", lw=1)],
               [r"Data ($1\sigma$ errors)", "Resolution-convolved model + background", "Fixed background"],
               ncol=3, frameon=False, loc="center", bbox_to_anchor=(0.53, 0.025))
    return fig


def main():
    args = plot_args("srmn2sb2_resolution_fit.pdf")
    endpoint = args.data_dir / "srmn2sb2_fit_endpoint"
    fig = build_figure(pd.read_csv(endpoint / "scan_predictions.csv"),
                       pd.read_csv(endpoint / "resolution_samples.csv.gz"),
                       pd.read_csv(endpoint / "dispersion_paths.csv"))
    fig.savefig(args.out)
    fig.savefig(args.out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
