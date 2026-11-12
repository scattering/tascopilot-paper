#!/usr/bin/env python
"""Replot the LSMO reference comparison from the saved S2 plotting snapshot.

Layout sized for a full-width manuscript figure.
Reduction, uncertainty propagation and peak extraction occur at export, not here.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm

from figure_inputs import plot_args, read_json

# One sequential colormap carries intensity in every panel, and nothing else
# uses colour to encode a second quantity: acquisition order is carried by the
# printed step number, and phase by marker shape.  Overlaying a second colour
# axis on a viridis map is what made the first version unreadable.
CMAP = "viridis"
ACCENT = "#e8112d"      # viridis contains no red, so a red line always reads

# (key, label, marker) -- shape, not colour, distinguishes the three phases
PHASES = [("space_filling", "raster 0-9", "o"),
          ("tasai_log_gp", "log-GP 10-24", "s"),
          ("tasai_fm_andie", "physics 25+", "D")]

CUT_COLOURS = ["#e8112d", "#0072b2", "#009e73", "#cc79a7"]


def draw_legacy_map(ax, scans, norm, half_width=0.025):
    mesh = None
    for s in scans:
        E, I = s["E"], s["I"]
        o = np.argsort(E)
        E, I = E[o], I[o]
        edges = np.concatenate(([E[0] - (E[1] - E[0]) / 2],
                                (E[:-1] + E[1:]) / 2,
                                [E[-1] + (E[-1] - E[-2]) / 2]))
        mesh = ax.pcolormesh(np.array([s["dq"] - half_width, s["dq"] + half_width]),
                             edges, I.reshape(-1, 1), norm=norm, cmap=CMAP,
                             shading="flat", zorder=1)
    return mesh


def main() -> int:
    args = plot_args("lsmo_tasai_sampling_vs_2017_map_jeff_review_20261103.pdf")
    args.marker_alpha, args.marker_size, args.no_title = 0.45, 80.0, True
    saved = read_json(args.data_dir / "figure_inputs/lsmo_comparison.json")
    scans = saved["legacy"]
    for scan in scans:
        for key in ("E", "I", "sigma"):
            scan[key] = np.asarray(scan[key], float)
    live = saved["live"]
    rlu = 2.0 * math.pi / saved["lattice_a"]
    D_leg, beta_leg = saved["D_legacy"], saved["beta_legacy"]

    allI = np.concatenate([s["I"] for s in scans] + [np.array([p["I"] for p in live])])
    norm = LogNorm(vmin=max(np.nanpercentile(allI, 2), 1.0), vmax=np.nanmax(allI))

    # Author at the intended print width, rather than shrinking a poster-wide
    # asset into the manuscript. Keep the maps tall enough to separate the
    # acquisition numbers; place the shared key outside their measured area.
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9.5, "axes.titlesize": 10,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8, "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.0, 7.0))
    maps = fig.add_gridspec(1, 2, left=0.10, right=0.88,
                           bottom=0.50, top=0.94, wspace=0.27)
    ax1, ax2 = (fig.add_subplot(maps[0, i]) for i in range(2))
    ax3 = fig.add_axes((0.10, 0.08, 0.78, 0.255))

    dgrid = np.linspace(0.0, 0.34, 200)
    qg = dgrid * rlu
    disp = D_leg * qg ** 2 * (1 - beta_leg * qg ** 2)

    lx = np.array([p["dq"] for p in live])
    ly = np.array([p["E"] for p in live])
    ls = np.array([p["step"] for p in live])
    li = np.array([p["I"] for p in live])

    # ------------------------------- (a) the path, in acquisition order
    # Markers are plain white here: the colour axis belongs to the 2017 map,
    # and the step number already says everything about order.
    mesh = draw_legacy_map(ax1, scans, norm)
    ax1.plot(dgrid, disp, "-", lw=1.6, color=ACCENT, zorder=3,
             label=f"2017 dispersion  D={D_leg:.0f}")
    ax1.plot(lx, ly, "-", lw=1.4, color="black", alpha=0.40, zorder=3.5)
    ax1.plot(lx, ly, "-", lw=0.7, color="white", alpha=0.85, zorder=4)
    # Translucent face, opaque outline: the outline keeps each point findable
    # while the 2017 map stays readable through the middle of it.
    for key, label, marker in PHASES:
        sel = [i for i, p in enumerate(live) if p["surrogate"] == key]
        if not sel:
            continue
        ax1.scatter(lx[sel], ly[sel], marker=marker, s=args.marker_size,
                    zorder=5, facecolors=(1.0, 1.0, 1.0, args.marker_alpha),
                    edgecolors="black", linewidths=0.75,
                    label=f"{label}  ({len(sel)})")
    # black text with a white halo, so the number is legible over a translucent
    # marker whichever part of the colour scale happens to be underneath it
    for i in range(len(live)):
        ax1.annotate(str(ls[i]), (lx[i], ly[i]), fontsize=7.0, zorder=6,
                     ha="center", va="center", color="black",
                     path_effects=[pe.withStroke(linewidth=1.2,
                                                 foreground="white")])
    ax1.set_xlim(0.02, 0.335)
    ax1.set_ylim(-3.5, 26)
    ax1.set_xlabel(r"$\delta q$   (r.l.u., longitudinal)")
    ax1.set_ylabel("E   (meV)")
    ax1.set_title("(a) Acquisition order", loc="left", fontweight="bold", pad=9)

    # ------------------- (b) live points on the legacy intensity scale
    # Same colormap, same norm, so a marker that matches its surroundings means
    # the 2026 and 2017 intensities agree there.
    mesh2 = draw_legacy_map(ax2, scans, norm)
    ax2.plot(dgrid, disp, "-", lw=1.6, color=ACCENT, zorder=3)
    for key, label, marker in PHASES:
        sel = [i for i, p in enumerate(live) if p["surrogate"] == key]
        if not sel:
            continue
        # deliberately NOT translucent: the fill IS the 2026 measurement, and
        # fading it would destroy the lightness comparison this panel exists
        # for.  Kept small instead, so the map stays visible around each one.
        ax2.scatter(lx[sel], ly[sel], c=li[sel], norm=norm, cmap=CMAP,
                    marker=marker, s=args.marker_size * 0.72, zorder=5,
                    edgecolors="black", linewidths=0.75, label=label)
    ax2.set_xlim(0.02, 0.335)
    ax2.set_ylim(-3.5, 26)
    ax2.set_xlabel(r"$\delta q$   (r.l.u., longitudinal)")
    ax2.set_ylabel("E   (meV)")
    ax2.set_title("(b) Measured 2026 intensity", loc="left", fontweight="bold", pad=9)
    colorbar_ax = fig.add_axes((0.905, 0.50, 0.022, 0.44))
    fig.colorbar(mesh2, cax=colorbar_ax,
                 label="intensity  (counts / 150k monitor)")
    # Both maps and the coloured markers use the exact same saved intensity
    # normalisation. One shared colourbar makes that comparison explicit.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.50, 0.435),
               ncol=4, frameon=False, columnspacing=1.0, handlelength=1.7,
               markerscale=0.85, fontsize=8)
    fig.text(0.50, 0.995, "2017 map in both panels; white path and numbers show acquisition order",
             ha="center", va="top", fontsize=8.5)

    # ----------------- (c) direct overlay where the two datasets coincide
    shared = []
    for s in scans:
        sel = [p for p in live if abs(p["dq"] - s["dq"]) < 0.005]
        if sel:
            shared.append((s, sel))
    for n, (s, sel) in enumerate(shared):
        colour = CUT_COLOURS[n % len(CUT_COLOURS)]
        o = np.argsort(s["E"])
        ax3.errorbar(s["E"][o], s["I"][o], s["sigma"][o], fmt="-", lw=1.1,
                     color=colour, alpha=0.9,
                     label=fr"2017  $\delta q$={s['dq']:.2f}")
        ax3.errorbar([p["E"] for p in sel], [p["I"] for p in sel],
                     [p["sigma"] for p in sel], fmt="o", ms=5, lw=0,
                     elinewidth=1.1, capsize=2, color=colour,
                     mec="black", mew=0.8,
                     label=fr"2026  $\delta q$={s['dq']:.2f}")
    ax3.set_xlabel("E   (meV)")
    ax3.set_ylabel("intensity  (counts / 150k monitor)")
    ax3.set_yscale("log")
    ax3.set_title("(c) Coincident scans: 2017 curves and 2026 points\n"
                  "same monitor normalization; no rescaling", loc="left", fontsize=9.5, pad=8)
    ax3.legend(fontsize=8, ncol=3, loc="upper right", framealpha=0.92,
               columnspacing=1.1, handlelength=1.8)
    ax3.grid(alpha=0.25, ls=":")

    covered = sum(1 for p in live
                  if any(abs(p["dq"] - s["dq"]) < 0.026
                         and s["E"].min() <= p["E"] <= s["E"].max()
                         for s in scans))
    if not args.no_title:
        fig.suptitle("How tasAI sampled the true dispersion   |   "
                     "2017 expt 23816 (300 K) vs live run live_20260911_202614   |   "
                     f"{covered} of {len(live)} live points fall inside the 2017 "
                     "measured region", fontsize=12)
    fig.savefig(args.out, dpi=300)
    fig.savefig(args.out.with_suffix(".png"), dpi=300)
    print(f"wrote {args.out}")
    print(f"live points inside the 2017 measured region: {covered}/{len(live)}")
    for s, sel in shared:
        print(f"  dq={s['dq']:.2f}: {len(sel)} live point(s) share this dq")
    return 0


if __name__ == "__main__":
    sys.exit(main())
