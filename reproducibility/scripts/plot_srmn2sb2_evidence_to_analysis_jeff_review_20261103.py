"""Figure: the SrMn2Sb2 (BT-7 exp 24858) evidence-to-analysis trace.
Panels: (a) 43 candidate facts from two user-supplied CaMn2Sb2 papers with human verdicts,
(b) alignment planning (GSAS-II ranked reflections) against the observed 2018-05-23 FindPeak scans,
(c) alignment state-machine epochs per session, (d) daily scan summaries, (e) session-separated
dispersion slices, and a provenance spine. Inputs are the S2 plotting snapshot;
its source manifest links the saved results to the public archive and source artifacts."""
from __future__ import annotations
import re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy.optimize import curve_fit
from figure_inputs import plot_args, read_json

args = plot_args("srmn2sb2_evidence_to_analysis_jeff_review_20261103.pdf")
OUT = args.out
snapshot = read_json(args.data_dir / "figure_inputs/srmn2sb2_evidence.json")
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5, "legend.fontsize": 8.8,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42})

# ---------- helpers ----------
def gauss(x, a, c, w, b): return a * np.exp(-0.5 * ((x - c) / w) ** 2) + b
def fit_center(x, y):
    p0 = [y.max() - y.min(), x[np.argmax(y)], (x.max() - x.min()) / 6, y.min()]
    try:
        p, _ = curve_fit(gauss, x, y, p0=p0, sigma=np.sqrt(np.maximum(y, 1)), maxfev=20000)
        return p
    except Exception:
        return None

# ---------- data ----------
facts = [r for r in snapshot["fact_counts"] for _ in range(r["count"])]
edges = [dict(verdict=k) for k, count in snapshot["edge_counts"].items() for _ in range(count)]
days = pd.DataFrame(snapshot["days"])
disp = pd.read_csv(args.data_dir / "figure_inputs/srmn2sb2_dispersion.csv.gz")
states = snapshot["states"]

# observed FindPeak A4 scans on 2018-05-23 and the reflection each one targeted (from the console log / ScanDescr)
FINDPEAKS = [("fpx840982", "0 0 2"), ("fpx841000", "1 1 0"), ("fpx840998", "0 0 4")]  # (0 0 1) fpx840997 was a 12-point aborted search, not shown
pred = {r["hkl"]: r["planned_two_theta_deg"] for r in snapshot["findpeaks"]}

# Reflow for full manuscript width: two columns, then paired maps and a
# two-row provenance chain. The scientific arrays and fit recipe are unchanged.
fig = plt.figure(figsize=(8.8, 9.1))
gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1.05, 1.15, 0.85], hspace=0.78, wspace=0.32,
                      left=0.08, right=0.97, top=0.94, bottom=0.035)

# ---------- (a) candidate facts with human verdicts ----------
ax = fig.add_subplot(gs[0, 0])
VERD = [("correct", "#2e7d32", "correct"), ("not_supported", "#c62828", "not supported by quote"),
        ("wrong_field", "#ef6c00", "wrong field"), ("too_vague", "#f9a825", "too vague"),
        ("ocr_or_chunking_problem", "#757575", "OCR / chunking problem")]
color = {k: c for k, c, _ in VERD}
papers = [("paper-camn2sb2-bridges-powder", "Bridges et al. 2009\n(powder diffraction)"),
          ("paper-camn2sb2-mcnally-spinwaves", "McNally et al. 2015\n(spin waves)")]
order = [k for k, _, _ in VERD]
y0 = 4.4
for pid, name in papers:
    rows = [r for r in facts if r["paper_id"] == pid]
    rows.sort(key=lambda r: (order.index(r["type_specific_status"]), not r["magnetic"]))
    for i, r in enumerate(rows):
        xi, yi = i % 12, y0 - (i // 12) * 0.5
        ax.add_patch(Rectangle((xi, yi), 0.9, 0.42, color=color[r["type_specific_status"]]))
        if r["magnetic"]:
            ax.add_patch(Rectangle((xi, yi), 0.9, 0.42, fill=False, lw=1.4, ec="k"))
    ax.text(-0.3, y0 + 0.5, f"{name}: {len(rows)} facts, {sum(r['verdict']=='positive' for r in rows)} correct",
            fontsize=8.8, va="bottom")
    y0 -= 2.5
# edges
ax.text(-0.3, y0 + 0.5, f"{len(edges)} proposed cross-paper relation edges\n"
        f"{sum(e['verdict']=='positive' for e in edges)} accepted; {sum(e['verdict']=='negative' for e in edges)} rejected; "
        f"{sum(e['verdict']=='inconclusive' for e in edges)} inconclusive", fontsize=8.8, va="bottom")
ECOL = {"positive": "#2e7d32", "negative": "#c62828", "inconclusive": "#f9a825"}
for i, e in enumerate(sorted(edges, key=lambda e: e["verdict"])):
    ax.add_patch(Rectangle((i, y0), 0.9, 0.42, color=ECOL[e["verdict"]]))
ax.set_xlim(-0.4, 12.2); ax.set_ylim(-0.8, 6.5); ax.axis("off")
handles = [Rectangle((0, 0), 1, 1, color=c) for _, c, _ in VERD] + [Rectangle((0, 0), 1, 1, fill=False, ec="k", lw=1.4)]
fig.legend(handles, ["correct", "not supported by quote", "wrong field", "too vague", "OCR / chunking problem",
                   "magnetic fact type\n(model prior)"], loc="upper left",
          bbox_to_anchor=(0.075, 0.759), ncol=2, frameon=False, handlelength=1.0,
          columnspacing=0.7, borderaxespad=0, fontsize=9)
ax.set_title("(a) Analog literature: 43 candidate facts\nTwo user-supplied papers; human review\n(gemma3:4b; 11/43 correct)", loc="left")

# ---------- (b) planning vs observed FindPeak scans ----------
ax = fig.add_subplot(gs[0, 1])
off = 0
for name, hkl in FINDPEAKS:
    saved = next(r for r in snapshot["findpeaks"] if r["filename"] == f"{name}.bt7")
    x, y = np.asarray(saved["x"]), np.asarray(saved["rate"])
    span = max(y.max() - y.min(), 1)
    yn = (y - y.min()) / span  # the offset is exact; only the scale touches the uncertainty
    yn_err = np.asarray(saved["display_error"])  # propagated by tasreduction at export
    p = fit_center(x, y)
    ax.errorbar(x - pred[hkl], yn + off, yerr=yn_err, fmt="o-", ms=3, lw=0.9, color="C0", ecolor="#7f8c8d", elinewidth=0.7, capsize=0)
    if p is not None:
        xf = np.linspace(x.min(), x.max(), 200)
        ax.plot(xf - pred[hkl], (gauss(xf, *p) - y.min()) / max(y.max() - y.min(), 1) + off, "-", lw=1, color="k", alpha=0.6)
        ax.text(2.4, off + 0.55, f"({hkl})\nplanned 2$\\theta$ {pred[hkl]:.2f}$^\\circ$\nfound {p[1]:.2f}$^\\circ$ ({p[1]-pred[hkl]:+.2f})",
                fontsize=8.8, va="center")
    off += 1.15
ax.axvline(0, color="grey", ls="--", lw=0.9)
ax.set_xlim(-2.2, 5.8); ax.set_yticks([])
ax.set_xlabel("A4 $-$ planned 2$\\theta$ (deg)")
ax.set_ylabel("FindPeak scans\n(normalized, offset)")
ax.set_title("(b) GSAS-II-ranked reflections\nPlanning vs. observed FindPeak history\n2018-05-23", loc="left")

# ---------- (c) alignment state machine epochs ----------
ax = fig.add_subplot(gs[1, 0])
STATES = [("reflection_localization", "#1565c0", "reflection localization"),
          ("second_reflection_or_plane_flattening", "#6a1b9a", "second reflection / plane flattening"),
          ("fine_alignment_or_sample_centering", "#2e7d32", "fine alignment / centering"),
          ("alignment_frame_redefined_or_reset_suspected", "#ef6c00", "frame reset suspected"),
          ("alignment_activity_observed", "#bdbdbd", "activity, unclassified")]
scol = {k: c for k, c, _ in STATES}
for row, (s, lab) in enumerate((("may", "May 2018 (initial mount)"), ("july", "July 2018 (remount)"))):
    ep = states[s]["state_epochs"]
    n = len(ep)
    for i, e in enumerate(ep):
        st = e["current_state_estimate"]["state"]
        ax.add_patch(Rectangle((i / n, 1.2 - 1.5 * row), 1 / n, 0.55, color=scol.get(st, "#bdbdbd"), ec="white", lw=0.4))
    summ = states[s]["summary"]
    ax.text(0, 1.2 - 1.5 * row + 0.62, f"{lab}\n{summ['event_count']} events $\\rightarrow$ {n} epochs; "
            f"{summ['flagged_contradictions']} contradictions", fontsize=8.8)
ax.set_xlim(0, 1); ax.set_ylim(-0.4, 2.7); ax.set_xticks([0, 1]); ax.set_xticklabels(["start", "end"]); ax.set_yticks([])
fig.legend([Rectangle((0, 0), 1, 1, color=c) for _, c, _ in STATES],
           ["reflection localization", "second reflection /\nplane flattening", "fine alignment / centering",
            "frame reset suspected", "activity, unclassified"],
          loc="upper left", bbox_to_anchor=(0.075, 0.500), frameon=False, ncol=2,
          handlelength=1, borderaxespad=0, fontsize=8.8, labelspacing=0.2, columnspacing=0.6)
ax.set_title("(c) Alignment-state epochs\nSuggest-only; rocking to software state", loc="left")

# ---------- (d) daily summaries ----------
ax = fig.add_subplot(gs[1, 1])
def family_counts(s: str) -> dict:
    out = {"alignment / rocking": 0, "constant-Q energy": 0, "fixed-E Q scans": 0, "elastic / other": 0}
    for part in s.split(";"):
        m = re.match(r"\s*(.*?):\s*(\d+)", part)
        if not m: continue
        name, n = m.group(1), int(m.group(2))
        if "alignment" in name: out["alignment / rocking"] += n
        elif "constant-Q" in name: out["constant-Q energy"] += n
        elif "fixed-E" in name: out["fixed-E Q scans"] += n
        else: out["elastic / other"] += n
    return out
fam = pd.DataFrame([family_counts(s) for s in days["top_families"]], index=days["date"])
xs = np.arange(len(fam)); xs[5:] += 1  # visual gap between sessions
bottom = np.zeros(len(fam))
for col, c in zip(fam.columns, ["#ef6c00", "#1565c0", "#2e7d32", "#9e9e9e"]):
    ax.bar(xs, fam[col], bottom=bottom, color=c, label=col, width=0.8); bottom += fam[col].to_numpy()
ax.set_xticks(xs); ax.set_xticklabels([d[5:] for d in fam.index], rotation=60)
ax.set_ylabel("scans per day"); ax.set_ylim(0, 118); ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8)
ax.set_title(f"(d) Daily summaries: {len(fam)} days; {int(days['scan_count'].sum())} scans\nOne experiment-level narrative", loc="left")

# ---------- (e) session-separated dispersion slices ----------
sub = gs[2, :].subgridspec(1, 2, wspace=0.14)
low = disp[(disp["Temp"] < 30) & (disp["Monitor"] > 0) & (disp["E"] >= 1.0)]
hh = low[(low["scan_variable"].str.lower() == "qy") & ((low["H"] - low["K"]).abs() < 0.05) & ((low["L"] - 2).abs() < 0.12)]
vmin, vmax = np.percentile(hh["I_over_M"][hh["I_over_M"] > 0], [5, 99.5])
for j, (sess, title) in enumerate((("2018-05", "session 1 (May, $E_f$ = 14.7 / 28 meV)"), ("2018-07", "session 2 (July remount, $E_f$ = 28 meV)"))):
    ax = fig.add_subplot(sub[0, j])
    g = hh[hh["date"].str.startswith(sess)]
    sc = ax.scatter(g["H"], g["E"], c=np.clip(g["I_over_M"], vmin, vmax), s=6, cmap="viridis",
                    norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax), linewidths=0)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 36); ax.set_xlabel("$(H\\,H\\,2)$, $H$ (r.l.u.)")
    if j == 0: ax.set_ylabel("energy transfer (meV)")
    else: ax.set_yticklabels([])
    session_title = "(e) May: $E_f$ = 14.7 / 28 meV" if j == 0 else "July remount: $E_f$ = 28 meV"
    ax.set_title(f"{session_title}; $T<30$ K\n{len(g)} points; {g['filename'].nunique()} scans", fontsize=9.5)
cb = fig.colorbar(sc, ax=fig.axes[-2:], pad=0.015, fraction=0.025); cb.set_label("counts / monitor", fontsize=8.8)

# ---------- spine ----------
ax = fig.add_subplot(gs[3, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
n_fact_ok = sum(r["verdict"] == "positive" for r in facts)
SPINE = [("2 user-supplied\nCaMn$_2$Sb$_2$ PDFs", "private copies,\noutside the repo", "user input"),
         (f"{len(facts)} typed facts,\n{len(edges)} relation edges", f"{n_fact_ok} facts, 1 edge accepted\nafter human review", "analog prior; candidate"),
         ("ranked reflections,\nscan ranges", "GSAS-II $|F|^2$ boundary;\nAl-background warnings", "retrospective plan"),
         ("1573 console/scan\nevents; 66 alignment epochs", "state estimates, replay plan\nwith stop points", "observed history"),
         ("11 daily reports,\n1 experiment narrative", "409 archived scans,\nsource lineage kept", "derived summary"),
         (f"{len(disp)} inelastic points,\n28 panels", "two sessions kept apart", "derived map"),
         ("20 scans, 799 points,\n$\\chi^2_\\nu$ = 1.764", "pySpinW + QMC64 resolution;\n4 start/normalization cases", "local endpoint")]
w, gap = 0.229, 0.025
for i, (head, sub_, cls) in enumerate(SPINE):
    column, row = (i, 0) if i < 4 else (i - 4, 1)
    x0, y0 = 0.004 + column * (w + gap), 0.56 - 0.57 * row
    ax.add_patch(FancyBboxPatch((x0, y0), w, 0.46, boxstyle="round,pad=0.006", fc="#f3f6fb", ec="#4a6fa5", lw=0.9, clip_on=False))
    ax.text(x0 + w / 2, y0 + 0.34, f"{i+1}. {head}", ha="center", va="center", fontsize=8.2, fontweight="bold")
    ax.text(x0 + w / 2, y0 + 0.11, sub_, ha="center", va="center", fontsize=8.8, color="#333")
    ax.text(x0 + w / 2, y0 - 0.065, cls, ha="center", va="center", fontsize=8.8, style="italic", color="#4a6fa5")
    if i not in (3, 6):
        ax.add_patch(FancyArrowPatch((x0 + w + 0.002, y0 + 0.23), (x0 + w + gap - 0.002, y0 + 0.23), arrowstyle="-|>", mutation_scale=9, color="#4a6fa5", lw=0.9))
fig.text(0.075, 0.184, "Provenance spine: source lineage and evidence class; steps 1-7 continue below. No instrument actions.", fontsize=9, va="bottom")

# Dedicated strips separate the legends from the next panels at printed width.
for panel, position in zip(fig.axes, [
        [0.075, 0.780, 0.400, 0.160], [0.570, 0.780, 0.400, 0.160],
        [0.075, 0.515, 0.400, 0.130], [0.570, 0.515, 0.400, 0.130],
        [0.075, 0.250, 0.390, 0.140], [0.535, 0.250, 0.390, 0.140],
        [0.935, 0.250, 0.018, 0.140], [0.075, 0.025, 0.895, 0.140]]):
    panel.set_position(position)

fig.savefig(OUT, bbox_inches="tight"); fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
print("wrote", OUT)
