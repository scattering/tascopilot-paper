# Figure Reproducibility

This folder contains drawing sources for Figures 1-7 and the plotting code
and numerical inputs for Figures 8-11. It is independent of the core
installation and needs no private corpus, cluster account, manuscript,
or instrument software. The two author-supplied S1 panels are also included.

## Quick Start: Figures 8-11

From this directory, with Python 3.11:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -B scripts/verify_package.py
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B scripts/replay_figures.py --out outputs/reviewer-run
```

The replay requires a new output directory and produces four PDFs, four PNGs,
and `replay_result.json`. It runs from an unrelated working directory with
an empty home directory and checks that TAS packages are not installed.
Only dependency installation needs network access.

To compare rendered output with the supplied figure-only references:

```sh
.venv/bin/python -B scripts/replay_figures.py --compare-reference --out outputs/reference-check
```

This additionally needs Poppler's `pdftoppm`. It compares 120-dpi RGB rasters,
not PDF bytes. Font, OS, or dependency differences can affect rendering.
The references must not be replaced to conceal a mismatch.

The plotting dependencies are NumPy 2.4.6, pandas 3.0.5, SciPy 1.17.1, and
Matplotlib 3.10.9. `requirements-lock.txt` also pins transitive dependencies.

## Diagrams: Figures 1-7

The exact drawing-only TikZ sources are under `diagrams/`, with their source
revisions and hashes in `diagrams/figure_sources.json`.
[DIAGRAMS.md](DIAGRAMS.md) lists TeX dependencies and figure mappings.

```sh
python3 -B scripts/build_diagrams.py --check-dependencies
python3 -B scripts/build_diagrams.py --render --out outputs/diagrams
```

This emits seven figure-only PDFs and PNGs; omit `--render` without Poppler.
Generated files stay under ignored `outputs/`. No manuscript source is needed.

## Numerical Figures and Inputs

These are the authoritative plotting scripts. Each accepts `--data-dir`
and `--out`; defaults are relative to this package, not the working directory.

| Figure | Script under `scripts/` | Inputs under `data/` |
| --- | --- | --- |
| 8, evidence to analysis | `plot_srmn2sb2_evidence_to_analysis_jeff_review_20261103.py` | `figure_inputs/srmn2sb2_evidence.json`, `figure_inputs/srmn2sb2_dispersion.csv.gz` |
| 9, resolution-aware fit | `plot_srmn2sb2_resolution_fit.py` | `srmn2sb2_fit_endpoint/scan_predictions.csv`, `resolution_samples.csv.gz`, `dispersion_paths.csv` |
| 10, LSMO alignment | `plot_lsmo_live_alignment_chain_jeff_review_20261103.py` | `figure_inputs/lsmo_alignment.json` |
| 11, adaptive/reference comparison | `plot_lsmo_sampling_comparison_jeff_review_20261103.py` | `figure_inputs/lsmo_comparison.json` |

For example:

```sh
.venv/bin/python -B scripts/plot_srmn2sb2_resolution_fit.py
```

The plot calculations and rendering match the preserved source scripts.
The shared path helper was adapted for this standalone layout, and the
Figure 10 heading uses the manuscript's name for the system, TASCopilot.

## Numerical Provenance

The [LSMO fit replay](LSMO_FIT.md) reruns the Table 4 optimization on all
40 saved monitor-normalized live points. Its configuration, reference output,
and comparison command are included; TAS-AI is a separate pinned dependency.

The [fit endpoint](data/srmn2sb2_fit_endpoint/README.md) contains 799 fitted
points in 20 scans, 51,136 resolution samples, parameters, fixed backgrounds,
and predictions. Figure replay uses those saved results. The separate
[endpoint recalculation guide](ENDPOINT_RECALCULATION.md) documents the
scientific runtime and exporter for recomputing predictions at the fitted
parameters; it does not rerun optimization.

`PROVENANCE.json` records source revisions and original hashes.
`SHA256SUMS` verifies delivered files. The historical archive checksum list
under `provenance/` identifies the origin of numerical files; that archive
is not needed or distributed. Release adaptations replace personal paths
and internal prose, without changing measured values or predictions.
The reduced CSV's `path` column is relative to the NCNR archive.

Figure 8 contains aggregate literature-review counts only. Its saved map has
10,522 rows; the selection `E >= 1 meV` and `Monitor > 0` yields the
reported 10,353 inelastic points. The difference comprises 148 zero-energy
rows, 12 negative-energy rows, and nine rows between 0 and 1 meV.

Figure 9 uses saved model predictions and conditional energy widths.
Resolution ellipses are approximate two-standard-deviation instrument
projections, not parameter-confidence regions. Alignment curves are display
fits. Figure 11 uses monitor-normalized measurements with saved one-sigma
uncertainties; its phase labels were reconstructed from archive filenames
and step thresholds.

Raw measurements remain in the [NCNR BT-7 archive](https://ncnr.nist.gov/pub/ncnrdata/bt7/):
SrMn2Sb2 in 201804/24858 and 201806/24858, LSMO reference in 201706/23816,
and the live case in 202608/14. The preserved source manifest records hashes
for 405 raw files. Archive access is not needed for figure replay.

## S1 Panels

The unmodified JPEGs in `reference_figures/s1/` were supplied by Jeffrey W.
Lynn on 2026-11-06. The author-reported sample mass is 0.5 g and Neel
temperature is 114.6(6) K. The order-parameter image labels beta = 0.21;
the contour uses the 170 K measurement as background.

`provenance/s1_temperature.json` records attribution, scientific context,
and hashes. These are supplied panels, not generated by the replay scripts.
Their original numerical fitting files are not included.

## Verification

Tests check file integrity, source hashes, portable paths, aggregate-only
review data, scan/sample joins, uncertainties, resolution covariance, S1
attribution, and all seven diagram sources. `VERIFICATION.json` records
completed checks. Re-run the commands above after changes.

Distribute the files enumerated by `SHA256SUMS` and that checksum file itself.
Exclude environments, outputs, caches, and source working trees. No nested
code archive is required. `LICENSE` covers software; data and supplied
figures retain their source attribution.
