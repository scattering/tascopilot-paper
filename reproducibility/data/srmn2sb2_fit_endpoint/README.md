# SrMn2Sb2 Fit Endpoint

Saved solution of the reported 20-scan resolution-convolved intensity fit:
BT-7 experiment 24858, center005 start, per-scan normalization, QMC64,
single in-plane spin orientation, and the documented legacy resolution
scenario. The 799 retained points give nllf = 683.569864454994 and reduced
chi-squared = 1.7640512631096619 (775 nominal degrees of freedom).

| File | Contents |
| --- | --- |
| `dispersion_points_input.csv.gz` | Reduced archive measurements from which the 20 scans are selected |
| `scan_peak_fits_backgrounds.json` | Preceding scan-local fits; `best.background` supplies each fixed background |
| `endpoint_center005_per_scan_qmc64.par` | Complete Bumps parameter file, including the per-scan scales |
| `endpoint_fit_provenance.json` | Model, selection, resolution assumptions, and input hashes |
| `scan_predictions.csv` | Measured intensities, uncertainties, predictions, scales, backgrounds, and conditional energy widths |
| `resolution_samples.csv.gz` | 64 QMC resolution samples per fitted point |
| `dispersion_paths.csv` | Spin-wave modes and perpendicular intensities along HH2, HH3, and 00L |
| `export_provenance.json`, `result_sha256.txt` | Export configuration and checksums |
| `export_srmn2sb2_scan_predictions.py` | Saved-endpoint exporter |
| `RESULTS_stampede_qmc64.md` | Initialization and normalization comparison |

[Figure replay](../../README.md) uses the saved predictions.
[Endpoint recalculation](../../ENDPOINT_RECALCULATION.md) recomputes them
with the separately installed scientific runtime.

The reduced CSV's `path` field is relative to the
[public NCNR archive](https://ncnr.nist.gov/pub/ncnrdata/).
All other CSV fields, fitted values, predictions, and resolution samples are
unchanged. Provenance records retain original input hashes and identify the
path-only adaptation; they do not require access to an author's filesystem.
