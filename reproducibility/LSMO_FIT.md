# Replay the live LSMO fit (Table 4)

The reported quartic fit uses all 40 live measurements in
`data/figure_inputs/lsmo_comparison.json`, the same monitor-normalized points
shown in Figure 11. It gives **D = 131.9 meV A^2**, beta = 2.00 A^2
(the imposed upper bound), and chi2 / 35 = 149.4.

## Method and inputs

`lsmo_fit_config.json` fixes the input hash, model hash, external TAS-AI
revision, lattice constant (3.9005 A), and temperature (300 K). The replay
uses H = 0, K = 1 + dq, L = 0 and the saved E, I, and sigma for every live
record in acquisition order. Coordinates were reconstructed from the
public archive filenames. Intensities and uncertainties are
`150000 * detector_counts / monitor_counts` and
`150000 * sqrt(max(detector_counts, 1)) / monitor_counts`.
The monitor is treated as the normalization, not an additional error term.
The archive origin and raw-file hashes are in `provenance/source_manifest.json`.

The model and optimizer are the released `../scripts/tasai_fm_magnon.py`;
the replay imports them without copying their scientific implementation.
It fits E(q) = D q^2 (1 - beta q^2), q = 2 pi dq / a, with a damped-oscillator
intensity, Bose population factor, Mn form factor, amplitude, width, and
constant background. Least squares uses the saved counting uncertainties
and the model's default initialization and bounds (0 <= beta <= 2 A^2).
There is no instrumental-resolution convolution or empirical higher-order
monitor correction.

## Run

Use the pinned Python 3.11 figure environment from [README.md](README.md).
TAS-AI remains a separate dependency. Obtain the public source at revision
`1f68c59bedc53972bfd4383f8ac6d10644177444` from
[usnistgov/tasai](https://github.com/usnistgov/tasai/tree/1f68c59bedc53972bfd4383f8ac6d10644177444).
An anonymous download, from this directory, is:

```sh
mkdir -p outputs/lsmo-fit-runtime
curl -fL -o outputs/tasai-1f68c59.tar.gz \
  https://api.github.com/repos/usnistgov/tasai/tarball/1f68c59bedc53972bfd4383f8ac6d10644177444
tar -xzf outputs/tasai-1f68c59.tar.gz -C outputs/lsmo-fit-runtime --strip-components=1
.venv/bin/python -B scripts/replay_lsmo_fit.py \
  --tasai-root outputs/lsmo-fit-runtime --check \
  --out outputs/lsmo-fit/replay_result.json
```

After dependency staging the fit is offline. No instrument connection,
private corpus, author home directory, or manuscript is required. The script
also runs from another working directory using absolute paths. An existing
output is not overwritten. Missing or mismatched dependencies produce an
error rather than substitute results.

## Outputs and checks

`results/lsmo_live_fit.json` contains the reference fit: all five parameters,
bounds and active bounds, degrees of freedom, chi-squared, 40 predictions,
40 standardized residuals, input/model hashes, command, and runtime versions.
`--check` compares the configuration, parameters, objective and every
prediction (relative tolerance 1e-4; absolute tolerance 1e-4). This tolerance
allows small numerical differences while preserving the reported precision.
The reference is not overwritten by a check.

```sh
.venv/bin/python -B -m unittest discover -s tests -v
```

These tests verify the saved input, result arithmetic, parameter bounds,
normalization metadata, and mismatch detection. The `--check` command above
is the optimizer replay test and requires the separate TAS-AI source.
Figure 11's overlaid dispersion curve is the independent 2017 peak-position
fit; this replay does not change that curve or any figure input.
