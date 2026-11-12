# Recalculate the saved SrMn2Sb2 endpoint

This route recalculates model predictions, resolution samples, and dispersion
paths from the 20-scan data and saved complete Bumps parameter file. It is
more than plot replay, but **does not refit or rerun optimization**. The
lightweight replay environment intentionally does not contain this scientific
runtime. The commands were run after the release was assembled and reproduced
the delivered endpoint within the tolerance given below.

## Required software

The repository root, one level above this directory, supplies:

- `scripts/srmn2sb2_official_python_intensity_core.py`
- `scripts/srmn2sb2_official_python_intensity_bumps_model.py`

The endpoint's `export_provenance.json` records their exact SHA-256 hashes:

```text
310f2a0ec514ccef7428539c621169fd241779e7c441047a613170cd2f4721bb  srmn2sb2_official_python_intensity_core.py
2bc05206b30fdf77cd86d967e6efd0fa85f010cf096a56aa656b9c1e77b47e0f  srmn2sb2_official_python_intensity_bumps_model.py
```

These hashes matched the audited upstream source. Check the separately
exported root files before running; do not substitute a newer model silently.

External scientific implementations remain separate, not vendored here:

- Official `pySpinW` at commit
  `82c14f35579d309c1a1a816940920a25d3254897`, with its own build/runtime
  dependencies. Use the **Python intensity backend**, not Rust intensities.
- `rescalculator`, providing `rescalculator.TASResolution`.
- `lattice-calculator`, providing `lattice_calculator.Lattice`.
- Stock Bumps 1.0.3 (the historical endpoint runtime), NumPy, pandas, and SciPy;
  Matplotlib for the model's optional plotting interface.

Importing pySpinW at this commit also loads its viewer, so its declared
Python dependencies, including `imageio`, must be installed. A clean source
checkout on `PYTHONPATH` is sufficient for this route: the calculation uses
`use_rust=False`, and the Rust extension need not be built.

The original cluster environment used CPython 3.12. The four figure-replay
pins do not constitute a complete lock of that scientific environment.
The historical export does not record exact revisions of every external resolution/lattice
dependency, so record their installed revisions, versions, and build details
with a recalculation. If these tools or the specified upstream pySpinW source
are unavailable, report recalculation as unavailable; do not substitute fake
predictions or fall back to another backend.

Use a separate environment, not `.venv` from the plot quick start. A local
installation outline, after staging and reviewing the three external source
checkouts, is:

```sh
python3.12 -m venv outputs/endpoint-env
FIT_PY="$PWD/outputs/endpoint-env/bin/python"
PYSPINW_ROOT=/absolute/path/to/pinned/pySpinW
RESCALCULATOR_ROOT=/absolute/path/to/rescalculator
LATTICE_ROOT=/absolute/path/to/lattice-calculator
test "$(git -C "$PYSPINW_ROOT" rev-parse HEAD)" = 82c14f35579d309c1a1a816940920a25d3254897
test -z "$(git -C "$PYSPINW_ROOT" status --porcelain)"
"$FIT_PY" -m pip install 'bumps==1.0.3' numpy pandas scipy matplotlib
"$FIT_PY" -m pip install "$LATTICE_ROOT" "$RESCALCULATOR_ROOT" "$PYSPINW_ROOT"
"$FIT_PY" -c 'import bumps, pyspinw; from rescalculator import TASResolution; from lattice_calculator import Lattice; print(bumps.__version__)'
```

Installation may need Rust/build prerequisites according to the pinned
pySpinW project's own instructions; building an extension does not authorize
using its intensity backend. This outline is not a verified portable lock or
an automatic installer. Preserve source trees and use a separate checkout
when a different revision is required.

The historical root script
`scripts/preflight_srmn2sb2_official_python_intensity.py` is **not a portable
fresh-machine gate**: its defaults require the exact Stampede compiled
extension hash and isolated wheel layout, even though intensity evaluation
uses Python. A different OS/Python build legitimately has a different
extension hash. Do not remove/bypass the scientific checks to make it pass.
Validate and record a platform-appropriate build separately. The core also
checks the pinned Python spin-wave signature and source hash; preserve those
checks. Process-pool scheduling and available memory/CPU resources must be
appropriate for 51,136 QMC samples.

## Recalculation command

Run from this `reproducibility` directory with `FIT_PY` set to the separately
prepared runtime's Python. This command only writes a new local output folder.

```sh
R="$PWD"
CODE_ROOT="$(cd .. && pwd)"
ENDPOINT="$R/data/srmn2sb2_fit_endpoint"
OUT="$R/outputs/endpoint-recalculation"
test ! -e "$OUT"
mkdir -p "$OUT/input"
gzip -dc "$ENDPOINT/dispersion_points_input.csv.gz" > "$OUT/input/dispersion_points.csv"
export SRMN_PYSPINW_SCHEDULING=official_processpool
"$FIT_PY" "$ENDPOINT/export_srmn2sb2_scan_predictions.py" \
  --model "$CODE_ROOT/scripts/srmn2sb2_official_python_intensity_bumps_model.py" \
  --core "$CODE_ROOT/scripts/srmn2sb2_official_python_intensity_core.py" \
  --data "$OUT/input/dispersion_points.csv" \
  --backgrounds "$ENDPOINT/scan_peak_fits_backgrounds.json" \
  --pars "$ENDPOINT/endpoint_center005_per_scan_qmc64.par" \
  --seed center005 --normalization per_scan --qmc 64 \
  --out "$OUT/results"
"$FIT_PY" -m pip freeze > "$OUT/runtime-requirements.txt"
```

Use `set -e` when putting this sequence in a shell script so failed checks
stop execution. The uncompressed input is generated from the packaged gzip
only in the output directory, preserving the released uncompressed input
hash and avoiding edits to delivered inputs. The model uses pandas
`read_csv`, so it can also read the packaged gzip directly, but the resulting
export input-file hash then refers to compressed rather than original bytes.

The packaged exporter fixes `single_y`, `legacy`, `full`, and
`official_processpool` as model options and loads the complete `.par` with
Bumps `load_best`. It writes:

- `scan_predictions.csv`: data, uncertainties, predictions, signal,
  background, scale, and conditional energy width for 799 points.
- `resolution_samples.csv`: 64 samples per point, 51,136 rows total.
- `dispersion_paths.csv`: spin-wave modes/intensities on HH2, HH3, and 00L.
- `export_provenance.json`: input/code hashes, configuration, and objective.

Archive-relative paths inside the reduced input do not need to exist for
the saved **legacy** resolution scenario. Switching to `header_horizontal`
is a different analysis and requires explicit raw-file remapping and reviewed
header metadata. Do not infer missing collimation/mosaic information.

## Compare with the delivered endpoint

The original exported endpoint reports:

```text
npoints = 799
nllf = 683.569864454994
reduced_chi_squared = 1.7640512631096619
```

Agreement of `nllf` and `reduced_chi_squared` with these values to a relative
tolerance of 1e-3 counts as reproduction; the last digits depend on the
platform's numerical libraries. A recalculation on a different platform
(macOS arm64, CPython 3.11.16, Bumps 1.0.3) agreed within this tolerance.

Compare the new JSON to `data/srmn2sb2_fit_endpoint/export_provenance.json` and
the new CSVs to the corresponding delivered files (decompress the reference
resolution CSV for reading). Check scan/point ordering, all measured values
and uncertainties, model predictions, backgrounds/scales, conditional energy
widths, dispersion modes, and the complete QMC cloud, not just the scalar
objective. Record maximum differences and the runtime; platform differences
must be evaluated rather than silently accepted. Replot only after inspecting
those comparisons. Keep the delivered endpoint unchanged as the reference.

For a full new optimization, use the separate root model and documented
scientific workflow with explicitly reviewed optimizer settings, runtime,
and starting parameters. `RESULTS_stampede_qmc64.md` summarizes the
initialization and normalization comparison. No live instrument
control or private paper corpus is needed for this endpoint recalculation.
