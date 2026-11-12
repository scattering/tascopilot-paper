# Figure Source Data Workflow

`find-figure-source-data` searches a TAS data-file catalog for candidate raw scans/files that could have produced a published figure panel. It is an auditable candidate-search workflow, not a definitive attribution engine.

## Purpose

Use this workflow when a scientist has a figure panel, manually supplied figure constraints, and wants ranked candidate source data. The workflow compares cataloged scan/file metadata against date, instrument, mode, sample/composition, experimenter, Q range, temperature range, PSD temperature-series likelihood, and optional paper-to-experiment mappings.

The workflow preserves uncertainty. It must not state that files made a figure panel unless data are hydrated, metadata are explicit, reconstruction succeeds, and a human reviews the evidence.

## Example Query

The checked-in LuFe/Mn/O example is:

```bash
examples/figure_queries/lu_femn_o3_bt7_psd_2013_panel_k.yaml
```

It records the supplied mismatch: `LuFe0.6Mn0.4O3` corresponds to `x_Mn = 0.4`, while panel `(k)` is labeled `x_Mn = 0.5`. The workflow searches both interpretations: panel `(k)`/`x_Mn=0.5` and the user-sample interpretation, likely panel `(j)`/`x_Mn=0.4`.

## Run With Existing Catalog

```bash
PYTHONPATH=packages/tas_copilot_contracts/src:packages/tas_copilot_core/src \
python -m tas_copilot_core.cli workflow find-figure-source-data \
  --query-file examples/figure_queries/lu_femn_o3_bt7_psd_2013_panel_k.yaml \
  --catalog [CATALOG_DB] \
  --output-dir [OUTPUT_DIR]
```

Add reconstruction only after candidate files are hydrated and Q/T/PSD metadata are explicit:

```bash
PYTHONPATH=packages/tas_copilot_contracts/src:packages/tas_copilot_core/src \
python -m tas_copilot_core.cli workflow find-figure-source-data \
  --query-file examples/figure_queries/lu_femn_o3_bt7_psd_2013_panel_k.yaml \
  --catalog [CATALOG_DB] \
  --output-dir [OUTPUT_DIR] \
  --attempt-reconstruction \
  --max-reconstruction-candidates 5
```

## Run With Data Root

If a catalog is not ready, provide a data root. The workflow first composes the existing TAS directory catalog path, then searches the resulting SQLite catalog.

```bash
PYTHONPATH=packages/tas_copilot_contracts/src:packages/tas_copilot_core/src \
python -m tas_copilot_core.cli workflow find-figure-source-data \
  --data-root [TAS_DATA_DIR] \
  --catalog [CATALOG_DB] \
  --output-dir [OUTPUT_DIR] \
  --query-file examples/figure_queries/lu_femn_o3_bt7_psd_2013_panel_k.yaml
```

## Direct CLI Constraints

The query YAML can be replaced or supplemented with flags:

```bash
python -m tas_copilot_core.cli workflow find-figure-source-data \
  --catalog [CATALOG_DB] \
  --output-dir [OUTPUT_DIR] \
  --instrument BT-7 \
  --mode PSD \
  --sample-type powder \
  --material-family "LuFe_{1-x}Mn_xO3" \
  --sample "LuFe0.6Mn0.4O3" \
  --target-panel k \
  --target-panel-xmn 0.5 \
  --alternate-panel j \
  --alternate-panel-xmn 0.4 \
  --date 2013-05-17 \
  --date 2013-06-01 \
  --experimenter Disseler \
  --experimenter Ratcliff \
  --experimenter Lynn \
  --experimenter Cheong \
  --q-min 1.10 \
  --q-max 1.45 \
  --temperature-min 0 \
  --temperature-max 200 \
  --annotated-q 1.24
```

## Scoring

The score is deterministic and transparent. It combines date, instrument, PSD mode, sample type, composition aliases, experimenter names, Q coverage, temperature coverage, PSD temperature-series likelihood, and paper-mapping evidence. Each score contribution is recorded as evidence. Composition mismatches are penalized but not discarded.

Confidence labels are coarse:

- `high`: strong metadata match, still requiring reconstruction/human review for figure attribution.
- `medium`: plausible candidate with some missing or weak evidence.
- `low`: sparse or partial metadata.
- `insufficient`: no meaningful evidence.

## Reconstruction Rules

Reconstruction is optional. It only runs when requested. It fails closed if:

- source files are path-only;
- the reader plugin is unavailable;
- Q axis is missing;
- temperature axis is missing;
- PSD intensity arrays are missing;
- PSD channel-to-A4/sample two-theta calibration would need to be guessed.

Synthetic test fixtures may store explicit `psd_map` metadata in the catalog. Real data should be reconstructed through reader plugins and explicit metadata only.

## Report Artifacts

The workflow writes:

- `figure_source_query.json`
- `figure_source_candidates.json`
- `rejected_candidates.json`
- `figure_source_report.json`
- `figure_source_report.md`
- `candidate_table.csv`
- `candidate_table.md`
- `audit/audit.jsonl`
- `reconstructions/` when reconstruction succeeds

Generated reports and reconstructions from real data should remain local unless explicitly sanitized.

The Markdown report starts with a decision summary. It identifies the strongest candidate branch, states why it is still only candidate source data, lists rejected incompatible branches such as nonzero-energy line scans for elastic queries, and repeats the PSD calibration blocker when Q coverage is missing or uncalibrated.

## Manual Review

Start with `candidate_table.md`, then inspect the top candidate scan IDs and file IDs. Hydrate path-only files before reconstruction. A source-data claim should only be made after reconstructing an `I(Q,T)` representation and comparing it against the figure constraints.

This workflow differs from `scan-brief` and `experiment-brief`: those summarize known catalog objects. `find-figure-source-data` searches for uncertain source candidates for a figure panel and preserves competing interpretations.
