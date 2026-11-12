# TASCopilot: Code and Data for Reproducibility

Companion software and data for **TASCopilot: A provenance-preserving
assistant for triple-axis neutron scattering**.

This repository contains the review snapshot, not the manuscript. It starts
with a single initial commit and does not import development history.

## Start Here

| Review task | Location |
| --- | --- |
| Generate the instrument and workflow diagrams (Figures 1-7) | [Diagram build instructions](reproducibility/README.md) |
| Reproduce the four main-text data figures (Figures 8-11) offline | [Figure replay instructions](reproducibility/README.md) |
| Refit the 40 live LSMO points and reproduce Table 4 | [LSMO fit replay](reproducibility/LSMO_FIT.md) |
| Inspect the 20-scan fit, parameters, backgrounds, predictions, and resolution samples | [Fit endpoint data](reproducibility/data/srmn2sb2_fit_endpoint/) |
| Inspect the supplied temperature-dependent S1 panels | [S1 panels and provenance](reproducibility/README.md#s1-panels) |
| Exercise contracts, permission checks, provenance, and synthetic retrieval | Core quick start below |
| Inspect workflow implementations and external-tool boundaries | [Core source](packages/tas_copilot_core/src/tas_copilot_core/), [contracts](packages/tas_copilot_contracts/src/tas_copilot_contracts/), [integration manifest](configs/integrations/external_tools.yaml) |
| Check release content, source revisions, and verification scope | [Release provenance](RELEASE_PROVENANCE.json), [verification](VERIFICATION.md) |

## Figure Reproduction

The `reproducibility/` folder contains the drawing code for Figures 1-7 and
the plotting code and saved numerical inputs for Figures 8-11. The diagram
build uses LaTeX/TikZ; the data-figure replay below uses Python. Neither
requires the manuscript source. See the [figure-by-figure guide](reproducibility/README.md)
for commands, dependencies, input provenance, and verification.

### Offline Data-Figure Replay

The figure package is independent of the core installation and scientific
plugins. With Python 3.11, from the repository root:

```sh
cd reproducibility
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -B scripts/verify_package.py
.venv/bin/python -B -m unittest discover -s tests -v
.venv/bin/python -B scripts/replay_figures.py --out outputs/reviewer-run
```

Only dependency installation needs a network connection. Numerical figure
inputs, uncertainties, the saved fit endpoint, and four figure-only reference
PDFs are included. This replay does not rerun the optimizer or acquire data.
The separate [reproducibility guide](reproducibility/README.md) explains
recalculating predictions with the external scientific packages.

## Core Quick Start

Use a separate Python 3.11 environment for the core. Run from the repository
root so that the bundled configurations and synthetic fixtures are found:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e packages/tas_copilot_contracts -e packages/tas_copilot_core
python -m pip install -r tests/requirements.txt
export TAS_COPILOT_PLUGIN_MODULES=
export TAS_COPILOT_AUDIT_DIR="$PWD/local_runs/audit"
python -m pytest -q -rs
python -m tas_copilot_core.cli tools doctor
python -m tas_copilot_core.cli corpus load-fixtures \
  --fixtures data/corpus_fixtures --catalog local_runs/synthetic_corpus
python -m tas_copilot_core.cli corpus report --catalog local_runs/synthetic_corpus
```

The synthetic corpus is deliberately small and contains no publisher text.
Absent scientific plugins are reported as unavailable; the core does not
substitute fake numerical results. Generated catalogs and audit logs belong
in the ignored `local_runs/` directory.

The default test collection reports explicit skips for absent external tools
and optional runtimes. A smaller smoke suite is also available; see
[reviewer testing](docs/reviewer_testing.md) and [verification scope](VERIFICATION.md).

## Scientific Dependencies

TASCopilot coordinates scientific tools through typed contracts and adapters.
Their implementations remain in their own repositories. The
[integration manifest](configs/integrations/external_tools.yaml) lists
repository locations, imports, and availability behavior. Examples include
BT-7 file readers, `tasreduction`, `rescalculator`, the pySpinW adapter, and
upstream pySpinW. The SrMn2Sb2 model and Bumps objective are supplied under
[`scripts/`](scripts/); recomputing predictions additionally needs the
scientific runtime documented with the endpoint.

The six companion plugins use the reviewed release revisions recorded in
[`review_revisions.json`](configs/integrations/review_revisions.json).
These are separate repositories, not vendored implementations.

The reviewer commands above are read-only or create local derived artifacts.
They do not connect to an instrument or execute measurements. Existing
permission gates and unavailable-tool behavior are retained.

## Data and Redistribution

Public raw measurements are hosted by the NCNR:

- SrMn2Sb2 experiment 24858: [201804 session](https://ncnr.nist.gov/pub/ncnrdata/bt7/201804/24858/) and [201806 session](https://ncnr.nist.gov/pub/ncnrdata/bt7/201806/24858/).
- LSMO reference: [201706/23816](https://ncnr.nist.gov/pub/ncnrdata/bt7/201706/23816/).
- LSMO live case: [202608/14](https://ncnr.nist.gov/pub/ncnrdata/bt7/202608/14/).

Included derived data carry source and checksum records. Paths are portable;
original source hashes identify the inputs before path-only adaptations.
Author-supplied S1 image panels retain their attribution;
their original numerical fitting files are not included.

No manuscript source or compiled manuscript, publisher PDFs, full literature
corpus, extracted private paper text, credentials, or local experiment
catalogs are distributed. Aggregate literature-review counts and synthetic
test records are included. Code is under the [MIT license](LICENSE).
Third-party ontology material retains its [own license and attribution](configs/ontology/vendor/magmo/).
