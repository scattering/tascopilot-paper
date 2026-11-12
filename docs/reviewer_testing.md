# Reviewer Testing

Use a fresh Python 3.11 or newer environment from the repository root:

```sh
python -m pip install -e packages/tas_copilot_contracts -e packages/tas_copilot_core
python -m pip install -r tests/requirements.txt
python -m pytest -q -rs
```

The default collection covers contracts, registry permissions, provenance,
synthetic retrieval, ontology validation, workflow composition, and the
LuFe/Mn/O known-answer regression. Optional integrations report explicit
prerequisite skips, not passes. No tests connect to an instrument.

A smaller core smoke suite remains available:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest @tests/core_suite.txt -q -rs
```

Generated audit files and derived artifacts are ignored. The synthetic
`data/catalogs/` records are labeled software fixtures, not experimental data.
Historical PSD registry entries retain their reviewed calibration values;
unresolved features are not classified as physical or artifacts by the registry.

## Optional Integrations

External-checkout tests use `TAS_COPILOT_TEST_PLUGIN_ROOT`, never an author's
home directory. Place checkouts under that root with these directory names:
`bt7grouping`, `tas-data-readers`, `tasreduction`, `rescalculator`,
`spurion`, `bt7-paper-mapping`, `psdplanner`, and `tas-pyspinw-adapter`.
Install each plugin's dependencies in the test environment before opting in.

```sh
export TAS_COPILOT_TEST_PLUGIN_ROOT=/absolute/path/to/plugin-checkouts
python -m pytest tests/test_external_plugin_contract_matrix.py -q -rs
```

Header hydration needs `bt7grouping`; counting-uncertainty tests need
`tasreduction`. Web tests need FastAPI, Uvicorn, and HTTPX. GSAS-II tests
require an explicit `TAS_GSASII_RUNTIME` configuration.

The paper release excludes facility-specific deployment scripts and unrelated
experiment settings. Their retained generic boundary tests use synthetic
inputs; integration tests needing omitted servers or cluster scripts skip
with the missing prerequisite named. Scientific implementations remain in
the repositories listed by `configs/integrations/external_tools.yaml`.

The standalone [figure tests](../reproducibility/README.md) have a separate,
pinned environment and do not need any scientific plugins.
