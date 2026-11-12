# Paper/Figure/Panel Catalog

Paper, figure, panel, caption, and source-support fixtures live in a document-side JSONL catalog. This is separate from the TAS SQLite data-file catalog.

Load the LuFe/Mn/O paper/figure fixture:

```bash
python -m tas_copilot_core.cli corpus load-fixtures \
  --fixtures tests/fixtures/lu_femn_o3_paper_figure \
  --catalog /tmp/lu_femn_o3_corpus_catalog
```

Report counts:

```bash
python -m tas_copilot_core.cli corpus report \
  --catalog /tmp/lu_femn_o3_corpus_catalog
```

The fixture contains one synthetic paper, one figure, panels `i/j/k`, a caption chunk, a textbook placeholder, a paper-to-experiment mapping, candidate panel-source records, and source-support records.

Compile a panel into a source-triage query:

```bash
python -m tas_copilot_core.cli corpus compile-figure-query \
  --corpus-catalog /tmp/lu_femn_o3_corpus_catalog \
  --paper-id paper-lu-femn-o3-synthetic \
  --figure-id fig-lu-femn-o3-1 \
  --panel-id k \
  --user-sample "LuFe0.6Mn0.4O3" \
  --output /tmp/panel_k_query.json
```

Validate deterministic fixture-level source support:

```bash
python -m tas_copilot_core.cli corpus validate-source-support \
  --catalog /tmp/lu_femn_o3_corpus_catalog \
  --json
```
