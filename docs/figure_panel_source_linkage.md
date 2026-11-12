# Figure Panel Source Linkage

`link-figure-panel-to-source` converts panel metadata into source-triage constraints, ranks candidate source data, and writes proposed panel-source links.

Example:

```bash
python -m tas_copilot_core.cli workflow link-figure-panel-to-source \
  --corpus-catalog /tmp/lu_femn_o3_corpus_catalog \
  --data-catalog eval/figure_source_lu_femn_o3_triage.yaml \
  --paper-id paper-lu-femn-o3-synthetic \
  --figure-id fig-lu-femn-o3-1 \
  --panel-id i \
  --output-dir /tmp/lu_femn_o3_panel_i_linkage
```

The workflow compiles the panel metadata into a source-triage query, preserves panel/sample mismatch warnings, runs deterministic source triage, writes proposed candidate links, records rejected candidates, and includes source-support placeholder validation.

Candidate links are proposed only. Accepted source links require reconstruction/provenance or human review.
