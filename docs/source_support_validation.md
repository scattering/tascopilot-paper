# Source Support Validation

Source-support validation is a deterministic placeholder layer for the pilot corpus. It is not a general citation verifier and it does not use an LLM judge.

Run:

```bash
python -m tas_copilot_core.cli corpus validate-source-support \
  --catalog [CORPUS_CATALOG] \
  --output-dir [OUTPUT_DIR]/source_support \
  --json
```

The validator checks fixture-level claims such as:

- panel labels support panel compositions;
- user formula `LuFe0.6Mn0.4O3` supports `x_Mn = 0.4`;
- methods chunks support instrument and detector-mode constraints when explicitly encoded;
- candidate panel-source links remain candidate/proposed unless reconstruction or human review supports acceptance.

Results are classified as supported, unsupported, ambiguous, missing source, or requiring human review.

## Boundary

Source-support records are evidence, not final authority. A proposed source-data link must not become an accepted source link without reconstruction/provenance or human review.

