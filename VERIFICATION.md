# Verification

## Core Software

On 12 November 2026, the default test collection passed with Python 3.12:
**760 passed, 83 skipped**. FastAPI, Uvicorn, and HTTPX were installed to
include the web tests. Skips name optional scientific tools, separate
runtimes, or deployment scripts that are not part of this release.

```sh
python -m pytest -q -rs
```

Tests cover contracts, permission enforcement, provenance, synthetic retrieval,
workflow composition, ontology validation, dashboard routes, and portable
configuration. The release does not rely on an author's sibling checkouts.
See [reviewer testing](docs/reviewer_testing.md) for optional integrations.

The six companion plugin suites passed 83 tests with three optional-runtime
or external-fixture skips. Their selected core integration tests passed
15 tests with two skips for other external repositories. Tested plugin
revisions are in `configs/integrations/review_revisions.json`.

## Figures and Data

The [figure package](reproducibility/README.md) has independent tests,
checksums, and a pinned replay environment. Its
[verification record](reproducibility/VERIFICATION.json) records completed
diagram and plot checks.

On 12 November 2026, all 42 reproducibility-package tests passed and Figures 8-11
reproduced the supplied reference rasters exactly at 120 dpi.

The [Table 4 LSMO fit replay](reproducibility/LSMO_FIT.md) was run with the
pinned Python environment and an anonymous download of the pinned TAS-AI
source. All five parameters and all 40 predictions matched the saved result
from an unrelated working directory. Missing dependencies and attempts to
overwrite an existing output were rejected.

The reported endpoint contains 799 fitted points in 20 scans and 51,136
resolution samples. Personal paths in the full reduced CSV were replaced by
archive-relative paths. All other fields in all 10,522 rows are unchanged;
the tests enforce their canonical checksum. Saved predictions, uncertainties,
parameters, and resolution samples retain their source hashes. The figure
references are the manuscript's figure files. The plot scripts match their
sources except for the Figure 10 heading, which uses the manuscript's name
for the system; `reproducibility/PROVENANCE.json` records both hashes.

Figure replay uses saved results. Full spin-wave recalculation and
optimization are separate workflows; see the
[endpoint instructions](reproducibility/ENDPOINT_RECALCULATION.md).
The endpoint recalculation was run on 12 November 2026 and reproduced the
saved objective within the documented tolerance.
S1 contains the two original author-supplied images, not a new fit.

## Release Content

```sh
python tools/audit_release.py --self-test
python tools/audit_release.py
```

The auditor checks the staged or committed Git index, including decompressed
CSV data. It rejects manuscript files, handoffs, opaque archives, private
catalogs outside the synthetic-fixture allowlist, personal paths, and known
credential patterns. It reports issue types without printing matching secret
values. Pattern checks supplement review of the file inventory.

The repository contains code, derived data, synthetic fixtures, and reviewer
documentation. It excludes manuscripts, publisher PDFs, full literature
corpora, private deployment files, internal notes, and development history.
Original source hashes and release adaptations are in
`RELEASE_PROVENANCE.json`.
