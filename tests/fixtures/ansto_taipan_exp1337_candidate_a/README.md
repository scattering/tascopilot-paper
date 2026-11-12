# ANSTO TAIPAN exp1337 Candidate A Fixture

This is a synthetic/sanitized data-side fixture for the first non-BT-7 known-answer regression.

It is based on local ANSTO TAIPAN `exp1337` metadata inspection, but it contains no raw `.dat`, `.nx.hdf`, `.ILL`, image, script, or private derived data. The committed records preserve only the minimal metadata needed to test cross-facility routing:

- instrument: ANSTO TAIPAN
- experiment id: 1337
- material signal from scan titles: `DyTiO`, interpreted as candidate `Dy2Ti2O7`
- candidate known-answer family: Dy2Ti2O7 003 longitudinal scans at 45 mK and 800 mK
- counter-evidence: the parent folder name suggests Ho2Ti2O7, so folder-derived material identity must not override scan-title evidence
- control scans: DyTiO 002 and 222 reflections

The known-answer behavior is intentionally candidate-only. Paper linkage is not accepted from this fixture because the current scoped source is data-directory-backed only.

