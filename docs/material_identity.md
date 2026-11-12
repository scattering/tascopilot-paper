# Material Identity And Formula Handling

Material identity is a shared service for dashboards, experiment planning, source triage, corpus retrieval, ICSD lookup, and structure-factor suggestions. It is intentionally separated from scan logic so every workflow uses the same parsing policy and provenance.

## Library Policy

The preferred parser stack is:

1. `pymatgen.core.Composition`
   - Canonical parser for inorganic formulas when installed.
   - Preferred for normalized formulas, reduced formulas, anonymous formulas, element amounts, and fractional composition.
2. `ase.formula.Formula`
   - Lightweight fallback and cross-check.
   - Useful for simple formulas such as `SrTiO3`, `Al2O3`, and `Ho2Ti2O7`.
3. Narrow TAS filename/material-family grammar
   - Beamline shorthand interpreter, not a chemistry parser.
   - Handles curated patterns such as `LuFe75Mn25O3 -> x_Mn = 0.25`.
   - Preserves ambiguity for strings such as `LuFeMnO3` and `LuMnFeO3`.
4. Optional text/entity extractors
   - `ChemDataExtractor` or similar tools may later extract material mentions from papers/proposals.
   - Those mentions are candidate evidence, not accepted identity.

If pymatgen is missing and ASE cannot parse a common decimal solid-solution formula such as `La0.7Sr0.3MnO3`, the core uses a narrow decimal stoichiometry fallback. This fallback is explicitly labeled and is not a general chemistry parser.

## Commands

Check available parser backends:

```bash
python -m tas_copilot_core.cli material doctor
```

Parse ordinary formulas:

```bash
python -m tas_copilot_core.cli material parse --text "SrTiO3" --json
python -m tas_copilot_core.cli material parse --text "La0.7Sr0.3MnO3" --json
```

Parse TAS filename shorthand:

```bash
python -m tas_copilot_core.cli material parse \
  --text "LuFe75Mn25O3_Correlations102248.bt7" \
  --family lu_femn_o3 \
  --json
```

Compare identities:

```bash
python -m tas_copilot_core.cli material compare \
  --left "SrTiO3" \
  --right "TiO3Sr" \
  --json
```

The same operations are also exposed through the tool registry for workflows:

- `tas.material_identity.parse_formula`
- `tas.material_identity.parse_filename_composition`
- `tas.material_identity.compare_material_identity`

These are read-only tools. They shadow the manifest stubs when the core package is installed, so `tools doctor` should report them as available.
Source-triage and experiment-workspace workflows call material identity through this registry boundary, which preserves audit/provenance records and keeps missing parser/tool states fail-closed.

## Record Shape

Material identity records preserve:

- raw string;
- normalized formula;
- reduced formula;
- anonymous formula;
- element amounts;
- fractional composition;
- variable-composition fields such as `x_Mn`;
- parser backend and version;
- interpretation method;
- confidence;
- ambiguity reason;
- warnings;
- provenance.

## Ambiguity Rules

Ambiguous names remain ambiguous unless additional evidence resolves them. For example:

- `LuFe75Mn25O3` encodes `x_Mn = 0.25`.
- `LuFe0.6Mn0.4O3` encodes `x_Mn = 0.4`.
- `LuFeMnO3` does not encode Fe/Mn ratio.
- `LuMnFeO3` does not encode Fe/Mn ratio.

Panel labels, sample logs, proposal metadata, paper-to-experiment mappings, or human review may resolve ambiguity. The filename alone must not.

## Dashboard And Planning Use

Dashboards and planning briefs should use material identity as a filter and confidence signal:

- group scans by material family and composition when supported;
- distinguish related but different compounds;
- avoid false-family matches where possible;
- rank precedent scans by composition match;
- select relevant substrate/mount/background overlays;
- drive ICSD/GSAS-II structure candidate lookup.

The system should show probability-style evidence and caveats rather than treating formula parsing as proof that a scan belongs to a user’s sample.
