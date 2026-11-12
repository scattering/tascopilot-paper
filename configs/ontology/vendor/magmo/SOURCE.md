# MagMO (Magnetic Materials Ontology), vendored copy

- Source: https://github.com/emmo-repo/domain-magnetic-materials (file `magnetic-materials.ttl`)
- Version: 0.0.5 (EMMO-inferred 1.0.3); fetched 2026-09-18 from main at commit `70704234d6d0618a4daabd471aa2e124c16aa230`
- License: CC BY 4.0 (see LICENSE beside this file)
- Citation: W. Hortschitz et al., "A Community-Developed Domain Ontology for Magnetic
  Materials", arXiv:2609.11464 (2026). MaMMoS project, EU grant 101135546.

This file is the input of `scripts/generate_magmo_concepts.py`, which writes
`configs/ontology/magmo_generated.yaml`. Do not edit the generated file by hand;
local annotations go in `configs/ontology/magmo_overlay.yaml`. To upgrade: replace
the .ttl, update this note, rerun the generator, review the diff.
