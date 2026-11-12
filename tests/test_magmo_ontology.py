"""The MagMO-derived concept family: generated from the vendored ontology, loaded
through the seed's includes, mapped to local neutron concepts, and usable for
tagging paper facts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from tas_copilot_core.config import CoreConfig
from tas_copilot_core.ontology import load_external_mappings, load_external_vocabularies, load_ontology, validate_ontology_alignment
from tas_copilot_core.ontology.linker import match_concepts

REPO = Path(__file__).resolve().parents[1]


def _magmo(concepts):
    return [c for c in concepts if c.id.startswith("magmo_")]


def test_generated_family_is_current_and_loaded_through_includes() -> None:
    check = subprocess.run([sys.executable, str(REPO / "scripts/generate_magmo_concepts.py"), "--check"], capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    concepts = load_ontology(CoreConfig().ontology_path)
    magmo = _magmo(concepts)
    assert 80 <= len(magmo) <= 130
    assert {c.family for c in magmo} == {"magnetic_material_property", "magnetic_material_class", "crystallographic_metadata"}
    ids = [c.id for c in concepts]
    assert len(ids) == len(set(ids))
    # every generated concept keeps its EMMO IRI and a reviewed exactMatch to the vendored source
    for c in magmo:
        assert c.xrefs.exactMatch and c.xrefs.exactMatch[0].startswith("https://w3id.org/emmo/domain/magnetic-materials#")
        assert c.external_mappings and c.external_mappings[0].vocabulary_id == "magmo" and c.external_mappings[0].mapping_type == "exactMatch"
        assert c.retrieval_policy == "metadata_filter_only"


def test_data_and_unit_classes_are_excluded_and_overlay_applied() -> None:
    concepts = _magmo(load_ontology(CoreConfig().ontology_path))
    ids = {c.id for c in concepts}
    for absent in ("magmo_moke_data", "magmo_edx_counts", "magmo_xrd_two_theta_angles", "magmo_mass_magnetization_unit", "magmo_iec_entry"):
        assert absent not in ids
    ex = next(c for c in concepts if c.id == "magmo_exchange_stiffness_constant")
    assert "heisenberg exchange" in ex.negative_keywords and "magnon stiffness" in ex.description
    mat = next(c for c in concepts if c.id == "magmo_magnetic_material")
    assert mat.avoid_if_matches_too_broadly and "ferromagnet" not in mat.keywords


def test_short_symbols_never_become_keywords() -> None:
    for c in _magmo(load_ontology(CoreConfig().ontology_path)):
        for k in c.keywords:
            assert len(k) >= 6, (c.id, k)
        assert "magnet" not in c.keywords  # bare word would match 'magnetic' everywhere


def test_matching_tags_magnetic_property_facts_without_stealing_neutron_exchange() -> None:
    concepts = load_ontology(CoreConfig().ontology_path)
    text = "The spontaneous magnetization Ms = 3.7 mu_B per Mn and the uniaxial anisotropy constant Ku were reported for the thin film magnet."
    hit = {c.id for c in match_concepts(text, concepts)}
    assert {"magmo_spontaneous_magnetization", "magmo_uniaxial_anisotropy_constant", "magmo_thin_film_magnet"} <= hit
    neutron = "The Heisenberg exchange J1 = 10.3 meV was refined from the spin-wave dispersion; the exchange stiffness constant was not measured."
    hit = {c.id for c in match_concepts(neutron, concepts)}
    assert "magmo_exchange_stiffness_constant" not in hit  # negative keyword from the overlay


def test_registry_entry_and_seed_mappings_validate() -> None:
    cfg = CoreConfig()
    vocab = load_external_vocabularies(cfg.external_vocabularies_path)
    assert any(v.id == "magmo" and v.import_policy == "generated_concepts_from_vendored_copy" for v in vocab)
    mapping_sets = load_external_mappings(cfg.external_mappings_path)
    by_id = {m.local_concept_id: m for m in mapping_sets}
    assert by_id["spinwave_dispersion"].mappings[0].external_label == "ExchangeStiffnessConstant"
    assert all(m.mapping_type == "relatedMatch" for m in by_id["spin_gap"].mappings)
    assert by_id["magnetic_order_parameter"].mappings == [] and "No MagMO target" in (by_id["magnetic_order_parameter"].notes or "")
    magmo_in_structure = [m for m in by_id["magnetic_structure"].mappings if m.vocabulary_id == "magmo"]
    assert magmo_in_structure and magmo_in_structure[0].mapping_type == "broadMatch"
    report = validate_ontology_alignment(vocab, load_ontology(cfg.ontology_path), mapping_sets)
    assert not report.errors, report.errors


def test_vendored_source_note_and_license_present() -> None:
    vendor = REPO / "configs/ontology/vendor/magmo"
    assert (vendor / "LICENSE").read_text().startswith("Attribution 4.0 International")
    note = (vendor / "SOURCE.md").read_text()
    assert "arXiv:2609.11464" in note and "0.0.5" in note
    header = (REPO / "configs/ontology/magmo_generated.yaml").read_text().splitlines()[0]
    assert header.startswith("# GENERATED")


def test_new_fact_types_have_guidance() -> None:
    from tas_copilot_core.corpus.paper_facts import FACT_TYPES
    from tas_copilot_core.corpus.srmn2sb2_agentic_rag import AGENTIC_FACT_TYPE_GUIDANCE
    for ft in ("magnetic_anisotropy", "magnetization", "hysteresis_property"):
        assert ft in FACT_TYPES and ft in AGENTIC_FACT_TYPE_GUIDANCE
