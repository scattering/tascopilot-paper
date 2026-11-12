from tas_copilot_core.config import CoreConfig
from tas_copilot_core.ontology import load_external_vocabularies


REQUIRED_VOCABULARIES = {
    "panet",
    "nexus",
    "nexus_ontology",
    "cif",
    "corecif",
    "magcif",
    "qudt",
    "prov_o",
    "chameo",
    "emmo",
    "mdo",
    "optimade",
    "sosa_ssn",
    "dcat",
    "dcterms",
}


def test_external_vocabulary_registry_loads_required_ids():
    vocabularies = load_external_vocabularies(CoreConfig().external_vocabularies_path)
    ids = {item.id for item in vocabularies}

    assert REQUIRED_VOCABULARIES <= ids
    assert all(item.import_policy for item in vocabularies)
    assert all(item.canonical_url.startswith("http") for item in vocabularies)
