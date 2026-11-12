"""Policy documents: contract, derivation from a settings file, validation
against the instrument document and the ledger, the immutable store, and the
operator control that requests one."""
import importlib
import json
import os
from pathlib import Path

import pytest

from tas_copilot_core.workflows import andie_control as control
from tas_copilot_core.workflows import andie_policy as ap

ROOT = Path(__file__).resolve().parents[1]
settings_path = os.environ.get("TAS_COPILOT_SITE_SETTINGS")
if not settings_path or not Path(settings_path).is_file():
    pytest.skip("Site-policy integration requires explicit TAS_COPILOT_SITE_SETTINGS and its external analysis runtime", allow_module_level=True)
SETTINGS = json.loads(Path(settings_path).read_text())


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("andie_science")


def derived(registry, **changes):
    document = ap.policy_from_settings(SETTINGS, registry.resolve_model).model_dump(mode="json")
    document.update(changes)
    return document


def test_revision_zero_is_derived_from_the_settings(registry):
    policy = ap.policy_from_settings(SETTINGS, registry.resolve_model)
    assert policy.measured == ("001", "101")
    assert policy.science_pairs == [("001", "001")]          # the settings' model is 001 alone
    assert policy.diagnostics.reflections == ("100", "102") and policy.diagnostics.every_n_temperatures == 5
    assert policy.model.id == "tc_brillouin_001_v1"
    assert set(policy.prior) == {"Tc", "J", "amplitude_001", "background_001"}
    assert policy.sampler["burn"] == 1000
    assert policy.acquisition.eta == 0.7 and policy.acquisition.max_step_k == 25.0
    assert policy.acquisition.implementation == ap.DEFAULT_ACQUISITION
    ap.validate_policy(policy, SETTINGS, registry.resolve_model, known_acquisitions={ap.DEFAULT_ACQUISITION},
                       max_step_cap=control.MAX_STEP_K)


def test_hash_ignores_the_note_and_id_but_nothing_else(registry):
    base = ap.PolicyDocument.model_validate(derived(registry))
    renamed = ap.PolicyDocument.model_validate(derived(registry, policy_id="other", note="a remark"))
    assert renamed.hash == base.hash
    narrower = ap.PolicyDocument.model_validate(derived(registry, measured=["001"]))
    assert narrower.hash != base.hash
    tuned = derived(registry)
    tuned["acquisition"]["eta"] = 0.5
    assert ap.PolicyDocument.model_validate(tuned).hash != base.hash


@pytest.mark.parametrize("change, match", [
    ({"measured": ["101"]}, "belong to a measured reflection"),
    ({"measured": ["001", "001"]}, "distinct"),
    ({"measured": ["001", "100"]}, "both measured for science and diagnostic"),
    ({"schema_version": 2}, "schema_version"),
])
def test_internal_consistency_is_checked_by_the_contract(registry, change, match):
    with pytest.raises(ValueError, match=match):
        ap.PolicyDocument.model_validate(derived(registry, **change))


def test_validation_against_the_instrument_document(registry):
    with pytest.raises(ValueError, match="not in the scan catalog"):
        ap.validate_policy(derived(registry, measured=["001", "011"]), SETTINGS, registry.resolve_model)
    with pytest.raises(ValueError, match="not a diagnostic scan"):
        ap.validate_policy(derived(registry, diagnostics={"reflections": ["103"]}), SETTINGS, registry.resolve_model)
    # the science set must be exactly what the model predicts
    both = derived(registry, science=[{"reflection": "001", "component": "001"},
                                     {"reflection": "101", "component": "101_central"}])
    with pytest.raises(ValueError, match="predicts"):
        ap.validate_policy(both, SETTINGS, registry.resolve_model)
    both["model"] = {"id": "shared_tc_brillouin_pair_v1", "options": {}}
    with pytest.raises(ValueError, match="prior does not cover"):
        ap.validate_policy(both, SETTINGS, registry.resolve_model)
    both["prior"] = SETTINGS["inference"]["prior"]
    policy = ap.validate_policy(both, SETTINGS, registry.resolve_model)
    assert policy.science_labels == ("001", "101_central")
    with pytest.raises(ValueError, match="unknown acquisition"):
        ap.validate_policy(policy, SETTINGS, registry.resolve_model, known_acquisitions={"fixed_step"})
    big = derived(registry)
    big["acquisition"]["max_step_k"] = 40.0
    big["acquisition"]["fallback_step_k"] = 30.0
    with pytest.raises(ValueError, match="hard cap"):
        ap.validate_policy(big, SETTINGS, registry.resolve_model, max_step_cap=25.0)


def test_the_generic_model_takes_any_component_set(registry):
    document = derived(registry, measured=["001", "101"],
                       science=[{"reflection": "001", "component": "001"},
                                {"reflection": "101", "component": "101_central"}],
                       model={"id": "tc_brillouin_v1", "options": {"components": ["001", "101_central"]}})
    prior = SETTINGS["inference"]["prior"]
    document["prior"] = {"Tc": prior["Tc"], "J": prior["J"], "amplitude_001": prior["amplitude_001"],
                         "background_001": prior["background_001"],
                         "amplitude_101_central": prior["amplitude_101"], "background_101_central": prior["background_101"]}
    policy = ap.validate_policy(document, SETTINGS, registry.resolve_model)
    spec = registry.resolve_model(policy.model.model_dump())
    assert spec.parameters == ("Tc", "J", "amplitude_001", "background_001", "amplitude_101_central", "background_101_central")
    assert spec.observables == ("001", "101_central")


def test_changed_component_definitions_are_a_new_run_not_a_revision(registry):
    policy = derived(registry)
    ledger = {"001": "abc"}
    with pytest.raises(ValueError, match="start a new run"):
        ap.validate_policy(policy, SETTINGS, registry.resolve_model, ledger_definitions=ledger,
                           definition_hash=lambda reflection: "def")
    ap.validate_policy(policy, SETTINGS, registry.resolve_model, ledger_definitions=ledger,
                       definition_hash=lambda reflection: "abc")


def test_store_keeps_immutable_revisions_and_loads_by_hash(registry, tmp_path):
    store = ap.PolicyStore(tmp_path)
    first = store.save(derived(registry), requested_by="test")
    second = store.save(derived(registry, measured=["001"], policy_id="only-001"))
    again = store.save(derived(registry, note="same content, new note"))
    assert (first["revision"], second["revision"], again["revision"]) == (0, 1, 0)
    assert [r["revision"] for r in store.records()] == [0, 1]
    assert store.load(second).measured == ("001",) and store.load(second).policy_id == "only-001"
    assert store.load(first).hash == first["hash"]
    path = Path(store.records()[1]["path"])
    stored = json.loads(path.read_text())
    stored["measured"] = ["001", "101"]            # tampering is caught by the hash
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="does not match its hash"):
        store.load(second)
    with pytest.raises(FileNotFoundError):
        store.load({"revision": 7, "hash": "0" * 64})


def test_set_policy_validates_stores_and_references(registry, tmp_path):
    with pytest.raises(ValueError, match="no active run"):
        control.set_policy(tmp_path, derived(registry))
    control.start_run(tmp_path, start_k=110, stop_k=200)
    ctl = control.set_policy(tmp_path, derived(registry, policy_id="first"), settings=SETTINGS,
                             resolve_model=registry.resolve_model)
    assert ctl["policy"]["revision"] == 0 and ctl["policy"]["policy_id"] == "first"
    with pytest.raises(ValueError, match="not in the scan catalog"):
        control.set_policy(tmp_path, derived(registry, measured=["001", "011"]), settings=SETTINGS,
                           resolve_model=registry.resolve_model)
    assert control.read_control(tmp_path)["policy"]["revision"] == 0     # the rejected one changed nothing
    ctl = control.set_policy(tmp_path, derived(registry, measured=["001"], policy_id="second"))
    assert ctl["policy"]["revision"] == 1
    assert ap.PolicyStore(tmp_path).load(ctl["policy"]).measured == ("001",)
    log = [json.loads(line) for line in (tmp_path / "control_log.jsonl").read_text().splitlines()]
    assert log[-1]["action"].startswith("policy revision 1")
