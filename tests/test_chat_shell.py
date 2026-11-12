import json

from tas_copilot_core.chat_shell import ChatShell, SESSION_ARG_FIELDS, _fields_for_missing_input
from tas_copilot_core.llm_router import build_intent_menu, build_system_prompt


def make_shell(tmp_path, **kwargs):
    return ChatShell(tmp_path / "session", **kwargs)


def test_status_question_routes_and_renders(tmp_path):
    shell = make_shell(tmp_path)
    reply = shell.handle("Where are we high level, and are we ready for an agent?")
    assert "copilot_status" in reply
    assert "status: ok" in reply
    assert shell.pending is None
    state = json.loads((tmp_path / "session" / "session.json").read_text())
    assert state["turns"][0]["routed_intent"] == "copilot_status"


def test_missing_input_then_slot_fill_runs_workflow(tmp_path):
    shell = make_shell(tmp_path)
    reply = shell.handle("What scans should I try first for alignment and the 003 peak?")
    assert shell.pending is not None
    assert "I still need" in reply
    reply = shell.handle("fixture=tests/fixtures/proposal_retrospective/ansto_dto_thin_film")
    assert shell.pending is None
    assert "status: ok" in reply
    assert "not an approved run list" in reply.lower()
    # the supplied input persists for later turns
    assert shell.defaults["fixture"].endswith("ansto_dto_thin_film")


def test_bare_value_fills_single_field_missing_input(tmp_path):
    shell = make_shell(tmp_path)
    shell.handle("Could this narrow feature be a spurion or hot pixel?")
    assert shell.pending is not None
    assert shell._still_missing() == ["data_root"]
    # a bare path (no key=) is accepted when exactly one field can satisfy it
    data_dir = tmp_path / "experiment"
    data_dir.mkdir()
    reply = shell.handle(str(data_dir))
    assert shell.pending is None
    assert shell.defaults["data_root"] == str(data_dir)
    assert "status:" in reply


def test_invalid_path_is_rejected_and_cleared_not_looped(tmp_path):
    shell = make_shell(tmp_path)
    shell.handle("Could this narrow feature be a spurion or hot pixel?")
    reply = shell.handle(str(tmp_path / "no_such_experiment"))
    # the router re-reported data_root missing: the bad value must be dropped
    assert "not accepted" in reply
    assert "data_root" not in shell.defaults
    assert shell.pending is not None
    assert shell._still_missing() == ["data_root"]


def test_live_control_is_blocked_regardless_of_proposal(tmp_path):
    shell = make_shell(tmp_path, propose_intent=lambda question: {"intent": "experiment_review"})
    reply = shell.handle("Move motor A3 and start scan now.")
    assert "blocked_live_control" in reply
    assert "cannot execute" in reply.lower()


def test_proposer_failure_falls_back_to_deterministic(tmp_path):
    def broken(question):
        raise OSError("ollama unreachable")

    shell = make_shell(tmp_path, propose_intent=broken)
    reply = shell.handle("Where are we high level, and are we ready for an agent?")
    assert "copilot_status" in reply
    assert shell.turns[0]["llm_error"] == "ollama unreachable"


def test_unknown_proposed_intent_is_not_forced(tmp_path):
    shell = make_shell(tmp_path, propose_intent=lambda question: {"intent": "needs_more_context"})
    reply = shell.handle("Where are we high level, and are we ready for an agent?")
    # control intents are not forced through copilot_ask; deterministic routing answers
    assert "copilot_status" in reply


def test_cancel_clears_pending(tmp_path):
    shell = make_shell(tmp_path)
    shell.handle("Could this narrow feature be a spurion or hot pixel?")
    assert shell.pending is not None
    reply = shell.handle("cancel")
    assert shell.pending is None
    assert "Cancelled" in reply


def test_set_command_and_session_summary(tmp_path):
    shell = make_shell(tmp_path)
    reply = shell.handle("set data_root /tmp/example_experiment")
    assert "Set data_root" in reply
    state = json.loads(shell.handle("session"))
    assert state["defaults"]["data_root"] == "/tmp/example_experiment"


def test_narration_is_labeled_and_envelope_fed(tmp_path):
    seen = {}

    def narrator(question, envelope):
        seen["envelope"] = envelope
        return "short summary"

    shell = make_shell(tmp_path, narrate=narrator)
    reply = shell.handle("Where are we high level, and are we ready for an agent?")
    assert "model narration" in reply
    assert "short summary" in reply
    assert seen["envelope"].get("uncertainty_class")


def test_missing_input_alias_maps_to_session_fields():
    assert _fields_for_missing_input("data_root") == ["data_root"]
    fields = _fields_for_missing_input("fixture_or_proposal_evidence_dir")
    assert "fixture" in fields and "proposal_evidence_dir" in fields


def test_intent_menu_and_prompt_cover_all_examples():
    menu = build_intent_menu()
    names = [item["intent"] for item in menu]
    assert "blocked_live_control" in names and "needs_more_context" in names
    prompt = build_system_prompt(menu)
    for name in names:
        assert name in prompt
    assert "untrusted evidence" in prompt
    assert set(SESSION_ARG_FIELDS) >= {"data_root", "corpus_catalog", "fixture"}
