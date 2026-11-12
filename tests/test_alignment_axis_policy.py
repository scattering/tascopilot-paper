from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_core.workflows.alignment_axis_policy import apply_axis_policy, canonical_axis, load_axis_policy
from tas_copilot_core.workflows.alignment_state import alignment_fine_replay_plan, alignment_state_machine


def _policy(tmp_path: Path) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(
        "mode: reversible_only\n"
        "frozen_axes:\n"
        "  - axis: preSampleWidth\n    reason: readback reset\n"
        "  - axis: BkSltHght\n    reason: inelastic run\n"
        "  - axis: sgu\n    reason: not reversible\n"
        "  - axis: noSuchMotor\n    reason: typo\n",
        encoding="utf-8",
    )
    return path


def test_policy_aliases_resolve_and_unknown_axes_are_visible(tmp_path: Path) -> None:
    policy = load_axis_policy(_policy(tmp_path))
    assert policy["mode"] == "reversible_only"
    assert policy["frozen"]["pre_sample_slit_width"] == "readback reset"
    assert policy["frozen"]["post_sample_slit_height"] == "inelastic run"
    assert policy["frozen"]["upper_tilt"] == "not reversible"
    assert policy["unknown_axes"] == ["noSuchMotor"]
    assert canonical_axis("FrontSlitWidth") == "pre_sample_slit_width" == canonical_axis("preSampleSlitWidth")


def test_steps_on_frozen_axes_are_skipped_with_reason(tmp_path: Path) -> None:
    policy = load_axis_policy(_policy(tmp_path))
    steps = [
        {"step_id": "rock", "operation": "scan_motor", "motor": "A3"},
        {"step_id": "vslit", "operation": "scan_slit_pair", "slits": ["FrontSlitHeight", "BackSlitHeight"]},
        {"step_id": "tilt", "operation": "scan_motor", "motor": "SmplUTilt"},
    ]
    kept, skipped = apply_axis_policy(steps, policy)
    assert [s["step_id"] for s in kept] == ["rock"]
    assert [(s["step_id"], s["reason"]) for s in skipped] == [("vslit", "inelastic run"), ("tilt", "not reversible")]


def test_checked_in_bt7_policy_loads() -> None:
    policy = load_axis_policy(Path(__file__).resolve().parents[1] / "configs/alignment/axis_policy_bt7_202608_14.yaml")
    assert policy["unknown_axes"] == []
    assert set(policy["frozen"]) == {"pre_sample_slit_width", "pre_sample_slit_height", "post_sample_slit_width", "post_sample_slit_height", "lower_translation", "upper_translation"}


def test_fine_replay_plan_respects_axis_policy(tmp_path: Path) -> None:
    report = alignment_fine_replay_plan({"axis_policy": str(_policy(tmp_path)), "output_dir": str(tmp_path / "plan")})
    step_ids = [s["step_id"] for s in report["replay_plan"]["steps"]]
    assert step_ids == ["rocking_preview", "theta_two_theta_check"]
    assert {s["step_id"] for s in report["steps_skipped_by_axis_policy"]} == {"vertical_slit_preview", "horizontal_slit_preview"}
    assert report["axis_policy"]["frozen_axes"]["post_sample_slit_height"] == "inelastic run"


def test_state_machine_marks_checks_blocked_by_policy(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps({"event_id": "scan:slit", "event_type": "planned_scan", "source_type": "bt7_file", "date": "2026-09-12", "scanned_devices": ["BkSltHght"],
                    "alignment_stage": "slit_translation_optimization", "alignment_intent": "slit", "automation_level": "analysis_only",
                    "evidence_strength": "strong", "state_epoch_effect": "observes_or_refines_current_epoch", "warnings": []}) + "\n",
        encoding="utf-8",
    )
    report = alignment_state_machine({"events_jsonl": str(events), "axis_policy": str(_policy(tmp_path)), "output_dir": str(tmp_path / "sm")})
    assert report["axis_policy"]["frozen_axes"]["post_sample_slit_height"] == "inelastic run"
    actions = [a["action"] for a in report["recommended_next_actions"]]
    assert "respect_axis_policy" in actions
    blocked = [b for est in report["state_estimates"] for b in est.get("checks_blocked_by_axis_policy", [])]
    assert blocked and "slit" in blocked[0]["check"]
