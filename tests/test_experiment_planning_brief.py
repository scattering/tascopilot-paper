from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_core.cli import main
from tas_copilot_core.workflows.experiment_planning_brief import _parse_scan_recipe, experiment_planning_brief


def test_parse_scan_recipe_uses_header_scan_descr_ranges() -> None:
    recipe = _parse_scan_recipe(
        "Scan:SubID=18546:JType=VECTOR:Fixed=1:FixedE=14.7:Npts=27:Counts=145000.0:"
        "Prefac=2.0:DetectorType=Detector:CountType=Monitor:Filename=LSMO_220_Q0p15_:"
        "Range=Q=1.85~-2.15~0.0 1.85~-2.15~0.0 s:Range=E=2.0 15.0 s:Title=2m20_Q0p15"
    )

    assert recipe["jtype"] == "VECTOR"
    assert recipe["fixed_e"] == 14.7
    assert recipe["npts"] == 27
    assert recipe["counts"] == 145000.0
    assert recipe["count_type"] == "Monitor"
    assert recipe["title"] == "2m20_Q0p15"
    assert recipe["range_axes"] == ["E", "Q"]
    q_range = next(item for item in recipe["ranges"] if item["axis"] == "Q")
    e_range = next(item for item in recipe["ranges"] if item["axis"] == "E")
    assert q_range["first_point"] == [1.85, -2.15, 0.0]
    assert e_range["start"] == 2.0
    assert e_range["stop"] == 15.0


def test_parse_scan_recipe_handles_psd_temperature_and_a4_recipes() -> None:
    temp_recipe = _parse_scan_recipe(
        "Scan:SubID=18445:JType=ENV:Npts=81:Counts=200000.0:Prefac=1.0:"
        "DetectorType=Detector:CountType=Monitor:Filename=LuMnFeO3_29pt5_TdepWarm:"
        "Range=Temp=2.5 202.5 s:Title=TscanInc"
    )
    a4_recipe = _parse_scan_recipe(
        "Scan:SubID=18445:JType=ANGLE:Npts=29:Counts=100000.0:Prefac=1.0:"
        "DetectorType=Detector:CountType=Monitor:Filename=LuFe75Mn25O3_Correlations:"
        "Range=A4=25.0 32.0 s:Title=Corr"
    )

    assert temp_recipe["jtype"] == "ENV"
    assert temp_recipe["range_axes"] == ["Temp"]
    assert temp_recipe["ranges"][0]["start"] == 2.5
    assert temp_recipe["ranges"][0]["stop"] == 202.5
    assert a4_recipe["jtype"] == "ANGLE"
    assert a4_recipe["range_axes"] == ["A4"]
    assert a4_recipe["ranges"][0]["start"] == 25.0
    assert a4_recipe["ranges"][0]["stop"] == 32.0


def test_experiment_planning_brief_composes_precedent_and_replay(tmp_path: Path) -> None:
    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "experiment_id": "exp-lsmo",
                "title": "Summer school LSMO spin waves",
                "datafile_count": 12,
                "scan_count": 12,
                "path_clues": {"top_filename_prefixes": [{"value": "LSMO_220_Q0p15"}]},
                "hydrated_clue_summary": {
                    "top_values": {
                        "fixed_e": [{"value": "Ef 14.7", "count": 12}],
                        "temp_mean": [{"value": "300.0", "count": 5}, {"value": "345.0", "count": 4}],
                        "scan_variable": [{"value": "E", "count": 6}],
                    },
                    "scan_descr_values": {
                        "scan_description": [
                            {
                                "value": "Scan:SubID=18546:JType=VECTOR:Fixed=1:FixedE=14.7:Npts=27:Counts=145000.0:"
                                "DetectorType=Detector:CountType=Monitor:Filename=LSMO_220_Q0p15_:"
                                "Range=Q=1.85~-2.15~0.0 1.85~-2.15~0.0 s:Range=E=2.0 15.0 s:Title=2m20_Q0p15"
                            },
                            {"value": "Scan:SubID=18546:JType=VECTOR:Range=Q=0.1 0.3 transverse:Title=q-scan"},
                        ]
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replay = tmp_path / "replay.json"
    replay.write_text(
        json.dumps(
            {
                "status": "success",
                "scan_path": "synthetic.bt7",
                "scan_metadata": {"Filename": "synthetic", "InstrName": "BT7"},
                "measurement": {
                    "points": [
                        {"move_status": "success", "count_status": "success"},
                        {"move_status": "success", "count_status": "success"},
                    ]
                },
                "file_write": {"created_files": ["a.bt7"], "created_previews": {"a.bt7": ["# header"]}},
                "instrument_dag": {"node_count": 10, "control_input_count": 4, "readback_only_count": 6, "detector_or_counter_node_count": 2},
            }
        ),
        encoding="utf-8",
    )

    result = experiment_planning_brief(
        {
            "material": "La0.7Sr0.3MnO3",
            "goal": "measure spin waves and extract stiffness",
            "precedent_index": str(index),
            "replay_report": str(replay),
            "output_dir": str(tmp_path / "out"),
        }
    )

    assert result["status"] == "ok"
    assert result["evidence_channels"]["scan_precedents"]["candidate_count"] == 1
    assert result["evidence_channels"]["scan_precedents"]["user_facing_top_patterns"][0]["pattern_id"] == "precedent_pattern_01"
    assert "raw experiment identity" in result["evidence_channels"]["scan_precedents"]["user_facing_top_patterns"][0]["privacy_note"]
    candidate = result["evidence_channels"]["scan_precedents"]["top_candidates"][0]
    assert candidate["scan_recipe_examples"][0]["jtype"] == "VECTOR"
    assert {"Q", "E"}.issubset(set(candidate["scan_recipe_examples"][0]["range_axes"]))
    assert result["evidence_channels"]["replay"]["move_and_count_success_count"] == 2
    assert result["recommendations"]["scan_plan"][1]["suggested_offsets_rlu"]
    assert result["recommendations"]["precedent_explanation_policy"]["user_facing_patterns"]
    assert result["not_approved_operations"]
    assert Path(result["artifacts"]["json"]).exists()
    assert Path(result["artifacts"]["markdown"]).exists()


def test_experiment_planning_brief_cli(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(json.dumps({"status": "success", "measurement": {"points": []}, "file_write": {}, "instrument_dag": {}}), encoding="utf-8")

    code = main(
        [
            "workflow",
            "experiment-planning-brief",
            "--material",
            "LuFeMnO3",
            "--goal",
            "elastic PSD temperature map triage",
            "--replay-report",
            str(replay),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert code == 0
    payload = json.loads((tmp_path / "out" / "experiment_planning_brief.json").read_text(encoding="utf-8"))
    assert payload["workflow_name"] == "experiment-planning-brief"
    assert payload["evidence_channels"]["replay"]["simulation_only"] is True
