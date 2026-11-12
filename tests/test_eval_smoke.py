from tas_copilot_core.workflows.eval_smoke import eval_smoke


def test_eval_smoke_exercises_retrieval_ontology_similarity_and_mappings():
    result = eval_smoke()

    assert result["status"] == "pass"
    assert result["question_count"] == 5
    assert result["mapping_check_count"] == 2
    assert result["regression_check_count"] == 7
    assert len(result["retrieval_regressions"]) == 4
    assert result["failed_count"] == 0
    assert result["catalog_counts"]["chunks"] >= 2
    assert result["catalog_counts"]["scans"] >= 2
    assert result["questions"][0]["matched_concepts"]
    assert "paper-bt7-spinwave" in result["questions"][0]["paper_result_ids"]
    assert result["mappings"][0]["status"] == "pass"
    assert result["scan_similarity"][0]["candidate_scan_id"] == "scan-002"
    assert result["scan_similarity"][0]["similarity_score"] > 0.7
    cross_instrument = next(item for item in result["questions"] if "cross-instrument" in item["question"])
    assert "cross_instrument_tas_catalog" in cross_instrument["matched_concepts"]
    mapping_question = next(item for item in result["questions"] if "paper experiment mapping" in item["question"])
    assert "paper_experiment_mapping" in mapping_question["matched_concepts"]
    path_only_quality = next(item for item in result["regressions"] if item["name"] == "path_only_catalog_quality_warning")
    assert path_only_quality["status"] == "pass"
    assert path_only_quality["details"]["metadata_modes"] == {"path_only": 1}
    path_only_geometry = next(item for item in result["regressions"] if item["name"] == "path_only_geometry_safe_failure")
    assert path_only_geometry["actual_status"] == "missing_input"
    complete_geometry = next(item for item in result["regressions"] if item["name"] == "complete_lattice_geometry_reaches_registry")
    assert complete_geometry["actual_status"] in {"unavailable", "success"}
    assert complete_geometry["details"]["missing_fields"] == []
    assert complete_geometry["details"]["geometry_mode"] == "lattice"
    assert complete_geometry["details"]["result_tools"] == ["device_angles", "resolution", "spurions"]
    compatible_ub = next(item for item in result["regressions"] if item["name"] == "compatible_ub_geometry_fails_closed")
    assert compatible_ub["actual_status"] == "unavailable"
    assert compatible_ub["details"]["error_codes"] == ["unsupported_geometry_convention"]
    scan_brief = next(item for item in result["regressions"] if item["name"] == "scan_brief_writes_reports")
    assert scan_brief["details"]["artifact_json_exists"] is True
    experiment_brief = next(item for item in result["regressions"] if item["name"] == "experiment_brief_writes_reports")
    assert experiment_brief["details"]["paper_mapping_count"] == 1
    assert all(item["status"] == "pass" for item in result["retrieval_regressions"])
