from tas_copilot_contracts import ReductionCorrectionRef, ReductionDAGTemplate, ReductionInputRef
from tas_copilot_core.reduction_dag import (
    constant_q_energy_scan_preview_template,
    execute_reduction_dag,
    order_parameter_fit_branch_template,
    order_parameter_temperature_series_template,
    psd_channel_temperature_preview_template,
    replay_reduction_dag_run,
    reduction_cache_key,
)


def test_reduction_dag_contracts_validate_minimal_template():
    template = order_parameter_temperature_series_template()

    assert isinstance(template, ReductionDAGTemplate)
    assert template.default_target_node_id == "assemble_order_parameter_preview"
    assert "raw inputs are immutable" in template.raw_input_policy


def test_order_parameter_template_executes_preview_and_preserves_raw_inputs():
    template = order_parameter_temperature_series_template()
    inputs = {
        "scan_series": {
            "source_object_type": "scan_group",
            "source_object_id": "group-op-1",
            "scan_ids": ["scan-1", "scan-2", "scan-3"],
            "temperature_values": [5.0, 50.0, 100.0],
        }
    }

    result = execute_reduction_dag(template, inputs=inputs, corrections=[])

    assert result.status == "success"
    assert result.raw_inputs[0].immutable is True
    assert result.node_results[-1].output["preview_type"] == "order_parameter_temperature_series"
    assert result.node_results[-1].output["temperature_range"] == [5.0, 100.0]
    assert result.derived_artifacts[0]["lineage"]["corrections_are_separate_inputs"] is True


def test_psd_channel_temperature_template_blocks_q_claim_without_reviewed_calibration():
    template = psd_channel_temperature_preview_template()
    inputs = {
        "psd_scan": {
            "source_object_type": "scan",
            "source_object_id": "scan-psd-1",
            "temperature_values": [2.6, 10.0, 200.0],
            "psd_counts": [[10, 11, 12], [20, 21, 22], [30, 31, 32]],
        }
    }
    corrections = [
        ReductionCorrectionRef(
            correction_id="psd_efficiency",
            correction_type="psd_efficiency",
            status="placeholder",
            stale_ok_for_preview=True,
            parameters={"note": "synthetic fixture"},
        ),
        ReductionCorrectionRef(
            correction_id="psd_channel_to_a4",
            correction_type="psd_channel_to_a4",
            status="proposed",
            stale_ok_for_preview=True,
            parameters={"note": "not accepted"},
        ),
    ]

    result = execute_reduction_dag(template, inputs=inputs, corrections=corrections)

    assert result.status == "success"
    preview = result.node_results[-1].output
    assert preview["preview_type"] == "psd_channel_temperature"
    assert preview["shape"] == [3, 3]
    assert preview["q_temperature_status"] == "blocked_without_reviewed_channel_to_a4_calibration"
    assert "not a Q x temperature map" in " ".join(preview["warnings"])


def test_constant_q_energy_scan_template_executes_preview_without_fit_claim():
    template = constant_q_energy_scan_preview_template()
    inputs = {
        "energy_scan_series": {
            "source_object_type": "scan_group",
            "source_object_id": "group-e-1",
            "scan_ids": ["scan-e-1", "scan-e-2"],
            "q_values": [[1.85, -2.15, 0.0], [1.8, -2.2, 0.0]],
            "energy_values": [2.0, 17.0],
        }
    }

    result = execute_reduction_dag(template, inputs=inputs, corrections=[])

    assert result.status == "success"
    preview = result.node_results[-1].output
    assert preview["preview_type"] == "constant_q_energy_scan"
    assert preview["energy_range"] == [2.0, 17.0]
    assert preview["fit_status"] == "blocked_until_background_resolution_and_model_are_reviewed"
    assert result.derived_artifacts[0]["lineage"]["corrections_are_separate_inputs"] is True


def test_cache_key_changes_when_correction_version_changes():
    template = psd_channel_temperature_preview_template()
    node = next(item for item in template.nodes if item.node_id == "psd_efficiency_correction")
    inputs = {
        "psd_scan": {
            "source_object_type": "scan",
            "source_object_id": "scan-psd-1",
            "temperature_values": [2.6],
            "psd_counts": [[1, 2]],
        }
    }
    correction_v1 = {
        "psd_efficiency": ReductionCorrectionRef(
            correction_id="psd_efficiency",
            correction_type="psd_efficiency",
            status="placeholder",
            version="v1",
            parameters={"scale": [1.0, 0.9]},
        )
    }
    correction_v2 = {
        "psd_efficiency": ReductionCorrectionRef(
            correction_id="psd_efficiency",
            correction_type="psd_efficiency",
            status="placeholder",
            version="v2",
            parameters={"scale": [1.0, 0.9]},
        )
    }

    assert reduction_cache_key(template, node, inputs=inputs, corrections=correction_v1) != reduction_cache_key(
        template,
        node,
        inputs=inputs,
        corrections=correction_v2,
    )


def test_missing_required_scan_metadata_fails_closed():
    template = order_parameter_temperature_series_template()

    result = execute_reduction_dag(template, inputs={"scan_series": {"scan_ids": ["scan-1"]}}, corrections=[])

    assert result.status == "missing_input"
    assert "input:scan_series.temperature_values" in result.missing_inputs


def test_reduction_input_ref_can_be_supplied_without_mutating_raw_source():
    template = order_parameter_temperature_series_template()
    raw_ref = ReductionInputRef(
        input_id="scan_series",
        source_object_type="scan_group",
        source_object_id="group-op-1",
        content_hash="raw-hash",
        metadata={"scan_ids": ["scan-1"], "temperature_values": [10.0]},
    )
    inputs = {
        "scan_series": {
            "source_object_type": raw_ref.source_object_type,
            "source_object_id": raw_ref.source_object_id,
            "scan_ids": ["scan-1"],
            "temperature_values": [10.0],
        }
    }

    result = execute_reduction_dag(template, inputs=inputs, corrections=[])

    assert result.status == "success"
    assert result.raw_inputs[0].source_object_id == "group-op-1"
    assert result.raw_inputs[0].immutable is True


def test_order_parameter_fit_branch_records_model_branch_and_bic_tag():
    template = order_parameter_fit_branch_template(model_branch="single_peak_power_law", bic_tag="bic_not_computed")
    result = execute_reduction_dag(
        template,
        inputs={
            "scan_series": {
                "source_object_type": "scan_group",
                "source_object_id": "group-op-1",
                "scan_ids": ["scan-1", "scan-2", "scan-3", "scan-4"],
                "temperature_values": [10.0, 50.0, 100.0, 150.0],
            },
            "fit_evidence": {"model_branch": "single_peak_power_law", "bic_tag": "bic_not_computed"},
        },
        corrections=[],
    )

    assert result.status == "success"
    assert result.target_node_id == "order_parameter_model_branch"
    branch = result.node_results[-1].output
    assert branch["model_branch"] == "single_peak_power_law"
    assert branch["bic_tag"] == "bic_not_computed"
    assert result.derived_artifacts[-1]["artifact_type"] == "model_branch"


def test_replay_reduction_dag_run_reports_changed_cache_keys():
    template = order_parameter_temperature_series_template()
    inputs = {
        "scan_series": {
            "source_object_type": "scan_group",
            "source_object_id": "group-op-1",
            "scan_ids": ["scan-1", "scan-2", "scan-3", "scan-4"],
            "temperature_values": [10.0, 50.0, 100.0, 150.0],
        }
    }
    prior = execute_reduction_dag(template, inputs=inputs, corrections=[])

    replay = replay_reduction_dag_run(
        prior.model_dump(mode="json"),
        template=template,
        inputs=inputs,
        corrections=[],
        software_versions={"workflow_version": "changed"},
    )

    assert replay["prior_run_id"] == prior.run_id
    assert replay["changed_node_ids"]
