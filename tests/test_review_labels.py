from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tas_copilot_contracts import ProvenanceRecord, ReviewLabel, ReviewLabelBatch
from tas_copilot_core.catalogs import ReviewOutcomeSQLiteStore
from tas_copilot_core.cli import main
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.ontology import load_ontology
from tas_copilot_core.ontology.mappings import load_external_mappings
from tas_copilot_core.review_labels import backfill_review_labels, label_calibrated_score, label_informed_priors, review_label_calibration_diagnostics, stale_review_label_check
from tas_copilot_core.workflows.experiment_dashboard import _experiment_review_row_to_label


def test_review_label_contract_captures_dag_and_cache_versioning():
    label = ReviewLabel(
        object_type="order_parameter_candidate",
        object_id="candidate-1",
        source_workflow="fit-order-parameter",
        source_run_id="run-1",
        source_artifact_path="local_runs/review/evidence.json",
        source_artifact_hash="sha256:abc",
        verdict="positive",
        type_specific_status="fit_accepted_for_review",
        type_specific_allowed=["fit_accepted_for_review", "model_mismatch"],
        claim_state="candidate",
        evidence_channels_relied_on=["evidence_channel_reconstruction"],
        recommendation="calibrate_prior",
        uncertainty_class="source_grounded_candidate_evidence",
        workflow_version="fit-order-parameter-v0",
        tool_versions={"tas_copilot_core": "0.1"},
        calibration_state_hash="sha256:cal",
        correction_status_snapshot={"monitor_normalization": "available"},
        dag_run_id="dag-run-1",
        cache_key="cache-1",
        cache_scope="model_branch",
        reviewer_id="reviewer",
        reviewed_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        provenance=[ProvenanceRecord(source_type="workflow", source_id="run-1")],
    )

    assert label.cache_key == "cache-1"
    assert label.source_artifact_hash == "sha256:abc"
    assert label.correction_status_snapshot["monitor_normalization"] == "available"


def test_review_label_rejects_status_not_in_type_specific_allowed():
    with pytest.raises(ValueError):
        ReviewLabel(
            object_type="spurion_candidate",
            object_id="candidate-2",
            source_workflow="survey-inelastic-spurion-evidence",
            type_specific_status="likely_spurion",
            type_specific_allowed=["likely_real_feature"],
        )


def test_review_outcome_store_round_trips_batches_and_summarizes(tmp_path):
    label = ReviewLabel(
        label_id="label-1",
        object_type="paper_experiment_link",
        object_id="paper-1",
        object_secondary_id="experiment-1",
        source_workflow="link-paper-to-experiments",
        source_artifact_hash="sha256:source",
        verdict="negative",
        type_specific_status="not_enough_information",
        type_specific_allowed=["accepted", "rejected", "not_enough_information"],
        claim_state="rejected",
        evidence_channels_relied_on=["evidence_channel_paper_text", "evidence_channel_hydrated_metadata"],
        counter_evidence_observed=["counter_evidence_calibration_unreviewed"],
        recommendation="downweight",
        uncertainty_class="probability_style_triage_prior",
        dag_run_id="dag-1",
        cache_key="cache-1",
        cache_scope="full_dag",
    )
    batch = ReviewLabelBatch(batch_id="batch-1", source_template_paths=["template.jsonl"], labels=[label])
    store = ReviewOutcomeSQLiteStore(tmp_path / "review_outcomes.sqlite")

    store.upsert_batch(batch)
    labels = store.list_labels(object_type="paper_experiment_link", verdict="negative")
    summary = store.summary()

    assert labels == [label]
    assert summary["label_count"] == 1
    assert summary["batch_count"] == 1
    assert summary["reviewed_label_count"] == 1
    assert summary["by_object_type"] == {"paper_experiment_link": 1}
    assert summary["by_verdict"] == {"negative": 1}


def test_review_label_cli_summary_export_packet_and_stale_check(tmp_path, capsys):
    artifact = tmp_path / "evidence.json"
    artifact.write_text('{"old": true}', encoding="utf-8")
    label = ReviewLabel(
        label_id="label-cli-1",
        object_type="order_parameter_candidate",
        object_id="candidate-1",
        source_workflow="order-parameter-triage",
        source_artifact_path=str(artifact),
        source_artifact_hash="sha256:old",
        verdict="unreviewed",
        claim_state="proposed",
        dag_run_id="dag-old",
        cache_key="cache-old",
    )
    store_path = tmp_path / "review_outcomes.sqlite"
    ReviewOutcomeSQLiteStore(store_path).upsert_labels([label])

    assert main(["review", "labels", "summary", "--store", str(store_path), "--output-dir", str(tmp_path / "summary")]) == 0
    summary_out = capsys.readouterr().out
    assert "review-label-summary" in summary_out
    assert (tmp_path / "summary" / "review_label_summary.md").exists()

    export_path = tmp_path / "labels.jsonl"
    assert main(["review", "labels", "export", "--store", str(store_path), "--output", str(export_path), "--object-type", "order_parameter_candidate"]) == 0
    assert export_path.exists()
    assert "candidate-1" in export_path.read_text(encoding="utf-8")

    packet_dir = tmp_path / "packet"
    assert main(["review", "labels", "packet", "--store", str(store_path), "--output-dir", str(packet_dir), "--limit", "5"]) == 0
    assert (packet_dir / "review_label_packet.csv").exists()

    stale = stale_review_label_check({"store": str(store_path), "cache_key": "cache-new"})
    assert stale["stale_label_count"] == 1
    assert "source_artifact_hash_changed" in stale["results"][0]["stale_reasons"]
    assert "cache_key_changed" in stale["results"][0]["stale_reasons"]


def test_review_label_cli_ingests_paper_link_dashboard_export(tmp_path, capsys):
    source_report = tmp_path / "paper_link_report.json"
    source_report.write_text('{"workflow": "link-paper-to-experiments"}', encoding="utf-8")
    exported = tmp_path / "reviewed_rows.jsonl"
    exported.write_text(
        json.dumps(
            {
                "paper_id": "paper-1",
                "paper_title": "BT-7 paper",
                "candidate_experiment_id": "exp-1",
                "candidate_rank": 1,
                "candidate_status": "candidate",
                "probability_score": 0.71,
                "human_review_status": "accepted",
                "reviewer": "tester",
                "evidence_channels_relied_on": ["evidence_channel_panel_label"],
                "source_report": str(source_report),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "review_outcomes.sqlite"

    assert main(["review", "labels", "ingest", "--labels", str(exported), "--store", str(store_path), "--output-dir", str(tmp_path / "ingest")]) == 0
    output = capsys.readouterr().out
    assert "review-label-ingest" in output

    labels = ReviewOutcomeSQLiteStore(store_path).list_labels(object_type="paper_experiment_link")
    assert len(labels) == 1
    assert labels[0].verdict == "positive"
    assert labels[0].claim_state == "accepted"
    assert labels[0].evidence_channels_relied_on == ["evidence_channel_panel_label"]
    assert (tmp_path / "ingest" / "review_label_ingest.md").exists()


def test_review_label_ingests_figure_source_candidate_export(tmp_path):
    exported = tmp_path / "figure_source_review.jsonl"
    exported.write_text(
        json.dumps(
            {
                "paper_id": "lu-femn-o3",
                "figure_id": "fig-1",
                "panel_id": "panel-i",
                "candidate_id": "scan-group-x025",
                "candidate_type": "scan_group",
                "human_review_status": "needs_more_evidence",
                "reviewer": "tester",
                "evidence_channels_relied_on": ["evidence_channel_panel_label", "evidence_channel_material_identity"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "review_outcomes.sqlite"

    result = backfill_review_labels({"root": str(tmp_path), "store": str(store_path)})

    assert result["status"] == "ok"
    labels = ReviewOutcomeSQLiteStore(store_path).list_labels(object_type="figure_source_candidate")
    assert len(labels) == 1
    assert labels[0].object_id == "panel-i"
    assert labels[0].object_secondary_id == "scan-group-x025"
    assert labels[0].verdict == "inconclusive"
    assert labels[0].claim_state == "candidate"
    assert labels[0].uncertainty_class == "source_grounded_candidate_evidence"


def test_review_label_backfill_ingests_experiment_review_template_and_priors(tmp_path):
    root = tmp_path / "local_runs" / "experiment"
    root.mkdir(parents=True)
    template = root / "review_outcome_template.jsonl"
    template.write_text(
        json.dumps(
            {
                "object_type": "spurion_candidate",
                "object_id": "spurion-1",
                "workflow": "survey-inelastic-spurion-evidence",
                "review_status": "likely_spurion",
                "allowed_review_status_values": ["likely_spurion", "likely_real_feature"],
                "evidence_channels_relied_on": ["evidence_channel_scan_family"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "review_outcomes.sqlite"

    result = backfill_review_labels({"root": str(tmp_path / "local_runs"), "store": str(store_path), "output_dir": str(tmp_path / "backfill")})
    priors = label_informed_priors(store_path, object_type="spurion_candidate", source_workflow="survey-inelastic-spurion-evidence")

    assert result["ingested_label_count"] == 1
    assert result["by_object_type"] == {"spurion_candidate": 1}
    assert priors["status"] == "label_evidence_available"
    channel_prior = priors["channel_priors"]["evidence_channel_scan_family"]
    assert channel_prior["raw_positive_rate"] == 1.0
    assert channel_prior["positive_rate"] == pytest.approx(2.0 / 3.0)
    assert channel_prior["label_weight"] == pytest.approx(1.0 / 11.0)
    assert channel_prior["beta_binomial"]["n_eff_half"] == 10.0
    assert channel_prior["behavior_change_ready"] is False
    assert channel_prior["behavior_change_blockers"] == ["insufficient_positive_labels", "insufficient_negative_labels"]
    assert priors["behavior_change_readiness"]["status"] == "not_ready"
    assert (tmp_path / "backfill" / "review_label_backfill.md").exists()


def test_label_calibrated_score_applies_only_behavior_ready_channels(tmp_path):
    root = tmp_path / "local_runs" / "experiment"
    root.mkdir(parents=True)
    template = root / "review_outcome_template.jsonl"
    rows = []
    for index in range(20):
        rows.append(
            {
                "object_type": "spurion_candidate",
                "object_id": f"positive-{index}",
                "workflow": "survey-inelastic-spurion-evidence",
                "review_status": "likely_spurion",
                "allowed_review_status_values": ["likely_spurion", "likely_real_feature"],
                "evidence_channels_relied_on": ["evidence_channel_scan_family"],
            }
        )
    for index in range(10):
        rows.append(
            {
                "object_type": "spurion_candidate",
                "object_id": f"negative-{index}",
                "workflow": "survey-inelastic-spurion-evidence",
                "review_status": "likely_real_feature",
                "allowed_review_status_values": ["likely_spurion", "likely_real_feature"],
                "evidence_channels_relied_on": ["evidence_channel_scan_family"],
            }
        )
    template.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    store_path = tmp_path / "review_outcomes.sqlite"
    backfill_review_labels({"root": str(tmp_path / "local_runs"), "store": str(store_path)})
    priors = label_informed_priors(store_path, object_type="spurion_candidate", source_workflow="survey-inelastic-spurion-evidence")

    adjusted = label_calibrated_score(0.5, evidence_channels=["evidence_channel_scan_family"], priors=priors)

    assert priors["channel_priors"]["evidence_channel_scan_family"]["behavior_change_ready"] is True
    assert adjusted["applied"] is True
    assert adjusted["applied_score"] > adjusted["base_score"]
    assert adjusted["calibration_status"] == "weakly_calibrated_from_labels"


def test_review_label_backfill_ingests_alignment_review_template(tmp_path):
    root = tmp_path / "local_runs" / "alignment"
    root.mkdir(parents=True)
    template = root / "alignment_review_template.jsonl"
    template.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "object_type": "alignment_peak_candidate",
                        "object_id": "scan:fpx001.bt7",
                        "source_workflow": "alignment-state-epochs",
                        "verdict": "unreviewed",
                        "claim_state": "candidate",
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "object_type": "alignment_plane_candidate",
                        "object_id": "alignment-epoch-002:plane",
                        "source_workflow": "alignment-state-epochs",
                        "verdict": "unreviewed",
                        "claim_state": "candidate",
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "review_outcomes.sqlite"

    result = backfill_review_labels({"root": str(tmp_path / "local_runs"), "store": str(store_path), "output_dir": str(tmp_path / "backfill")})

    assert result["ingested_label_count"] == 2
    assert result["by_object_type"] == {"alignment_peak_candidate": 1, "alignment_plane_candidate": 1}
    peak = ReviewOutcomeSQLiteStore(store_path).list_labels(object_type="alignment_peak_candidate")[0]
    plane = ReviewOutcomeSQLiteStore(store_path).list_labels(object_type="alignment_plane_candidate")[0]
    assert peak.evidence_channels_relied_on == []
    assert plane.object_id == "alignment-epoch-002:plane"


def test_review_label_ingest_accepts_alignment_dashboard_export(tmp_path):
    exported = tmp_path / "alignment_reviewed_rows.jsonl"
    exported.write_text(
        json.dumps(
            {
                "label_id": "alignment-review-1",
                "object_type": "alignment_peak_candidate",
                "object_id": "scan:fpx001.bt7:peak",
                "source_workflow": "alignment-state-epochs",
                "source_artifact_path": "local_runs/alignment_state_epochs/alignment_review_template.jsonl",
                "source_artifact_hash": "sha256:alignment-template",
                "verdict": "negative",
                "type_specific_status": "rejected_powder_or_background",
                "type_specific_allowed": ["accepted_peak_assignment", "rejected_powder_or_background", "rejected_lambda_half_alias", "needs_more_context"],
                "claim_state": "rejected",
                "suggested_status": "candidate",
                "evidence_channels_relied_on": ["peak_fit", "background_material_overlap"],
                "recommendation": "downweight",
                "uncertainty_class": "probability_style_triage_prior",
                "review_notes": "Looks like an aluminum powder/background overlap, not the intended sample reflection.",
                "reviewer_id": "student",
                "metadata": {"alignment_stage": "first_or_second_reflection_localization"},
                "provenance": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    store_path = tmp_path / "review_outcomes.sqlite"

    assert main(["review", "labels", "ingest", "--labels", str(exported), "--store", str(store_path), "--output-dir", str(tmp_path / "ingest")]) == 0

    labels = ReviewOutcomeSQLiteStore(store_path).list_labels(object_type="alignment_peak_candidate")
    assert len(labels) == 1
    assert labels[0].verdict == "negative"
    assert labels[0].claim_state == "rejected"
    assert labels[0].type_specific_status == "rejected_powder_or_background"
    assert labels[0].evidence_channels_relied_on == ["peak_fit", "background_material_overlap"]


def test_review_label_calibration_diagnostics_reports_available_labels(tmp_path):
    store_path = tmp_path / "review_outcomes.sqlite"
    ReviewOutcomeSQLiteStore(store_path).upsert_labels(
        [
            ReviewLabel(
                object_type="paper_experiment_link",
                object_id="paper-1",
                object_secondary_id="experiment-1",
                source_workflow="link-paper-to-experiments",
                verdict="positive",
                claim_state="accepted",
            )
        ]
    )

    diagnostics = review_label_calibration_diagnostics(
        store_path,
        object_type="paper_experiment_link",
        source_workflow="link-paper-to-experiments",
    )

    assert diagnostics["status"] == "label_evidence_available"
    assert diagnostics["reviewed_label_count"] == 1


@pytest.mark.parametrize(
    ("object_type", "type_specific_status", "expected_verdict", "expected_claim_state"),
    [
        ("spurion_candidate", "likely_spurion", "positive", "accepted"),
        ("spurion_candidate", "likely_real_feature", "negative", "rejected"),
        ("spurion_candidate", "ambiguous", "inconclusive", "proposed"),
        ("spurion_candidate", "needs_more_data", "inconclusive", "proposed"),
        ("order_parameter_candidate", "accepted", "positive", "accepted"),
        ("order_parameter_candidate", "rejected", "negative", "rejected"),
        ("fit_handoff", "fit_accepted_for_review", "positive", "accepted"),
        ("fit_handoff", "model_mismatch", "negative", "rejected"),
        ("fit_handoff", "needs_alternate_model", "inconclusive", "proposed"),
    ],
)
def test_experiment_review_status_normalization_by_object_type(
    tmp_path,
    object_type,
    type_specific_status,
    expected_verdict,
    expected_claim_state,
):
    template = tmp_path / "review_outcome_template.jsonl"
    template.write_text("{}", encoding="utf-8")

    label = _experiment_review_row_to_label(
        {
            "object_type": object_type,
            "object_id": "candidate-1",
            "workflow": "experiment-review",
            "review_status": type_specific_status,
            "allowed_review_status_values": [type_specific_status],
            "suggested_status": "proposed",
        },
        template_path=template,
        source_hash="sha256:template",
    )

    assert label.verdict == expected_verdict
    assert label.claim_state == expected_claim_state


def test_experiment_review_row_preserves_reviewer_evidence_channels(tmp_path):
    template = tmp_path / "review_outcome_template.jsonl"
    template.write_text("{}", encoding="utf-8")

    label = _experiment_review_row_to_label(
        {
            "object_type": "spurion_candidate",
            "object_id": "spurion-1",
            "workflow": "survey-inelastic-spurion-evidence",
            "review_status": "likely_spurion",
            "allowed_review_status_values": ["likely_spurion"],
            "evidence_channels_relied_on": ["evidence_channel_reconstruction"],
        },
        template_path=template,
        source_hash="sha256:template",
    )

    assert label.evidence_channels_relied_on == ["evidence_channel_reconstruction"]


def test_decision_ontology_concepts_and_mappings_load():
    concepts = load_ontology(CoreConfig().ontology_path)
    ids = {concept.id for concept in concepts}
    assert "review_outcome" in ids
    assert "evidence_channel_reconstruction" in ids
    assert "uncertainty_class_private_precedent" in ids

    mappings = load_external_mappings(CoreConfig().external_mappings_path)
    mapping_ids = {mapping.local_concept_id for mapping in mappings}
    assert "review_outcome" in mapping_ids
    assert "claim_state_rejected" in mapping_ids
