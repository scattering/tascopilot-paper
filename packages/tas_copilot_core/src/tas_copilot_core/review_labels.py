from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord, ReviewLabel, ReviewLabelBatch
from tas_copilot_core.catalogs.review_outcomes_sqlite import ReviewOutcomeSQLiteStore
from tas_copilot_core.reporting import write_report_bundle

_BETA_BINOMIAL_PRIOR_ALPHA = 1.0
_BETA_BINOMIAL_PRIOR_BETA = 1.0
_BETA_BINOMIAL_N_EFF_HALF = 10.0
_BEHAVIOR_CHANGE_MIN_LABELS_PER_DIRECTION = 10


def review_label_calibration_diagnostics(
    store_path: str | Path | None,
    *,
    object_type: str | None = None,
    source_workflow: str | None = None,
) -> dict[str, Any]:
    if not store_path:
        return {
            "status": "not_provided",
            "calibration_status": "no_review_label_store",
            "message": "No review outcome store was supplied; ranking probabilities remain uncalibrated priors.",
        }
    path = Path(store_path).expanduser()
    if not path.exists():
        return {
            "status": "missing_input",
            "calibration_status": "missing_review_label_store",
            "store": str(path),
            "message": "Review outcome store path does not exist; ranking probabilities remain uncalibrated priors.",
        }
    labels = ReviewOutcomeSQLiteStore(path).list_labels(object_type=object_type, source_workflow=source_workflow)
    reviewed = [label for label in labels if label.verdict != "unreviewed"]
    return {
        "status": "label_evidence_available" if reviewed else "no_reviewed_labels",
        "calibration_status": "reviewed_labels_available" if reviewed else "needs_human_labels",
        "store": str(path),
        "object_type": object_type,
        "source_workflow": source_workflow,
        "label_count": len(labels),
        "reviewed_label_count": len(reviewed),
        "by_verdict": _count_labels(labels, "verdict"),
        "by_claim_state": _count_labels(labels, "claim_state"),
        "by_recommendation": _count_labels(labels, "recommendation"),
        "interpretation": (
            "Reviewed labels can inform local priors, but they are not accepted scientific claims by themselves."
            if reviewed
            else "No reviewed labels are available for this slice; keep scores labeled as heuristic priors."
        ),
    }


def label_informed_priors(
    store_path: str | Path | None,
    *,
    object_type: str | None = None,
    source_workflow: str | None = None,
) -> dict[str, Any]:
    """Summarize reviewed labels by evidence channel without changing rankings.

    This is an intentionally conservative calibration consumer: workflows can
    render these priors beside v0 heuristic priors, but should not overwrite
    ranking behavior until enough reviewed labels exist. The per-channel
    probabilities use the Beta-Binomial smoothing described in
    ``notes/calibration_math_sketch.md`` so that future behavior-change
    consumers do not accidentally treat one reviewed label as certainty.
    """

    diagnostics = review_label_calibration_diagnostics(store_path, object_type=object_type, source_workflow=source_workflow)
    if diagnostics.get("status") not in {"label_evidence_available", "no_reviewed_labels"}:
        return {
            "status": diagnostics.get("status"),
            "calibration_status": diagnostics.get("calibration_status"),
            "store": diagnostics.get("store"),
            "object_type": object_type,
            "source_workflow": source_workflow,
            "channel_priors": {},
            "interpretation": diagnostics.get("interpretation") or diagnostics.get("message"),
        }
    labels = ReviewOutcomeSQLiteStore(Path(store_path).expanduser()).list_labels(object_type=object_type, source_workflow=source_workflow) if store_path else []
    reviewed = [label for label in labels if label.verdict != "unreviewed"]
    by_channel: dict[str, dict[str, Any]] = {}
    for label in reviewed:
        channels = label.evidence_channels_relied_on or ["evidence_channel_unspecified"]
        for channel in channels:
            entry = by_channel.setdefault(
                channel,
                {
                    "reviewed_label_count": 0,
                    "positive": 0,
                    "negative": 0,
                    "inconclusive": 0,
                    "superseded": 0,
                    "objects": [],
                },
            )
            entry["reviewed_label_count"] += 1
            if label.verdict in entry:
                entry[label.verdict] += 1
            entry["objects"].append(f"{label.object_type}:{label.object_id}")
    for channel, entry in by_channel.items():
        total = entry["reviewed_label_count"]
        positive = int(entry["positive"])
        negative = int(entry["negative"])
        n_eff = positive + negative
        alpha = _BETA_BINOMIAL_PRIOR_ALPHA + positive
        beta = _BETA_BINOMIAL_PRIOR_BETA + negative
        posterior_positive = alpha / (alpha + beta)
        weight = n_eff / (n_eff + _BETA_BINOMIAL_N_EFF_HALF) if n_eff else 0.0
        entry["raw_positive_rate"] = positive / total if total else 0.0
        entry["raw_negative_rate"] = negative / total if total else 0.0
        entry["positive_rate"] = posterior_positive
        entry["negative_rate"] = 1.0 - posterior_positive
        entry["posterior_positive_probability"] = posterior_positive
        entry["label_weight"] = weight
        entry["weighted_log_odds_shift"] = weight * _safe_log_odds(posterior_positive)
        entry["beta_binomial"] = {
            "prior_alpha": _BETA_BINOMIAL_PRIOR_ALPHA,
            "prior_beta": _BETA_BINOMIAL_PRIOR_BETA,
            "alpha": alpha,
            "beta": beta,
            "n_eff": n_eff,
            "n_eff_half": _BETA_BINOMIAL_N_EFF_HALF,
        }
        entry["behavior_change_ready"] = positive >= _BEHAVIOR_CHANGE_MIN_LABELS_PER_DIRECTION and negative >= _BEHAVIOR_CHANGE_MIN_LABELS_PER_DIRECTION
        entry["minimum_labels_per_direction_for_behavior_change"] = _BEHAVIOR_CHANGE_MIN_LABELS_PER_DIRECTION
        entry["behavior_change_blockers"] = []
        if positive < _BEHAVIOR_CHANGE_MIN_LABELS_PER_DIRECTION:
            entry["behavior_change_blockers"].append("insufficient_positive_labels")
        if negative < _BEHAVIOR_CHANGE_MIN_LABELS_PER_DIRECTION:
            entry["behavior_change_blockers"].append("insufficient_negative_labels")
        entry["calibration_status"] = "human_labels_available" if total else "needs_human_labels"
        entry["objects"] = entry["objects"][:10]
    ready_channels = [channel for channel, entry in by_channel.items() if entry.get("behavior_change_ready")]
    return {
        "status": "label_evidence_available" if reviewed else "no_reviewed_labels",
        "calibration_status": "reviewed_labels_available" if reviewed else "needs_human_labels",
        "store": str(Path(store_path).expanduser()) if store_path else None,
        "object_type": object_type,
        "source_workflow": source_workflow,
        "reviewed_label_count": len(reviewed),
        "channel_priors": dict(sorted(by_channel.items())),
        "behavior_change_readiness": {
            "status": "ready" if ready_channels else "not_ready",
            "ready_channels": sorted(ready_channels),
            "minimum_labels_per_direction": _BEHAVIOR_CHANGE_MIN_LABELS_PER_DIRECTION,
        },
        "behavior_change": "none",
        "interpretation": "Label-informed priors are rendered as calibration evidence only; v0 ranking behavior is unchanged.",
    }


def label_calibrated_score(
    base_score: float,
    *,
    evidence_channels: list[str] | tuple[str, ...],
    priors: dict[str, Any],
    require_behavior_ready: bool = True,
) -> dict[str, Any]:
    """Combine a v0 triage score with reviewed-label channel priors.

    The returned payload always preserves the original score.  ``applied`` is
    true only when at least one cited channel meets the behavior-change gate,
    unless a caller explicitly opts into preview-only behavior by setting
    ``require_behavior_ready=False``.
    """

    channels = [str(channel) for channel in evidence_channels if channel]
    channel_priors = priors.get("channel_priors") or {}
    base = _clip_probability(base_score)
    log_odds = _safe_log_odds(base)
    contributions: list[dict[str, Any]] = []
    applied_channels: list[str] = []
    for channel in channels:
        prior = channel_priors.get(channel)
        if not prior:
            contributions.append(
                {
                    "evidence_channel": channel,
                    "status": "no_reviewed_labels",
                    "applied": False,
                    "log_odds_shift": 0.0,
                }
            )
            continue
        ready = bool(prior.get("behavior_change_ready"))
        weighted_shift = float(prior.get("weighted_log_odds_shift") or 0.0)
        can_apply = ready or not require_behavior_ready
        if can_apply and weighted_shift:
            log_odds += weighted_shift
            applied_channels.append(channel)
        contributions.append(
            {
                "evidence_channel": channel,
                "status": "ready" if ready else "preview_only",
                "applied": bool(can_apply and weighted_shift),
                "positive": int(prior.get("positive") or 0),
                "negative": int(prior.get("negative") or 0),
                "posterior_positive_probability": prior.get("posterior_positive_probability"),
                "label_weight": prior.get("label_weight"),
                "log_odds_shift": weighted_shift if can_apply else 0.0,
                "preview_log_odds_shift": weighted_shift,
                "behavior_change_blockers": prior.get("behavior_change_blockers") or [],
            }
        )
    label_score = _logistic(log_odds)
    return {
        "base_score": round(base, 6),
        "label_informed_score": round(label_score, 6),
        "applied_score": round(label_score if applied_channels else base, 6),
        "applied": bool(applied_channels),
        "applied_channels": applied_channels,
        "evidence_channels": channels,
        "contributions": contributions,
        "calibration_status": "weakly_calibrated_from_labels" if applied_channels else "label_preview_or_no_matching_labels",
        "behavior_policy": (
            "Applied only channels meeting the behavior-change minimum-label gate."
            if require_behavior_ready
            else "Preview mode allowed sparse label channels to adjust the score."
        ),
    }


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-9, max(1e-9, float(value)))


def _safe_log_odds(probability: float) -> float:
    value = _clip_probability(probability)
    return math.log(value / (1.0 - value))


def _logistic(log_odds: float) -> float:
    if log_odds >= 0:
        z = math.exp(-log_odds)
        return 1.0 / (1.0 + z)
    z = math.exp(log_odds)
    return z / (1.0 + z)


def review_label_summary(arguments: dict[str, Any]) -> dict[str, Any]:
    path = Path(arguments["store"]).expanduser()
    if not path.exists():
        return _missing("review-label-summary", ["store"], f"Review outcome store does not exist: {path}")
    store = ReviewOutcomeSQLiteStore(path)
    summary = store.summary()
    labels = store.list_labels(limit=int(arguments.get("sample_limit") or 10))
    summary.update(
        {
            "workflow_name": "review-label-summary",
            "status": "ok",
            "store": str(path),
            "sample_labels": [label.model_dump(mode="json") for label in labels],
            "safe_conclusion": "Review labels are calibration evidence and review state. They do not convert candidate claims into accepted scientific conclusions without the underlying reviewed evidence.",
        }
    )
    output_dir = arguments.get("output_dir")
    if output_dir:
        summary["artifacts"] = write_report_bundle(
            Path(output_dir).expanduser(),
            summary,
            markdown_renderer=_review_label_summary_markdown,
            json_filename="review_label_summary.json",
            markdown_filename="review_label_summary.md",
        )
    return summary


def export_review_labels(arguments: dict[str, Any]) -> dict[str, Any]:
    path = Path(arguments["store"]).expanduser()
    if not path.exists():
        return _missing("review-label-export", ["store"], f"Review outcome store does not exist: {path}")
    output = Path(arguments["output"]).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = ReviewOutcomeSQLiteStore(path).list_labels(
        object_type=arguments.get("object_type"),
        object_id=arguments.get("object_id"),
        verdict=arguments.get("verdict"),
        source_workflow=arguments.get("source_workflow"),
        limit=arguments.get("limit"),
    )
    if output.suffix.lower() == ".csv":
        _write_labels_csv(output, labels)
    else:
        _write_labels_jsonl(output, labels)
    return {
        "workflow_name": "review-label-export",
        "status": "ok",
        "store": str(path),
        "output": str(output),
        "label_count": len(labels),
        "filters": _label_filters(arguments),
        "safe_conclusion": "Exported labels preserve review state and provenance for calibration/review workflows.",
    }


def ingest_review_labels(arguments: dict[str, Any]) -> dict[str, Any]:
    labels_path = Path(arguments["labels"]).expanduser()
    store_path = Path(arguments["store"]).expanduser()
    if not labels_path.exists():
        return _missing("review-label-ingest", ["labels"], f"Review labels file does not exist: {labels_path}")
    rows = _load_jsonl_dicts(labels_path)
    labels: list[ReviewLabel] = []
    invalid_rows: list[dict[str, Any]] = []
    source_hash = _path_hash(labels_path)
    for index, row in enumerate(rows, start=1):
        try:
            labels.append(_review_label_from_row(row, source_path=labels_path, source_hash=source_hash, row_number=index))
        except Exception as exc:
            invalid = dict(row)
            invalid["review_row_number"] = index
            invalid["validation_error"] = f"{type(exc).__name__}: {exc}"
            invalid_rows.append(invalid)
    batch = ReviewLabelBatch(
        batch_id=f"review-label-batch-{hashlib.sha256(str(labels_path).encode('utf-8')).hexdigest()[:16]}",
        source_template_paths=[str(labels_path)],
        labels=labels,
        provenance=[ProvenanceRecord(source_type="review_labels", source_id=str(labels_path), method="review_label_unified_ingest")],
    )
    ReviewOutcomeSQLiteStore(store_path).upsert_batch(batch)
    report = {
        "workflow_name": "review-label-ingest",
        "status": "ok" if not invalid_rows else "partial",
        "labels": str(labels_path),
        "store": str(store_path),
        "input_row_count": len(rows),
        "ingested_label_count": len(labels),
        "invalid_row_count": len(invalid_rows),
        "by_verdict": _count_labels(labels, "verdict"),
        "by_object_type": _count_labels(labels, "object_type"),
        "safe_conclusion": "Ingested review labels are calibration evidence. They do not accept scientific claims without the linked reviewed evidence.",
    }
    output_dir = arguments.get("output_dir")
    if output_dir:
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        valid_path = out / "review_labels.ingested.jsonl"
        invalid_path = out / "review_labels.invalid.jsonl"
        _write_labels_jsonl(valid_path, labels)
        _write_jsonl_dicts(invalid_path, invalid_rows)
        report["generated_artifacts"] = {"ingested_labels_jsonl": str(valid_path), "invalid_labels_jsonl": str(invalid_path)}
        report["artifacts"] = write_report_bundle(
            out,
            report,
            markdown_renderer=_review_label_ingest_markdown,
            json_filename="review_label_ingest.json",
            markdown_filename="review_label_ingest.md",
        )
    return report


def backfill_review_labels(arguments: dict[str, Any]) -> dict[str, Any]:
    root = Path(arguments.get("root") or "local_runs").expanduser()
    store_path = Path(arguments["store"]).expanduser()
    patterns = arguments.get("patterns") or [
        "review_outcome_template.jsonl",
        "paper_experiment_review_template.jsonl",
        "figure_source_review_template.jsonl",
        "figure_source_review.jsonl",
        "figure_source_candidate_review.jsonl",
        "alignment_review_template.jsonl",
    ]
    if not root.exists():
        return _missing("review-label-backfill", ["root"], f"Backfill root does not exist: {root}")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    files = sorted(set(files))
    all_labels: list[ReviewLabel] = []
    invalid_rows: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    for labels_path in files:
        rows = _load_jsonl_dicts(labels_path)
        labels: list[ReviewLabel] = []
        source_hash = _path_hash(labels_path)
        for index, row in enumerate(rows, start=1):
            try:
                labels.append(_review_label_from_row(row, source_path=labels_path, source_hash=source_hash, row_number=index))
            except Exception as exc:
                invalid = dict(row)
                invalid["source_template_path"] = str(labels_path)
                invalid["review_row_number"] = index
                invalid["validation_error"] = f"{type(exc).__name__}: {exc}"
                invalid_rows.append(invalid)
        if labels:
            batch = ReviewLabelBatch(
                batch_id=f"review-label-backfill-{hashlib.sha256(str(labels_path).encode('utf-8')).hexdigest()[:16]}",
                source_template_paths=[str(labels_path)],
                labels=labels,
                provenance=[ProvenanceRecord(source_type="review_template", source_id=str(labels_path), method="review_label_backfill")],
            )
            ReviewOutcomeSQLiteStore(store_path).upsert_batch(batch)
            all_labels.extend(labels)
        file_reports.append({"path": str(labels_path), "row_count": len(rows), "ingested_label_count": len(labels)})
    report = {
        "workflow_name": "review-label-backfill",
        "status": "ok" if not invalid_rows else "partial",
        "root": str(root),
        "store": str(store_path),
        "file_count": len(files),
        "input_row_count": sum(item["row_count"] for item in file_reports),
        "ingested_label_count": len(all_labels),
        "invalid_row_count": len(invalid_rows),
        "files": file_reports,
        "by_object_type": _count_labels(all_labels, "object_type"),
        "by_verdict": _count_labels(all_labels, "verdict"),
        "safe_conclusion": "Backfilled labels are calibration evidence only. They do not turn proposed candidates into accepted scientific claims.",
    }
    output_dir = arguments.get("output_dir")
    if output_dir:
        out = Path(output_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        _write_labels_jsonl(out / "review_labels.backfilled.jsonl", all_labels)
        _write_jsonl_dicts(out / "review_labels.backfill_invalid.jsonl", invalid_rows)
        report["artifacts"] = write_report_bundle(
            out,
            report,
            markdown_renderer=_review_label_backfill_markdown,
            json_filename="review_label_backfill.json",
            markdown_filename="review_label_backfill.md",
        )
    return report


def stale_review_label_check(arguments: dict[str, Any]) -> dict[str, Any]:
    path = Path(arguments["store"]).expanduser()
    if not path.exists():
        return _missing("review-label-stale-check", ["store"], f"Review outcome store does not exist: {path}")
    labels = ReviewOutcomeSQLiteStore(path).list_labels(
        object_type=arguments.get("object_type"),
        object_id=arguments.get("object_id"),
        verdict=arguments.get("verdict"),
        source_workflow=arguments.get("source_workflow"),
        limit=arguments.get("limit"),
    )
    current = {
        "workflow_version": arguments.get("workflow_version"),
        "tool_versions": arguments.get("tool_versions") or {},
        "calibration_state_hash": arguments.get("calibration_state_hash"),
        "dag_run_id": arguments.get("dag_run_id"),
        "cache_key": arguments.get("cache_key"),
    }
    results = [_stale_label_result(label, current=current) for label in labels]
    stale = [item for item in results if item["stale"]]
    report = {
        "workflow_name": "review-label-stale-check",
        "status": "ok",
        "store": str(path),
        "label_count": len(labels),
        "stale_label_count": len(stale),
        "current_state_checked": {key: value for key, value in current.items() if value},
        "results": results,
        "safe_conclusion": "Stale labels should not be used for calibration until reviewed against the current workflow/cache/calibration state.",
    }
    output_dir = arguments.get("output_dir")
    if output_dir:
        report["artifacts"] = write_report_bundle(
            Path(output_dir).expanduser(),
            report,
            markdown_renderer=_stale_check_markdown,
            json_filename="review_label_stale_check.json",
            markdown_filename="review_label_stale_check.md",
        )
    return report


def prepare_review_label_packet(arguments: dict[str, Any]) -> dict[str, Any]:
    path = Path(arguments["store"]).expanduser()
    if not path.exists():
        return _missing("review-label-packet", ["store"], f"Review outcome store does not exist: {path}")
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ReviewOutcomeSQLiteStore(path).list_labels(
        object_type=arguments.get("object_type"),
        source_workflow=arguments.get("source_workflow"),
        verdict=arguments.get("verdict") or "unreviewed",
        limit=int(arguments.get("limit") or 50),
    )
    jsonl_path = output_dir / "review_label_packet.jsonl"
    csv_path = output_dir / "review_label_packet.csv"
    _write_labels_jsonl(jsonl_path, labels)
    _write_labels_csv(csv_path, labels)
    report = {
        "workflow_name": "review-label-packet",
        "status": "ok",
        "store": str(path),
        "label_count": len(labels),
        "filters": _label_filters(arguments),
        "generated_artifacts": {
            "review_label_packet_jsonl": str(jsonl_path),
            "review_label_packet_csv": str(csv_path),
        },
        "review_instructions": [
            "Open the CSV or JSONL packet and assign type-specific statuses only where evidence has been inspected.",
            "Keep unreviewed rows unreviewed; absence of review is not negative evidence.",
            "Use review_notes for missing inputs, likely false positives, or changed calibration/workflow state.",
            "Re-ingest reviewed rows through a workflow-specific ingest path before using them to calibrate priors.",
        ],
        "safe_conclusion": "This packet is a local human-review aid. It does not mutate raw data, corpus records, or accepted source links.",
    }
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_review_packet_markdown,
        json_filename="review_label_packet_report.json",
        markdown_filename="review_label_packet_report.md",
    )
    return report


def _stale_label_result(label: ReviewLabel, *, current: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if label.claim_state == "stale":
        reasons.append("claim_state_stale")
    if label.source_artifact_path and label.source_artifact_hash:
        artifact = Path(label.source_artifact_path).expanduser()
        if not artifact.exists():
            reasons.append("source_artifact_missing")
        else:
            current_hash = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
            if current_hash != label.source_artifact_hash:
                reasons.append("source_artifact_hash_changed")
    for key in ("workflow_version", "calibration_state_hash", "dag_run_id", "cache_key"):
        current_value = current.get(key)
        label_value = getattr(label, key)
        if current_value and label_value and str(current_value) != str(label_value):
            reasons.append(f"{key}_changed")
    current_tools = current.get("tool_versions") or {}
    for tool, version in current_tools.items():
        if label.tool_versions.get(tool) and str(label.tool_versions[tool]) != str(version):
            reasons.append(f"tool_version_changed:{tool}")
    return {
        "label_id": label.label_id,
        "object_type": label.object_type,
        "object_id": label.object_id,
        "object_secondary_id": label.object_secondary_id,
        "verdict": label.verdict,
        "claim_state": label.claim_state,
        "source_workflow": label.source_workflow,
        "dag_run_id": label.dag_run_id,
        "cache_key": label.cache_key,
        "stale": bool(reasons),
        "stale_reasons": reasons,
    }


def _write_labels_jsonl(path: Path, labels: list[ReviewLabel]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label.model_dump(mode="json"), sort_keys=True) + "\n")


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _review_label_from_row(row: dict[str, Any], *, source_path: Path, source_hash: str, row_number: int) -> ReviewLabel:
    if {"object_type", "object_id", "source_workflow", "verdict"} <= set(row):
        return ReviewLabel.model_validate(row)
    if row.get("paper_id") and row.get("candidate_experiment_id"):
        return _paper_link_export_row_to_label(row, source_path=source_path, source_hash=source_hash, row_number=row_number)
    if row.get("object_type") and row.get("object_id") and row.get("workflow"):
        return _experiment_review_export_row_to_label(row, source_path=source_path, source_hash=source_hash, row_number=row_number)
    if row.get("candidate_id") and (row.get("panel_id") or row.get("figure_id") or row.get("candidate_type")):
        return _figure_source_candidate_export_row_to_label(row, source_path=source_path, source_hash=source_hash, row_number=row_number)
    raise ValueError("row is neither a ReviewLabel record nor a recognized dashboard/template review export row")


def _paper_link_export_row_to_label(row: dict[str, Any], *, source_path: Path, source_hash: str, row_number: int) -> ReviewLabel:
    status = str(row.get("human_review_status") or "unreviewed")
    paper_id = str(row.get("paper_id"))
    experiment_id = str(row.get("candidate_experiment_id"))
    source_artifact_path = row.get("source_report") or row.get("source_report_json") or str(source_path)
    artifact = Path(str(source_artifact_path)).expanduser()
    artifact_hash = _path_hash(artifact) if artifact.exists() else source_hash
    return ReviewLabel(
        label_id=f"review-label-{hashlib.sha256('|'.join([str(source_path), paper_id, experiment_id, str(row.get('candidate_rank') or row_number)]).encode('utf-8')).hexdigest()[:16]}",
        object_type="paper_experiment_link",
        object_id=paper_id,
        object_secondary_id=experiment_id,
        source_workflow="link-paper-to-experiments",
        source_artifact_path=str(source_artifact_path),
        source_artifact_hash=artifact_hash,
        verdict=_paper_link_status_to_verdict(status),
        type_specific_status=None if status == "unreviewed" else status,
        type_specific_allowed=[str(item) for item in row.get("allowed_human_review_statuses") or []],
        claim_state=_paper_link_status_to_claim_state(status),
        suggested_status=str(row.get("candidate_status") or "candidate"),
        # Reviewer-selected channels are calibration evidence and must survive
        # ingest; the paper/text+metadata pair is only the dashboard default.
        evidence_channels_relied_on=[
            str(item)
            for item in (
                row.get("evidence_channels_relied_on")
                or row.get("evidence_channels")
                or ["evidence_channel_paper_text", "evidence_channel_hydrated_metadata"]
            )
        ],
        recommendation=_paper_link_status_to_recommendation(status),
        uncertainty_class="probability_style_triage_prior",
        review_notes=str(row.get("review_notes") or ""),
        reviewer_id=row.get("reviewer"),
        reviewed_at=_parse_datetime(row.get("reviewed_at")),
        legacy_provenance=row.get("provenance"),
        provenance=[ProvenanceRecord(source_type="review_labels", source_id=str(source_path), method="paper_link_export_row_to_label")],
        metadata={
            "paper_title": row.get("paper_title"),
            "candidate_rank": row.get("candidate_rank"),
            "probability_score": row.get("probability_score"),
            "confidence_label": row.get("confidence_label"),
            "review_priority": row.get("review_priority"),
            "review_row_number": row_number,
        },
    )


def _paper_link_status_to_verdict(status: str) -> str:
    if status == "accepted":
        return "positive"
    # Duplicate means this row should not independently support the link. The
    # duplicate target can carry its own positive/negative/superseded label.
    if status in {"rejected", "duplicate"}:
        return "negative"
    if status in {"needs_more_evidence", "not_enough_information"}:
        return "inconclusive"
    return "unreviewed"


def _paper_link_status_to_claim_state(status: str) -> str:
    if status == "accepted":
        return "accepted"
    if status in {"rejected", "duplicate"}:
        return "rejected"
    return "candidate"


def _paper_link_status_to_recommendation(status: str) -> str:
    if status in {"rejected", "duplicate"}:
        return "downweight"
    if status in {"needs_more_evidence", "not_enough_information"}:
        return "escalate"
    if status == "accepted":
        return "calibrate_prior"
    return "no_action"


def _experiment_review_export_row_to_label(row: dict[str, Any], *, source_path: Path, source_hash: str, row_number: int) -> ReviewLabel:
    object_type = str(row.get("object_type") or "experiment_review")
    status = str(row.get("review_status") or row.get("type_specific_status") or "unreviewed")
    return ReviewLabel(
        label_id=f"review-label-{hashlib.sha256('|'.join([str(source_path), object_type, str(row.get('object_id')), str(row.get('workflow')), str(row_number)]).encode('utf-8')).hexdigest()[:16]}",
        object_type=object_type,
        object_id=str(row.get("object_id") or "unknown"),
        source_workflow=str(row.get("workflow") or "experiment-review"),
        source_artifact_path=str(source_path),
        source_artifact_hash=source_hash,
        verdict=_experiment_review_status_to_verdict(object_type, status),
        type_specific_status=None if status == "unreviewed" else status,
        type_specific_allowed=[str(item) for item in row.get("allowed_review_status_values") or row.get("type_specific_allowed") or []],
        claim_state=_experiment_review_status_to_claim_state(object_type, status),
        suggested_status=str(row.get("suggested_status") or "proposed"),
        evidence_channels_relied_on=[
            str(item)
            for item in (
                row.get("evidence_channels_relied_on")
                or row.get("evidence_channels")
                or _experiment_review_default_evidence_channels(object_type)
            )
        ],
        recommendation=_experiment_review_status_to_recommendation(object_type, status),
        uncertainty_class="probability_style_triage_prior",
        false_positive_reason=row.get("false_positive_reason") or None,
        review_notes=str(row.get("review_notes") or ""),
        workflow_version=row.get("workflow_version"),
        tool_versions={str(key): str(value) for key, value in (row.get("tool_versions") or {}).items()},
        correction_status_snapshot=row.get("correction_status_snapshot") or {},
        dag_run_id=row.get("dag_run_id"),
        cache_key=row.get("cache_key"),
        cache_scope="full_dag" if row.get("cache_key") else None,
        reviewer_id=row.get("reviewer") or row.get("reviewer_id"),
        reviewed_at=_parse_datetime(row.get("reviewed_at")),
        legacy_provenance=row.get("provenance"),
        provenance=[ProvenanceRecord(source_type="review_template", source_id=str(source_path), method="experiment_review_export_row_to_label")],
        metadata={"review_row_number": row_number},
    )


def _figure_source_candidate_export_row_to_label(row: dict[str, Any], *, source_path: Path, source_hash: str, row_number: int) -> ReviewLabel:
    status = str(row.get("human_review_status") or row.get("review_status") or "unreviewed")
    candidate_id = str(row.get("candidate_id"))
    panel_id = str(row.get("panel_id") or row.get("figure_id") or "unknown-panel")
    source_artifact_path = row.get("source_report") or row.get("source_report_json") or str(source_path)
    artifact = Path(str(source_artifact_path)).expanduser()
    artifact_hash = _path_hash(artifact) if artifact.exists() else source_hash
    return ReviewLabel(
        label_id=f"review-label-{hashlib.sha256('|'.join([str(source_path), panel_id, candidate_id, str(row.get('rank') or row.get('candidate_rank') or row_number)]).encode('utf-8')).hexdigest()[:16]}",
        object_type="figure_source_candidate",
        object_id=panel_id,
        object_secondary_id=candidate_id,
        source_workflow=str(row.get("source_workflow") or "figure-source-triage"),
        source_artifact_path=str(source_artifact_path),
        source_artifact_hash=artifact_hash,
        verdict=_figure_source_status_to_verdict(status),
        type_specific_status=None if status == "unreviewed" else status,
        type_specific_allowed=[
            str(item)
            for item in (
                row.get("allowed_human_review_statuses")
                or row.get("allowed_review_status_values")
                or ["accepted", "rejected", "needs_more_evidence", "ambiguous", "not_source", "duplicate"]
            )
        ],
        claim_state=_figure_source_status_to_claim_state(status),
        suggested_status=str(row.get("candidate_status") or row.get("status") or "candidate"),
        evidence_channels_relied_on=[
            str(item)
            for item in (
                row.get("evidence_channels_relied_on")
                or row.get("evidence_channels")
                or ["evidence_channel_panel_label", "evidence_channel_material_identity", "evidence_channel_scan_family"]
            )
        ],
        recommendation=_figure_source_status_to_recommendation(status),
        uncertainty_class="source_grounded_candidate_evidence",
        review_notes=str(row.get("review_notes") or ""),
        reviewer_id=row.get("reviewer") or row.get("reviewer_id"),
        reviewed_at=_parse_datetime(row.get("reviewed_at")),
        provenance=[ProvenanceRecord(source_type="review_labels", source_id=str(source_path), method="figure_source_candidate_export_row_to_label")],
        metadata={
            "paper_id": row.get("paper_id"),
            "figure_id": row.get("figure_id"),
            "candidate_type": row.get("candidate_type"),
            "score": row.get("score") or row.get("probability_score"),
            "confidence_label": row.get("confidence_label"),
            "review_row_number": row_number,
        },
    )


def _experiment_review_status_to_verdict(object_type: str, status: str) -> str:
    if object_type == "spurion_candidate":
        if status == "likely_spurion":
            return "positive"
        if status == "likely_real_feature":
            return "negative"
        if status in {"ambiguous", "needs_more_data"}:
            return "inconclusive"
        return "unreviewed"
    if status in {"accepted", "fit_accepted_for_review"}:
        return "positive"
    if status in {"rejected", "fit_rejected", "model_mismatch", "did_not_converge"}:
        return "negative"
    if status == "superseded":
        return "superseded"
    if status in {"needs_more_data", "ambiguous", "needs_alternate_model", "parser_issue", "no_action"}:
        return "inconclusive"
    return "unreviewed"


def _experiment_review_status_to_claim_state(object_type: str, status: str) -> str:
    if object_type == "spurion_candidate":
        if status == "likely_spurion":
            return "accepted"
        if status == "likely_real_feature":
            return "rejected"
        return "proposed"
    if status in {"accepted", "fit_accepted_for_review"}:
        return "accepted"
    if status in {"rejected", "fit_rejected", "model_mismatch", "did_not_converge"}:
        return "rejected"
    if status == "superseded":
        return "superseded"
    return "proposed"


def _experiment_review_status_to_recommendation(object_type: str, status: str) -> str:
    verdict = _experiment_review_status_to_verdict(object_type, status)
    if verdict == "positive":
        return "calibrate_prior"
    if verdict == "negative":
        return "downweight"
    if verdict == "inconclusive":
        return "escalate"
    return "no_action"


def _experiment_review_default_evidence_channels(object_type: str) -> list[str]:
    if object_type == "order_parameter_candidate":
        return ["evidence_channel_scan_family", "evidence_channel_repeatability", "evidence_channel_reconstruction"]
    if object_type == "spurion_candidate":
        return ["evidence_channel_scan_family", "evidence_channel_filter_state", "evidence_channel_collimation"]
    if object_type == "fit_handoff":
        return ["evidence_channel_reconstruction"]
    if object_type == "alignment_peak_candidate":
        return ["evidence_channel_scan_recipe", "evidence_channel_structure_factor", "evidence_channel_console_log"]
    if object_type == "alignment_plane_candidate":
        return ["evidence_channel_state_epoch", "evidence_channel_structure_factor", "evidence_channel_scan_recipe", "evidence_channel_console_log"]
    if object_type == "alignment_state_transition":
        return ["evidence_channel_state_epoch", "evidence_channel_console_log", "evidence_channel_scan_recipe"]
    return []


def _figure_source_status_to_verdict(status: str) -> str:
    if status == "accepted":
        return "positive"
    if status in {"rejected", "not_source", "duplicate"}:
        return "negative"
    if status in {"needs_more_evidence", "ambiguous"}:
        return "inconclusive"
    return "unreviewed"


def _figure_source_status_to_claim_state(status: str) -> str:
    if status == "accepted":
        return "accepted"
    if status in {"rejected", "not_source", "duplicate"}:
        return "rejected"
    return "candidate"


def _figure_source_status_to_recommendation(status: str) -> str:
    verdict = _figure_source_status_to_verdict(status)
    if verdict == "positive":
        return "calibrate_prior"
    if verdict == "negative":
        return "downweight"
    if verdict == "inconclusive":
        return "escalate"
    return "no_action"


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _path_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_labels_csv(path: Path, labels: list[ReviewLabel]) -> None:
    fields = [
        "label_id",
        "object_type",
        "object_id",
        "object_secondary_id",
        "source_workflow",
        "verdict",
        "type_specific_status",
        "claim_state",
        "recommendation",
        "uncertainty_class",
        "reviewer_id",
        "reviewed_at",
        "dag_run_id",
        "cache_key",
        "source_artifact_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            payload = label.model_dump(mode="json")
            writer.writerow({field: payload.get(field) for field in fields})


def _count_labels(labels: list[ReviewLabel], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        key = str(getattr(label, field) or "missing")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _label_filters(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        key: arguments.get(key)
        for key in ["object_type", "object_id", "verdict", "source_workflow", "limit"]
        if arguments.get(key) is not None
    }


def _review_label_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Review Label Summary",
        "",
        f"Status: `{report.get('status')}`",
        f"Store: `{report.get('store')}`",
        f"Labels: {report.get('label_count', 0)}",
        f"Reviewed labels: {report.get('reviewed_label_count', 0)}",
        "",
        "## Counts",
        f"- By object type: `{report.get('by_object_type', {})}`",
        f"- By verdict: `{report.get('by_verdict', {})}`",
        f"- By workflow: `{report.get('by_source_workflow', {})}`",
        "",
        "## Safe Conclusion",
        report.get("safe_conclusion", ""),
    ]
    return "\n".join(lines) + "\n"


def _stale_check_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Review Label Stale Check",
        "",
        f"Status: `{report.get('status')}`",
        f"Labels checked: {report.get('label_count', 0)}",
        f"Stale labels: {report.get('stale_label_count', 0)}",
        "",
        "## Stale Labels",
    ]
    for item in report.get("results", []):
        if item.get("stale"):
            lines.append(f"- `{item.get('label_id')}` {item.get('object_type')}:{item.get('object_id')} reasons={item.get('stale_reasons')}")
    lines.extend(["", "## Safe Conclusion", report.get("safe_conclusion", "")])
    return "\n".join(lines) + "\n"


def _review_packet_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Review Label Packet",
        "",
        f"Status: `{report.get('status')}`",
        f"Rows: {report.get('label_count', 0)}",
        "",
        "## Artifacts",
    ]
    for key, value in (report.get("generated_artifacts") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Instructions"])
    lines.extend(f"- {item}" for item in report.get("review_instructions", []))
    lines.extend(["", "## Safe Conclusion", report.get("safe_conclusion", "")])
    return "\n".join(lines) + "\n"


def _review_label_ingest_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Review Label Ingest",
        "",
        f"Status: `{report.get('status')}`",
        f"Input rows: {report.get('input_row_count', 0)}",
        f"Ingested labels: {report.get('ingested_label_count', 0)}",
        f"Invalid rows: {report.get('invalid_row_count', 0)}",
        "",
        "## Counts",
        f"- By object type: `{report.get('by_object_type', {})}`",
        f"- By verdict: `{report.get('by_verdict', {})}`",
        "",
        "## Safe Conclusion",
        report.get("safe_conclusion", ""),
    ]
    return "\n".join(lines) + "\n"


def _review_label_backfill_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Review Label Backfill",
        "",
        f"Status: `{report.get('status')}`",
        f"Root: `{report.get('root')}`",
        f"Store: `{report.get('store')}`",
        f"Files: {report.get('file_count', 0)}",
        f"Ingested labels: {report.get('ingested_label_count', 0)}",
        f"Invalid rows: {report.get('invalid_row_count', 0)}",
        "",
        "## Counts",
        f"- By object type: `{report.get('by_object_type', {})}`",
        f"- By verdict: `{report.get('by_verdict', {})}`",
        "",
        "## Safe Conclusion",
        report.get("safe_conclusion", ""),
    ]
    return "\n".join(lines) + "\n"


def _missing(workflow_name: str, fields: list[str], warning: str) -> dict[str, Any]:
    return {
        "workflow_name": workflow_name,
        "status": "missing_input",
        "missing_fields": fields,
        "warnings": [warning],
        "try": f"python -m tas_copilot_core.cli review labels {workflow_name.replace('review-label-', '')} --help",
    }
