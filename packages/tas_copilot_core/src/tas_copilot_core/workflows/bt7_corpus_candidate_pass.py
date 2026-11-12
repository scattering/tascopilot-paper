from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord
from tas_copilot_core.reporting import write_report_bundle
from tas_copilot_core.workflows.alignment_corpus_index import alignment_corpus_index
from tas_copilot_core.workflows.alignment_state import alignment_state_epochs, alignment_state_machine
from tas_copilot_core.workflows.experiment_dashboard import order_parameter_triage, survey_inelastic_spurion_evidence


def bt7_corpus_candidate_pass(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run local-only candidate passes over one or more hydrated BT-7 roots."""

    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/bt7_corpus_candidate_pass")).expanduser()
    data_roots = _coerce_roots(arguments.get("data_root") or arguments.get("data_roots"))
    if not data_roots:
        report = {
            "workflow_name": "bt7-corpus-candidate-pass",
            "status": "missing_input",
            "missing_inputs": ["data_root"],
            "try": "python -m tas_copilot_core.cli workflow bt7-corpus-candidate-pass --data-root /path/to/bt7 --output-dir local_runs/bt7_corpus_candidate_pass --max-files 0",
            "safe_conclusion": "No corpus pass was run because no local hydrated BT-7 data root was supplied.",
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="bt7_corpus_candidate_pass.json", markdown_filename="bt7_corpus_candidate_pass.md")
        return report

    output_dir.mkdir(parents=True, exist_ok=True)
    max_files = int(arguments.get("max_files", 0) or 0)
    post_year = int(arguments.get("post_year", 2010) or 0)
    review_label_store = arguments.get("review_label_store") or arguments.get("review_outcomes")
    per_root = []
    for root in data_roots:
        root_path = Path(root).expanduser()
        root_id = _safe_id(str(root_path))
        root_dir = output_dir / root_id
        if not root_path.exists():
            per_root.append({"data_root": str(root_path), "status": "missing_input", "missing_inputs": ["data_root"], "summary": {}})
            continue

        alignment = alignment_corpus_index(
            {
                "data_roots": [str(root_path)],
                "output_dir": str(root_dir / "alignment_corpus_index"),
                "max_files": max_files,
                "max_log_lines": int(arguments.get("max_log_lines", 5000) or 5000),
                "sample_limit": int(arguments.get("sample_limit", 50) or 50),
                "include_logs": bool(arguments.get("include_logs", True)),
            }
        )
        events_jsonl = (alignment.get("artifacts") or {}).get("events_jsonl")
        epochs = alignment_state_epochs({"events_jsonl": events_jsonl, "output_dir": str(root_dir / "alignment_state_epochs")}) if events_jsonl else {"status": "skipped", "summary": {}}
        machine = alignment_state_machine({"events_jsonl": events_jsonl, "output_dir": str(root_dir / "alignment_state_machine")}) if events_jsonl else {"status": "skipped", "summary": {}}
        spurions = survey_inelastic_spurion_evidence(
            {
                "data_root": str(root_path),
                "output_dir": str(root_dir / "spurion_survey"),
                "max_files": max_files,
                "post_year": post_year,
                "spurion_top_k": int(arguments.get("spurion_top_k", 100) or 100),
                "review_label_store": review_label_store,
            }
        )
        order = order_parameter_triage(
            {
                "data_root": str(root_path),
                "output_dir": str(root_dir / "order_parameter_triage"),
                "max_files": max_files,
                "post_year": int(arguments.get("order_post_year", 0) or 0),
                "min_scans": int(arguments.get("min_scans", 4) or 4),
                "top_k": int(arguments.get("order_top_k", 100) or 100),
                "review_label_store": review_label_store,
            }
        )
        per_root.append(
            {
                "data_root": str(root_path),
                "status": "ok",
                "summary": {
                    "alignment_event_count": ((alignment.get("summary") or {}).get("alignment_event_count")),
                    "alignment_epoch_count": ((epochs.get("summary") or {}).get("epoch_count")),
                    "alignment_state_estimate_count": ((machine.get("summary") or {}).get("state_estimate_count")),
                    "spurion_candidate_count": spurions.get("candidate_count"),
                    "order_parameter_candidate_count": order.get("candidate_count"),
                },
                "artifacts": {
                    "alignment_index": (alignment.get("artifacts") or {}).get("json"),
                    "alignment_events_jsonl": events_jsonl,
                    "alignment_epochs": (epochs.get("artifacts") or {}).get("json"),
                    "alignment_state_machine": (machine.get("artifacts") or {}).get("json"),
                    "spurion_survey": (spurions.get("artifacts") or {}).get("json"),
                    "order_parameter_triage": (order.get("artifacts") or {}).get("json"),
                },
                "label_calibration": {
                    "spurion": (spurions.get("ranking_diagnostics") or {}).get("label_calibration_summary"),
                    "order_parameter": order.get("label_calibration_summary"),
                },
                "top_candidates": {
                    "spurion": _top_ids(spurions.get("candidates") or [], "candidate_id", "likelihood_score"),
                    "order_parameter": _top_ids(order.get("candidates") or [], "candidate_id", "score"),
                },
            }
        )

    summary = {
        "root_count": len(data_roots),
        "completed_root_count": sum(1 for item in per_root if item.get("status") == "ok"),
        "alignment_event_count": sum(int((item.get("summary") or {}).get("alignment_event_count") or 0) for item in per_root),
        "spurion_candidate_count": sum(int((item.get("summary") or {}).get("spurion_candidate_count") or 0) for item in per_root),
        "order_parameter_candidate_count": sum(int((item.get("summary") or {}).get("order_parameter_candidate_count") or 0) for item in per_root),
    }
    report = {
        "workflow_name": "bt7-corpus-candidate-pass",
        "status": "ok" if summary["completed_root_count"] else "missing_input",
        "data_roots": [str(Path(root).expanduser()) for root in data_roots],
        "max_files": max_files,
        "post_year": post_year,
        "review_label_store": str(Path(review_label_store).expanduser()) if review_label_store else None,
        "summary": summary,
        "per_root": per_root,
        "safe_conclusion": "Ran local-only candidate discovery passes for alignment evidence, spurion triage, and order-parameter triage. Outputs are review aids, not accepted scientific conclusions.",
        "warnings": [
            "This workflow writes derived reports only under the requested output directory; do not commit real-data outputs.",
            "Label-calibrated scores apply only when reviewed labels meet the minimum behavior-change gate.",
            "Full-corpus runs can be expensive; use TACC/Vista or a local cluster for large roots.",
        ],
        "provenance": [ProvenanceRecord(source_type="data_roots", source_id=json.dumps(data_roots), method="bt7_corpus_candidate_pass").model_dump(mode="json")],
    }
    report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="bt7_corpus_candidate_pass.json", markdown_filename="bt7_corpus_candidate_pass.md")
    return report


def _coerce_roots(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return [str(value)]


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)[-90:].strip("_") or "root"


def _top_ids(rows: list[dict[str, Any]], id_key: str, score_key: str) -> list[dict[str, Any]]:
    output = []
    for row in rows[:10]:
        output.append({"id": row.get(id_key), "score": row.get(score_key), "path": row.get("path")})
    return output


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BT-7 Corpus Candidate Pass",
        "",
        f"Status: `{report.get('status')}`",
        f"Roots: {len(report.get('data_roots') or [])}",
        f"Max files: `{report.get('max_files')}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in (report.get("summary") or {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Per Root", ""])
    for item in report.get("per_root") or []:
        lines.append(f"### `{item.get('data_root')}`")
        for key, value in (item.get("summary") or {}).items():
            lines.append(f"- `{key}`: {value}")
        artifacts = item.get("artifacts") or {}
        if artifacts:
            lines.append("- Artifacts:")
            for name, path in artifacts.items():
                if path:
                    lines.append(f"  - `{name}`: `{path}`")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report.get("warnings") or [])
    return "\n".join(lines) + "\n"
