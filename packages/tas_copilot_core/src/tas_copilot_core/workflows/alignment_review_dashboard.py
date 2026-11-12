from __future__ import annotations

import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord, ReviewLabel, ReviewLabelBatch
from tas_copilot_core.catalogs import ReviewOutcomeSQLiteStore
from tas_copilot_core.reporting import write_report_bundle


def build_alignment_review_dashboard(arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a browser review dashboard for alignment-state review rows."""

    arguments = arguments or {}
    output_dir = Path(str(arguments.get("output_dir") or "local_runs/alignment_review_dashboard")).expanduser()
    review_template = Path(str(arguments.get("review_template") or "")).expanduser()
    if not review_template.exists():
        report = {
            "workflow_name": "alignment-review-dashboard",
            "status": "missing_input",
            "missing_inputs": ["review_template"],
            "try": "python -m tas_copilot_core.cli workflow alignment-review-dashboard --review-template local_runs/alignment_state_epochs/alignment_review_template.jsonl --output-dir local_runs/alignment_review_dashboard",
            "safe_conclusion": "No dashboard was built because the alignment review template was missing.",
        }
        report["artifacts"] = write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="alignment_review_dashboard_report.json", markdown_filename="alignment_review_dashboard_report.md")
        return report

    rows = _load_jsonl(review_template)
    records = [_dashboard_record(row, index=index + 1) for index, row in enumerate(rows)]
    labels = [_record_to_label(record, source_template=review_template) for record in records]
    max_dashboard_rows_arg = arguments.get("max_dashboard_rows")
    max_dashboard_rows = 5000 if max_dashboard_rows_arg is None else int(max_dashboard_rows_arg)
    if max_dashboard_rows <= 0:
        max_dashboard_rows = len(records)
    dashboard_records = records[:max_dashboard_rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    html_filename = str(arguments.get("html_filename") or "alignment_review_dashboard.html")
    html_path = output_dir / html_filename
    csv_path = output_dir / "alignment_review_rows.csv"
    summary_path = output_dir / "alignment_review_summary.json"
    store_path = output_dir / "review_outcomes.sqlite"

    _write_csv(csv_path, records)
    summary = _summary(records)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    html_path.write_text(
        _html_page(
            {
                "records": dashboard_records,
                "summary": summary,
                "source_review_template": str(review_template),
                "total_review_rows": len(records),
                "dashboard_review_rows": len(dashboard_records),
                "max_dashboard_rows": max_dashboard_rows,
            }
        ),
        encoding="utf-8",
    )
    ReviewOutcomeSQLiteStore(store_path).upsert_batch(
        ReviewLabelBatch(
            batch_id=f"alignment-review-dashboard-{_stable_hash(str(review_template))}",
            source_template_paths=[str(review_template)],
            labels=labels,
            provenance=[ProvenanceRecord(source_type="alignment_review_template", source_id=str(review_template), method="alignment_review_dashboard_dual_write")],
        )
    )

    report = {
        "workflow_name": "alignment-review-dashboard",
        "status": "ok",
        "source_review_template": str(review_template),
        "review_row_count": len(records),
        "dashboard_row_count": len(dashboard_records),
        "dashboard_row_limit": max_dashboard_rows,
        "summary": summary,
        "html": str(html_path),
        "safe_conclusion": "This dashboard is a local human-review aid for alignment-state hypotheses. Labels are calibration evidence, not accepted alignment truth.",
        "next_actions": [
            "Open the HTML dashboard and label state transitions, plane candidates, and peak assignments.",
            "For large full-corpus passes, build a smaller per-session or filtered review template before detailed labeling.",
            "Export reviewed JSONL from the page.",
            "Ingest the export with `python -m tas_copilot_core.cli review labels ingest --labels [EXPORT.jsonl] --store local_runs/review_outcomes.sqlite`.",
        ],
        "artifacts": {
            "html": str(html_path),
            "review_rows_csv": str(csv_path),
            "review_summary_json": str(summary_path),
            "review_outcomes_sqlite": str(store_path),
        },
        "provenance": [ProvenanceRecord(source_type="alignment_review_template", source_id=str(review_template), method="build_alignment_review_dashboard").model_dump(mode="json")],
    }
    report["artifacts"].update(write_report_bundle(output_dir, report, markdown_renderer=_markdown, json_filename="alignment_review_dashboard_report.json", markdown_filename="alignment_review_dashboard_report.md"))
    return report


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _dashboard_record(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    object_type = str(row.get("object_type") or "alignment_review")
    object_id = str(row.get("object_id") or f"row-{index}")
    allowed = [str(item) for item in row.get("type_specific_allowed") or []]
    if not allowed:
        allowed = _allowed_statuses(object_type)
    event_ref = metadata.get("event_ref") or metadata.get("filename") or metadata.get("state") or metadata.get("dominant_stage") or ""
    return {
        "review_id": row.get("label_id") or f"alignment::{_stable_hash('|'.join([object_type, object_id, str(index)]))}",
        "row_number": index,
        "object_type": object_type,
        "object_id": object_id,
        "source_workflow": str(row.get("source_workflow") or "alignment-state-epochs"),
        "source_artifact_path": row.get("source_artifact_path"),
        "source_artifact_hash": row.get("source_artifact_hash"),
        "verdict": str(row.get("verdict") or "unreviewed"),
        "type_specific_status": row.get("type_specific_status"),
        "type_specific_allowed": allowed,
        "claim_state": str(row.get("claim_state") or "candidate"),
        "suggested_status": str(row.get("suggested_status") or "candidate"),
        "evidence_channels_relied_on": [str(item) for item in row.get("evidence_channels_relied_on") or []],
        "recommendation": str(row.get("recommendation") or "no_action"),
        "uncertainty_class": str(row.get("uncertainty_class") or "probability_style_triage_prior"),
        "review_notes": str(row.get("review_notes") or ""),
        "metadata": metadata,
        "display": {
            "stage": metadata.get("alignment_stage") or metadata.get("dominant_stage") or metadata.get("state") or "unknown",
            "confidence": metadata.get("confidence"),
            "devices": sorted((metadata.get("devices") or {}).keys()) if isinstance(metadata.get("devices"), dict) else metadata.get("scanned_devices") or [],
            "break_reasons": metadata.get("break_reasons") or [],
            "event_ref": event_ref,
        },
        "review_questions": _review_questions(object_type),
        "cautions": _cautions(object_type),
        "provenance": row.get("provenance") or [],
    }


def _record_to_label(record: dict[str, Any], *, source_template: Path) -> ReviewLabel:
    return ReviewLabel(
        label_id=str(record["review_id"]),
        object_type=record["object_type"],
        object_id=record["object_id"],
        source_workflow=record["source_workflow"],
        source_artifact_path=str(record.get("source_artifact_path") or source_template),
        source_artifact_hash=record.get("source_artifact_hash") or _path_hash(source_template),
        verdict=record["verdict"],
        type_specific_status=record.get("type_specific_status"),
        type_specific_allowed=record["type_specific_allowed"],
        claim_state=record["claim_state"],
        suggested_status=record["suggested_status"],
        evidence_channels_relied_on=record["evidence_channels_relied_on"],
        recommendation=record["recommendation"],
        uncertainty_class=record["uncertainty_class"],
        review_notes=record["review_notes"],
        legacy_provenance=record.get("provenance"),
        metadata={"dashboard_row_number": record["row_number"], **(record.get("metadata") or {})},
        provenance=[ProvenanceRecord(source_type="alignment_review_dashboard", source_id=str(source_template), method="record_to_label")],
    )


def _allowed_statuses(object_type: str) -> list[str]:
    if object_type == "alignment_state_transition":
        return ["accepted_continuous_epoch", "rejected_epoch_break", "needs_more_context"]
    if object_type == "alignment_plane_candidate":
        return ["accepted_plane_hypothesis", "rejected_plane_hypothesis", "needs_third_reflection", "needs_more_context"]
    if object_type == "alignment_peak_candidate":
        return ["accepted_peak_assignment", "rejected_powder_or_background", "rejected_lambda_half_alias", "needs_more_context"]
    return ["accepted", "rejected", "needs_more_context"]


def _review_questions(object_type: str) -> list[str]:
    if object_type == "alignment_state_transition":
        return [
            "Does this epoch look continuous, or is there evidence that the sample was removed, remounted, or the A3 frame reset?",
            "Do console-log events and scan recipes agree on the alignment stage?",
        ]
    if object_type == "alignment_plane_candidate":
        return [
            "Is this a plausible scattering-plane hypothesis from the observed tilt/second-reflection sequence?",
            "Is a third reflection or later real-scan lattice state needed before accepting the plane?",
        ]
    if object_type == "alignment_peak_candidate":
        return [
            "Is the peak assignment plausible, or could it be powder/background/lambda-half?",
            "Does the scan recipe, A4 window, and later lattice/scattering-plane state support the assignment?",
        ]
    return ["Is this alignment interpretation supported by the linked evidence?"]


def _cautions(object_type: str) -> list[str]:
    base = [
        "Alignment labels are review evidence, not approved instrument actions.",
        "Sample removal/remounting and rate-meter searches can be missing from logs.",
    ]
    if object_type == "alignment_peak_candidate":
        base.append("Powder/background, lambda-half, twins, and out-of-plane projections require material/filter/geometry evidence.")
    if object_type == "alignment_plane_candidate":
        base.append("Plane hypotheses should remain provisional without enough independent reflections.")
    return base


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(records),
        "by_object_type": dict(Counter(row["object_type"] for row in records)),
        "by_suggested_status": dict(Counter(row["suggested_status"] for row in records)),
        "by_stage": dict(Counter(str((row.get("display") or {}).get("stage") or "unknown") for row in records)),
        "by_verdict": dict(Counter(row["verdict"] for row in records)),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = ["review_id", "row_number", "object_type", "object_id", "suggested_status", "stage", "confidence", "devices", "verdict", "type_specific_allowed"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            display = record.get("display") or {}
            writer.writerow(
                {
                    "review_id": record.get("review_id"),
                    "row_number": record.get("row_number"),
                    "object_type": record.get("object_type"),
                    "object_id": record.get("object_id"),
                    "suggested_status": record.get("suggested_status"),
                    "stage": display.get("stage"),
                    "confidence": display.get("confidence"),
                    "devices": json.dumps(display.get("devices") or []),
                    "verdict": record.get("verdict"),
                    "type_specific_allowed": json.dumps(record.get("type_specific_allowed") or []),
                }
            )


def _html_page(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, sort_keys=True).replace("</script>", "<\\/script>")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BT-7 Alignment Review Dashboard</title>
<style>
:root {{ --bg:#f6f8fb; --card:#fff; --ink:#18212f; --muted:#596579; --line:#d8dee9; --accent:#174ea6; --ok:#166534; --bad:#991b1b; --warn:#92400e; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--bg); }}
header {{ padding:20px 28px; background:#fff; border-bottom:1px solid var(--line); position:sticky; top:0; z-index:10; }}
h1 {{ margin:0 0 6px; font-size:24px; }}
p {{ margin:0 0 8px; line-height:1.45; }}
main {{ display:grid; grid-template-columns:380px minmax(0,1fr); gap:16px; padding:16px; }}
aside, section {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px; }}
.controls {{ display:grid; gap:10px; }}
label {{ display:grid; gap:4px; font-size:13px; color:var(--muted); }}
input, select, textarea, button {{ font:inherit; }}
input, select, textarea {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:7px 8px; background:#fff; }}
textarea {{ min-height:96px; resize:vertical; }}
button {{ border:1px solid var(--accent); background:var(--accent); color:#fff; border-radius:6px; padding:7px 10px; cursor:pointer; }}
button.secondary {{ background:#fff; color:var(--accent); }}
button.ghost {{ background:#fff; color:var(--ink); border-color:var(--line); }}
.metric-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin:12px 0; }}
.metric {{ border:1px solid var(--line); border-radius:6px; padding:8px; background:#fbfcfe; }}
.metric b {{ display:block; font-size:20px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th, td {{ border-bottom:1px solid var(--line); text-align:left; vertical-align:top; padding:7px; }}
th {{ position:sticky; top:78px; background:#f0f3f8; z-index:5; }}
tr.selected {{ outline:2px solid var(--accent); outline-offset:-2px; }}
.badge {{ display:inline-block; border:1px solid var(--line); border-radius:999px; padding:1px 7px; font-size:12px; background:#fff; margin:1px 2px 1px 0; }}
.badge.positive, .badge.accepted {{ color:var(--ok); border-color:#bbf7d0; background:#f0fdf4; }}
.badge.negative, .badge.rejected {{ color:var(--bad); border-color:#fecaca; background:#fff5f5; }}
.badge.inconclusive, .badge.needs_more_context, .badge.needs_third_reflection {{ color:var(--warn); border-color:#fed7aa; background:#fff7ed; }}
.small {{ color:var(--muted); font-size:12px; }}
.detail-list {{ margin:8px 0; padding-left:18px; }}
.detail-list li {{ margin:4px 0; }}
@media (max-width:980px) {{ main {{ grid-template-columns:1fr; }} th {{ top:0; }} header {{ position:static; }} }}
</style>
</head>
<body>
<header>
<h1>BT-7 Alignment Review Dashboard</h1>
<p>Review alignment-state transitions, plane hypotheses, and peak assignments from local scan/log evidence. Labels are calibration evidence, not approved instrument actions.</p>
<p class="small">Dashboard rows: <code>{payload.get("dashboard_review_rows", len(payload.get("records", [])))}</code> of <code>{payload.get("total_review_rows", len(payload.get("records", [])))}</code>. For large full-corpus passes, label one real session or a filtered packet at a time.</p>
<p class="small">Source: <code>{html.escape(str(payload.get("source_review_template") or ""))}</code></p>
</header>
<main>
<aside>
  <div class="controls">
    <label>Text filter <input id="textFilter" type="search" placeholder="object, stage, device, filename"></label>
    <label>Object type <select id="typeFilter"><option value="">all</option><option>alignment_state_transition</option><option>alignment_plane_candidate</option><option>alignment_peak_candidate</option></select></label>
    <label>Stage <input id="stageFilter" type="search" placeholder="fine, gross, plane..."></label>
    <label>Review status <select id="statusFilter"><option value="">all</option><option>unreviewed</option><option>positive</option><option>negative</option><option>inconclusive</option></select></label>
    <button id="exportReviewed">Export reviewed JSONL</button>
    <button id="exportAll" class="secondary">Export all rows JSONL</button>
    <button id="showAll" class="secondary">Show all rows</button>
    <button id="clearLocal" class="ghost">Clear local page edits</button>
  </div>
  <div class="metric-grid">
    <div class="metric"><b id="visibleCount">0</b><span>visible</span></div>
    <div class="metric"><b id="reviewedCount">0</b><span>reviewed</span></div>
    <div class="metric"><b id="positiveCount">0</b><span>positive</span></div>
    <div class="metric"><b id="negativeCount">0</b><span>negative</span></div>
  </div>
  <p id="renderNote" class="small"></p>
  <p id="renderError" class="small" style="display:none;color:var(--bad);"></p>
  <h3>Selected row</h3>
  <div id="detail" class="small">Select a row to review.</div>
</aside>
<section>
  <table id="reviewTable">
    <thead><tr><th>#</th><th>Object</th><th>Stage</th><th>Devices</th><th>Status</th><th>Evidence</th></tr></thead>
    <tbody></tbody>
  </table>
</section>
</main>
<script id="payload" type="application/json">{payload_json}</script>
<script>
const payload = JSON.parse(document.getElementById('payload').textContent);
const records = Array.isArray(payload.records) ? payload.records : [];
const renderLimit = 500;
const storageKey = 'tas-alignment-review::' + payload.source_review_template;
let edits = {{}};
try {{ edits = JSON.parse(localStorage.getItem(storageKey) || '{{}}') || {{}}; }} catch (err) {{ edits = {{}}; localStorage.removeItem(storageKey); }}
let selectedId = null;
function escapeHtml(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function mergedRecord(r) {{ return {{...r, ...(edits[r.review_id] || {{}})}}; }}
function statusToVerdict(status) {{
  if (!status || status === 'unreviewed') return 'unreviewed';
  if (status.startsWith('accepted_')) return 'positive';
  if (status.startsWith('rejected_')) return 'negative';
  if (status.startsWith('needs_')) return 'inconclusive';
  return status === 'accepted' ? 'positive' : status === 'rejected' ? 'negative' : 'inconclusive';
}}
function statusToClaimState(status) {{
  if (statusToVerdict(status) === 'positive') return 'accepted';
  if (statusToVerdict(status) === 'negative') return 'rejected';
  return 'candidate';
}}
function statusToRecommendation(status) {{
  const verdict = statusToVerdict(status);
  if (verdict === 'positive') return 'calibrate_prior';
  if (verdict === 'negative') return 'downweight';
  if (verdict === 'inconclusive') return 'escalate';
  return 'no_action';
}}
function badge(text, cls='') {{ return `<span class="badge ${{cls}}">${{escapeHtml(text || '')}}</span>`; }}
function rowText(r) {{ return [r.object_type,r.object_id,r.suggested_status,r.display?.stage,(r.display?.devices||[]).join(' '),r.display?.event_ref,(r.evidence_channels_relied_on||[]).join(' ')].join(' ').toLowerCase(); }}
function passes(r) {{
  const m = mergedRecord(r);
  const q = document.getElementById('textFilter').value.toLowerCase();
  const type = document.getElementById('typeFilter').value;
  const stage = document.getElementById('stageFilter').value.toLowerCase();
  const status = document.getElementById('statusFilter').value;
  const verdict = m.verdict || statusToVerdict(m.type_specific_status);
  return (!q || rowText(r).includes(q)) && (!type || r.object_type === type) && (!stage || String(r.display?.stage || '').toLowerCase().includes(stage)) && (!status || verdict === status);
}}
function render() {{
  const visible = records.filter(passes);
  const visiblePage = visible.slice(0, renderLimit);
  document.querySelector('#reviewTable tbody').innerHTML = visiblePage.map(r => {{
    const m = mergedRecord(r);
    const verdict = m.verdict || statusToVerdict(m.type_specific_status);
    const devices = Array.isArray(r.display?.devices) ? r.display.devices.join(', ') : String(r.display?.devices || '');
    const ev = (r.evidence_channels_relied_on || []).slice(0, 4).map(e => badge(e)).join('');
    return `<tr data-id="${{escapeHtml(r.review_id)}}" class="${{selectedId === r.review_id ? 'selected' : ''}}">
      <td>${{r.row_number}}</td>
      <td><b>${{escapeHtml(r.object_type)}}</b><br><code>${{escapeHtml(r.object_id)}}</code><br><span class="small">${{escapeHtml(r.suggested_status)}}</span></td>
      <td>${{escapeHtml(r.display?.stage || 'unknown')}}<br><span class="small">${{escapeHtml(r.display?.event_ref || '')}}</span></td>
      <td>${{escapeHtml(devices)}}</td>
      <td>${{badge(verdict, verdict)}}<br><span class="small">${{escapeHtml(m.type_specific_status || 'unreviewed')}}</span></td>
      <td>${{ev}}</td></tr>`;
  }}).join('');
  document.querySelectorAll('#reviewTable tbody tr').forEach(tr => tr.addEventListener('click', () => selectRow(tr.dataset.id)));
  const merged = records.map(mergedRecord);
  document.getElementById('visibleCount').textContent = visible.length;
  document.getElementById('reviewedCount').textContent = merged.filter(r => (r.verdict || 'unreviewed') !== 'unreviewed').length;
  document.getElementById('positiveCount').textContent = merged.filter(r => (r.verdict || '') === 'positive').length;
  document.getElementById('negativeCount').textContent = merged.filter(r => (r.verdict || '') === 'negative').length;
  document.getElementById('renderNote').textContent = visible.length > renderLimit ? `Showing first ${{renderLimit}} matching rows. Narrow the filters to review a smaller session slice.` : `Showing ${{visible.length}} matching rows.`;
}}
function saveEdit(id, patch) {{ edits[id] = {{...(edits[id] || {{}}), ...patch}}; localStorage.setItem(storageKey, JSON.stringify(edits)); render(); selectRow(id); }}
function selectRow(id) {{
  selectedId = id;
  const base = records.find(r => r.review_id === id);
  if (!base) return;
  const r = mergedRecord(base);
  const options = ['unreviewed', ...(base.type_specific_allowed || [])].map(s => `<option value="${{s}}" ${{(r.type_specific_status || 'unreviewed') === s ? 'selected' : ''}}>${{s}}</option>`).join('');
  const display = base.display || {{}};
  document.getElementById('detail').innerHTML = `<p><b>${{escapeHtml(base.object_type)}}</b><br><code>${{escapeHtml(base.object_id)}}</code></p>
    <p>Stage: <code>${{escapeHtml(display.stage || 'unknown')}}</code><br>Confidence: <code>${{escapeHtml(display.confidence ?? 'unknown')}}</code><br>Devices: <code>${{escapeHtml((display.devices || []).join(', '))}}</code></p>
    <label>Status <select id="editStatus">${{options}}</select></label>
    <label>Reviewer <input id="editReviewer" value="${{escapeHtml(r.reviewer_id || '')}}" placeholder="name"></label>
    <label>Notes <textarea id="editNotes" placeholder="What evidence supports or rejects this interpretation?">${{escapeHtml(r.review_notes || '')}}</textarea></label>
    <button id="saveRow">Save row locally</button>
    <h4>Review questions</h4><ul class="detail-list">${{(base.review_questions || []).map(e => `<li>${{escapeHtml(e)}}</li>`).join('')}}</ul>
    <h4>Cautions</h4><ul class="detail-list">${{(base.cautions || []).map(e => `<li>${{escapeHtml(e)}}</li>`).join('')}}</ul>
    <h4>Metadata</h4><pre>${{escapeHtml(JSON.stringify(base.metadata || {{}}, null, 2))}}</pre>`;
  document.getElementById('saveRow').addEventListener('click', () => {{
    const status = document.getElementById('editStatus').value;
    saveEdit(id, {{
      type_specific_status: status === 'unreviewed' ? null : status,
      verdict: statusToVerdict(status),
      claim_state: statusToClaimState(status),
      recommendation: statusToRecommendation(status),
      reviewer_id: document.getElementById('editReviewer').value || null,
      reviewed_at: status === 'unreviewed' ? null : new Date().toISOString(),
      review_notes: document.getElementById('editNotes').value || ''
    }});
  }});
  render();
}}
function exportRows(reviewedOnly) {{
  const rows = records.map(mergedRecord).filter(r => !reviewedOnly || (r.verdict || 'unreviewed') !== 'unreviewed').map(r => ({{
    label_id:r.review_id, object_type:r.object_type, object_id:r.object_id, source_workflow:r.source_workflow,
    source_artifact_path:r.source_artifact_path, source_artifact_hash:r.source_artifact_hash,
    verdict:r.verdict || 'unreviewed', type_specific_status:r.type_specific_status || null, type_specific_allowed:r.type_specific_allowed || [],
    claim_state:r.claim_state || statusToClaimState(r.type_specific_status), suggested_status:r.suggested_status || 'candidate',
    evidence_channels_relied_on:r.evidence_channels_relied_on || [], recommendation:r.recommendation || statusToRecommendation(r.type_specific_status),
    uncertainty_class:r.uncertainty_class || 'probability_style_triage_prior', review_notes:r.review_notes || '',
    reviewer_id:r.reviewer_id || null, reviewed_at:r.reviewed_at || null, metadata:r.metadata || {{}}, provenance:r.provenance || []
  }}));
  const blob = new Blob([rows.map(x => JSON.stringify(x)).join('\\n') + (rows.length ? '\\n' : '')], {{type:'application/jsonl'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = reviewedOnly ? 'alignment_reviewed_rows.jsonl' : 'alignment_review_all_rows.jsonl'; a.click(); URL.revokeObjectURL(a.href);
}}
['textFilter','typeFilter','stageFilter','statusFilter'].forEach(id => document.getElementById(id).addEventListener('input', render));
document.getElementById('exportReviewed').addEventListener('click', () => exportRows(true));
document.getElementById('exportAll').addEventListener('click', () => exportRows(false));
document.getElementById('showAll').addEventListener('click', () => {{ document.getElementById('textFilter').value=''; document.getElementById('typeFilter').value=''; document.getElementById('stageFilter').value=''; document.getElementById('statusFilter').value=''; render(); }});
document.getElementById('clearLocal').addEventListener('click', () => {{ if (confirm('Clear local edits for this review page?')) {{ edits={{}}; localStorage.removeItem(storageKey); selectedId=null; render(); document.getElementById('detail').textContent='Select a row to review.'; }} }});
try {{ render(); }} catch (err) {{ document.getElementById('renderError').style.display='block'; document.getElementById('renderError').textContent='Interactive render failed: ' + err.message; }}
</script>
</body>
</html>
"""


def _fallback_row(record: dict[str, Any]) -> str:
    display = record.get("display") or {}
    devices = display.get("devices") or []
    evidence = "".join(f"<span class=\"badge\">{html.escape(str(item))}</span>" for item in (record.get("evidence_channels_relied_on") or [])[:4])
    return (
        f"<tr data-id=\"{html.escape(str(record.get('review_id')))}\">"
        f"<td>{record.get('row_number')}</td>"
        f"<td><b>{html.escape(str(record.get('object_type')))}</b><br><code>{html.escape(str(record.get('object_id')))}</code></td>"
        f"<td>{html.escape(str(display.get('stage') or 'unknown'))}</td>"
        f"<td>{html.escape(', '.join(str(item) for item in devices))}</td>"
        f"<td><span class=\"badge {html.escape(str(record.get('verdict') or 'unreviewed'))}\">{html.escape(str(record.get('verdict') or 'unreviewed'))}</span></td>"
        f"<td>{evidence}</td>"
        "</tr>"
    )


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Alignment Review Dashboard",
        "",
        f"Status: `{report.get('status')}`",
        f"Rows: `{report.get('review_row_count', 0)}`",
        f"HTML: `{report.get('html')}`",
        f"Template: `{report.get('source_review_template')}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in (report.get("summary") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Next Actions", ""])
    lines.extend(f"- {item}" for item in report.get("next_actions") or [])
    lines.extend(["", report.get("safe_conclusion", "")])
    return "\n".join(lines) + "\n"


def _path_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
