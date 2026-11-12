from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_contracts import FigureAxisConstraint, FigurePanelConstraint, PaperExperimentLink, ProvenanceRecord, SourceChunk, TextbookSection
from tas_copilot_core.catalogs.jsonl_catalog import JsonlCatalog
from tas_copilot_core.material_identity import parse_material_text


def load_corpus_fixtures(fixtures: str | Path, catalog: str | Path) -> dict[str, Any]:
    fixture_dir = Path(fixtures)
    catalog_dir = Path(catalog)
    if not fixture_dir.exists():
        return {
            "status": "missing_input",
            "errors": [{"code": "missing_fixture_dir", "message": f"fixture directory does not exist: {fixture_dir}"}],
        }
    chunks = JsonlCatalog(fixture_dir / "source_chunks.jsonl", SourceChunk).load()
    textbook_sections = JsonlCatalog(fixture_dir / "textbook_sections.jsonl", TextbookSection).load()
    mappings = JsonlCatalog(fixture_dir / "mappings.jsonl", PaperExperimentLink).load()
    JsonlCatalog(catalog_dir / "source_chunks.jsonl", SourceChunk).save(chunks)
    JsonlCatalog(catalog_dir / "textbook_sections.jsonl", TextbookSection).save(textbook_sections)
    JsonlCatalog(catalog_dir / "mappings.jsonl", PaperExperimentLink).save(mappings)
    for filename in [
        "documents.jsonl",
        "papers.jsonl",
        "figures.jsonl",
        "figure_panels.jsonl",
        "figure_captions.jsonl",
        "claims.jsonl",
        "tables.jsonl",
        "panel_source_candidates.jsonl",
        "panel_source_links.jsonl",
        "source_support.jsonl",
    ]:
        _copy_jsonl_fixture(fixture_dir / filename, catalog_dir / filename)
    report = corpus_fixture_report(catalog_dir)
    report.update({"status": "ok", "fixture_dir": str(fixture_dir), "catalog_dir": str(catalog_dir)})
    return report


def corpus_fixture_report(catalog: str | Path) -> dict[str, Any]:
    catalog_dir = Path(catalog)
    chunks = JsonlCatalog(catalog_dir / "source_chunks.jsonl", SourceChunk).load()
    textbook_sections = JsonlCatalog(catalog_dir / "textbook_sections.jsonl", TextbookSection).load()
    mappings = JsonlCatalog(catalog_dir / "mappings.jsonl", PaperExperimentLink).load()
    raw_counts = {path.stem: len(_load_jsonl_dicts(path)) for path in catalog_dir.glob("*.jsonl") if path.name not in {"source_chunks.jsonl", "textbook_sections.jsonl", "mappings.jsonl"}}
    paper_chunks = [chunk for chunk in chunks if chunk.source_type == "paper"]
    textbook_chunks = [chunk for chunk in chunks if chunk.source_type == "textbook"]
    provenance_records = sum(1 for item in [*chunks, *textbook_sections, *mappings] if item.provenance)
    total_records = len(chunks) + len(textbook_sections) + len(mappings)
    return {
        "status": "ok" if catalog_dir.exists() else "missing_input",
        "catalog_dir": str(catalog_dir),
        "paper_chunks": len(paper_chunks),
        "textbook_chunks": len(textbook_chunks),
        "source_chunks": len(chunks),
        "textbook_sections": len(textbook_sections),
        "mappings": len(mappings),
        "papers": raw_counts.get("papers", 0),
        "figures": raw_counts.get("figures", 0),
        "figure_panels": raw_counts.get("figure_panels", 0),
        "figure_captions": raw_counts.get("figure_captions", 0),
        "panel_source_candidates": raw_counts.get("panel_source_candidates", 0),
        "panel_source_links": raw_counts.get("panel_source_links", 0),
        "source_support_records": raw_counts.get("source_support", 0),
        "provenance_coverage": 1.0 if total_records == 0 else round(provenance_records / total_records, 3),
        "source_support_validation": "ok" if provenance_records == total_records and total_records > 0 else "warning",
        "separation_policy": "paper/textbook corpus fixtures are JSONL document catalogs; TAS data-file catalogs remain SQLite and separate",
    }


def compile_figure_query(
    corpus_catalog: str | Path,
    paper_id: str | None,
    figure_id: str,
    panel_id: str,
    output: str | Path | None = None,
    user_sample: str | None = None,
) -> dict[str, Any]:
    catalog_dir = Path(corpus_catalog)
    papers = _load_jsonl_dicts(catalog_dir / "papers.jsonl")
    figures = _load_jsonl_dicts(catalog_dir / "figures.jsonl")
    panels = _load_jsonl_dicts(catalog_dir / "figure_panels.jsonl")
    figure = next((item for item in figures if item.get("figure_id") == figure_id and (paper_id is None or item.get("paper_id") == paper_id)), None)
    panel = next((item for item in panels if item.get("figure_id") == figure_id and item.get("panel_id") == panel_id), None)
    if figure is None or panel is None:
        return {
            "status": "missing_input",
            "missing_fields": [field for field, value in [("figure", figure), ("figure_panel", panel)] if value is None],
            "warnings": ["A paper/figure/panel fixture record is required to compile a figure-source query."],
        }
    paper = next((item for item in papers if item.get("paper_id") == figure.get("paper_id")), {})
    target_x = _panel_x(panel)
    sample = user_sample or ""
    sample_identity = parse_material_text(sample) if sample else None
    axes = panel.get("axes") or []
    q_axis = _axis(axes, "Q")
    temperature_axis = _axis(axes, "temperature")
    annotations = panel.get("annotations") or []
    alternate_panels = []
    for other in panels:
        if other.get("figure_id") != figure_id or other.get("panel_id") == panel_id:
            continue
        x_value = _panel_x(other)
        if x_value is not None:
            alternate_panels.append(FigurePanelConstraint(panel_id=str(other["panel_id"]), composition={"x_mn": x_value}).model_dump(mode="json"))
    warnings = []
    if sample_identity is not None and sample_identity.x_variables.get("x_Mn") is not None and target_x is not None and abs(float(sample_identity.x_variables["x_Mn"]) - float(target_x)) > 0.025:
        likely = next((item for item in panels if item.get("figure_id") == figure_id and abs(float(_panel_x(item) or -999) - float(sample_identity.x_variables["x_Mn"])) <= 0.025), None)
        warnings.append(
            f"Panel/sample mismatch: user sample {sample} has x_Mn={float(sample_identity.x_variables['x_Mn']):g}, "
            f"while panel {panel_id} has x_Mn={float(target_x):g}."
            + (f" The user sample is more consistent with panel {likely.get('panel_id')}." if likely else "")
        )
    query = {
        "query_id": f"{figure_id}_{panel_id}_source_query",
        "paper_id": figure.get("paper_id"),
        "figure_id": figure_id,
        "target_panel": panel_id,
        "target_panel_label": (panel.get("panel_label") or {}).get("label_text") or panel.get("label"),
        "alternate_panels": alternate_panels,
        "material_family": "LuFe_{1-x}Mn_xO3",
        "user_sample_formula": sample,
        "target_composition": {"x_mn": target_x, "measurement_kind": panel.get("measurement_kind")},
        "instrument": paper.get("instrument") or "BT-7",
        "measurement_mode": panel.get("detector_mode") or "PSD",
        "measurement_kind": panel.get("measurement_kind"),
        "required_energy_transfer": 0 if str(panel.get("measurement_kind") or "").lower() == "elastic" else None,
        "expected_source_families": ["single_file_psd_temperature_map", "multi_file_psd_angle_temperature_series"] if str(panel.get("detector_mode") or "").upper() == "PSD" else [],
        "excluded_source_families": ["single_detector_inelastic_line_scan", "nonzero_energy_line_scan"] if str(panel.get("measurement_kind") or "").lower() == "elastic" else [],
        "sample_type": "powder",
        "q_range": q_axis,
        "temperature_range": temperature_axis,
        "annotated_q_values": [item.get("value") for item in annotations if str(item.get("axis_name")).lower() == "q" and item.get("value") is not None],
        "notes": [
            "Compiled from document-side paper/figure/panel fixture metadata.",
            "Candidate source links remain proposed until reconstruction/provenance or human review.",
            *warnings,
        ],
        "provenance": [
            ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="compile_figure_query").model_dump(mode="json"),
            ProvenanceRecord(source_type="figure_panel", source_id=f"{figure_id}:{panel_id}", method="compile_figure_query").model_dump(mode="json"),
        ],
    }
    payload = {"status": "ok", "query": query, "panel": panel, "figure": figure, "warnings": warnings}
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in {".yaml", ".yml"}:
            output_path.write_text(yaml.safe_dump(query, sort_keys=False), encoding="utf-8")
        else:
            output_path.write_text(json.dumps(query, indent=2, sort_keys=True), encoding="utf-8")
        payload["output"] = str(output_path)
    return payload


def validate_source_support(catalog: str | Path) -> dict[str, Any]:
    catalog_dir = Path(catalog)
    panels = _load_jsonl_dicts(catalog_dir / "figure_panels.jsonl")
    support_records = _load_jsonl_dicts(catalog_dir / "source_support.jsonl")
    links = _load_jsonl_dicts(catalog_dir / "panel_source_links.jsonl")
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for panel in panels:
        x_value = _panel_x(panel)
        label_x = (panel.get("panel_label") or {}).get("parsed_fields", {}).get("x_mn")
        claim = {"claim": f"panel_{panel.get('panel_id')}_x_mn", "expected": x_value, "label": label_x, "panel_id": panel.get("panel_id")}
        if x_value is not None and label_x is not None and abs(float(x_value) - float(label_x)) <= 0.025:
            supported.append(claim)
        elif x_value is None or label_x is None:
            ambiguous.append({**claim, "reason": "missing panel composition or parsed label"})
        else:
            unsupported.append({**claim, "reason": "panel composition and label disagree"})
    sample = parse_material_text("LuFe0.6Mn0.4O3")
    supported.append({"claim": "user_sample_lufe06mn04_x_mn", "expected": 0.4, "actual": sample.x_variables.get("x_Mn"), "source": "material_identity"})
    for link in links:
        if link.get("status") in {"accepted", "reviewed"} and (link.get("reconstruction_required") or link.get("human_review_required")):
            unsupported.append({"claim": "accepted_source_link_without_review", "link": link, "reason": "accepted/reviewed links must clear reconstruction and human-review requirements"})
        elif link.get("status") in {"candidate", "proposed"}:
            supported.append({"claim": "candidate_link_remains_non_definitive", "link": link.get("source_object_id"), "status": link.get("status")})
    missing = []
    if not support_records:
        missing.append("source_support")
    total_claims = len(supported) + len(unsupported) + len(ambiguous)
    return {
        "status": "ok" if not unsupported and not missing else "warning",
        "catalog_dir": str(catalog_dir),
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "ambiguous_claims": ambiguous,
        "missing_source_records": missing,
        "source_support_record_count": len(support_records),
        "claim_count": total_claims,
        "warnings": ["Source-support validation is deterministic fixture validation, not a general LLM source-support grader."],
        "provenance": [ProvenanceRecord(source_type="corpus_catalog", source_id=str(catalog_dir), method="validate_source_support").model_dump(mode="json")],
    }


def _copy_jsonl_fixture(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    records = _load_jsonl_dicts(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _panel_x(panel: dict[str, Any]) -> float | None:
    composition = panel.get("composition") or {}
    value = composition.get("value") if isinstance(composition, dict) else None
    return float(value) if value is not None else None


def _axis(axes: list[dict[str, Any]], name: str) -> dict[str, Any]:
    axis = next((item for item in axes if str(item.get("axis_name")).lower() == name.lower()), {})
    return FigureAxisConstraint(
        axis_name=name,
        min_value=axis.get("min_value"),
        max_value=axis.get("max_value"),
        unit=axis.get("unit"),
    ).model_dump(mode="json")
