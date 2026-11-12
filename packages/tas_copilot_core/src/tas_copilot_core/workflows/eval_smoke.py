from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import yaml

from tas_copilot_contracts import DataFile, EnergyTransfer, Experiment, InstrumentConfig, PaperExperimentLink, QPoint, Scan, SourceChunk, UnitValue
from tas_copilot_core.catalogs.data_catalog_quality import data_catalog_quality_report
from tas_copilot_core.catalogs.jsonl_catalog import JsonlCatalog
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.ontology import load_ontology, supernode_diagnostics
from tas_copilot_core.ontology.linker import match_concepts
from tas_copilot_core.retrieval import HybridRetriever, KeywordRetriever, compare_scans
from tas_copilot_core.workflows.check_geometry import check_geometry
from tas_copilot_core.workflows.check_resolution import check_resolution
from tas_copilot_core.workflows.experiment_brief import experiment_brief
from tas_copilot_core.workflows.find_experiments import find_experiments
from tas_copilot_core.workflows.find_papers import find_papers
from tas_copilot_core.workflows.scan_brief import scan_brief


def eval_smoke(arguments: dict | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    config = CoreConfig()
    eval_dir = Path(arguments.get("eval_dir") or config.root / "eval")
    catalog_dir = Path(arguments.get("catalog_dir") or config.catalog_dir)
    limit = int(arguments.get("limit", 5))

    questions = _load_yaml(eval_dir / "sample_questions.yaml").get("questions", [])
    golden_mappings = _load_yaml(eval_dir / "golden_mappings.yaml").get("mappings", [])
    regression_tasks = _load_yaml(eval_dir / "regression_tasks.yaml").get("tasks", [])
    retrieval_tasks = _load_yaml(eval_dir / "retrieval_regression_questions.yaml").get("tasks", [])
    concepts = load_ontology(config.ontology_path)
    chunks = JsonlCatalog(catalog_dir / "source_chunks.jsonl", SourceChunk).load()
    scans = JsonlCatalog(catalog_dir / "scans.jsonl", Scan).load()
    experiments = JsonlCatalog(catalog_dir / "experiments.jsonl", Experiment).load()
    mappings = JsonlCatalog(catalog_dir / "mappings.jsonl", PaperExperimentLink).load()
    hybrid = HybridRetriever(KeywordRetriever(chunks), concepts=concepts)

    question_reports = []
    for question in questions:
        paper_results = find_papers(question, limit=limit)
        experiment_results = find_experiments(question, limit=limit)
        hybrid_results = hybrid.search(question, limit=limit)
        matched_concepts = [concept.id for concept in match_concepts(question, concepts)]
        question_reports.append(
            {
                "question": question,
                "matched_concepts": matched_concepts,
                "paper_result_ids": [result.source_id for result in paper_results],
                "experiment_result_ids": [result.source_id for result in experiment_results],
                "hybrid_result_ids": [result.source_id for result in hybrid_results],
                "status": "pass" if (paper_results or experiment_results or hybrid_results) and matched_concepts else "fail",
            }
        )

    scan_similarity = []
    if len(scans) >= 2:
        for candidate in scans[1:limit]:
            comparison = compare_scans(scans[0], candidate)
            comparison["reference_scan_id"] = scans[0].scan_id
            comparison["candidate_scan_id"] = candidate.scan_id
            scan_similarity.append(comparison)

    mapping_reports = []
    by_key = {(mapping.paper_id, mapping.experiment_id): mapping for mapping in mappings}
    for expected in golden_mappings:
        key = (expected.get("paper_id"), expected.get("experiment_id"))
        actual = by_key.get(key)
        mapping_reports.append(
            {
                "paper_id": key[0],
                "experiment_id": key[1],
                "expected_status": expected.get("status"),
                "actual_status": actual.status if actual else None,
                "confidence": actual.confidence if actual else None,
                "status": "pass" if actual and (expected.get("status") is None or actual.status == expected.get("status")) else "fail",
            }
        )

    regression_reports = _run_regression_tasks(regression_tasks)
    retrieval_reports = _run_retrieval_tasks(retrieval_tasks, chunks, scans, experiments, mappings)

    ontology_warnings = supernode_diagnostics(
        [chunk.text for chunk in chunks] + [_scan_text(scan) for scan in scans] + [_experiment_text(experiment) for experiment in experiments],
        concepts,
        max_fraction=float(arguments.get("max_concept_fraction", 0.8)),
    )
    checks = question_reports + mapping_reports + regression_reports + retrieval_reports
    failed = [item for item in checks if item.get("status") != "pass"]
    return {
        "status": "pass" if not failed and question_reports else "fail",
        "question_count": len(question_reports),
        "mapping_check_count": len(mapping_reports),
        "regression_check_count": len(regression_reports),
        "failed_count": len(failed),
        "questions": question_reports,
        "scan_similarity": scan_similarity,
        "mappings": mapping_reports,
        "regressions": regression_reports,
        "retrieval_regressions": retrieval_reports,
        "ontology_warning_count": len(ontology_warnings),
        "ontology_warnings": ontology_warnings,
        "catalog_counts": {
            "chunks": len(chunks),
            "experiments": len(experiments),
            "scans": len(scans),
            "mappings": len(mappings),
        },
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _run_regression_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    with tempfile.TemporaryDirectory(prefix="tascopilot-eval-") as tmp:
        fixture_db = _path_only_fixture_catalog(Path(tmp) / "path-only.sqlite")
        complete_geometry_db = _complete_lattice_fixture_catalog(Path(tmp) / "complete-lattice.sqlite")
        compatible_ub_db = _compatible_ub_fixture_catalog(Path(tmp) / "compatible-ub.sqlite")
        brief_db = _brief_fixture_catalog(Path(tmp) / "brief.sqlite")
        for task in tasks:
            name = task.get("name") or "unnamed"
            workflow = task.get("workflow")
            expected = task.get("expected_status")
            observed = None
            details: dict[str, Any] = {}
            if workflow == "check-resolution":
                result = check_resolution({})
                observed = result.status.value
                details = {"tool_namespace": result.tool_namespace, "tool_name": result.tool_name}
            elif workflow == "catalog-quality" and task.get("fixture") == "path_only_scan":
                result = data_catalog_quality_report(fixture_db)
                observed = result["status"]
                details = {
                    "metadata_modes": result["diagnostics"]["metadata_modes"],
                    "warnings": result["warnings"],
                }
            elif workflow == "check-geometry" and task.get("fixture") == "path_only_scan":
                result = check_geometry({"scan_id": "scan-hfir-0059", "db_path": str(fixture_db)})
                observed = result["status"]
                details = {
                    "missing_fields": result.get("missing_fields", []),
                    "geometry_mode": result.get("geometry_mode"),
                }
            elif workflow == "check-geometry" and task.get("fixture") == "complete_lattice_scan":
                result = check_geometry({"scan_id": "scan-geometry-001", "db_path": str(complete_geometry_db)})
                observed = result["status"]
                details = {
                    "missing_fields": result.get("missing_fields", []),
                    "geometry_mode": result.get("geometry_mode"),
                    "result_tools": sorted(result.get("results", {}).keys()),
                }
            elif workflow == "check-geometry" and task.get("fixture") == "compatible_ub_scan":
                result = check_geometry({"scan_id": "scan-compatible-ub", "db_path": str(compatible_ub_db), "geometry_mode": "compatible_ub"})
                observed = result["status"]
                details = {
                    "geometry_mode": result.get("geometry_mode"),
                    "error_codes": sorted({error["code"] for item in result.get("results", {}).values() for error in item.get("errors", [])}),
                    "tool_audit_ids": result.get("tool_audit_ids", {}),
                }
            elif workflow == "scan-brief":
                output_dir = Path(tmp) / "scan-brief"
                result = scan_brief({"scan_id": "scan-brief-001", "catalog": str(brief_db), "output_dir": str(output_dir)})
                observed = "ok" if result.get("artifacts") and result.get("workflow_audit_id") and result.get("geometry_check") else "missing_artifacts"
                details = {
                    "status": result.get("status"),
                    "artifact_json_exists": (output_dir / "scan-brief-001_scan_brief.json").exists(),
                    "artifact_markdown_exists": (output_dir / "scan-brief-001_scan_brief.md").exists(),
                    "geometry_status": result.get("geometry_check", {}).get("status"),
                }
            elif workflow == "experiment-brief":
                output_dir = Path(tmp) / "experiment-brief"
                result = experiment_brief({"experiment_id": "exp-brief", "catalog": str(brief_db), "output_dir": str(output_dir)})
                observed = "ok" if result.get("artifacts") and result.get("workflow_audit_id") and result.get("paper_mapping_count", 0) >= 1 else "missing_artifacts"
                details = {
                    "status": result.get("status"),
                    "scan_count": result.get("scan_count"),
                    "paper_mapping_count": result.get("paper_mapping_count"),
                    "artifact_json_exists": (output_dir / "exp-brief_experiment_brief.json").exists(),
                    "artifact_markdown_exists": (output_dir / "exp-brief_experiment_brief.md").exists(),
                }
            else:
                observed = "not_implemented"
                details = {"workflow": workflow, "fixture": task.get("fixture")}
            reports.append(
                {
                    "name": name,
                    "workflow": workflow,
                    "expected_status": expected,
                    "actual_status": observed,
                    "status": "pass" if _status_matches(observed, task) else "fail",
                    "details": details,
                }
            )
    return reports


def _status_matches(observed: str | None, task: dict[str, Any]) -> bool:
    expected = task.get("expected_status")
    allowed = task.get("allowed_statuses")
    if allowed:
        return observed in set(allowed)
    return observed == expected


def _run_retrieval_tasks(
    tasks: list[dict[str, Any]],
    chunks: list[SourceChunk],
    scans: list[Scan],
    experiments: list[Experiment],
    mappings: list[PaperExperimentLink],
) -> list[dict[str, Any]]:
    reports = []
    haystack = []
    for chunk in chunks:
        haystack.append(("chunk", chunk.chunk_id, f"{chunk.title or ''} {chunk.text} {chunk.source_type} {chunk.source_id}", chunk.provenance))
    for scan in scans:
        haystack.append(("scan", scan.scan_id, _scan_text(scan), scan.provenance))
    for experiment in experiments:
        haystack.append(("experiment", experiment.experiment_id, _experiment_text(experiment), experiment.provenance))
    for mapping in mappings:
        haystack.append(("mapping", f"{mapping.paper_id}:{mapping.experiment_id}", " ".join([mapping.paper_id, mapping.experiment_id, " ".join(mapping.evidence), mapping.status]), mapping.provenance))
    for task in tasks:
        query_terms = {term.lower() for term in str(task.get("query") or "").replace("-", " ").split() if term}
        scored = []
        for source_type, source_id, text, provenance in haystack:
            text_lower = text.lower()
            score = sum(1 for term in query_terms if term in text_lower)
            if score:
                scored.append(
                    {
                        "source_type": source_type,
                        "source_id": source_id,
                        "score": score,
                        "provenance_present": bool(provenance),
                    }
                )
        scored = sorted(scored, key=lambda item: item["score"], reverse=True)[: int(task.get("top_k", 5))]
        result_ids = {item["source_id"] for item in scored}
        expected = set(task.get("expected_source_ids", []))
        forbidden = set(task.get("forbidden_source_ids", []))
        provenance_required = bool(task.get("require_provenance", True))
        provenance_ok = (not provenance_required) or all(item["provenance_present"] for item in scored if item["source_id"] in expected)
        status = "pass" if expected.issubset(result_ids) and not (forbidden & result_ids) and provenance_ok else "fail"
        reports.append(
            {
                "name": task.get("name") or "unnamed",
                "query": task.get("query"),
                "expected_source_ids": sorted(expected),
                "forbidden_source_ids": sorted(forbidden),
                "result_ids": [item["source_id"] for item in scored],
                "status": status,
                "details": {"top_results": scored, "provenance_ok": provenance_ok},
            }
        )
    return reports


def _path_only_fixture_catalog(path: Path) -> Path:
    catalog = SQLiteDataCatalog(path)
    experiment = Experiment(experiment_id="exp-hfir", instrument="HB-1A", metadata={"facility": "HFIR", "native_experiment_id": "1083"})
    scan = Scan(
        scan_id="scan-hfir-0059",
        experiment_id="exp-hfir",
        instrument="HB-1A",
        metadata={"facility": "HFIR", "native_scan_id": "0059", "raw_metadata": {"metadata_mode": "path_only"}},
    )
    datafile = DataFile(
        datafile_id="df-hfir-0059",
        path="/box/HB1A_exp1083_scan0059.dat",
        scan_id="scan-hfir-0059",
        experiment_id="exp-hfir",
        metadata={
            "facility": "HFIR",
            "instrument": "HB-1A",
            "native_format": "HB-1A dat",
            "native_scan_id": "0059",
            "native_experiment_id": "1083",
            "source_adapter": "fixture",
            "raw_metadata": {"metadata_mode": "path_only"},
        },
    )
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])
    return path


def _complete_lattice_fixture_catalog(path: Path) -> Path:
    catalog = SQLiteDataCatalog(path)
    experiment = Experiment(experiment_id="exp-geometry", instrument="BT-7", metadata={"facility": "NCNR", "native_experiment_id": "synthetic"})
    scan = Scan(
        scan_id="scan-geometry-001",
        experiment_id="exp-geometry",
        instrument="BT-7",
        scan_type="constant-Q scan",
        q_center=QPoint(h=1.0, k=0.0, l=0.0),
        energy_transfer=EnergyTransfer(value=UnitValue(value=0.0, unit="meV")),
        instrument_config=InstrumentConfig(
            instrument="BT-7",
            fixed_energy_mode="fixed_final_energy",
            fixed_energy=UnitValue(value=14.7, unit="meV"),
            monochromator="PG(002)",
            analyzer="PG(002)",
            collimation="40-40-40-80",
        ),
        metadata={
            "facility": "NCNR",
            "native_scan_id": "synthetic-001",
            "lattice": {
                "a": 6.28,
                "b": 6.28,
                "c": 6.28,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
                "orient1": [[1.0, 0.0, 0.0]],
                "orient2": [[0.0, 1.0, 0.0]],
            },
            "raw_metadata": {"metadata_mode": "synthetic_fixture"},
        },
    )
    datafile = DataFile(
        datafile_id="df-geometry-001",
        path="/synthetic/geometry_scan.bt7",
        scan_id="scan-geometry-001",
        experiment_id="exp-geometry",
        metadata={
            "facility": "NCNR",
            "instrument": "BT-7",
            "native_format": "synthetic TAS fixture",
            "native_scan_id": "synthetic-001",
            "native_experiment_id": "synthetic",
            "source_adapter": "eval_fixture",
            "raw_metadata": {"metadata_mode": "synthetic_fixture"},
        },
    )
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])
    return path


def _compatible_ub_fixture_catalog(path: Path) -> Path:
    catalog = SQLiteDataCatalog(path)
    experiment = Experiment(experiment_id="exp-compatible-ub", instrument="TRIAX", metadata={"facility": "MURR", "native_experiment_id": "153"})
    scan = Scan(
        scan_id="scan-compatible-ub",
        experiment_id="exp-compatible-ub",
        instrument="TRIAX",
        scan_type="constant-Q scan",
        q_center=QPoint(h=1.0, k=0.0, l=0.0),
        energy_transfer=EnergyTransfer(value=UnitValue(value=0.0, unit="meV")),
        instrument_config=InstrumentConfig(
            instrument="TRIAX",
            fixed_energy_mode="fixed_final_energy",
            fixed_energy=UnitValue(value=14.7, unit="meV"),
            monochromator="PG(002)",
            analyzer="PG(002)",
        ),
        metadata={
            "facility": "MURR",
            "native_scan_id": "0023",
            "ub_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "ub_convention": "hfir_murr_instrument_control",
            "raw_metadata": {"metadata_mode": "synthetic_fixture"},
        },
    )
    datafile = DataFile(
        datafile_id="df-compatible-ub",
        path="/synthetic/TRIAX_exp0153_scan0023.dat",
        scan_id=scan.scan_id,
        experiment_id=experiment.experiment_id,
        metadata={
            "facility": "MURR",
            "instrument": "TRIAX",
            "native_format": "TRIAX dat",
            "native_scan_id": "0023",
            "native_experiment_id": "153",
            "source_adapter": "eval_fixture",
            "raw_metadata": {"metadata_mode": "synthetic_fixture"},
        },
    )
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan])
    catalog.upsert_datafiles([datafile])
    return path


def _brief_fixture_catalog(path: Path) -> Path:
    catalog = SQLiteDataCatalog(path)
    experiment = Experiment(experiment_id="exp-brief", instrument="BT-7", title="Brief fixture experiment", metadata={"facility": "NCNR"})
    scan = Scan(
        scan_id="scan-brief-001",
        experiment_id=experiment.experiment_id,
        instrument="BT-7",
        scan_type="constant-Q scan",
        q_center=QPoint(h=1.0, k=0.0, l=0.0),
        energy_transfer=EnergyTransfer(value=UnitValue(value=0.0, unit="meV")),
        instrument_config=InstrumentConfig(
            instrument="BT-7",
            fixed_energy_mode="fixed_final_energy",
            fixed_energy=UnitValue(value=14.7, unit="meV"),
            monochromator="PG(002)",
            analyzer="PG(002)",
            collimation="40-40-40-80",
        ),
        metadata={
            "lattice": {"a": 6.28, "b": 6.28, "c": 6.28, "alpha": 90, "beta": 90, "gamma": 90, "orient1": [[1, 0, 0]], "orient2": [[0, 1, 0]]},
            "raw_metadata": {"metadata_mode": "synthetic_fixture"},
        },
    )
    similar = Scan(scan_id="scan-brief-002", experiment_id=experiment.experiment_id, instrument="BT-7", scan_type="constant-Q scan", q_center=QPoint(h=1.05, k=0, l=0), energy_transfer=scan.energy_transfer, instrument_config=scan.instrument_config, metadata=scan.metadata)
    datafile = DataFile(datafile_id="df-brief-001", path="/synthetic/scan001.bt7", scan_id=scan.scan_id, experiment_id=experiment.experiment_id, metadata={"facility": "NCNR", "instrument": "BT-7", "native_scan_id": "001"})
    mapping = PaperExperimentLink(paper_id="paper-fixture", experiment_id=experiment.experiment_id, confidence=0.75, evidence=["fixture mapping"], mapping_method="fixture", status="candidate")
    catalog.upsert_experiments([experiment])
    catalog.upsert_scans([scan, similar])
    catalog.upsert_datafiles([datafile])
    catalog.upsert_paper_experiment_links([mapping])
    return path


def _scan_text(scan: Scan) -> str:
    return " ".join([scan.scan_type or "", scan.physics_target or "", " ".join(scan.concepts), str(scan.metadata)])


def _experiment_text(experiment: Experiment) -> str:
    return " ".join([experiment.title or "", " ".join(experiment.concepts), str(experiment.metadata)])
