from __future__ import annotations

from pathlib import Path

from tas_copilot_contracts import Experiment, RetrievalResult, Scan

from tas_copilot_core.audit import provenance_for
from tas_copilot_core.catalogs.jsonl_catalog import JsonlCatalog
from tas_copilot_core.config import CoreConfig


def find_experiments(query: str, limit: int = 10) -> list[RetrievalResult]:
    config = CoreConfig()
    experiments = JsonlCatalog(Path(config.catalog_dir) / "experiments.jsonl", Experiment).load()
    scans = JsonlCatalog(Path(config.catalog_dir) / "scans.jsonl", Scan).load()
    terms = {term.lower() for term in query.split()}
    results: list[RetrievalResult] = []
    for experiment in experiments:
        text = " ".join([experiment.title or "", " ".join(experiment.concepts), str(experiment.metadata)]).lower()
        matched = sorted(term for term in terms if term in text)
        if matched:
            results.append(RetrievalResult(source_type="experiment", source_id=experiment.experiment_id, title=experiment.title, score=len(matched), matched_terms=matched, retrieval_method="metadata", provenance=[provenance_for("metadata", "experiment", experiment.experiment_id)]))
    for scan in scans:
        text = " ".join([scan.scan_type or "", scan.physics_target or "", " ".join(scan.concepts), str(scan.metadata)]).lower()
        matched = sorted(term for term in terms if term in text)
        if matched:
            results.append(RetrievalResult(source_type="scan", source_id=scan.scan_id, title=scan.scan_type, score=len(matched), matched_terms=matched, retrieval_method="metadata", provenance=[provenance_for("metadata", "scan", scan.scan_id)]))
    return sorted(results, key=lambda item: item.score, reverse=True)[:limit]

