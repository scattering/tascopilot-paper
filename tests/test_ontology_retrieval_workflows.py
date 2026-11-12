from pathlib import Path

from tas_copilot_contracts import InstrumentConfig, QPoint, Scan
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.ontology import load_ontology, supernode_diagnostics
from tas_copilot_core.ontology.linker import match_concepts
from tas_copilot_core.retrieval import HybridRetriever, KeywordRetriever
from tas_copilot_core.retrieval.similarity import compare_scans
from tas_copilot_core.catalogs.jsonl_catalog import JsonlCatalog
from tas_copilot_core.workflows.check_resolution import check_resolution
from tas_copilot_core.workflows.check_spurions import check_spurions
from tas_copilot_core.workflows.find_experiments import find_experiments
from tas_copilot_core.workflows.find_papers import find_papers
from tas_copilot_core.workflows.plan_spinwave import plan_spinwave
from tas_copilot_core.workflows.group_bt7_scans import group_bt7_scans
from tas_copilot_contracts import SourceChunk, ToolResultStatus


def test_ontology_loads_and_matches_concepts():
    concepts = load_ontology(CoreConfig().ontology_path)
    assert len(concepts) >= 40
    matches = match_concepts("constant-Q spin-wave scan with PG filter", concepts)
    ids = {concept.id for concept in matches}
    assert "constant_q_scan" in ids
    assert "spinwave_dispersion" in ids


def test_supernode_diagnostics_runs():
    concepts = load_ontology(CoreConfig().ontology_path)
    warnings = supernode_diagnostics(["constant-Q scan", "PG filter"], concepts, max_fraction=0.5)
    assert isinstance(warnings, list)


def test_keyword_and_hybrid_retrieval():
    chunks = JsonlCatalog(Path(CoreConfig().catalog_dir) / "source_chunks.jsonl", SourceChunk).load()
    concepts = load_ontology(CoreConfig().ontology_path)
    keyword = KeywordRetriever(chunks)
    assert keyword.search("spin-wave PG filter")
    hybrid = HybridRetriever(keyword, concepts=concepts)
    results = hybrid.search("constant-Q spin-wave")
    assert results[0].retrieval_method == "hybrid_normalized"
    assert results[0].matched_concepts


def test_scan_similarity_dimensions():
    config = InstrumentConfig(monochromator="PG(002)", analyzer="PG", collimation="open-50-50-120", fixed_energy_mode="fixed final energy")
    left = Scan(scan_id="a", scan_type="constant-Q scan", q_center=QPoint(h=1, k=0, l=0), instrument_config=config, physics_target="spin-wave dispersion")
    right = Scan(scan_id="b", scan_type="constant-Q scan", q_center=QPoint(h=1.05, k=0, l=0), instrument_config=config, physics_target="spin-wave dispersion")
    result = compare_scans(left, right)
    assert result["similarity_score"] > 0.7
    assert "q_center" in result["matched_dimensions"]


def test_find_workflows_return_typed_results():
    assert find_papers("BT-7 spin-wave")
    assert find_experiments("spin-wave")


def test_tool_workflows_return_unavailable_and_plan_is_not_live():
    assert check_resolution({}).status in {ToolResultStatus.unavailable, ToolResultStatus.error, ToolResultStatus.validation_failed}
    assert check_spurions({}).status in {ToolResultStatus.unavailable, ToolResultStatus.error, ToolResultStatus.validation_failed}
    grouping_result = group_bt7_scans({"root_dirs": ["/tmp/no-such-bt7-dir"]})
    assert grouping_result.status in {ToolResultStatus.unavailable, ToolResultStatus.success}
    if grouping_result.status == ToolResultStatus.success:
        assert grouping_result.output["record_count"] == 0
    result = plan_spinwave({"material": "example"})
    assert result.status == ToolResultStatus.unavailable
    assert result.tool_namespace == "tas.tasai"
