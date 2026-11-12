from __future__ import annotations

from pathlib import Path

from tas_copilot_contracts import SourceChunk

from tas_copilot_core.audit import JsonlAuditLogger
from tas_copilot_core.catalogs.jsonl_catalog import JsonlCatalog
from tas_copilot_core.config import CoreConfig
from tas_copilot_core.ontology.loader import load_ontology
from tas_copilot_core.registry import ToolRegistry, ToolRunner, load_tool_manifest
from tas_copilot_core.registry.plugin_discovery import discover_entrypoint_plugins
from tas_copilot_core.retrieval import GraphRetriever, HybridRetriever, KeywordRetriever


def default_runner(config: CoreConfig | None = None) -> ToolRunner:
    config = config or CoreConfig()
    registry = ToolRegistry()
    registry.register_many(load_tool_manifest(config.tools_manifest_path))
    for plugin in discover_entrypoint_plugins():
        try:
            specs = plugin.get_tool_specs()
            for spec in specs:
                registry.register(spec, plugin.run_tool)
        except Exception:
            continue
    return ToolRunner(registry=registry, audit_logger=JsonlAuditLogger(config.audit_dir))


def default_retriever(config: CoreConfig | None = None) -> HybridRetriever:
    config = config or CoreConfig()
    catalog_dir = Path(config.catalog_dir)
    chunks = JsonlCatalog(catalog_dir / "source_chunks.jsonl", SourceChunk).load()
    concepts = load_ontology(config.ontology_path)
    # Serve the typed evidence graph when the catalog has one; review status
    # (accepted/proposed/rejected) folds into scores via the GraphRetriever.
    graph_retriever = GraphRetriever(catalog_dir) if (catalog_dir / "evidence_graph_edges.jsonl").exists() else None
    return HybridRetriever(KeywordRetriever(chunks), graph_retriever=graph_retriever, concepts=concepts)
