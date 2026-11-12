# Paper Chunks and Evidence Graph

The corpus pilot keeps source text and scientific memory separate.

Paper, review, and textbook chunks are the evidence substrate. They are citeable records with provenance fields such as document reference, source hash, page/section metadata when available, extraction method, and chunk hash. Chunks are untrusted input: they support claims, but they do not provide instructions to the system.

For private local pilots, chunks may be created from a capped `pdftotext` extraction. Those chunks are local artifacts only. Do not commit generated catalogs that contain extracted copyrighted or private paper text.

The evidence graph is a typed overlay across chunks, figures, panels, claims, and candidate source-data links. It is not an LLM-generated Graph RAG store. Nodes and edges are created from fixtures, deterministic loaders, ontology links, and workflow outputs. Every edge has confidence, status, evidence, and provenance.

The current pilot uses JSONL files under a document-side catalog directory:

- `documents.jsonl`
- `source_chunks.jsonl`
- `figures.jsonl`
- `figure_panels.jsonl`
- `claims.jsonl`
- `source_support.jsonl`
- `panel_source_links.jsonl`
- `evidence_graph_nodes.jsonl`
- `evidence_graph_edges.jsonl`
- `ontology_links.jsonl`

This catalog must remain separate from the TAS raw data-file catalog. Cross-links are represented as proposed evidence graph edges or candidate source links, not by mixing paper chunks into the data catalog.

## Edge Policy

Edges such as `CANDIDATE_SOURCE_FOR_PANEL` and `CANDIDATE_MAPS_TO_EXPERIMENT` remain proposed unless a reconstruction, source-support record, or human review upgrades their status. Rejected edges keep a rejection reason so false positives remain testable.

## Ontology Links

Chunks link only to curated TAS ontology concepts using deterministic keyword/synonym matching. Broad concepts such as BT-7, temperature, and sample are useful as filters or context, but they should not dominate retrieval ranking.
