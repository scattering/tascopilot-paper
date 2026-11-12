# Reviewer Architecture

The implementation is a federated decision-support system. The core owns
contracts, discovery, permissions, audit records, catalogs, retrieval, validation,
and workflow composition. External repositories own scientific tools.

| Responsibility | Source directory |
| --- | --- |
| ToolCall, ToolSpec, ToolResult and provenance contracts | `packages/tas_copilot_contracts/src/tas_copilot_contracts/` |
| Plugin discovery, permissions and runtime boundaries | `packages/tas_copilot_core/src/tas_copilot_core/registry/` |
| Catalogs and document-side evidence | `packages/tas_copilot_core/src/tas_copilot_core/catalogs/` and `corpus/` |
| Curated ontology and typed retrieval | `packages/tas_copilot_core/src/tas_copilot_core/ontology/` and `retrieval/` |
| Deterministic validation | `packages/tas_copilot_core/src/tas_copilot_core/validation/` |
| Read-only, planning and simulation workflow composition | `packages/tas_copilot_core/src/tas_copilot_core/workflows/` |

The registry returns unavailable results when optional tools are absent. Source
records and retrieved text are evidence, not executable instructions. Candidate
source links, inferred metadata, and planning recommendations retain their
uncertainty and provenance; they are not approved operations.

The full source includes permission-gated operational interfaces for review.
Their inclusion is not authorization to use them. The documented core tests use
fixtures and mocks, and do not send commands to hardware.

See [Reviewer Testing](reviewer_testing.md) for the self-contained suite,
[Paper/Figure/Panel Catalog](paper_figure_panel_catalog.md) for document records,
[Figure Source Data Workflow](figure_source_data_workflow.md) for candidate search,
and [Source Support Validation](source_support_validation.md) for evidence checks.
Internal development narratives, site runbooks, and historical planning notes
are intentionally not part of the reviewer documentation collection.
