from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_audit_dir() -> Path:
    # Headless deployments (read-only source trees) point audit output
    # elsewhere via TAS_COPILOT_AUDIT_DIR.
    override = os.environ.get("TAS_COPILOT_AUDIT_DIR")
    if override:
        return Path(override).expanduser()
    return repo_root() / "data" / "audit"


@dataclass(frozen=True)
class CoreConfig:
    root: Path = repo_root()
    catalog_dir: Path = repo_root() / "data" / "catalogs"
    data_catalog_db_path: Path = repo_root() / "data" / "catalogs" / "bt7_data_catalog.sqlite"
    audit_dir: Path = field(default_factory=_default_audit_dir)
    ontology_path: Path = repo_root() / "configs" / "ontology" / "tas_bt7_seed.yaml"
    external_vocabularies_path: Path = repo_root() / "configs" / "ontology" / "external_vocabularies.yaml"
    external_mappings_path: Path = repo_root() / "configs" / "ontology" / "external_mappings_seed.yaml"
    tools_manifest_path: Path = repo_root() / "configs" / "tools" / "example_tools.yaml"
