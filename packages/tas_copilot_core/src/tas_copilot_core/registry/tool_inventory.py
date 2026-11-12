from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_contracts import Availability, ToolSpec

from tas_copilot_core.config import CoreConfig

from .manifest_loader import load_tool_manifest
from .plugin_discovery import discover_entrypoint_plugins


@dataclass
class InventoryRecord:
    spec: ToolSpec
    source: str
    plugin_name: str | None = None
    handler_available: bool = False

    def status(self) -> str:
        if self.spec.availability == Availability.disabled or not self.spec.enabled_by_default:
            return "disabled"
        if self.spec.availability != Availability.available:
            return "unavailable"
        return "available" if self.handler_available else "unavailable"

    def key(self) -> tuple[str, str]:
        return (self.spec.namespace, self.spec.name)


@dataclass
class ToolInventory:
    records: list[InventoryRecord] = field(default_factory=list)
    plugin_errors: list[dict[str, str]] = field(default_factory=list)

    def effective_records(self) -> list[InventoryRecord]:
        by_key: dict[tuple[str, str], list[InventoryRecord]] = defaultdict(list)
        for record in self.records:
            by_key[record.key()].append(record)
        effective: list[InventoryRecord] = []
        for records in by_key.values():
            handler_records = [record for record in records if record.handler_available]
            effective.extend(handler_records or records)
        return effective

    def shadowed_manifest_stubs(self) -> list[dict[str, Any]]:
        shadowed = []
        by_key: dict[tuple[str, str], list[InventoryRecord]] = defaultdict(list)
        for record in self.records:
            by_key[record.key()].append(record)
        for (namespace, name), records in sorted(by_key.items()):
            handler_sources = [record.source for record in records if record.handler_available]
            manifest_sources = [record.source for record in records if not record.handler_available]
            if handler_sources and manifest_sources:
                shadowed.append(
                    {
                        "namespace": namespace,
                        "name": name,
                        "manifest_sources": manifest_sources,
                        "handler_sources": handler_sources,
                    }
                )
        return shadowed

    def duplicates(self) -> list[dict[str, Any]]:
        by_key: dict[tuple[str, str], list[InventoryRecord]] = defaultdict(list)
        for record in self.effective_records():
            by_key[record.key()].append(record)
        duplicates = []
        for (namespace, name), records in sorted(by_key.items()):
            if len(records) > 1:
                duplicates.append(
                    {
                        "namespace": namespace,
                        "name": name,
                        "sources": [record.source for record in records],
                        "handler_sources": [record.source for record in records if record.handler_available],
                    }
                )
        return duplicates

    def as_dict(self) -> dict[str, Any]:
        tools = []
        effective_ids = {id(record) for record in self.effective_records()}
        shadowed_by_key = {
            (item["namespace"], item["name"]): item["handler_sources"] for item in self.shadowed_manifest_stubs()
        }
        for record in sorted(self.records, key=lambda item: (item.spec.namespace, item.spec.name, item.source)):
            spec = record.spec
            tools.append(
                {
                    "namespace": spec.namespace,
                    "name": spec.name,
                    "version": spec.version,
                    "status": record.status(),
                    "source": record.source,
                    "plugin_name": record.plugin_name,
                    "availability": spec.availability.value,
                    "implementation_type": spec.implementation_type.value,
                    "permissions_required": [permission.value for permission in spec.permissions_required],
                    "enabled_by_default": spec.enabled_by_default,
                    "required_dependencies": spec.required_dependencies,
                    "owner_repo": spec.owner_repo,
                    "description": spec.description,
                    "effective": id(record) in effective_ids,
                    "shadowed_by": shadowed_by_key.get((spec.namespace, spec.name), []) if id(record) not in effective_ids else [],
                }
            )
        effective = self.effective_records()
        return {
            "tools": tools,
            "summary": {
                "total": len(effective),
                "registered_records": len(tools),
                "available": sum(1 for record in effective if record.status() == "available"),
                "unavailable": sum(1 for record in effective if record.status() == "unavailable"),
                "disabled": sum(1 for record in effective if record.status() == "disabled"),
                "duplicates": len(self.duplicates()),
                "shadowed_manifest_stubs": len(self.shadowed_manifest_stubs()),
                "plugin_errors": len(self.plugin_errors),
            },
            "duplicates": self.duplicates(),
            "shadowed_manifest_stubs": self.shadowed_manifest_stubs(),
            "plugin_errors": self.plugin_errors,
        }


def build_tool_inventory(config: CoreConfig | None = None) -> ToolInventory:
    config = config or CoreConfig()
    inventory = ToolInventory()
    manifest_path = Path(config.tools_manifest_path)
    for spec in load_tool_manifest(manifest_path):
        inventory.records.append(InventoryRecord(spec=spec, source=str(manifest_path), handler_available=False))

    for plugin in discover_entrypoint_plugins():
        plugin_name = getattr(plugin, "__name__", plugin.__class__.__name__)
        try:
            specs = plugin.get_tool_specs()
        except Exception as exc:
            inventory.plugin_errors.append({"plugin": plugin_name, "error": str(exc)})
            continue
        for spec in specs:
            inventory.records.append(
                InventoryRecord(
                    spec=ToolSpec.model_validate(spec.model_dump(mode="json") if hasattr(spec, "model_dump") else spec),
                    source=f"plugin:{plugin_name}",
                    plugin_name=plugin_name,
                    handler_available=hasattr(plugin, "run_tool"),
                )
            )
    return inventory


def doctor_report(inventory: ToolInventory, runtime_profile_path: str | None = None) -> str:
    data = inventory.as_dict()
    lines = ["TASCopilot tool doctor", ""]
    summary = data["summary"]
    lines.append(
        f"Total tools: {summary['total']} | available: {summary['available']} | unavailable: {summary['unavailable']} | disabled: {summary['disabled']} | duplicates: {summary['duplicates']}"
    )
    lines.extend(_capability_lines(data))
    profile_warnings = _runtime_profile_warning_lines(runtime_profile_path)
    if profile_warnings:
        lines.append("")
        lines.append("Runtime profile warnings:")
        lines.extend(profile_warnings)
    if data["plugin_errors"]:
        lines.append("")
        lines.append("Plugin errors:")
        for error in data["plugin_errors"]:
            lines.append(f"- {error['plugin']}: {error['error']}")
    if data["duplicates"]:
        lines.append("")
        lines.append("Conflicting duplicate tool registrations:")
        for duplicate in data["duplicates"]:
            lines.append(f"- {duplicate['namespace']}.{duplicate['name']}: {', '.join(duplicate['sources'])}")
    if data["shadowed_manifest_stubs"]:
        lines.append("")
        lines.append("Manifest stubs shadowed by installed plugins:")
        for shadowed in data["shadowed_manifest_stubs"]:
            lines.append(
                f"- {shadowed['namespace']}.{shadowed['name']}: {', '.join(shadowed['manifest_sources'])} -> {', '.join(shadowed['handler_sources'])}"
            )
    lines.append("")
    lines.append("Tools:")
    for tool in data["tools"]:
        if not tool["effective"]:
            continue
        permissions = ",".join(tool["permissions_required"])
        lines.append(f"- {tool['status']}: {tool['namespace']}.{tool['name']} [{permissions}] via {tool['source']}")
    missing = [tool for tool in data["tools"] if tool["effective"] and tool["status"] == "unavailable" and tool["owner_repo"]]
    if missing:
        lines.append("")
        lines.append("Suggested installs/registers for optional tools:")
        seen: set[str] = set()
        for tool in missing:
            owner = tool["owner_repo"]
            if owner and owner not in seen:
                seen.add(owner)
                lines.append(f"- install plugin repo in editable mode or expose entry point: {owner}")
    return "\n".join(lines)


def _runtime_profile_warning_lines(runtime_profile_path: str | None) -> list[str]:
    if not runtime_profile_path:
        return []
    path = Path(runtime_profile_path).expanduser()
    lines: list[str] = []
    if path.name.endswith(".example.yaml"):
        lines.append(f"- {path}: example runtime profile selected; copy it to a local/private profile before real jobs.")
    if not path.exists():
        lines.append(f"- {path}: runtime profile file is missing.")
        return lines
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        lines.append(f"- {path}: could not parse runtime profile: {exc}")
        return lines
    controlpath = str(payload.get("controlpath") or "")
    if "williamratcliff" in controlpath:
        lines.append(f"- {path}: ControlPath contains a hard-coded username; prefer `/tmp/<cluster>-{{user}}.sock`.")
    payload_text = str(payload).lower()
    if "[local" in payload_text or "placeholder" in payload_text:
        lines.append(f"- {path}: profile appears to contain placeholder values.")
    return lines


def _capability_lines(data: dict) -> list[str]:
    effective = [tool for tool in data["tools"] if tool["effective"]]
    available = {(tool["namespace"], tool["name"]) for tool in effective if tool["status"] == "available"}
    disabled = {(tool["namespace"], tool["name"]) for tool in effective if tool["status"] == "disabled"}

    def has(namespace: str, name: str) -> bool:
        return (namespace, name) in available

    lines = ["", "What you can do today:"]
    data_reader_ready = any(namespace in {"tas.data_reader", "tas.bt7"} for namespace, _ in available)
    material_ready = any(namespace == "tas.material_identity" for namespace, _ in available)
    paper_mapping_ready = has("tas.paper_mapping", "map_paper_to_experiments")
    psd_calibration_blocked = bool(
        disabled
        & {
            ("tas.psd_calibration", "channel_to_detector_angle"),
            ("tas.psd_calibration", "detector_angle_to_q"),
        }
    )
    lines.append(
        "- Start from a situation: `python -m tas_copilot_core.cli workflow copilot-ask --list-intents`"
    )
    lines.append(
        "- Review labels: `python -m tas_copilot_core.cli review labels summary --store [review_outcomes.sqlite]`"
    )
    lines.append(
        "- Paper/corpus fixtures and review dashboards: available from core commands; private PDFs remain local-only."
    )
    lines.append(
        "- BT-7/TAS data inspection and dashboards: "
        + ("reader tools available" if data_reader_ready else "reader plugins not visible; fixture/path-only workflows may still run")
    )
    lines.append(
        "- Material/formula handling: "
        + ("material identity tools available" if material_ready else "optional parser/plugin not visible; ambiguous formulas stay ambiguous")
    )
    lines.append(
        "- Paper-to-experiment linkage: "
        + ("mapping plugin available" if paper_mapping_ready else "core heuristic linkage available; external paper-mapping plugin not visible")
    )
    lines.append(
        "- PSD calibration/QxT: "
        + ("reviewed-calibration boundary is disabled/fail-closed" if psd_calibration_blocked else "check psd-calibration commands before using")
    )
    lines.append("- Live control: blocked by policy; planning/simulation outputs are not approved operations.")
    lines.extend(["", "Runnable workflow starts today:"])
    workflow_rows = [
        {
            "name": "copilot status/examples",
            "status": "ready",
            "command": 'python -m tas_copilot_core.cli workflow copilot-ask --question "What can you do?" --output-dir local_runs/copilot_status',
        },
        {
            "name": "local web workbench",
            "status": "ready_if_web_extra_installed",
            "command": "python -m tas_copilot_core.cli serve --port 8765",
        },
        {
            "name": "review label summary/backfill",
            "status": "ready_with_review_store_or_local_runs",
            "command": "python -m tas_copilot_core.cli review labels backfill --root local_runs --store local_runs/review_outcomes.sqlite",
        },
        {
            "name": "experiment review / overnight summary / dashboard",
            "status": "ready_with_data_root" if data_reader_ready else "needs_data_root_and_reader_plugin",
            "command": "python -m tas_copilot_core.cli workflow experiment-review --data-root [DATA_ROOT] --output-dir local_runs/experiment_review",
        },
        {
            "name": "full BT-7 alignment/spurion/order-parameter candidate pass",
            "status": "ready_with_hydrated_bt7_root",
            "command": "python -m tas_copilot_core.cli workflow bt7-corpus-candidate-pass --data-root [BT7_DATA_ROOT] --output-dir local_runs/bt7_corpus_candidate_pass --max-files 0",
        },
        {
            "name": "alignment review dashboard",
            "status": "ready_with_alignment_review_template",
            "command": "python -m tas_copilot_core.cli workflow alignment-review-dashboard --review-template [ALIGNMENT_REVIEW_TEMPLATE.jsonl] --output-dir local_runs/alignment_review_dashboard",
        },
        {
            "name": "paper-to-experiment linkage",
            "status": "ready_with_catalogs" if paper_mapping_ready else "core_heuristics_ready_with_catalogs",
            "command": "python -m tas_copilot_core.cli workflow link-paper-to-experiments --corpus-catalog [CORPUS] --data-catalog [DATA] --paper-id [PAPER_ID] --output-dir local_runs/paper_links",
        },
        {
            "name": "source triage / figure-source linkage",
            "status": "ready_with_synthetic_or_data_catalog",
            "command": "python -m tas_copilot_core.cli workflow figure-source-triage --catalog [DATA_CATALOG] --query-file [QUERY] --output-dir local_runs/source_triage",
        },
        {
            "name": "PSD QxT reconstruction",
            "status": "blocked_without_reviewed_psd_calibration",
            "command": "python -m tas_copilot_core.cli psd-calibration report --registry [CALIBRATION_REGISTRY]",
        },
        {
            "name": "NICE BT-7 simulator replay",
            "status": "ready_for_dry_run_or_mock; nice_container_session reuses isolated NICE service; nice_remote requires host NICE client",
            "command": "python -m tas_copilot_core.cli workflow nice-bt7-replay --replay-plan [NICE_REPLAY_PLAN] --executor nice_container_session --nice-checkout [NICE_CHECKOUT] --output-dir local_runs/nice_bt7_replay",
        },
        {
            "name": "tasAI adaptive NICE simulation",
            "status": "ready_with_fixture_backends; use nice_container_session for persistent NICE simulator; full ANDiE can run through andie_container",
            "command": "python -m tas_copilot_core.cli workflow tasai-adaptive-nice-sim --policy-backend andie_container --andie-container-workspace [ANDIE_CONTAINER_WORKSPACE] --andie-source-root [ANDIE_SOURCE_ROOT] --nice-executor nice_container_session --nice-checkout [NICE_CHECKOUT] --max-steps 5 --output-dir local_runs/tasai_adaptive_nice_sim",
        },
    ]
    for row in workflow_rows:
        lines.append(f"- {row['status']}: {row['name']} -> `{row['command']}`")
    return lines
