"""Replay serialized numerical DAGs through the current registry and policy."""

from __future__ import annotations

from pathlib import Path

from tas_copilot_contracts import ReductionCorrectionRef, ReductionDAGTemplate
from tas_copilot_core.reduction_dag import replay_reduction_dag_run
from tas_copilot_core.reporting import write_report_bundle

from .common import default_runner


def replay_numerical_reduction(manifest: dict, arguments: dict, runner=None) -> dict:
    saved = manifest["replay"]
    template = ReductionDAGTemplate.model_validate(saved["template"])
    inputs = {**saved["inputs"], **arguments.get("input_overrides", {})}
    overrides = arguments.get("correction_overrides", {})
    corrections = [
        ReductionCorrectionRef.model_validate({**value, **overrides.get(value["correction_id"], {})})
        for value in saved.get("corrections", [])
    ]
    unknown = set(overrides) - {correction.correction_id for correction in corrections}
    if unknown:
        raise ValueError(f"unknown correction overrides: {sorted(unknown)}")
    versions = {**saved.get("software_versions", {}), **arguments.get("software_versions", {})}
    if arguments.get("workflow_version"):
        versions["workflow_version"] = arguments["workflow_version"]
    replay = replay_reduction_dag_run(
        manifest["reduction_dag"], template=template, inputs=inputs,
        corrections=corrections, software_versions=versions,
        runner=runner or default_runner(), cache_dir=arguments.get("cache_dir"),
    )
    report = {
        "workflow_name": "dag-run-replay", "status": replay["status"],
        "cache_replay": replay, "reduction_dag": replay["new_result"],
        "replay": {
            "template": template.model_dump(mode="json"), "inputs": inputs,
            "corrections": [value.model_dump(mode="json") for value in corrections],
            "software_versions": versions,
        },
        "safe_conclusion": (
            f"Numerical DAG replay status: {replay['status']}; "
            f"{len(replay['cache_hit_node_ids'])} cached node result(s) reused, "
            f"{replay['changed_count']} changed node fingerprint(s)."
        ),
    }
    report["artifacts"] = write_report_bundle(
        Path(arguments["output_dir"]), report,
        markdown_renderer=lambda value: f"# Numerical Reduction Replay\n\n{value['safe_conclusion']}\n",
        json_filename="dag_run_replay.json", markdown_filename="dag_run_replay.md",
    )
    return report
