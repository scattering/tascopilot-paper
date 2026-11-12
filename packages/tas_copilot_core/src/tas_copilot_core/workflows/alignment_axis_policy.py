"""Axis policy for alignment: which motors must NOT be optimized in this run, and why.

An alignment is not always allowed to touch every axis. On 2026-09-12 (BT-7 exp 14,
inelastic, nobody at the instrument) the pre-sample slit readbacks had been reset
to zero by the motor computer, the post-sample slits are not optimized for inelastic
work anyway, and translations are not reversible from a distance. The state machine
and the live driver must know this: a frozen axis is never scanned, never
recommended, and every skipped step is recorded with its reason so the report says
what was left alone and why, instead of silently omitting it.

Policy file (YAML or JSON):

    mode: reversible_only            # free text; recorded in reports
    frozen_axes:
      - axis: preSampleWidth         # any alias of the axis (ICE or NICE spelling)
        reason: readback reset to 0 after the motor-computer power cycle (2026-09-12)
      - axis: postSampleSlitHeight
        reason: inelastic run; post-sample slits are not optimized
      - axis: sampleLTrans
        reason: not reversible without someone at the instrument
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# One canonical name per axis, with every spelling seen in ICE and NICE files. A
# policy naming any alias freezes the whole group.
AXIS_ALIASES: dict[str, set[str]] = {
    "a3": {"a3", "A3", "sampleTheta", "sampleTheta.softPosition"},
    "a4": {"a4", "A4", "sampleTwoTheta", "sampleTwoTheta.softPosition"},
    "lower_tilt": {"sampleLTilt", "SmplLTilt", "sgl", "lowerTilt"},
    "upper_tilt": {"sampleUTilt", "SmplUTilt", "sgu", "upperTilt"},
    "lower_translation": {"sampleLTrans", "SmplLTrn", "lowerTranslation"},
    "upper_translation": {"sampleUTrans", "SmplUTrn", "upperTranslation"},
    "sample_elevation": {"SmplElev", "sampleElevation", "sampleTableTrans"},
    "pre_sample_slit_width": {"preSampleWidth", "preSampleSlitWidth", "FrontSlitWidth", "SmplWdth"},
    "pre_sample_slit_height": {"preSampleHeight", "preSampleSlitHeight", "FrontSlitHeight", "SmplHght"},
    "post_sample_slit_width": {"postSampleWidth", "postSampleSlitWidth", "BackSlitWidth", "BkSltWdth"},
    "post_sample_slit_height": {"postSampleHeight", "postSampleSlitHeight", "BackSlitHeight", "BkSltHght"},
}
_ALIAS_TO_CANONICAL = {alias.lower(): canonical for canonical, aliases in AXIS_ALIASES.items() for alias in aliases | {canonical}}


def canonical_axis(name: Any) -> str:
    text = str(name or "").strip()
    return _ALIAS_TO_CANONICAL.get(text.lower(), text)


def load_axis_policy(path: str | Path | None) -> dict[str, Any]:
    """Return {"mode", "frozen": {canonical: reason}, "source", "unknown_axes"}; an
    empty policy when path is None. Unknown axis names are kept (frozen under their
    own name) and listed so a typo is visible rather than silently ineffective."""
    if not path:
        return {"mode": None, "frozen": {}, "source": None, "unknown_axes": []}
    policy_path = Path(str(path)).expanduser()
    text = policy_path.read_text(encoding="utf-8")
    if policy_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml

            payload = yaml.safe_load(text) or {}
        except ImportError:  # the NICE-client interpreter has no PyYAML; JSON policies work there
            payload = json.loads(text)
    frozen: dict[str, str] = {}
    unknown: list[str] = []
    for item in list(payload.get("frozen_axes") or []):
        if isinstance(item, str):
            item = {"axis": item, "reason": "frozen by policy"}
        name = str(item.get("axis") or "").strip()
        if not name:
            continue
        canonical = canonical_axis(name)
        if canonical == name and name.lower() not in _ALIAS_TO_CANONICAL:
            unknown.append(name)
        frozen[canonical] = str(item.get("reason") or "frozen by policy")
    return {"mode": payload.get("mode"), "frozen": frozen, "source": str(policy_path), "unknown_axes": unknown}


def frozen_reason(policy: dict[str, Any], axis: Any) -> str | None:
    return (policy or {}).get("frozen", {}).get(canonical_axis(axis))


def step_axes(step: dict[str, Any]) -> list[str]:
    axes: list[str] = []
    for key in ("motor", "motors", "slits", "nodes", "axes"):
        value = step.get(key)
        if isinstance(value, str):
            axes.append(value)
        elif isinstance(value, list):
            axes.extend(str(v) for v in value)
    return axes


def apply_axis_policy(steps: list[dict[str, Any]], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split steps into (kept, skipped); a step is skipped when any axis it moves is frozen.
    Each skipped entry carries the step id, the frozen axis and the reason."""
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for step in steps:
        hit = [(axis, frozen_reason(policy, axis)) for axis in step_axes(step) if frozen_reason(policy, axis)]
        if hit:
            skipped.append({"step_id": step.get("step_id") or step.get("stage_id"), "frozen_axis": hit[0][0], "canonical_axis": canonical_axis(hit[0][0]), "reason": hit[0][1], "operation": step.get("operation")})
        else:
            kept.append(step)
    return kept, skipped


def policy_summary(policy: dict[str, Any]) -> dict[str, Any]:
    return {"mode": (policy or {}).get("mode"), "source": (policy or {}).get("source"), "frozen_axes": dict((policy or {}).get("frozen") or {}), "unknown_axes": list((policy or {}).get("unknown_axes") or [])}


__all__ = ["AXIS_ALIASES", "canonical_axis", "load_axis_policy", "frozen_reason", "step_axes", "apply_axis_policy", "policy_summary"]
