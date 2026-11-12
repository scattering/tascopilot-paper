"""Operator controls for an autonomous ANDiE temperature run.

The web UI and the run loop share this module. The UI never talks to NICE: it
writes a request here, and the run loop (the only process that owns the
instrument) reads it between temperatures. Every change is validated, written
atomically under a lock, and appended to a log.

Files in the run root:
    control.json       what the operator wants (start/stop, auto, injections,
                       the policy revision to apply at the next temperature)
    control_log.jsonl  every change, in order
    status.json        what the run loop is doing, written by the loop
    policy/            immutable policy revisions (andie_policy.PolicyStore)
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import glob
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

CONTROL = "control.json"
LOG = "control_log.jsonl"
STATUS = "status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    os.replace(temporary, path)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


@contextmanager
def _locked(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".control.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _temperature(value, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive temperature in K")
    return number


def read_control(root: str | Path) -> dict[str, Any]:
    return _read_json(Path(root) / CONTROL, {})


def read_status(root: str | Path) -> dict[str, Any]:
    return _read_json(Path(root) / STATUS, {})


def write_status(root: str | Path, **fields) -> dict[str, Any]:
    """Run loop only: replace status.json with the given fields plus a time."""
    status = {**fields, "updated_at": _now()}
    _write_json(Path(root) / STATUS, status)
    return status


def _change(root: str | Path, action: str, mutate) -> dict[str, Any]:
    root = Path(root)
    with _locked(root):
        control = _read_json(root / CONTROL, {})
        mutate(control)
        control["revision"] = int(control.get("revision", 0)) + 1
        control["updated_at"] = _now()
        _write_json(root / CONTROL, control)
        with (root / LOG).open("a") as log:
            log.write(json.dumps({"at": control["updated_at"], "action": action,
                                  "revision": control["revision"]}, sort_keys=True) + "\n")
        return dict(control)


SEED_NAME = re.compile(r"_(?P<reflection>\d{3})_tt_T(?P<temperature>\d+(?:p\d+)?)K")


def expand_seed_files(entries) -> list[str]:
    """Paths or glob patterns -> existing files, absolute, in order, no repeats."""
    files: list[str] = []
    for entry in entries:
        entry = str(entry).strip()
        if not entry:
            continue
        matches = sorted(glob.glob(os.path.expanduser(entry)))
        if not matches:
            raise ValueError(f"no file matches {entry}")
        files.extend(str(Path(m).resolve()) for m in matches if str(Path(m).resolve()) not in files)
    return files


def seed_groups(files) -> dict[float, dict[str, list[str]]]:
    """Earlier files grouped by nominal temperature and reflection, from their
    names (the run's convention, e.g. sample_001_tt_T110K1370.dat.bt7). Several
    files of one reflection at one temperature are kept together; the run adds
    them (a repeat measurement improves the statistics, it is not a new point)."""
    groups: dict[float, dict[str, list[str]]] = {}
    for name in files:
        match = SEED_NAME.search(Path(name).name)
        if not match:
            raise ValueError(f"cannot tell reflection and temperature from the name {Path(name).name}")
        temperature = float(match["temperature"].replace("p", "."))
        groups.setdefault(temperature, {}).setdefault(match["reflection"], []).append(str(name))
    return dict(sorted(groups.items()))


# What a run leaves in its root for the UI; moved aside when a new run starts there.
RUN_RECORDS = ("series.json", "display", "seeds.json", "tc_history.jsonl",
               "order_parameter_band.json", "posterior_latest.npz",
               "tc_history_earlier.jsonl", "earlier_tc", "tc_history_recomputed.jsonl", "recomputed_tc",
               "policy")


def _archive_previous(root: Path) -> None:
    ledger = root / "series.json"
    if not ledger.exists():
        return
    destination = root / "previous_runs" / str(json.loads(ledger.read_text()).get("run_id"))
    destination.mkdir(parents=True, exist_ok=True)
    for name in RUN_RECORDS:
        if (root / name).exists():
            os.replace(root / name, destination / name)


MAX_STEP_K = 25.0


def start_run(root, *, start_k, stop_k, autonomous=True, seed_files=(), diagnostic_cadence=5,
              fallback_step_k=10.0, limits=(1.0, 1000.0), policy=None, measured=()) -> dict[str, Any]:
    """Ask the loop to start a warming series from start_k up to stop_k.

    ``seed_files`` (paths or glob patterns) are earlier scans the run fits and
    uses as its first temperatures; every temperature used must have all the
    ``measured`` reflections (001 and 101 unless given). ``diagnostic_cadence``:
    diagnostics at every n-th temperature of this run. ``fallback_step_k``: how
    far to warm when the posterior is not converged (flat data cannot place
    Tc). ``policy``: the initial policy document (``andie_policy``); without
    one the loop derives revision 0 from its settings file.
    """
    start, stop = _temperature(start_k, "start"), _temperature(stop_k, "stop")
    low, high = map(float, limits)
    if not low <= start < stop <= high:
        raise ValueError(f"need {low:g} <= start < stop <= {high:g} K (warming only)")
    cadence = _cadence(diagnostic_cadence)
    fallback = _fallback_step(fallback_step_k)
    files = expand_seed_files(seed_files)
    groups = seed_groups(files)
    needed = set(measured) if measured else {"001", "101"}
    if files and not any(needed <= set(g) for g in groups.values()):
        raise ValueError("no temperature in the seed files has every measured reflection (%s)" % ", ".join(sorted(needed)))
    initial = None
    if policy is not None:
        from .andie_policy import PolicyDocument
        initial = PolicyDocument.model_validate(policy) if not isinstance(policy, PolicyDocument) else policy

    def mutate(control):
        if control.get("state") == "running":
            raise ValueError("a run is already active; stop it first")
        _archive_previous(Path(root))
        control.clear()
        control.update(run_id="run-" + uuid4().hex[:12], state="running", start_k=start, stop_k=stop,
                       autonomous=bool(autonomous), stop_requested=False, injections=[],
                       accepted_proposal=None, seed_files=files, diagnostic_cadence=cadence,
                       diagnostic_next=False, fallback_step_k=fallback, started_at=_now(), policy=None)
        if initial is not None:
            from .andie_policy import PolicyStore
            control["policy"] = PolicyStore(root).save(initial, requested_by="start_run")
    return _change(root, f"start {start:g}->{stop:g} K autonomous={bool(autonomous)} "
                         f"seed files={len(files)} diagnostics every {cadence} fallback step {fallback:g} K", mutate)


def record_policy(root, reference) -> dict[str, Any]:
    """Run loop: record the policy it derived for a run started without one."""
    def mutate(control):
        if control.get("state") != "running":
            raise ValueError("no active run")
        control["policy"] = dict(reference)
    return _change(root, f"policy revision {reference['revision']} ({reference['hash'][:12]}) derived from settings", mutate)


def set_policy(root, document, *, settings=None, resolve_model=None, known_acquisitions=(),
               requested_by="operator") -> dict[str, Any]:
    """Ask the loop to apply this policy document from the next temperature on.

    The document is validated (its own consistency always; against the
    instrument document and the model registry when ``settings`` and
    ``resolve_model`` are given, as the dashboard does), stored as an immutable
    revision under ``policy/`` and referenced from the control file. The loop
    validates it again against the ledger before applying it.
    """
    from .andie_policy import PolicyDocument, PolicyStore, validate_policy
    if settings is not None and resolve_model is not None:
        policy = validate_policy(document, settings, resolve_model, known_acquisitions=known_acquisitions,
                                 max_step_cap=MAX_STEP_K)
    else:
        policy = document if isinstance(document, PolicyDocument) else PolicyDocument.model_validate(document)
        if policy.acquisition.max_step_k > MAX_STEP_K + 1e-9:
            raise ValueError(f"acquisition.max_step_k exceeds the hard cap of {MAX_STEP_K:g} K")
    if read_control(root).get("state") != "running":
        raise ValueError("no active run")
    reference = PolicyStore(root).save(policy, requested_by=requested_by)

    def mutate(control):
        if control.get("state") != "running":
            raise ValueError("no active run")
        control["policy"] = reference
    return _change(root, f"policy revision {reference['revision']} ({reference['hash'][:12]}) "
                         f"{policy.policy_id or ''}: measured {','.join(policy.measured)}; "
                         f"science {','.join(policy.science_labels)}; model {policy.model.id}", mutate)


def _fallback_step(value) -> float:
    step = float(value)
    if not math.isfinite(step) or not 0 < step <= MAX_STEP_K:
        raise ValueError(f"the fallback step must be above 0 and at most {MAX_STEP_K:g} K")
    return step


def set_fallback_step(root, step_k) -> dict[str, Any]:
    """How far to warm while the posterior is not converged."""
    step = _fallback_step(step_k)

    def mutate(control):
        if control.get("state") != "running":
            raise ValueError("no active run")
        control["fallback_step_k"] = step
    return _change(root, f"fallback step {step:g} K", mutate)


def set_stop(root, stop_k, *, limits=(1.0, 1000.0)) -> dict[str, Any]:
    """Move the stop temperature of the active run; the loop seeks Tc up to it
    from the next temperature."""
    stop = _temperature(stop_k, "stop")

    def mutate(control):
        if control.get("state") != "running":
            raise ValueError("no active run")
        if not float(control["start_k"]) < stop <= float(limits[1]):
            raise ValueError(f"need {float(control['start_k']):g} < stop <= {float(limits[1]):g} K")
        control["stop_k"] = stop
    return _change(root, f"stop {stop:g} K", mutate)


def _cadence(value) -> int:
    cadence = int(value)
    if cadence < 1 or cadence != float(value):
        raise ValueError("diagnostic cadence must be a whole number of temperatures, at least 1")
    return cadence


def set_diagnostic_cadence(root, cadence) -> dict[str, Any]:
    """100/102 at every n-th temperature, counted from the last diagnostics."""
    cadence = _cadence(cadence)

    def mutate(control):
        if control.get("state") != "running":
            raise ValueError("no active run")
        control["diagnostic_cadence"] = cadence
    return _change(root, f"diagnostics every {cadence}", mutate)


def request_diagnostic_next(root) -> dict[str, Any]:
    """100/102 at the next temperature; the cadence then counts from there."""
    def mutate(control):
        if control.get("state") != "running":
            raise ValueError("no active run")
        control["diagnostic_next"] = True
    return _change(root, "diagnostics at the next temperature", mutate)


def clear_diagnostic_next(root) -> None:
    def mutate(control):
        control["diagnostic_next"] = False
    _change(root, "diagnostics scheduled", mutate)


def stop_run(root) -> dict[str, Any]:
    """Ask the loop to stop after the scan in progress (no mid-scan abort)."""
    def mutate(control):
        if control.get("state") != "running":
            raise ValueError("no active run")
        control["stop_requested"] = True
    return _change(root, "stop requested", mutate)


def set_autonomous(root, autonomous: bool) -> dict[str, Any]:
    def mutate(control):
        control["autonomous"] = bool(autonomous)
    return _change(root, f"autonomous={bool(autonomous)}", mutate)


def inject(root, temperature_k, *, cooling=False, current_temperature_k=None) -> dict[str, Any]:
    """Queue a temperature to measure next, ahead of any proposal.

    Warming only unless ``cooling`` is set explicitly (e.g. a hysteresis check).
    """
    target = _temperature(temperature_k, "injected temperature")
    if current_temperature_k is not None and not cooling and target <= float(current_temperature_k):
        raise ValueError(f"{target:g} K is not above the current {float(current_temperature_k):g} K; "
                         "tick 'cooling leg' to measure it deliberately")

    def mutate(control):
        if control.get("state") != "running":
            raise ValueError("no active run")
        if target > float(control["stop_k"]):
            raise ValueError(f"{target:g} K is above the stop temperature {float(control['stop_k']):g} K; "
                             "raise the stop first")
        control.setdefault("injections", []).append(
            {"id": "inject-" + uuid4().hex[:8], "temperature_k": target, "cooling": bool(cooling),
             "requested_at": _now(), "consumed": False})
    return _change(root, f"inject {target:g} K cooling={bool(cooling)}", mutate)


def accept_proposal(root, point_id: str) -> dict[str, Any]:
    """Manual mode: approve the loop's proposed temperature."""
    if not point_id:
        raise ValueError("a proposal id is required")

    def mutate(control):
        if control.get("state") != "running":
            raise ValueError("no active run")
        control["accepted_proposal"] = str(point_id)
    return _change(root, f"accept {point_id}", mutate)


def next_injection(root) -> dict[str, Any] | None:
    """Run loop: the oldest unconsumed injection, if any."""
    for item in read_control(root).get("injections", []):
        if not item.get("consumed"):
            return dict(item)
    return None


def consume_injection(root, injection_id: str) -> None:
    def mutate(control):
        for item in control.get("injections", []):
            if item["id"] == injection_id:
                item["consumed"] = True
                item["consumed_at"] = _now()
                return
        raise ValueError(f"unknown injection {injection_id}")
    _change(root, f"consumed {injection_id}", mutate)


def clear_acceptance(root) -> None:
    def mutate(control):
        control["accepted_proposal"] = None
    _change(root, "proposal acted on", mutate)


def finish_run(root, reason: str) -> None:
    """Run loop: mark the run finished so a new one can start."""
    def mutate(control):
        control["state"] = "finished"
        control["finished_reason"] = reason
        control["finished_at"] = _now()
    _change(root, f"finished: {reason}", mutate)
