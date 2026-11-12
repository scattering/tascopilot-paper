"""Durable, offline-safe coordination of paired-area temperature measurements.

Scientific models and inference live in external adapters. This module owns
only validated observations, action identity, cadence and posterior provenance.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from tas_copilot_contracts.base import ContractModel


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class ComponentAreaObservation(ContractModel):
    """Fitted science areas of one temperature block: N labelled components with
    their joint covariance. The fixed 001/101-central pair is the N = 2 case,
    and ledgers written with it load unchanged."""
    action_id: str
    temperature_k: float = Field(gt=0)
    component_ids: tuple[str, ...] = Field(min_length=1)
    areas: tuple[float, ...] = Field(min_length=1)
    covariance: tuple[tuple[float, ...], ...] = Field(min_length=1)
    scan_ids: tuple[str, ...] = Field(min_length=1)
    normalization: str = Field(min_length=1)
    coordinate: Literal["A3", "A4"]
    source_hashes: tuple[str, ...] = Field(min_length=1)
    component_temperatures_k: tuple[float, ...] | None = None
    temperature_ranges_k: tuple[tuple[float, float], ...] | None = None
    # One reflection per component and the hash of the component definitions
    # its scan was fitted with; absent in records written before the
    # generalization (which were always 001 / 101 fitted under one setting).
    reflections: tuple[str, ...] | None = None
    definition_hashes: tuple[str, ...] | None = None
    quality_flags: tuple[str, ...] = ()
    fit_provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_covariance(self):
        n = len(self.component_ids)
        if len(set(self.component_ids)) != n or any(not c for c in self.component_ids):
            raise ValueError("component ids must be unique nonempty labels")
        if len(self.areas) != n or len(self.covariance) != n or any(len(row) != n for row in self.covariance):
            raise ValueError("areas and covariance must match the number of components")
        values = [self.temperature_k, *self.areas, *(x for row in self.covariance for x in row)]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("area observations must be finite")
        for i in range(n):
            for j in range(i + 1, n):
                if not math.isclose(self.covariance[i][j], self.covariance[j][i], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("area covariance must be symmetric")
        if not _positive_definite([list(row) for row in self.covariance]):
            raise ValueError("area covariance must be positive definite")
        if len(self.scan_ids) != n or len(set(self.scan_ids)) != n or any(not x for x in self.scan_ids):
            raise ValueError(f"{n} distinct completed scan identities are required")
        if len(self.source_hashes) != n or any(not x for x in self.source_hashes):
            raise ValueError("source hashes are required")
        for name in ("reflections", "definition_hashes"):
            value = getattr(self, name)
            if value is not None and (len(value) != n or any(not x for x in value)):
                raise ValueError(f"{name} must name every component")
        if self.component_temperatures_k is not None:
            if len(self.component_temperatures_k) != n or any(not math.isfinite(t) or t <= 0 for t in self.component_temperatures_k):
                raise ValueError("component temperatures must be finite and positive")
        if self.temperature_ranges_k is not None:
            if len(self.temperature_ranges_k) != n:
                raise ValueError("temperature ranges must cover every component")
            for i, (lo, hi) in enumerate(self.temperature_ranges_k):
                if not all(math.isfinite(t) and t > 0 for t in (lo, hi)) or lo > hi:
                    raise ValueError("temperature ranges must be finite, positive and ordered")
                if self.component_temperatures_k is not None and not lo <= self.component_temperatures_k[i] <= hi:
                    raise ValueError("component temperature must lie inside its recorded range")
        return self

    def submatrix(self, labels):
        """Areas and covariance of the named components, in that order; None
        for a label this observation does not contain."""
        index = {label: i for i, label in enumerate(self.component_ids)}
        if any(label not in index for label in labels):
            return None
        rows = [index[label] for label in labels]
        return ([self.areas[i] for i in rows], [[self.covariance[i][j] for j in rows] for i in rows])


def project_observation(observation: ComponentAreaObservation, science) -> ComponentAreaObservation:
    """The observation restricted to the ``science`` components ``[(reflection,
    label), ...]`` in that order; components left out stay in the provenance as
    measured fits. Identity when the science set is the observation's own."""
    labels = [label for _, label in science]
    if list(observation.component_ids) == labels:
        return observation
    index = {label: i for i, label in enumerate(observation.component_ids)}
    missing = [label for label in labels if label not in index]
    if missing:
        raise ValueError("observation lacks science components " + ", ".join(missing))
    rows = [index[label] for label in labels]
    pick = lambda values: None if values is None else tuple(values[i] for i in rows)  # noqa: E731
    provenance = dict(observation.fit_provenance)
    fits = provenance.get("full_profile_fits")
    if fits is not None and len(fits) == len(observation.component_ids):
        reflections = observation.reflections or tuple(r for r, _ in (("001", "001"), ("101", "101_central")))[:len(fits)]
        measured = dict(provenance.get("measured_fits", {}))
        for i, fit in enumerate(fits):
            if i not in rows:
                measured[reflections[i]] = fit
        provenance.update(full_profile_fits=[fits[i] for i in rows], measured_fits=measured,
                          science=[list(pair) for pair in science])
    return ComponentAreaObservation(
        action_id=observation.action_id, temperature_k=observation.temperature_k,
        component_ids=tuple(labels), reflections=tuple(reflection for reflection, _ in science),
        areas=pick(observation.areas),
        covariance=tuple(tuple(observation.covariance[i][j] for j in rows) for i in rows),
        scan_ids=pick(observation.scan_ids), normalization=observation.normalization,
        coordinate=observation.coordinate, source_hashes=pick(observation.source_hashes),
        component_temperatures_k=pick(observation.component_temperatures_k),
        temperature_ranges_k=pick(observation.temperature_ranges_k),
        definition_hashes=pick(observation.definition_hashes),
        quality_flags=observation.quality_flags, fit_provenance=provenance)


def _positive_definite(matrix) -> bool:
    """Cholesky by hand (no numpy in the contracts path)."""
    n = len(matrix)
    chol = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = matrix[i][j] - sum(chol[i][k] * chol[j][k] for k in range(j))
            if i == j:
                if total <= 0:
                    return False
                chol[i][j] = math.sqrt(total)
            else:
                chol[i][j] = total / chol[j][j]
    return True


class PairedAreaObservation(ComponentAreaObservation):
    """The fixed 001 / 101-central pair."""
    component_ids: tuple[Literal["001"], Literal["101_central"]] = ("001", "101_central")
    areas: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    scan_ids: tuple[str, str]
    source_hashes: tuple[str, str]
    component_temperatures_k: tuple[float, float] | None = None
    temperature_ranges_k: tuple[tuple[float, float], tuple[float, float]] | None = None


class ActionCost(ContractModel):
    """Disjoint wall-time terms of one temperature block over any measured set;
    ``scan_seconds`` is counting time per reflection, excluding the other terms."""
    scan_seconds: dict[str, float] = Field(min_length=1)
    within_scan_motion_seconds: float = Field(default=0, ge=0)
    reflection_switch_seconds: float = Field(default=0, ge=0)
    temperature_ramp_seconds: float = Field(default=0, ge=0)
    stabilization_seconds: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def finite_cost(self):
        values = [*self.scan_seconds.values(), self.within_scan_motion_seconds, self.reflection_switch_seconds,
                  self.temperature_ramp_seconds, self.stabilization_seconds]
        if not all(math.isfinite(v) for v in values) or any(v <= 0 for v in self.scan_seconds.values()):
            raise ValueError("costs must be finite and scan times positive")
        return self

    @property
    def measurement_seconds(self) -> float:
        return sum(self.scan_seconds.values()) + self.within_scan_motion_seconds

    @property
    def transition_seconds(self) -> float:
        return self.reflection_switch_seconds + self.temperature_ramp_seconds + self.stabilization_seconds

    @property
    def total_seconds(self) -> float:
        return self.measurement_seconds + self.transition_seconds


class PairedScanCost(ContractModel):
    """Disjoint wall-time terms; scan times exclude the other listed terms."""
    scan_001_seconds: float = Field(gt=0)
    scan_101_seconds: float = Field(gt=0)
    within_scan_motion_seconds: float = Field(default=0, ge=0)
    reflection_switch_seconds: float = Field(default=0, ge=0)
    temperature_ramp_seconds: float = Field(default=0, ge=0)
    stabilization_seconds: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def finite_cost(self):
        if not all(math.isfinite(v) for v in self.model_dump().values()):
            raise ValueError("costs must be finite")
        return self

    @property
    def measurement_seconds(self) -> float:
        return self.scan_001_seconds + self.scan_101_seconds + self.within_scan_motion_seconds

    @property
    def transition_seconds(self) -> float:
        return self.reflection_switch_seconds + self.temperature_ramp_seconds + self.stabilization_seconds

    @property
    def total_seconds(self) -> float:
        return self.measurement_seconds + self.transition_seconds


def action_cost(value) -> ActionCost | PairedScanCost:
    """Either cost form from a dict (the paired form is the two-reflection case)."""
    if isinstance(value, (ActionCost, PairedScanCost)):
        return value
    if "scan_seconds" in value:
        return ActionCost.model_validate(value)
    return PairedScanCost.model_validate(value)


class TemperatureSeries:
    """Single-owner persistent ledger. Callers reconcile execution separately.

    A unique action counts as one selected temperature. Repeating an action ID
    with identical input is idempotent. Only accepted complete science pairs
    advance inference; diagnostic records never enter its payload or hash.
    """
    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("schema_version") != 1 or self.state.get("run_id") != run_id:
                raise ValueError("incompatible run ledger")
        else:
            self.state = {"schema_version": 1, "run_id": run_id, "revision": 0,
                          "actions": [], "diagnostics": {}, "inference": None}
        # Older ledgers predate these; their behaviour is unchanged (100-200 K,
        # diagnostics at every fifth temperature).
        self.state.setdefault("tc_bounds", [100.0, 200.0])
        self.state.setdefault("diagnostic_cadence", 5)
        self.state.setdefault("diagnostic_anchor", 0)
        self.state.setdefault("diagnostic_requests", [])
        self.state.setdefault("policy_changes", [])
        self.state.setdefault("definition_changes", [])
        self.state.setdefault("superseded_observations", [])

    def note_policy(self, *, ordinal_from: int, revision: int, policy_hash: str, policy_id: str = "", at: str = ""):
        """Record that a policy applies from this ordinal on (provenance only;
        the policy is not part of the science hash because the data are unchanged)."""
        if not policy_hash:
            raise ValueError("a policy hash is required")
        changes = self.state["policy_changes"]
        if changes and changes[-1]["policy_hash"] == policy_hash:
            return
        changes.append({"ordinal_from": int(ordinal_from), "revision": int(revision),
                        "policy_hash": str(policy_hash), "policy_id": str(policy_id), "at": str(at)})
        self._save()

    def policy_at(self, ordinal: int) -> dict | None:
        """The policy change in force at this ordinal, if any was recorded."""
        current = None
        for change in self.state["policy_changes"]:
            if change["ordinal_from"] <= int(ordinal):
                current = change
        return current

    def definition_hashes(self) -> dict[str, str]:
        """Component label -> definition hash the ledger's observations are
        comparable under: the one earlier observations were fitted under, or
        the latest accepted definition change (accept_definition_change)."""
        found: dict[str, str] = {}
        for action in self.state["actions"]:
            obs = action["observation"]
            if obs and obs.get("definition_hashes"):
                for label, digest in zip(obs["component_ids"], obs["definition_hashes"]):
                    found.setdefault(label, digest)
        for change in self.state["definition_changes"]:
            for label in change["labels"]:
                found[label] = change["definition_hash"]
        return found

    def accept_definition_change(self, *, reflection: str, labels, definition_hash: str, mode: str,
                                 components=(), actions=(), detail: str = "", at: str = ""):
        """Record that a reflection's component definitions changed and how the
        ledger's earlier observations stay comparable: ``compatible`` (only bounds
        that no earlier fit sat on moved, so those fits are unchanged) or
        ``refit`` (each earlier observation re-fitted from its raw scans under the
        new definitions, see refit_observation)."""
        if mode not in {"compatible", "refit"}:
            raise ValueError("a definition change is either compatible or refit")
        if not definition_hash:
            raise ValueError("a definition hash is required")
        self.state["definition_changes"].append({
            "reflection": str(reflection), "labels": [str(label) for label in labels],
            "definition_hash": str(definition_hash), "mode": mode, "components": json.loads(json.dumps(list(components))),
            "actions": [str(a) for a in actions],
            "detail": str(detail), "at": str(at)})
        self._save()

    def refit_observation(self, observation: ComponentAreaObservation | dict, *, reason: str, at: str = ""):
        """Replace an accepted observation by a re-fit of the same raw scans
        (new component definitions). The replaced observation is kept in
        superseded_observations; the science hash changes, so the next
        inference uses the re-fitted areas."""
        obs = ComponentAreaObservation.model_validate(
            observation.model_dump(mode="json") if isinstance(observation, ComponentAreaObservation) else observation)
        if obs.quality_flags:
            raise ValueError("re-fitted observation has unresolved quality flags: " + ", ".join(obs.quality_flags))
        action = self._action(obs.action_id)
        old = action["observation"]
        if old is None:
            raise ValueError("only an accepted observation can be re-fitted")
        payload = obs.model_dump(mode="json")
        if any(list(payload[key]) != list(old[key]) for key in ("scan_ids", "source_hashes", "component_ids")):
            raise ValueError("a re-fit must use the same raw scans and the same components")
        self.state["superseded_observations"].append(
            {"action_id": obs.action_id, "observation": old, "reason": str(reason), "at": str(at)})
        action["observation"] = payload
        self.state["revision"] += 1
        self._save()

    @property
    def tc_bounds(self) -> tuple[float, float]:
        low, high = self.state["tc_bounds"]
        return float(low), float(high)

    def set_tc_bounds(self, bounds):
        low, high = (float(b) for b in bounds)
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
            raise ValueError("Tc bounds must be finite with 0 < low < high")
        if [low, high] != self.state["tc_bounds"]:
            # A posterior under other bounds goes stale by itself: the bounds
            # are part of the science hash every inference result must match.
            if any(not low <= a["temperature_k"] <= high for a in self.state["actions"]):
                raise ValueError("Tc bounds must contain every temperature already in the ledger")
            self.state["tc_bounds"] = [low, high]
            self._save()

    def _diagnostic_rule(self, ordinal: int) -> bool:
        cadence = int(self.state["diagnostic_cadence"])
        anchor = int(self.state["diagnostic_anchor"])
        return ordinal in self.state["diagnostic_requests"] or (
            ordinal > anchor and (ordinal - anchor) % cadence == 0)

    def _wants_diagnostics(self, action) -> bool:
        # Decided at selection; ledgers written before that fall back to the rule.
        return action.get("diagnostics", self._diagnostic_rule(action["ordinal"]))

    def set_diagnostic_cadence(self, cadence: int):
        """Every ``cadence`` temperatures, counted from the last diagnostics."""
        cadence = int(cadence)
        if cadence < 1:
            raise ValueError("diagnostic cadence must be at least 1")
        if cadence != self.state["diagnostic_cadence"]:
            done = [a["ordinal"] for a in self.state["actions"] if a.get("diagnostics")]
            self.state["diagnostic_anchor"] = max([int(self.state["diagnostic_anchor"]), *done])
            self.state["diagnostic_cadence"] = cadence
            self._save()

    def request_diagnostic_at(self, ordinal: int):
        """Diagnostics at this temperature; the cadence restarts from it."""
        ordinal = int(ordinal)
        if ordinal not in self.state["diagnostic_requests"]:
            self.state["diagnostic_requests"].append(ordinal)
        self.state["diagnostic_anchor"] = ordinal
        self._save()

    def anchor_diagnostics_after(self, ordinal: int):
        """Seeded temperatures do not count toward the cadence."""
        self.state["diagnostic_anchor"] = int(ordinal)
        self._save()

    def next_diagnostic_ordinal(self, after: int) -> int:
        ordinal = int(after) + 1
        while not self._diagnostic_rule(ordinal):
            ordinal += 1
        return ordinal

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self.state, sort_keys=True, indent=2, allow_nan=False))
        os.replace(temp, self.path)

    def _action(self, action_id):
        for action in self.state["actions"]:
            if action["action_id"] == action_id:
                return action
        raise KeyError(action_id)

    def select(self, action_id: str, temperature_k: float, scan_request_hash: str,
               temperature_tolerance_k: float = 0.0) -> dict:
        low, high = self.tc_bounds
        if not action_id or not scan_request_hash or not math.isfinite(temperature_k) or not low <= temperature_k <= high:
            raise ValueError(f"selection requires identity, request hash and temperature in {low:g}-{high:g} K")
        if not math.isfinite(temperature_tolerance_k) or temperature_tolerance_k < 0:
            raise ValueError("temperature tolerance must be explicit, finite and nonnegative")
        identity = {"action_id": action_id, "temperature_k": temperature_k,
                    "scan_request_hash": scan_request_hash,
                    "temperature_tolerance_k": temperature_tolerance_k}
        for previous in self.state["actions"]:
            if previous["action_id"] == action_id:
                if any(previous[k] != v for k, v in identity.items()):
                    raise ValueError("action ID reused with different inputs")
                return dict(previous)
        if any(a["observation"] is None for a in self.state["actions"]):
            raise RuntimeError("reconcile the unfinished science pair before selecting again")
        if self.diagnostics_due():
            raise RuntimeError("complete the scheduled diagnostics before selecting again")
        ordinal = len(self.state["actions"]) + 1
        action = {**identity, "ordinal": ordinal, "observation": None,
                  "diagnostics": self._diagnostic_rule(ordinal)}
        self.state["actions"].append(action)
        self._save()
        return dict(action)

    def accept_pair(self, observation: ComponentAreaObservation | dict):
        """The 001 / 101-central pair; see accept_observation for any component set."""
        return self.accept_observation(PairedAreaObservation.model_validate(
            observation.model_dump(mode="json") if isinstance(observation, ComponentAreaObservation) else observation))

    def accept_observation(self, observation: ComponentAreaObservation | dict):
        obs = ComponentAreaObservation.model_validate(
            observation.model_dump(mode="json") if isinstance(observation, ComponentAreaObservation) else observation)
        if obs.quality_flags:
            raise ValueError("science observation has unresolved quality flags: " + ", ".join(obs.quality_flags))
        action = self._action(obs.action_id)
        payload = obs.model_dump(mode="json")
        if action["observation"] is not None:
            if content_hash(action["observation"]) != content_hash(payload):
                raise ValueError("accepted science observation cannot be silently replaced")
            return
        previous = [a["observation"] for a in self.state["actions"] if a["observation"]]
        prior_scans = {scan for row in previous for scan in row["scan_ids"]}
        if prior_scans.intersection(obs.scan_ids):
            raise ValueError("a source scan cannot be counted as another observation")
        if previous and any(payload[k] != previous[0][k] for k in ("normalization", "coordinate")):
            raise ValueError("incompatible area units/scan coordinate require explicit conversion")
        if obs.definition_hashes:
            earlier = self.definition_hashes()
            changed = [label for label, digest in zip(obs.component_ids, obs.definition_hashes)
                       if earlier.get(label, digest) != digest]
            if changed:
                raise ValueError("component definitions changed since earlier observations of "
                                 + ", ".join(changed) + "; areas are not comparable without a re-fit")
        action["observation"] = payload
        self.state["revision"] += 1
        self._save()

    def diagnostics_due(self) -> list[str]:
        return [a["action_id"] for a in self.state["actions"]
                if a["observation"] is not None and self._wants_diagnostics(a)
                and a["action_id"] not in self.state["diagnostics"]]

    def accept_diagnostics(self, action_id: str, results: dict[str, Any], expected=("100", "102")):
        action = self._action(action_id)
        if (not self._wants_diagnostics(action) or action["observation"] is None
                or set(results) != set(expected)):
            raise ValueError("diagnostics require every diagnostic reflection (%s) at a temperature the cadence selects"
                             % "/".join(expected))
        old = self.state["diagnostics"].get(action_id)
        if old is not None and content_hash(old) != content_hash(results):
            raise ValueError("diagnostic record already exists with different results")
        self.state["diagnostics"][action_id] = results
        self._save()

    def science_payload(self) -> dict:
        return {"schema_version": 1, "run_id": self.run_id, "tc_bounds": list(self.tc_bounds),
                "revision": self.state["revision"],
                "observations": [a["observation"] for a in self.state["actions"] if a["observation"]]}

    def science_hash(self) -> str:
        return content_hash(self.science_payload())

    def register_inference(self, request_id: str, input_hash: str):
        if not request_id or not input_hash:
            raise ValueError("inference request identity is required")
        self.state["active_inference"] = {"request_id": request_id, "input_hash": input_hash,
                                          "science_hash": self.science_hash()}
        self._save()

    def accept_inference(self, request_id: str, science_hash: str, result_reference: dict):
        if science_hash != self.science_hash():
            raise ValueError("stale inference result: science inputs changed")
        active = self.state.get("active_inference")
        if (not active or active["request_id"] != request_id or active["science_hash"] != science_hash
                or active["input_hash"] != result_reference.get("input_hash")):
            raise ValueError("inactive inference request or changed model/prior configuration")
        self.state["inference"] = {"request_id": request_id, "science_hash": science_hash,
                                   "result_reference": result_reference}
        self._save()


class TemperatureSeriesCoordinator:
    """Compose existing scan/job boundaries; never chooses a live backend.

    ``fit_pair(request, result)`` is an injected external scientific adapter.
    It must return a PairedAreaObservation after fitting the complete profiles.
    This class neither imports numerical models nor opens an instrument/SSH
    connection. The selected executors retain their own explicit gates.
    """
    PAIR = (("001", "001"), ("101", "101_central"))

    def __init__(self, series: TemperatureSeries, scan_executor, fit_pair=None, fit_diagnostics=None,
                 fit_science=None):
        """``fit_pair(request, result)`` fits the fixed pair; ``fit_science(request,
        result, science)`` fits any science set, ``science`` being
        ``[(reflection, component), ...]``. Either may be given; the pair case
        falls back to ``fit_pair`` when ``fit_science`` is absent."""
        self.series, self.scans, self.fit_pair = series, scan_executor, fit_pair
        self.fit_diagnostics, self.fit_science = fit_diagnostics, fit_science

    def acquire_pair(self, request):
        """The 001 / 101-central science action."""
        return self.acquire_science(request, measured=("001", "101"), science=self.PAIR)

    def acquire_science(self, request, *, measured, science):
        """One temperature block: the ``measured`` reflections, of which the
        ``science`` components (reflection, label) enter the ledger's observation.
        Every science reflection must be measured; measured reflections outside
        the science set are fitted and recorded through the fit provenance only."""
        from .andie_scan_execution import request_hash
        measured, science = tuple(measured), tuple(tuple(pair) for pair in science)
        if request.run_id != self.series.run_id:
            raise ValueError("scan request belongs to another experiment")
        if not measured or len(set(measured)) != len(measured) or not science:
            raise ValueError("a science action needs a nonempty measured set and at least one science component")
        if len(request.scans) != len(measured) or {s.reflection for s in request.scans} != set(measured):
            raise ValueError("science action requires exactly the measured reflections " + ", ".join(measured))
        specs = {s.reflection: s for s in request.scans}
        if any(reflection not in specs for reflection, _ in science):
            raise ValueError("every science component must belong to a measured reflection")
        for reflection, label in science:
            spec = specs[reflection]
            if spec.role != "inference" or [k for k, v in spec.component_roles.items() if v == "inference"] != [label]:
                raise ValueError(f"only intended {label} may be the inference component of {reflection}")
        existing = next((a for a in self.series.state["actions"] if a["action_id"] == request.action_id), None)
        expected_index = existing["ordinal"] if existing else len(self.series.state["actions"]) + 1
        if expected_index != request.selected_temperature_index:
            raise ValueError("selected temperature index does not match durable cadence")
        action = self.series.select(request.action_id, request.temperature_k, request_hash(request))
        result = self.scans.submit(request)
        if result.status not in {"complete", "failed", "refused", "cancelled", "unavailable", "blocked"}:
            result = self.scans.reconcile(request)
        if result.status == "complete" and action["observation"] is None:
            observation = self.fit_for(request, result, measured=measured, science=science)
            ordered_ids = tuple(specs[reflection].scan_id for reflection, _ in science)
            if (observation.action_id != request.action_id or observation.scan_ids != ordered_ids
                    or observation.component_ids != tuple(label for _, label in science)
                    or observation.fit_provenance.get("request_hash") != request_hash(request)):
                raise ValueError("fit result does not belong to this scan request")
            self.series.accept_observation(observation)
        return result

    def fit_for(self, request, result, *, measured, science) -> ComponentAreaObservation:
        """The science observation for a completed request: through
        ``fit_science`` when given; otherwise, for the measured 001/101 pair,
        through ``fit_pair`` with the pair observation projected onto the
        science components (001 alone keeps 101 central as a measured fit)."""
        measured, science = tuple(measured), [tuple(pair) for pair in science]
        if self.fit_science is not None:
            fitted = self.fit_science(request, result, science)
        elif self.fit_pair is not None and set(measured) == {"001", "101"} and all(pair in self.PAIR for pair in science):
            fitted = self.fit_pair(request, result)
        else:
            raise ValueError("a fit_science callback is required for this science set")
        observation = ComponentAreaObservation.model_validate(
            fitted.model_dump(mode="json") if isinstance(fitted, ComponentAreaObservation) else fitted)
        return project_observation(observation, science)

    def acquire_diagnostics(self, science_action_id, request, expected=("100", "102")):
        action = self.series._action(science_action_id)
        if (science_action_id not in self.series.diagnostics_due()
                and science_action_id not in self.series.state["diagnostics"]):
            raise ValueError("diagnostics are not due at this selected temperature")
        if (request.run_id != self.series.run_id or request.temperature_k != action["temperature_k"]
                or request.selected_temperature_index != action["ordinal"]):
            raise ValueError("diagnostics must belong to the selected temperature")
        expected = tuple(expected)
        if (len(request.scans) != len(expected) or {s.reflection for s in request.scans} != set(expected)
                or any(s.role != "diagnostic" for s in request.scans)):
            raise ValueError("diagnostic action requires exactly " + ", ".join(expected))
        result = self.scans.submit(request)
        if result.status not in {"complete", "failed", "refused", "cancelled", "unavailable", "blocked"}:
            result = self.scans.reconcile(request)
        if result.status == "complete":
            by_id = {s.scan_id: s for s in result.scans}
            records = {spec.reflection: {"raw_profile": by_id[spec.scan_id].model_dump(mode="json"),
                       "analysis_status": "not_requested"} for spec in request.scans}
            if self.fit_diagnostics is not None:
                try:
                    fitted = self.fit_diagnostics(request, result)
                    if set(fitted) != set(expected):
                        raise ValueError("diagnostic fitter must return " + ", ".join(expected))
                    for reflection in records:
                        records[reflection].update(analysis_status="fitted", fit=fitted[reflection])
                except Exception as exc:
                    for row in records.values():
                        row.update(analysis_status="failed", reason=str(exc))
            # Failed diagnostic fitting is recorded, never an inference input.
            self.series.accept_diagnostics(science_action_id, records, expected)
        return result

    def inference_request(self, *, parameter_names, model, prior, sampler, seed, provenance=None):
        from .andie_inference_jobs import InferenceJobRequest
        science = self.series.science_payload()
        if not science["observations"]:
            raise ValueError("accepted science observations are required")
        request = InferenceJobRequest(
            payload={**science, "science_hash": self.series.science_hash(), "model": model,
                     "prior": prior, "sampler": sampler, "seed": seed},
            parameter_names=parameter_names, provenance=provenance or {})
        self.series.register_inference(request.request_id, request.input_hash)
        return request

    def accept_job_result(self, request, result, *, job_id, artifact_root):
        result.validate(request, job_id, Path(artifact_root))
        if result.status != "completed":
            raise ValueError("inference did not complete successfully")
        self.series.accept_inference(request.request_id, request.payload["science_hash"], result.to_dict())

    @staticmethod
    def assess_posterior_quality(diagnostics, *, max_gelman_r=1.2, min_acceptance_percent=5.0):
        """Automatic acceptance test so an unattended run can plan.

        Accepts only when every parameter has a finite Gelman-Rubin R at or
        below ``max_gelman_r`` and the mean DREAM acceptance is at least
        ``min_acceptance_percent``. ``diagnostics`` is the worker's result
        diagnostics (``bumps_gelman_r``, ``mean_acceptance_percent``). The
        evidence string records the numbers either way.
        """
        gelman = diagnostics.get("bumps_gelman_r") or {}
        acceptance = diagnostics.get("mean_acceptance_percent")
        problems = []
        if not gelman:
            problems.append("no Gelman-Rubin statistics")
        bad = {name: value for name, value in gelman.items()
               if value is None or not math.isfinite(value) or value > max_gelman_r}
        if bad:
            problems.append("R above %g: %s" % (max_gelman_r, ", ".join(
                "%s=%s" % (k, "nan" if v is None else "%.3f" % v) for k, v in sorted(bad.items()))))
        if acceptance is None or not math.isfinite(acceptance) or acceptance < min_acceptance_percent:
            problems.append("mean acceptance %s%% below %g%%" % (acceptance, min_acceptance_percent))
        worst = max((v for v in gelman.values() if v is not None and math.isfinite(v)), default=float("nan"))
        evidence = "auto: max Gelman R %.3f (limit %g), mean acceptance %s%% (min %g%%)" % (
            worst, max_gelman_r, acceptance, min_acceptance_percent)
        if problems:
            evidence += "; rejected: " + "; ".join(problems)
        return {"accepted": not problems, "evidence": evidence}

    def plan_next(self, *, posterior_samples, parameter_names, candidates,
                  current_temperature_k, policy, posterior_quality, seed=1, eta=.7):
        """Call the external policy only on a current, quality-accepted posterior."""
        inference = self.series.state.get("inference")
        if not inference or inference["science_hash"] != self.series.science_hash():
            raise ValueError("a current completed inference result is required")
        active = self.series.state.get("active_inference", {})
        if active.get("request_id") != inference["request_id"]:
            raise ValueError("the latest inference configuration has not completed")
        if self.series.diagnostics_due() or any(a["observation"] is None for a in self.series.state["actions"]):
            raise RuntimeError("finish/reconcile the current temperature block before planning another")
        expected_names = inference["result_reference"].get("parameter_names")
        if expected_names is None or list(parameter_names) != list(expected_names):
            raise ValueError("posterior parameter order differs from the accepted result")
        payload = {"science_hash": self.series.science_hash(),
                   "posterior_science_hash": inference["science_hash"],
                   "posterior_samples": posterior_samples, "parameter_names": parameter_names,
                   "candidates": candidates, "current_temperature_k": current_temperature_k,
                   "tc_bounds": list(self.series.tc_bounds),
                   "posterior_quality": posterior_quality, "seed": seed, "eta": eta}
        proposal = policy(payload)
        if proposal.get("science_hash") != self.series.science_hash():
            raise ValueError("policy returned a stale or unidentified proposal")
        if proposal.get("status") == "planned":
            candidate = next((c for c in candidates if c["point_id"] == proposal.get("point_id")), None)
            if candidate is None or proposal.get("temperature_k") != candidate["temperature_k"]:
                raise ValueError("policy proposed an unregistered temperature action")
        return proposal
