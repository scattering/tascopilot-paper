"""The scientific policy of an ANDiE run as a versioned, hashed document.

A policy says what is measured at each temperature, which fitted components
enter the likelihood, which model, prior and sampler infer Tc from them, how a
posterior is accepted, and how the next temperature is chosen. The run loop
re-reads the active policy at every temperature boundary
(``docs/andie_policy.md``); this module owns its contract, its validation
against the instrument document and the ledger, its hash, and the immutable
store of revisions in a run root. It imports no scientific implementation:
model observables and parameters come from a registry callback.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable

from pydantic import Field, model_validator
from tas_copilot_contracts.base import ContractModel

POLICY_DIR = "policy"
DEFAULT_ACQUISITION = "tasai_tc_information_rate"
HASH_EXCLUDED = ("note", "policy_id")


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class ScienceComponent(ContractModel):
    reflection: str = Field(min_length=1)
    component: str = Field(min_length=1)


class PolicyDiagnostics(ContractModel):
    reflections: tuple[str, ...] = ("100", "102")
    every_n_temperatures: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def distinct(self):
        if len(set(self.reflections)) != len(self.reflections) or any(not r for r in self.reflections):
            raise ValueError("diagnostic reflections must be distinct nonempty names")
        return self


class PolicyModel(ContractModel):
    id: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class PolicyAcceptance(ContractModel):
    max_gelman_r: float = Field(default=1.2, gt=1.0)
    min_acceptance_percent: float = Field(default=5.0, ge=0.0, le=100.0)


class PolicyAcquisition(ContractModel):
    implementation: str = Field(default=DEFAULT_ACQUISITION, min_length=1)
    eta: float = Field(default=0.7, gt=0.0, le=1.0)
    candidate_step_k: float = Field(default=2.0, gt=0.0)
    max_step_k: float = Field(default=25.0, gt=0.0)
    fallback_step_k: float = Field(default=10.0, gt=0.0)
    warm_only: bool = True

    @model_validator(mode="after")
    def finite(self):
        for name in ("eta", "candidate_step_k", "max_step_k", "fallback_step_k"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"acquisition.{name} must be finite")
        if self.candidate_step_k > self.max_step_k or self.fallback_step_k > self.max_step_k:
            raise ValueError("candidate and fallback steps cannot exceed max_step_k")
        return self


class PolicyDocument(ContractModel):
    """What the run does from the next temperature on. Immutable once stored."""
    schema_version: int = 1
    policy_id: str = Field(default="", max_length=120)
    note: str = Field(default="", max_length=4000)
    measured: tuple[str, ...] = Field(min_length=1)
    science: tuple[ScienceComponent, ...] = Field(min_length=1)
    diagnostics: PolicyDiagnostics = Field(default_factory=PolicyDiagnostics)
    model: PolicyModel
    prior: dict[str, dict[str, Any]] = Field(min_length=1)
    sampler: dict[str, int] = Field(default_factory=lambda: {"samples": 10000, "burn": 100, "pop": 10, "thin": 1})
    acceptance: PolicyAcceptance = Field(default_factory=PolicyAcceptance)
    acquisition: PolicyAcquisition = Field(default_factory=PolicyAcquisition)

    @model_validator(mode="after")
    def consistent(self):
        if self.schema_version != 1:
            raise ValueError("unsupported policy schema_version")
        if len(set(self.measured)) != len(self.measured) or any(not r for r in self.measured):
            raise ValueError("measured reflections must be distinct nonempty names")
        labels = [c.component for c in self.science]
        if len(set(labels)) != len(labels):
            raise ValueError("science components must be distinct")
        if any(c.reflection not in self.measured for c in self.science):
            raise ValueError("every science component must belong to a measured reflection")
        per_reflection = {}
        for c in self.science:
            per_reflection.setdefault(c.reflection, []).append(c.component)
        if any(len(v) > 1 for v in per_reflection.values()):
            raise ValueError("one science component per measured reflection")
        if set(self.measured) & set(self.diagnostics.reflections):
            raise ValueError("a reflection cannot be both measured for science and diagnostic")
        for name, settings in self.prior.items():
            if not isinstance(settings, dict) or "bounds" not in settings or "initial" not in settings:
                raise ValueError(f"prior for {name} needs bounds and initial")
            lo, hi = (float(b) for b in settings["bounds"])
            initial = float(settings["initial"])
            if not all(math.isfinite(v) for v in (lo, hi, initial)) or not lo < hi:
                raise ValueError(f"prior for {name} must have finite ordered bounds")
            if name != "Tc" and not lo <= initial <= hi:
                raise ValueError(f"initial value of {name} must lie within its bounds")
            if settings.get("distribution", "uniform") != "uniform":
                raise ValueError("only bounded uniform priors are supported")
        for key, value in self.sampler.items():
            if key not in {"samples", "burn", "pop", "thin"}:
                raise ValueError("sampler supports samples, burn, pop and thin only")
            if type(value) is not int or value < (0 if key == "burn" else 1):
                raise ValueError(f"sampler.{key} must be a nonnegative integer")
        return self

    @property
    def science_pairs(self) -> list[tuple[str, str]]:
        return [(c.reflection, c.component) for c in self.science]

    @property
    def science_labels(self) -> tuple[str, ...]:
        return tuple(c.component for c in self.science)

    def hashable(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        for key in HASH_EXCLUDED:
            value.pop(key, None)
        return value

    @property
    def hash(self) -> str:
        return content_hash(self.hashable())


def policy_hash(document: PolicyDocument | dict) -> str:
    return (document if isinstance(document, PolicyDocument) else PolicyDocument.model_validate(document)).hash


# ------------------------------------------------------------- derivation
def _science_for(model_spec, catalog: dict) -> list[dict]:
    """The (reflection, component) pairs a model's observables name in the catalog."""
    pairs = []
    for label in model_spec.observables:
        reflections = [name for name, scan in catalog.items()
                       if scan.get("role") == "inference" and scan.get("component_roles", {}).get(label) == "inference"]
        if len(reflections) != 1:
            raise ValueError(f"the catalog must declare exactly one inference scan for component {label}")
        pairs.append({"reflection": reflections[0], "component": label})
    return pairs


def policy_from_settings(settings: dict, resolve_model: Callable[[Any], Any], *, policy_id: str = "from-settings",
                         note: str = "") -> PolicyDocument:
    """Revision 0 of a run started from a settings file (no policy given):
    measured = every inference scan of the catalog, science = the settings'
    model's observables, the rest copied from the settings' inference,
    acquisition and diagnostics sections."""
    catalog = settings["scans"]
    model = {"id": settings["inference"].get("model", "shared_tc_brillouin_pair_v1"),
             "options": dict(settings["inference"].get("model_options") or {})}
    spec = resolve_model(model)
    measured = [name for name, scan in catalog.items() if scan.get("role") == "inference"]
    diagnostics = {"reflections": [name for name, scan in catalog.items() if scan.get("role") == "diagnostic"],
                   "every_n_temperatures": int(settings.get("diagnostics", {}).get("every_n_temperatures", 5))}
    prior = {name: dict(settings["inference"]["prior"][name]) for name in spec.parameters}
    acquisition = {k: v for k, v in settings.get("acquisition", {}).items() if not k.startswith("_")}
    acquisition.setdefault("implementation", DEFAULT_ACQUISITION)
    return PolicyDocument(policy_id=policy_id, note=note, measured=tuple(measured), science=_science_for(spec, catalog),
                          diagnostics=diagnostics, model=model, prior=prior,
                          sampler=dict(settings["inference"]["sampler"]), acquisition=acquisition)


# ------------------------------------------------------------- validation
def validate_policy(document: PolicyDocument | dict, settings: dict, resolve_model: Callable[[Any], Any], *,
                    known_acquisitions=(), max_step_cap: float | None = None,
                    ledger_definitions: dict[str, str] | None = None,
                    definition_hash: Callable[[str], str] | None = None) -> PolicyDocument:
    """A policy is valid for a run when its reflections exist in the instrument
    document's catalog with the roles it assumes, its science components are
    exactly the model's observables, its prior covers the model's parameters,
    its acquisition is registered and within the hard step cap, and its
    science components were not fitted under other definitions earlier in the
    ledger. Raises ValueError with the reason; returns the validated document."""
    policy = document if isinstance(document, PolicyDocument) else PolicyDocument.model_validate(document)
    catalog = settings["scans"]
    for reflection in policy.measured:
        scan = catalog.get(reflection)
        if scan is None:
            raise ValueError(f"measured reflection {reflection} is not in the scan catalog")
        if scan.get("role") != "inference":
            raise ValueError(f"catalog scan {reflection} is not an inference scan")
    for reflection in policy.diagnostics.reflections:
        scan = catalog.get(reflection)
        if scan is None or scan.get("role") != "diagnostic":
            raise ValueError(f"diagnostic reflection {reflection} is not a diagnostic scan of the catalog")
    for reflection, label in policy.science_pairs:
        roles = catalog[reflection].get("component_roles", {})
        if [k for k, v in roles.items() if v == "inference"] != [label]:
            raise ValueError(f"{label} must be the single inference component of catalog scan {reflection}")
    spec = resolve_model(policy.model.model_dump())
    if tuple(spec.observables) != policy.science_labels:
        raise ValueError(f"model {policy.model.id} predicts {list(spec.observables)}, the policy's science "
                         f"components are {list(policy.science_labels)}")
    missing = [name for name in spec.parameters if name not in policy.prior]
    if missing:
        raise ValueError("prior does not cover model parameters: " + ", ".join(missing))
    if known_acquisitions and policy.acquisition.implementation not in known_acquisitions:
        raise ValueError(f"unknown acquisition implementation {policy.acquisition.implementation!r}; "
                         f"known: {', '.join(sorted(known_acquisitions))}")
    if max_step_cap is not None and policy.acquisition.max_step_k > max_step_cap + 1e-9:
        raise ValueError(f"acquisition.max_step_k exceeds the run's hard cap of {max_step_cap:g} K")
    if ledger_definitions and definition_hash is not None:
        changed = [label for reflection, label in policy.science_pairs
                   if label in ledger_definitions and ledger_definitions[label] != definition_hash(reflection)]
        if changed:
            raise ValueError("component definitions of " + ", ".join(changed)
                             + " differ from those earlier observations were fitted under; start a new run")
    return policy


# ------------------------------------------------------------- the store
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PolicyStore:
    """Immutable policy revisions under ``<root>/policy/<revision>_<hash12>.json``."""

    def __init__(self, root: str | Path):
        self.dir = Path(root) / POLICY_DIR

    def records(self) -> list[dict[str, Any]]:
        if not self.dir.exists():
            return []
        out = []
        for path in sorted(self.dir.glob("*.json")):
            match = re.fullmatch(r"(\d{3})_([0-9a-f]{12})\.json", path.name)
            if match:
                out.append({"revision": int(match[1]), "hash12": match[2], "path": str(path)})
        return out

    def save(self, document: PolicyDocument | dict, *, requested_by: str = "") -> dict[str, Any]:
        """Store a new revision (or return the existing record of an identical document)."""
        policy = document if isinstance(document, PolicyDocument) else PolicyDocument.model_validate(document)
        digest = policy.hash
        for record in self.records():
            if record["hash12"] == digest[:12] and json.loads(Path(record["path"]).read_text()).get("_hash") == digest:
                return self.reference(record["revision"], digest, policy)
        revision = max([r["revision"] for r in self.records()], default=-1) + 1
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{revision:03d}_{digest[:12]}.json"
        payload = {**policy.model_dump(mode="json"), "_hash": digest, "_revision": revision,
                   "_stored_at": _now(), "_requested_by": requested_by}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        os.replace(temporary, path)
        return self.reference(revision, digest, policy)

    @staticmethod
    def reference(revision: int, digest: str, policy: PolicyDocument) -> dict[str, Any]:
        return {"revision": int(revision), "hash": digest, "policy_id": policy.policy_id, "requested_at": _now()}

    def load(self, reference: dict[str, Any]) -> PolicyDocument:
        """The document a control-file reference names; its hash must match."""
        digest = str(reference["hash"])
        for record in self.records():
            if record["revision"] == int(reference["revision"]) and record["hash12"] == digest[:12]:
                stored = json.loads(Path(record["path"]).read_text())
                document = {k: v for k, v in stored.items() if not k.startswith("_")}
                policy = PolicyDocument.model_validate(document)
                if policy.hash != digest or stored.get("_hash") != digest:
                    raise ValueError(f"policy revision {reference['revision']} does not match its hash")
                return policy
        raise FileNotFoundError(f"policy revision {reference['revision']} ({digest[:12]}) is not in {self.dir}")
