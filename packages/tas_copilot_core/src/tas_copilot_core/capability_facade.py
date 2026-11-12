"""Ontology-routed capability facade over the bounded router intents.

The flat MCP surface exposes one tool per intent. This facade exposes ~5
high-level scientific capabilities instead (plan_measurement / interpret /
model / plan / verify): the model proposes a capability plus a scientific
concept, and the routing table - expanded through the TAS ontology's labels
and keywords - disposes which intent runs. Every resolved call goes through
the same hardened ``mcp_gateway.dispatch`` seam (duplicate-call detection,
rate limiting, exception envelopes, path confinement), so the facade adds
routing vocabulary, never new execution paths.

This exists as an A/B experiment (HANDOFF.md 2026-07-10): whether a small
capability vocabulary beats the flat 21-tool surface for local models is an
empirical question for the route-quality evals, not doctrine.

SAFETY: no capability may be named or routed such that instrument execution
becomes expressible. Measurement is planned (plan_measurement), never
performed; ``load_routing`` fails closed on forbidden capability names and
on intents unknown to the router.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tas_copilot_core.workflows.copilot_router import COPILOT_INTENT_EXAMPLES

# Names that would make instrument execution expressible at the capability
# layer. The check is by name because the routing table is config: a future
# edit must not be able to smuggle an execution-shaped capability past review.
FORBIDDEN_CAPABILITY_NAMES = {"measure", "execute", "run", "control", "drive", "move", "count"}

# Extra flat arguments the capability tools accept beyond the routed
# intents' own fields.
_CONCEPT_FIELD = {
    "type": "string",
    "description": (
        "The scientific concept this request is about, in your own words "
        "(e.g. 'spin wave dispersion coverage', 'spurion near 12.5 meV'). "
        "Routing matches it against the TAS ontology's terms."
    ),
}


def default_routing_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "capability_facade" / "routing.yaml"


def default_ontology_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "ontology" / "tas_bt7_seed.yaml"


def load_routing(path: str | Path | None = None, *, ontology_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the routing table; expand ontology terms. Fail closed."""
    routing_path = Path(path) if path else default_routing_path()
    raw = yaml.safe_load(routing_path.read_text(encoding="utf-8")) or {}
    capabilities = raw.get("capabilities") or {}
    if not capabilities:
        raise ValueError(f"No capabilities defined in {routing_path}")

    from tas_copilot_core.ontology.loader import load_ontology

    concepts_by_id = {concept.id: concept for concept in load_ontology(Path(ontology_path) if ontology_path else default_ontology_path())}
    known_intents = {str(item["intent"]) for item in COPILOT_INTENT_EXAMPLES}

    validated: dict[str, Any] = {}
    for name, spec in capabilities.items():
        lowered = str(name).lower()
        if lowered in FORBIDDEN_CAPABILITY_NAMES:
            raise ValueError(f"Forbidden capability name {name!r}: instrument execution must not be expressible (use plan_measurement).")
        routes = []
        for route in spec.get("routes") or []:
            intent = str(route.get("intent") or "")
            if intent not in known_intents:
                raise ValueError(f"Capability {name!r} routes to unknown intent {intent!r}.")
            terms: list[str] = [str(phrase) for phrase in route.get("phrases") or []]
            for concept_id in route.get("ontology_concepts") or []:
                concept = concepts_by_id.get(str(concept_id))
                if concept is None:
                    raise ValueError(f"Capability {name!r} route {intent!r} names unknown ontology concept {concept_id!r}.")
                terms.append(concept.label)
                terms.extend(concept.keywords)
            if not terms:
                raise ValueError(f"Capability {name!r} route {intent!r} has no matchable terms.")
            routes.append({"intent": intent, "terms": terms, "ontology_concepts": list(route.get("ontology_concepts") or [])})
        validated[str(name)] = {"description": str(spec.get("description") or ""), "routes": routes}
    return validated


def resolve(capability: str, concept: str, routing: dict[str, Any]) -> dict[str, Any]:
    """Match a concept against a capability's routes. Fail closed on unknowns and ties."""
    spec = routing.get(capability)
    if spec is None:
        return {"status": "error", "error": f"Unknown capability {capability!r}.", "known_capabilities": sorted(routing)}
    concept_lower = " ".join(str(concept or "").lower().split())
    if not concept_lower:
        return {
            "status": "missing_input",
            "missing_inputs": ["concept"],
            "error": "concept is required.",
            "available_concepts": _concept_menu(spec),
        }
    best: dict[str, Any] | None = None
    best_score = 0
    tied_intents: set[str] = set()
    for route in spec["routes"]:
        for term in route["terms"]:
            term_lower = term.lower()
            if term_lower in concept_lower or concept_lower in term_lower:
                score = len(term_lower)
                if score > best_score:
                    best, best_score, tied_intents = {"intent": route["intent"], "matched_term": term}, score, {route["intent"]}
                elif score == best_score and best is not None:
                    tied_intents.add(route["intent"])
    if best is None:
        return {
            "status": "error",
            "error": f"Concept {concept!r} did not match any {capability} route; pick from the available concepts.",
            "available_concepts": _concept_menu(spec),
        }
    if len(tied_intents) > 1:
        return {
            "status": "error",
            "error": f"Concept {concept!r} is ambiguous between intents {sorted(tied_intents)}; restate it more specifically.",
            "available_concepts": _concept_menu(spec),
        }
    return {"status": "ok", **best}


def _concept_menu(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"intent": route["intent"], "example_terms": route["terms"][:6]} for route in spec["routes"]]


def capability_tools(routing: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """MCP tool definitions for the capability surface (plus concept menus)."""
    from tas_copilot_core.mcp_gateway import INPUT_FIELDS, SAFETY_NOTE, _allowed_fields

    routing = routing or load_routing()
    tools: list[dict[str, Any]] = []
    for name, spec in routing.items():
        fields: set[str] = set()
        for route in spec["routes"]:
            fields.update(_allowed_fields(route["intent"]))
        properties: dict[str, Any] = {"concept": dict(_CONCEPT_FIELD)}
        for field in sorted(fields):
            if field in INPUT_FIELDS:
                properties[field] = dict(INPUT_FIELDS[field])
        menu = "; ".join(f"{route['intent']}: {', '.join(route['terms'][:4])}" for route in spec["routes"])
        tools.append(
            {
                "name": name,
                "description": f"{spec['description']} Routes by concept -> {menu}. {SAFETY_NOTE}",
                "inputSchema": {"type": "object", "properties": properties, "required": ["concept"], "additionalProperties": False},
            }
        )
    return tools


def dispatch_capability(name: str, arguments: dict[str, Any] | None, runs_root: str | Path, *, routing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a capability call to an intent and run it through the hardened gateway seam."""
    from tas_copilot_core.mcp_gateway import dispatch

    routing = routing or load_routing()
    arguments = dict(arguments or {})
    concept = str(arguments.pop("concept", "") or "")
    resolution = resolve(name, concept, routing)
    if resolution.get("status") != "ok":
        return resolution
    response = dispatch(resolution["intent"], arguments, runs_root)
    response["capability_routing"] = {
        "capability": name,
        "concept": concept,
        "resolved_intent": resolution["intent"],
        "matched_term": resolution["matched_term"],
    }
    return response
