"""Interactive chat shell over the bounded copilot router.

The shell holds typed session state so that ``missing_input`` responses from
``copilot_ask`` become follow-up questions: the router reports what is
missing, the shell asks the user, remembers the supplied values, and re-runs
the pending question. Intent selection is optionally proposed by a local
model through :mod:`tas_copilot_core.llm_router` ("LLM proposes, registry
disposes"); with no model, or whenever the proposal layer fails, routing
falls back to the deterministic keyword router — answering exactly as the
CLI does today.

Output is a deterministic rendering of the structured ``answer_envelope``.
An optional narration callable may rephrase the envelope (house rules live in
``llm_router.NARRATION_HOUSE_RULES``); the narration is clearly labeled and
appended after, never instead of, the deterministic rendering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from tas_copilot_core.workflows.copilot_router import COPILOT_INTENT_EXAMPLES, copilot_ask
from tas_copilot_core.llm_fencing import check_narration_grounding, get_house_rule_constant

# Request fields a user may supply interactively and that persist across turns.
SESSION_ARG_FIELDS: tuple[str, ...] = (
    "corpus_catalog",
    "data_catalog",
    "data_root",
    "precedent_index",
    "paper_id",
    "figure_id",
    "proposal_evidence_dir",
    "actual_evidence_dir",
    "fixture",
    "experiment_config",
)


def _fields_for_missing_input(name: str) -> list[str]:
    """Map a router missing-input label to the request fields that satisfy it.

    Labels are mostly field names, but some are prose alternatives such as
    "fixture_or_proposal_evidence_dir" or "precedent_index or local hydrated
    BT-7 index"; offer every session field the label mentions.
    """

    mentioned = [field for field in SESSION_ARG_FIELDS if field in name]
    return mentioned or [name]


class ChatShell:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        propose_intent: Callable[[str], dict[str, Any]] | None = None,
        narrate: Callable[[str, dict[str, Any]], str] | None = None,
        session_defaults: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.propose_intent = propose_intent
        self.narrate = narrate
        self.defaults: dict[str, Any] = {
            key: value for key, value in (session_defaults or {}).items() if key in SESSION_ARG_FIELDS and value
        }
        self.pending: dict[str, Any] | None = None
        self.turns: list[dict[str, Any]] = []

    # -- public entry point -------------------------------------------------

    def handle(self, user_text: str) -> str:
        user_text = user_text.strip()
        if not user_text:
            return "Ask a question, or use `set <field> <value>`, `session`, or `cancel`."
        command = self._handle_command(user_text)
        if command is not None:
            return command
        if self.pending is not None:
            reply = self._fill_pending(user_text)
        else:
            reply = self._ask(user_text)
        self._save_session()
        return reply

    # -- commands -----------------------------------------------------------

    def _handle_command(self, text: str) -> str | None:
        lowered = text.lower()
        if lowered == "session":
            state = {"defaults": self.defaults, "pending": self._pending_summary(), "turns": len(self.turns)}
            return json.dumps(state, indent=2)
        if lowered == "cancel":
            if self.pending is None:
                return "Nothing pending to cancel."
            question = self.pending["question"]
            self.pending = None
            self._save_session()
            return f"Cancelled the pending question: {question!r}."
        if lowered.startswith("set "):
            parts = text.split(None, 2)
            if len(parts) < 3 or parts[1] not in SESSION_ARG_FIELDS:
                known = ", ".join(SESSION_ARG_FIELDS)
                return f"Usage: set <field> <value>. Known fields: {known}."
            self.defaults[parts[1]] = parts[2].strip()
            self._save_session()
            if self.pending is not None and not self._still_missing():
                return self._rerun_pending()
            return f"Set {parts[1]}. " + (self._missing_prompt() if self.pending else "")
        return None

    # -- slot filling -------------------------------------------------------

    def _still_missing(self) -> list[str]:
        assert self.pending is not None
        missing = []
        for name in self.pending["missing_inputs"]:
            if not any(self.defaults.get(field) for field in _fields_for_missing_input(name)):
                missing.append(name)
        return missing

    def _fill_pending(self, text: str) -> str:
        assert self.pending is not None
        filled = False
        for token in text.split():
            if "=" in token:
                key, _, value = token.partition("=")
                if key in SESSION_ARG_FIELDS and value:
                    self.defaults[key] = value
                    filled = True
        if not filled:
            missing = self._still_missing()
            candidates = _fields_for_missing_input(missing[0]) if missing else []
            if len(candidates) == 1 and candidates[0] in SESSION_ARG_FIELDS:
                self.defaults[candidates[0]] = text
                filled = True
        if not filled:
            return (
                "I could not match that to a missing input. Reply with `field=value` "
                "(for example `data_root=/path/to/experiment`), `set <field> <value>`, or `cancel`.\n"
                + self._missing_prompt()
            )
        if self._still_missing():
            return self._missing_prompt()
        return self._rerun_pending()

    def _rerun_pending(self) -> str:
        assert self.pending is not None
        pending = self.pending
        self.pending = None
        return self._ask(pending["question"], forced_intent=pending.get("intent"))

    def _missing_prompt(self) -> str:
        missing = self._still_missing() if self.pending else []
        if not missing:
            return ""
        lines = ["To run that, I still need:"]
        for name in missing:
            fields = _fields_for_missing_input(name)
            hint = " or ".join(f"`{field}=<value>`" for field in fields)
            lines.append(f"  - {name} (reply {hint})")
        lines.append("(or `cancel`)")
        return "\n".join(lines)

    def _pending_summary(self) -> dict[str, Any] | None:
        if self.pending is None:
            return None
        return {
            "question": self.pending["question"],
            "intent": self.pending.get("intent"),
            "missing_inputs": self._still_missing(),
        }

    # -- routing ------------------------------------------------------------

    def _ask(self, question: str, forced_intent: str | None = None) -> str:
        proposal: dict[str, Any] = {}
        proposal_error: str | None = None
        intent = forced_intent
        if intent is None and self.propose_intent is not None:
            try:
                proposal = self.propose_intent(question) or {}
                intent = proposal.get("intent")
            except Exception as error:  # any transport/parse failure -> deterministic fallback
                proposal_error = str(error)
                intent = None
        known_intents = {item["intent"] for item in COPILOT_INTENT_EXAMPLES}
        request: dict[str, Any] = {key: value for key, value in self.defaults.items() if value}
        request["question"] = question
        request["output_dir"] = str(self.output_dir / f"turn_{len(self.turns) + 1:03d}")
        if intent in known_intents:
            request["intent"] = intent
        result = copilot_ask(request)
        turn = {
            "question": question,
            "llm_intent": proposal.get("intent") if proposal else forced_intent,
            "llm_reason": proposal.get("reason"),
            "llm_error": proposal_error,
            "routed_intent": (result.get("route") or {}).get("intent"),
            "status": result.get("status"),
            "output_dir": request["output_dir"],
        }
        self.turns.append(turn)
        if result.get("status") == "missing_input":
            # Route-level missing inputs live in result.missing_inputs; when the
            # route was runnable but a workflow rejected its inputs (for example
            # a data_root path that does not exist), the names are per-call
            # missing_fields instead.
            missing = list(result.get("missing_inputs") or [])
            if not missing:
                for call in result.get("routed_workflows") or []:
                    missing.extend(call.get("missing_fields") or [])
            missing = list(dict.fromkeys(missing))
            if not any(field in SESSION_ARG_FIELDS for name in missing for field in _fields_for_missing_input(name)):
                self.pending = None
                return self._render(result, question)
            self.pending = {
                "question": question,
                "intent": intent if intent in known_intents else None,
                "missing_inputs": missing,
            }
            # A reported-missing input that the session already "supplies" means
            # the value did not validate (typically a path that does not exist).
            # Drop it so the user can re-enter instead of looping silently.
            rejected = []
            for name in self.pending["missing_inputs"]:
                for field in _fields_for_missing_input(name):
                    if self.defaults.pop(field, None) is not None:
                        rejected.append(field)
            note = ""
            if rejected:
                note = "\nThe supplied value(s) for " + ", ".join(sorted(set(rejected))) + " were not accepted (check the path exists) — please re-enter."
            return self._render(result, question) + "\n" + note + "\n" + self._missing_prompt()
        return self._render(result, question)

    # -- rendering ----------------------------------------------------------

    def _render(self, result: dict[str, Any], question: str) -> str:
        route = result.get("route") or {}
        envelope = result.get("answer_envelope") or {}
        lines = [str(result.get("answer") or "").strip()]
        lines.append("")
        confidence = route.get("confidence")
        confidence_text = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
        lines.append(f"[route: {route.get('intent')} (confidence {confidence_text}) | status: {result.get('status')}]")
        evidence = envelope.get("evidence") or []
        if evidence:
            lines.append("evidence:")
            lines.extend(f"  - {item}" for item in evidence[:8])
        for label, key in (("next", "safe_next_actions"), ("not approved", "not_approved_operations"), ("limitations", "limitations")):
            values = result.get(key) or []
            if values:
                lines.append(f"{label}:")
                lines.extend(f"  - {value}" for value in values[:5])
        artifacts = result.get("artifacts") or {}
        if artifacts:
            lines.append("artifacts:")
            lines.extend(f"  - {name}: {path}" for name, path in list(artifacts.items())[:5])
        if self.narrate is not None and result.get("status") != "blocked":
            try:
                narration = self.narrate(question, envelope)
            except Exception as error:
                narration = f"(narration unavailable: {error})"
            if narration:
                # Check narration grounding
                is_grounded, violations = check_narration_grounding(
                    narration,
                    json.dumps(envelope),
                    envelope.get("uncertainty_class", "planning_evidence")
                )
                if is_grounded:
                    lines.append("")
                    lines.append("model narration (rendered from the envelope above; planning evidence, not findings):")
                    lines.append(narration)
                else:
                    # Narration failed grounding check - fall back to deterministic rendering
                    lines.append("")
                    lines.append("(narration failed grounding check; using deterministic rendering)")
                    lines.append("  - Violations: " + "; ".join(violations))
        return "\n".join(line for line in lines if line is not None)

    # -- durable state ------------------------------------------------------

    def _save_session(self) -> None:
        state = {
            "defaults": self.defaults,
            "pending": self._pending_summary(),
            "turns": self.turns,
        }
        (self.output_dir / "session.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_chat_repl(
    output_dir: str,
    *,
    model: str | None = None,
    ollama_url: str = "http://127.0.0.1:11434",
    narrate: bool = False,
    session_defaults: dict[str, Any] | None = None,
    timeout: float = 300.0,
) -> int:
    """Blocking stdin/stdout REPL used by ``tas-copilot chat``."""

    propose = None
    narrator = None
    if model:
        from tas_copilot_core.llm_router import make_ollama_proposer, ollama_narrate

        propose = make_ollama_proposer(ollama_url, model, timeout=timeout)
        if narrate:
            narrator = lambda question, envelope: ollama_narrate(ollama_url, model, question, envelope, timeout=timeout)
    shell = ChatShell(output_dir, propose_intent=propose, narrate=narrator, session_defaults=session_defaults)
    routing = f"LLM-proposed ({model})" if model else "deterministic"
    print(f"TASCopilot chat shell — routing: {routing}; planning/simulation only, no live control.")
    print("Commands: set <field> <value> | session | cancel | quit")
    while True:
        try:
            text = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if text.strip().lower() in {"quit", "exit"}:
            return 0
        print(shell.handle(text))
