#!/usr/bin/env python
"""Score an Ollama-served local model on bounded-copilot route quality.

"LLM proposes, registry disposes": the model only picks an intent for each
suite question; the deterministic router then executes that intent through
``copilot_ask`` (forced-intent path) and the existing route-quality scorer
judges the end-to-end result. The report keeps raw model intent accuracy
separate from system-level route accuracy, because the router's fail-closed
behavior can mask model mistakes (e.g. live-control questions are blocked no
matter what intent the model proposes).

Suite tasks may set ``llm_only: true`` to score intent selection without
executing the router (used by the prompt-injection suite), and
``acceptable_intents`` for safe-but-suboptimal picks that should count as a
pass with a warning (e.g. over-refusing to ``blocked_live_control``).

Requires only the core-package import path and a reachable Ollama server:

  PYTHONPATH=packages/tas_copilot_core/src:packages/tas_copilot_contracts/src \
    python scripts/ollama_route_eval.py \
    --suite eval/copilot_router_route_quality.yaml \
    --model gemma3:4b --output-dir local_runs/ollama_route_eval

"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_core.llm_router import build_intent_menu, build_system_prompt, ollama_pick_intent, openai_compatible_chat
from tas_copilot_core.workflows.copilot_router import (
    _fraction,
    _score_route_quality_task,
    copilot_ask,
    copilot_intent_examples,
)


def vllm_pick_intent(
    base_url: str,
    model: str,
    question: str,
    *,
    system_prompt: str | None = None,
    intent_names: list[str] | None = None,
    api_key: str | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Ask the vLLM (OpenAI-compatible) model for one intent; returns {intent, reason, latency_s, ...}."""
    menu = build_intent_menu()
    names = intent_names if intent_names is not None else [item["intent"] for item in menu]
    prompt = system_prompt if system_prompt is not None else build_system_prompt(menu)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }
    # response_format to enforce JSON output
    payload["response_format"] = {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": names},
            "reason": {"type": "string"},
        },
        "required": ["intent"],
    }
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        # network or HTTP error
        raise RuntimeError(f"vLLM request failed: {exc}") from exc
    body["_latency_s"] = round(time.monotonic() - started, 3)
    # Extract content from choices[0].message.content (OpenAI-compatible)
    # However, to stay consistent with paper_facts.py extraction pattern, we try:
    content = str((body.get("message") or {}).get("content") or "{}")
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = {}
    return {
        "intent": parsed.get("intent"),
        "reason": parsed.get("reason"),
        "latency_s": body.get("_latency_s"),
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = Path(args.suite).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    menu = build_intent_menu()
    system_prompt = build_system_prompt(menu)
    intent_names = [item["intent"] for item in menu]
    example_intents = {item["intent"] for item in copilot_intent_examples()}

    checks: list[dict[str, Any]] = []
    for index, task in enumerate(suite.get("tasks") or [], start=1):
        task_name = str(task.get("name") or f"task_{index}")
        question = str(task.get("question") or (task.get("arguments") or {}).get("question") or "")
        expected_intent = task.get("expected_intent")
        acceptable = set(task.get("acceptable_intents") or [])
        proposal: dict[str, Any] = {}
        proposal_error = None
        for attempt in (1, 2):
            try:
                if args.provider == "ollama":
                    proposal = ollama_pick_intent(
                        args.ollama_url, args.model, question,
                        system_prompt=system_prompt, intent_names=intent_names, timeout=args.timeout,
                    )
                else:  # vllm
                    proposal = vllm_pick_intent(
                        args.vllm_url, args.model, question,
                        system_prompt=system_prompt, intent_names=intent_names, api_key=args.api_key, timeout=args.timeout,
                    )
                proposal_error = None
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
                proposal_error = f"attempt {attempt}: {error}"
                time.sleep(2)
        llm_intent = proposal.get("intent")
        intent_exact = bool(expected_intent) and llm_intent == expected_intent
        intent_acceptable = llm_intent in acceptable
        llm_only = bool(task.get("llm_only")) or args.intent_only

        if llm_only:
            failures = []
            warnings = []
            if proposal_error:
                failures.append(f"vLLM call failed: {proposal_error}" if args.provider == "vllm" else f"ollama call failed: {proposal_error}")
            elif not intent_exact and not intent_acceptable:
                failures.append(f"expected intent {expected_intent}, model picked {llm_intent}")
            if intent_acceptable and not intent_exact:
                warnings.append(f"model picked acceptable fallback {llm_intent} instead of {expected_intent}")
            check = {
                "name": task_name,
                "status": "fail" if failures else "pass",
                "question": question,
                "expected_intent": expected_intent,
                "failures": failures,
                "warnings": warnings,
                "executed": False,
            }
        else:
            request = dict(task.get("arguments") or {})
            request.setdefault("question", question)
            request["output_dir"] = str(output_dir / task_name)
            if llm_intent in example_intents:
                request["intent"] = llm_intent
            result = copilot_ask(request)
            check = _score_route_quality_task(task_name, task, result)
            check["executed"] = True
            check["intent_forced"] = llm_intent in example_intents
        check.update(
            {
                "llm_intent": llm_intent,
                "llm_reason": proposal.get("reason"),
                "llm_intent_correct": intent_exact,
                "llm_intent_acceptable": intent_exact or intent_acceptable,
                "llm_latency_s": proposal.get("latency_s"),
                "llm_error": proposal_error,
            }
        )
        checks.append(check)
        print(f"[{check['status']:>4}] {task_name}: expected={expected_intent} llm={llm_intent} ({proposal.get('latency_s')}s)", flush=True)

    executed = [check for check in checks if check.get("executed")]
    pass_count = sum(1 for check in checks if check["status"] == "pass")
    report = {
        "workflow_name": f"{args.provider}-route-quality-eval",
        "status": "ok" if pass_count == len(checks) else "failed",
        "model": args.model,
        "suite": str(suite_path),
        "suite_id": suite.get("suite_id"),
        "task_count": len(checks),
        "pass_count": pass_count,
        "fail_count": len(checks) - pass_count,
        "metrics": {
            "llm_intent_accuracy": _fraction(checks, "llm_intent_correct"),
            "llm_intent_acceptable_rate": _fraction(checks, "llm_intent_acceptable"),
            "system_route_accuracy": (sum(1 for check in executed if check["status"] == "pass") / len(executed)) if executed else None,
            "grounded_answer_rate": _fraction(executed, "grounded_answer") if executed else None,
            "envelope_completeness_rate": _fraction(executed, "answer_envelope_complete") if executed else None,
            "refusal_boundary_rate": _fraction(executed, "refusal_boundary_present") if executed else None,
            "mean_llm_latency_s": round(sum(check.get("llm_latency_s") or 0.0 for check in checks) / len(checks), 3) if checks else None,
        },
        "checks": checks,
        "safe_conclusion": (
            "Model intent accuracy is a routing-layer measurement only; system-level passes still flow "
            "through the deterministic router's validation, live-control block, and missing-input handling."
        ),
    }
    slug = args.model.replace(":", "_").replace("/", "_")
    json_path = output_dir / f"{args.provider}_route_eval_{slug}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_lines = [
        f"# {args.provider.upper()} Route-Quality Eval",
        "",
        f"Model: `{args.model}` | Suite: `{report['suite_id']}` | Status: `{report['status']}`",
        f"Tasks: {report['task_count']} | Passed: {report['pass_count']} | Failed: {report['fail_count']}",
        "",
        "## Metrics",
        "",
    ]
    md_lines.extend(f"- {key}: {value}" for key, value in report["metrics"].items())
    md_lines.extend(["", "## Tasks", ""])
    for check in checks:
        md_lines.append(f"- `{check['name']}`: {check['status']} (expected `{check.get('expected_intent')}`, model `{check.get('llm_intent')}`)")
        for failure in check.get("failures") or []:
            md_lines.append(f"  - failure: {failure}")
        for warning in check.get("warnings") or []:
            md_lines.append(f"  - warning: {warning}")
    (output_dir / f"{args.provider}_route_eval_{slug}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Base URL for Ollama provider")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000", help="Base URL for vLLM OpenAI-compatible provider")
    parser.add_argument("--api-key", default=None, help="API key for vLLM provider (if required)")
    parser.add_argument("--provider", choices=["ollama", "vllm"], default="ollama", help="LLM provider to use")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--intent-only", action="store_true", help="Score intent selection without executing the router.")
    args = parser.parse_args()
    report = run_suite(args)
    print(json.dumps({key: report[key] for key in ("status", "model", "suite_id", "pass_count", "fail_count", "metrics")}, indent=2))
    return 0 if not any(check.get("llm_error") for check in report["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())