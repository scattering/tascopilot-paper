"""Named LLM endpoint profiles: switch cleanly between Vista and NIST.

Runtime profiles under ``configs/runtime_profiles/`` describe compute sites
(schedulers, paths). LLM endpoint profiles under ``configs/llm_endpoints/``
describe where bounded extraction sends model calls: the provider transport
(Ollama or an OpenAI-compatible/vLLM service), the base URL, the model, and
which environment variables supply them. Secrets never live in the files;
they are resolved from the environment at call time.

The same profile name works for every bounded extraction surface
(``extract-paper-facts``, ``srmn2sb2-agentic-rag-loop``,
``evidence-graph-propose``), so moving a student workflow from Vista Gemma to
the NIST vLLM service is ``--llm-endpoint vista_ollama`` versus
``--llm-endpoint nist_vllm`` with the environment set, not a code change.
Explicit CLI flags always win over profile values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROVIDERS = {"ollama", "vllm"}
_ENDPOINTS_DIRNAME = "llm_endpoints"


def llm_endpoints_dir() -> Path:
    """Locate configs/llm_endpoints relative to the repository layout."""
    override = os.environ.get("TAS_LLM_ENDPOINTS_DIR")
    if override:
        return Path(override).expanduser()
    # packages/tas_copilot_core/src/tas_copilot_core/llm_endpoints.py -> repo root
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "configs" / _ENDPOINTS_DIRNAME


def available_llm_endpoints() -> list[str]:
    directory = llm_endpoints_dir()
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def load_llm_endpoint(name_or_path: str) -> dict[str, Any]:
    """Resolve a named endpoint profile plus environment into call settings.

    Returns {endpoint_id, provider, base_url, model, api_key, missing:
    [env var hints]} and raises ValueError for unknown profiles or providers
    (fail closed with the available names, never a guess).
    """

    candidate = Path(name_or_path).expanduser()
    if candidate.suffix in {".yaml", ".yml"} and candidate.is_file():
        path = candidate
    else:
        path = llm_endpoints_dir() / f"{name_or_path}.yaml"
        if not path.is_file():
            known = ", ".join(available_llm_endpoints()) or "(none found)"
            raise ValueError(f"Unknown LLM endpoint profile {name_or_path!r}. Available: {known}.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    provider = str(raw.get("provider") or "")
    if provider not in PROVIDERS:
        raise ValueError(f"LLM endpoint {path.stem!r} has unsupported provider {provider!r}; allowed: {sorted(PROVIDERS)}.")

    def resolve(env_key: str, default_key: str, example_key: str | None = None) -> tuple[str | None, str | None]:
        env_name = raw.get(env_key)
        if env_name:
            value = os.environ.get(str(env_name))
            if value:
                return value, None
        default = raw.get(default_key)
        if default:
            return str(default), None
        hint = str(env_name) if env_name else None
        if hint and example_key and raw.get(example_key):
            hint = f"{hint} (e.g. {raw.get(example_key)})"
        return None, hint

    base_url, base_missing = resolve("base_url_env", "base_url_default", "base_url_example")
    model, model_missing = resolve("model_env", "model_default")
    api_key = os.environ.get(str(raw["api_key_env"])) if raw.get("api_key_env") else None

    missing = [hint for hint in (base_missing, model_missing) if hint]
    return {
        "endpoint_id": str(raw.get("endpoint_id") or path.stem),
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "missing": missing,
    }


def apply_llm_endpoint(arguments: dict[str, Any], name_or_path: str | None) -> dict[str, Any]:
    """Fill provider/url/model/key defaults from a profile without overriding
    anything the caller passed explicitly. Missing env-derived values become a
    structured missing_input rather than a partial configuration."""

    if not name_or_path:
        return arguments
    endpoint = load_llm_endpoint(name_or_path)
    if endpoint["missing"]:
        raise ValueError(
            f"LLM endpoint {endpoint['endpoint_id']!r} is missing environment values: "
            + "; ".join(f"set {hint}" for hint in endpoint["missing"])
        )
    arguments = dict(arguments)
    # CLI parsers default provider to "none"; a named endpoint supplies it.
    if arguments.get("provider") in (None, "", "none"):
        arguments["provider"] = endpoint["provider"]
    if not arguments.get("model"):
        arguments["model"] = endpoint["model"]
    if endpoint["provider"] == "ollama":
        if not arguments.get("ollama_url"):
            arguments["ollama_url"] = endpoint["base_url"]
    else:
        if not arguments.get("vllm_url"):
            arguments["vllm_url"] = endpoint["base_url"]
        if not arguments.get("api_key") and endpoint["api_key"]:
            arguments["api_key"] = endpoint["api_key"]
    arguments.setdefault("llm_endpoint_id", endpoint["endpoint_id"])
    return arguments
