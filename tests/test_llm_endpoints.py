from __future__ import annotations

import json
from pathlib import Path

import pytest

from tas_copilot_core.llm_endpoints import apply_llm_endpoint, available_llm_endpoints, load_llm_endpoint


def test_committed_profiles_load_and_cover_both_sites() -> None:
    names = available_llm_endpoints()
    assert {"nist_vllm", "vista_ollama", "local_ollama"} <= set(names)
    vista = load_llm_endpoint("vista_ollama")
    assert vista["provider"] == "ollama"
    assert vista["base_url"] == "http://127.0.0.1:11434"
    assert vista["model"] == "gemma3:4b"
    assert not vista["missing"]


def test_nist_profile_resolves_from_environment(monkeypatch) -> None:
    monkeypatch.delenv("NIST_VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("NIST_VLLM_MODEL", raising=False)
    monkeypatch.delenv("NIST_VLLM_API_KEY", raising=False)
    # Without the environment, the profile reports what is missing (no
    # committed URL/model defaults on purpose).
    endpoint = load_llm_endpoint("nist_vllm")
    assert endpoint["provider"] == "vllm"
    assert endpoint["base_url"] is None and endpoint["model"] is None
    assert len(endpoint["missing"]) == 2
    assert any("NIST_VLLM_BASE_URL" in item for item in endpoint["missing"])

    monkeypatch.setenv("NIST_VLLM_BASE_URL", "https://example.nist.gov/api")
    monkeypatch.setenv("NIST_VLLM_MODEL", "gemma-3-27b-it")
    monkeypatch.setenv("NIST_VLLM_API_KEY", "sk-test")
    endpoint = load_llm_endpoint("nist_vllm")
    assert endpoint["base_url"] == "https://example.nist.gov/api"
    assert endpoint["model"] == "gemma-3-27b-it"
    assert endpoint["api_key"] == "sk-test"
    assert not endpoint["missing"]


def test_unknown_profile_fails_closed_with_available_names() -> None:
    with pytest.raises(ValueError) as excinfo:
        load_llm_endpoint("does-not-exist")
    assert "Available:" in str(excinfo.value)
    assert "nist_vllm" in str(excinfo.value)


def test_apply_llm_endpoint_fills_without_overriding(monkeypatch) -> None:
    monkeypatch.setenv("NIST_VLLM_BASE_URL", "https://example.nist.gov/api")
    monkeypatch.setenv("NIST_VLLM_MODEL", "gemma-3-27b-it")
    monkeypatch.setenv("NIST_VLLM_API_KEY", "sk-test")

    # CLI-default provider "none" is replaced by the profile provider.
    filled = apply_llm_endpoint({"provider": "none", "model": None, "vllm_url": None, "api_key": None}, "nist_vllm")
    assert filled["provider"] == "vllm"
    assert filled["vllm_url"] == "https://example.nist.gov/api"
    assert filled["model"] == "gemma-3-27b-it"
    assert filled["api_key"] == "sk-test"
    assert filled["llm_endpoint_id"] == "nist_vllm"

    # Explicit flags always win over profile values.
    overridden = apply_llm_endpoint(
        {"provider": "vllm", "model": "other-model", "vllm_url": "https://other/api", "api_key": "sk-mine"},
        "nist_vllm",
    )
    assert overridden["model"] == "other-model"
    assert overridden["vllm_url"] == "https://other/api"
    assert overridden["api_key"] == "sk-mine"


def test_apply_llm_endpoint_missing_env_is_an_error(monkeypatch) -> None:
    monkeypatch.delenv("NIST_VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("NIST_VLLM_MODEL", raising=False)
    with pytest.raises(ValueError) as excinfo:
        apply_llm_endpoint({"provider": "none"}, "nist_vllm")
    assert "NIST_VLLM_BASE_URL" in str(excinfo.value)


def test_custom_profile_path_and_dir_override(tmp_path: Path, monkeypatch) -> None:
    profile = tmp_path / "my_endpoint.yaml"
    profile.write_text(
        json.dumps(
            {
                "endpoint_id": "my_endpoint",
                "provider": "ollama",
                "base_url_default": "http://127.0.0.1:9999",
                "model_default": "gemma3:4b",
            }
        ),
        encoding="utf-8",
    )
    endpoint = load_llm_endpoint(str(profile))
    assert endpoint["base_url"] == "http://127.0.0.1:9999"

    monkeypatch.setenv("TAS_LLM_ENDPOINTS_DIR", str(tmp_path))
    assert available_llm_endpoints() == ["my_endpoint"]


def test_agentic_loop_accepts_vllm_provider(tmp_path: Path, monkeypatch) -> None:
    """The agentic loop is endpoint-agnostic: the same bounded packet goes to
    Ollama or an OpenAI-compatible service."""

    from tas_copilot_core.corpus import srmn2sb2_agentic_rag as loop_module
    from tas_copilot_core.corpus.srmn2sb2_agentic_rag import run_srmn2sb2_agentic_rag_loop

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "papers.jsonl").write_text(
        json.dumps({"paper_id": "paper-x", "title": "Paper X"}) + "\n", encoding="utf-8"
    )
    (catalog / "source_chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "chunk-x-1",
                "paper_id": "paper-x",
                "text": "The magnetic structure has ordered moment 3.4 muB with propagation vector k=(0,0,0).",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen = {}

    def fake_openai_chat(*, base_url, model, messages, api_key=None, response_format=None, max_tokens=4096, timeout=300.0):
        seen["base_url"] = base_url
        seen["model"] = model
        seen["api_key"] = api_key
        content = json.dumps(
            {
                "facts": [
                    {
                        "fact_type": "magnetic_structure",
                        "answer_status": "supported",
                        "value": {"text": "antiferromagnetic order"},
                        "confidence": 0.9,
                        "evidence": [{"chunk_id": "chunk-x-1", "quote": "The magnetic structure has ordered moment"}],
                    }
                ]
            }
        )
        return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(loop_module, "openai_compatible_chat", fake_openai_chat)
    report = run_srmn2sb2_agentic_rag_loop(
        {
            "catalog": str(catalog),
            "output_dir": str(tmp_path / "out"),
            "provider": "vllm",
            "model": "gemma-3-27b-it",
            "vllm_url": "https://example.nist.gov/api",
            "api_key": "sk-test",
            "max_targets": 1,
            "max_followups": 0,
        }
    )

    assert seen["base_url"] == "https://example.nist.gov/api"
    assert seen["model"] == "gemma-3-27b-it"
    assert seen["api_key"] == "sk-test"
    assert report["new_fact_count"] >= 1
    facts = [json.loads(line) for line in (catalog / "paper_facts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(fact["fact_type"] == "magnetic_structure" for fact in facts)


def test_agentic_loop_vllm_requires_url(tmp_path: Path) -> None:
    from tas_copilot_core.corpus.srmn2sb2_agentic_rag import run_srmn2sb2_agentic_rag_loop

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "source_chunks.jsonl").write_text(
        json.dumps({"chunk_id": "c", "paper_id": "p", "text": "text"}) + "\n", encoding="utf-8"
    )
    report = run_srmn2sb2_agentic_rag_loop(
        {"catalog": str(catalog), "output_dir": str(tmp_path / "out"), "provider": "vllm", "model": "m"}
    )
    assert report["status"] == "missing_input"
