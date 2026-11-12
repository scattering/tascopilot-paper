from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_core.corpus import srmn2sb2_agentic_rag as srmn_module
from tas_copilot_core.corpus.srmn2sb2_agentic_rag import build_srmn2sb2_agentic_rag_catalog, run_srmn2sb2_agentic_rag_loop, srmn2sb2_agentic_rag_report


def test_srmn2sb2_agentic_rag_catalog_consumes_docling_markdown(tmp_path: Path) -> None:
    bridges = tmp_path / "001_camn2sb2_bridges_poulton.md"
    bridges.write_text(
        "# Magnetic order in CaMn2Sb2\n\n"
        "Powder neutron diffraction measurements determine the magnetic structure and ordered moment.\n"
        "The Mn moments lie in the ab plane.\n",
        encoding="utf-8",
    )
    mcnally = tmp_path / "002_camnsb2_igor_inelastic.md"
    mcnally.write_text(
        "# CaMn2Sb2 spin waves\n\n"
        "Inelastic neutron scattering on a triple-axis spectrometer was analyzed with a Heisenberg Hamiltonian.\n"
        "The model includes exchange constants J1 and J2 and easy-plane anisotropy.\n",
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog"

    result = build_srmn2sb2_agentic_rag_catalog(
        {
            "catalog": str(catalog),
            "output_dir": str(tmp_path / "build_report"),
            "source_markdowns": [str(bridges), str(mcnally)],
        }
    )

    assert result["status"] == "ok"
    assert result["paper_count"] == 2
    assert result["manifest"]["direct_publication_status"] == "unpublished_scattering_case"
    assert {row["paper_id"] for row in _jsonl(catalog / "papers.jsonl")} == {
        "paper-camn2sb2-bridges-powder",
        "paper-camn2sb2-mcnally-spinwaves",
    }
    documents = _jsonl(catalog / "documents.jsonl")
    assert all(row["metadata"]["is_direct_srmn2sb2_scattering_publication"] is False for row in documents)
    chunks = _jsonl(catalog / "source_chunks.jsonl")
    assert chunks
    assert {row["extraction_method"] for row in chunks} == {"provided_markdown_fixture"}


def test_srmn2sb2_agentic_rag_report_separates_analog_and_direct_evidence(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    _write_jsonl(
        catalog / "papers.jsonl",
        [{"paper_id": "paper-camn2sb2-mcnally-spinwaves", "title": "CaMn2Sb2 spin waves"}],
    )
    _write_jsonl(
        catalog / "source_chunks.jsonl",
        [{"chunk_id": "chunk-1", "paper_id": "paper-camn2sb2-mcnally-spinwaves", "text": "J1 and J2 Hamiltonian"}],
    )
    (catalog / "srmn2sb2_agentic_rag_manifest.json").write_text(
        json.dumps(
            {
                "case_id": "srmn2sb2_agentic_rag_ca_analog_v0",
                "direct_publication_status": "unpublished_scattering_case",
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        catalog / "paper_facts.jsonl",
        [
            {
                "paper_id": "paper-camn2sb2-mcnally-spinwaves",
                "fact_type": "source_data_clue",
                "answer_status": "supported",
                "value": {"text": "Heisenberg Hamiltonian with J1 and J2"},
                "source_chunk_ids": ["chunk-1"],
            }
        ],
    )
    case_note = tmp_path / "case.md"
    case_note.write_text("SrMn2Sb2 direct BT-7 experiment 24858 context uses a CaMn2Sb2-like Hamiltonian with J1.", encoding="utf-8")

    result = srmn2sb2_agentic_rag_report({"catalog": str(catalog), "output_dir": str(tmp_path / "report"), "sr_case_note": str(case_note)})

    assert result["status"] == "ok"
    answer = result["answer"]
    assert answer["direct_srmn2sb2_scattering_publication_found"] is False
    assert answer["ca_mn2sb2_analog_evidence_available"] is True
    assert answer["sr_mn2sb2_direct_experiment_evidence_available"] is True
    assert "CaMn2Sb2 as analog literature evidence" in answer["short_answer"]
    assert Path(result["artifacts"]["html"]).exists()


def test_srmn2sb2_agentic_rag_loop_dry_run_selects_targets(tmp_path: Path) -> None:
    catalog = _build_agentic_loop_fixture(tmp_path)

    result = run_srmn2sb2_agentic_rag_loop(
        {"catalog": str(catalog), "output_dir": str(tmp_path / "loop"), "provider": "none", "max_targets": 2}
    )

    assert result["status"] == "needs_model_or_review"
    assert result["new_fact_count"] == 0
    assert result["target_count"] == 4  # 2 targets x 2 papers
    assert {row["target_id"] for row in result["target_reports"]} == {"magnetic_structure", "hamiltonian"}
    assert all(row["selected_chunk_ids"] for row in result["target_reports"])
    assert Path(result["artifacts"]["html"]).exists()


def test_srmn2sb2_agentic_rag_loop_with_mocked_ollama_writes_targeted_facts(tmp_path: Path, monkeypatch) -> None:
    catalog = _build_agentic_loop_fixture(tmp_path)
    calls: list[dict] = []

    def fake_ollama_chat(*args, **_kwargs):
        packet = json.loads(args[2][1]["content"])
        calls.append(packet)
        target_id = packet["agentic_target"]["target_id"]
        chunk_id = packet["chunks"][0]["chunk_id"]
        if target_id == "hamiltonian":
            fact = {
                "fact_type": "exchange_interaction",
                "answer_status": "supported",
                "value": {"text": "Heisenberg Hamiltonian with J1 and J2 exchange terms"},
                "confidence": 0.86,
                "evidence": [{"chunk_id": chunk_id, "quote": "Heisenberg Hamiltonian includes J1 and J2 exchange"}],
            }
        else:
            fact = {
                "fact_type": "magnetic_structure",
                "answer_status": "supported",
                "value": {"text": "Neel-like magnetic structure on corrugated honeycomb Mn layers"},
                "confidence": 0.82,
                "evidence": [{"chunk_id": chunk_id, "quote": "ordered moment lies in the ab plane"}],
            }
        return {"message": {"content": json.dumps({"facts": [fact]})}}

    monkeypatch.setattr(srmn_module, "ollama_chat", fake_ollama_chat)
    result = run_srmn2sb2_agentic_rag_loop(
        {
            "catalog": str(catalog),
            "output_dir": str(tmp_path / "loop_model"),
            "provider": "ollama",
            "model": "mock-gemma",
            "max_targets": 2,
            "max_followups": 0,
        }
    )

    assert result["status"] == "ok"
    assert result["new_fact_count"] == 4
    assert len(calls) == 4
    facts = _jsonl(catalog / "paper_facts.jsonl")
    assert {fact["fact_type"] for fact in facts} >= {"magnetic_structure", "exchange_interaction"}
    assert all(fact["extractor"] == "bounded_agentic_ollama" for fact in facts)
    assert result["coverage"]["target_coverage"]["hamiltonian"]["exchange_interaction"] == "supported"
    assert result["coverage"]["target_coverage"]["magnetic_structure"]["magnetic_structure"] == "supported"


def _build_agentic_loop_fixture(tmp_path: Path) -> Path:
    bridges = tmp_path / "001_camn2sb2_bridges_poulton.md"
    bridges.write_text(
        "# Magnetic order in CaMn2Sb2\n\n"
        "Powder neutron diffraction measurements determine the magnetic structure. "
        "The ordered moment lies in the ab plane and the structure is CaAl2Si2-type.\n",
        encoding="utf-8",
    )
    mcnally = tmp_path / "002_camnsb2_igor_inelastic.md"
    mcnally.write_text(
        "# CaMn2Sb2 spin waves\n\n"
        "Inelastic neutron scattering on BT-7 measured spin waves. "
        "The Heisenberg Hamiltonian includes J1 and J2 exchange and easy-plane anisotropy.\n",
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog_loop"
    build_srmn2sb2_agentic_rag_catalog(
        {
            "catalog": str(catalog),
            "output_dir": str(tmp_path / "build_loop"),
            "source_markdowns": [str(bridges), str(mcnally)],
        }
    )
    return catalog


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
