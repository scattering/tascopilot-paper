"""Corpus hygiene found on the 48-paper gold set: documents whose text belongs to
another paper, duplicate ingests, repeated facts, malformed values, and the two
retyping patterns. Fact ids must survive every fix so labels still resolve."""
from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_core.corpus.catalog_integrity import apply_catalog_quarantine, catalog_integrity_report, pdf_integrity_report
from tas_copilot_core.corpus.paper_facts import malformed_value_reason, normalize_paper_fact_response, revalidate_paper_facts


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "")


def _catalog(tmp_path: Path) -> Path:
    cat = tmp_path / "catalog"; cat.mkdir()
    cms = "EUROPEAN ORGANIZATION FOR NUCLEAR RESEARCH CERN CMS LHCb observation of the rare Bs to mu mu decay from the combined analysis"
    _jsonl(cat / "papers.jsonl", [
        {"paper_id": "p-good", "title": "Spin waves in LaFeAsO measured with neutrons"},
        {"paper_id": "p-bad1", "title": "Topological RPdBi half-Heusler semimetals"},
        {"paper_id": "p-bad2", "title": "Pressure-induced transitions in a Mott insulator"},
        {"paper_id": "p-dup-a", "title": "Driven spin wave modes in XY ferromagnet"},
        {"paper_id": "p-dup-b", "title": "Driven spin wave modes in XY ferromagnet"},
    ])
    _jsonl(cat / "source_chunks.jsonl", [
        {"chunk_id": "c-good-1", "paper_id": "p-good", "source_hash": "h-good", "chunk_type": "experimental", "text": "Spin waves in LaFeAsO were measured with neutrons on HB3 at HFIR. The sample was cooled to 5 K. Neutron scattering experiments used the BT-7 spectrometer at the NIST Center for Neutron Research (NCNR)."},
        {"chunk_id": "c-bad1-t", "paper_id": "p-bad1", "source_hash": "h-cms", "chunk_type": "title", "text": "Topological RPdBi half-Heusler semimetals"},
        {"chunk_id": "c-bad1-1", "paper_id": "p-bad1", "source_hash": "h-cms", "chunk_type": "experimental", "text": cms},
        {"chunk_id": "c-bad2-t", "paper_id": "p-bad2", "source_hash": "h-cms", "chunk_type": "title", "text": "Pressure-induced transitions in a Mott insulator"},
        {"chunk_id": "c-bad2-1", "paper_id": "p-bad2", "source_hash": "h-cms", "chunk_type": "experimental", "text": cms},
        {"chunk_id": "c-dupa-1", "paper_id": "p-dup-a", "source_hash": "h-xy", "chunk_type": "experimental", "text": "Driven spin wave modes in the XY ferromagnet were studied by Monte Carlo simulation."},
        {"chunk_id": "c-dupb-1", "paper_id": "p-dup-b", "source_hash": "h-xy", "chunk_type": "experimental", "text": "Driven spin wave modes in the XY ferromagnet were studied by Monte Carlo simulation."},
    ])
    return cat


def test_integrity_report_separates_mismatch_from_duplicate_ingest(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    report = catalog_integrity_report(cat)
    docs = report["documents"]
    assert docs["p-good"]["integrity_status"] == "ok"
    assert docs["p-bad1"]["integrity_status"] == "content_mismatch" and docs["p-bad2"]["integrity_status"] == "content_mismatch"
    assert docs["p-dup-a"]["integrity_status"] == "duplicate_ingest" and docs["p-dup-b"]["integrity_status"] == "duplicate_ingest"
    assert report["quarantine_paper_ids"] == ["p-bad1", "p-bad2"]


def test_quarantine_stamps_facts_and_keeps_ids(tmp_path: Path) -> None:
    cat = _catalog(tmp_path)
    _jsonl(cat / "paper_facts.jsonl", [
        {"fact_id": "f1", "paper_id": "p-bad1", "fact_type": "energy_setting", "answer_status": "supported", "value": {"text": "7 TeV"}, "evidence": [], "warnings": [], "metadata": {}},
        {"fact_id": "f2", "paper_id": "p-good", "fact_type": "instrument", "answer_status": "supported", "value": {"text": "HB3"}, "evidence": [], "warnings": [], "metadata": {}},
    ])
    result = apply_catalog_quarantine(cat)
    assert result["facts_stamped"] == 1 and result["quarantined_papers"] == 2
    rows = {json.loads(l)["fact_id"]: json.loads(l) for l in (cat / "paper_facts.jsonl").read_text().splitlines()}
    assert rows["f1"]["metadata"]["integrity_status"] == "content_mismatch" and rows["f1"]["human_review_required"] is True
    assert "integrity_status" not in rows["f2"]["metadata"]
    assert (cat / "quarantined_paper_ids.txt").read_text().split() == ["p-bad1", "p-bad2"]
    papers = {json.loads(l)["paper_id"]: json.loads(l) for l in (cat / "papers.jsonl").read_text().splitlines()}
    assert papers["p-bad1"]["integrity_status"] == "content_mismatch"


def test_pdf_integrity_groups_files_by_content(tmp_path: Path) -> None:
    root = tmp_path / "pdfs"; root.mkdir()
    (root / "Topological_RPdBi_half-Heusler_semimetals.pdf").write_bytes(b"%PDF cms paper")
    (root / "Pressure-induced_transitions_in_a_Mott_insulator.pdf").write_bytes(b"%PDF cms paper")
    (root / "Spin_waves_in_LaFeAsO.pdf").write_bytes(b"%PDF lafeaso")
    (root / "Spin_waves_in_LaFeAsO_copy.pdf").write_bytes(b"%PDF lafeaso")
    (root / "10.1103_physrevb.106.214438.pdf").write_bytes(b"%PDF cms paper")  # DOI-named copy of the bad file
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("title,doi,downloaded_pdf_path\nTopological RPdBi half-Heusler semimetals,10.1/a,x/Topological_RPdBi_half-Heusler_semimetals.pdf\nPressure-induced transitions in a Mott insulator,10.1/b,\n")
    report = pdf_integrity_report(root, manifest_csv=manifest, first_page_text={"Spin_waves_in_LaFeAsO.pdf": "Spin waves in LaFeAsO measured", "Spin_waves_in_LaFeAsO_copy.pdf": "Spin waves in LaFeAsO copy"})
    assert report["pdf_count"] == 5 and report["bad_download_count"] == 3
    assert report["refetch_dois"] == ["10.1/a", "10.1/b", "10.1103/physrevb.106.214438"]  # manifest path, manifest title, DOI-shaped name


def test_malformed_values_duplicates_and_retyping_keep_ids() -> None:
    chunks = [
        {"chunk_id": "c1", "paper_id": "p", "text": "Neutron scattering experiments were carried out on the triple-axis spectrometer BT-7 at the NIST Center for Neutron Research (NCNR). The sample orders antiferromagnetically below TN = 33 K. Data were collected between 5 K and 300 K."},
    ]
    packet = {"paper_id": "p", "document_id": "d", "chunks": chunks}
    raw = {"facts": [
        {"fact_type": "instrument", "answer_status": "supported", "value": {"text": "NCNR"}, "evidence": [{"chunk_id": "c1", "quote": "NIST Center for Neutron Research (NCNR)"}]},
        {"fact_type": "temperature_range", "answer_status": "supported", "value": {"text": "TN = 33 K"}, "evidence": [{"chunk_id": "c1", "quote": "orders antiferromagnetically below TN = 33 K"}]},
        {"fact_type": "temperature_range", "answer_status": "supported", "value": {"text": "between 5 K and 300 K"}, "evidence": [{"chunk_id": "c1", "quote": "collected between 5 K and 300 K"}]},
        {"fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}, "evidence": [{"chunk_id": "c1", "quote": "triple-axis spectrometer BT-7"}]},
        {"fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}, "evidence": [{"chunk_id": "c1", "quote": "spectrometer BT-7 at the NIST"}]},
        {"fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7; }, // Wait, the prompt asks for JSON"}, "evidence": []},
    ]}
    facts, errors = normalize_paper_fact_response(raw, packet=packet, provider="ollama", model="m")
    assert not errors and len(facts) == 6
    by_index = {f["metadata"]["raw_fact_index"]: f for f in facts}
    assert by_index[0]["fact_type"] == "facility" and by_index[0]["metadata"]["retyped_from"] == "instrument"
    assert by_index[1]["fact_type"] == "transition_temperature"
    assert by_index[2]["fact_type"] == "temperature_range"
    assert by_index[4]["metadata"]["duplicate_of"] == by_index[3]["fact_id"] and by_index[3]["metadata"]["merged_duplicates"] == 1
    assert len(by_index[3]["evidence"]) == 2
    assert by_index[5]["answer_status"] == "ambiguous" and by_index[5]["value"] == {} and "malformed_value" in by_index[5]["metadata"]
    # ids derive from the delivered value and index, so a rerun with the same raw output reproduces them
    facts2, _ = normalize_paper_fact_response(raw, packet=packet, provider="ollama", model="m")
    assert [f["fact_id"] for f in facts2] == [f["fact_id"] for f in facts]


def test_malformed_reason_catalogue() -> None:
    assert malformed_value_reason("B r u k e r  A P E X - I I  C C D  s y s t e m") == "letter-spaced text"
    assert malformed_value_reason("framework. framework. framework. framework. framework.") == "degenerate repetition (model loop)"
    assert malformed_value_reason("confidence=0.0; evidence=chunk_id=chunk-1") == "serialized record fields inside the value"
    assert malformed_value_reason("fixed final energy of 14.7 meV") is None
    assert malformed_value_reason("WAND 2, BT-4, and BT-7 - See Figure " + "details-" * 40 + "facilities") == "degenerate repetition (model loop)"
    assert malformed_value_reason("High Flux Isotope Reactor <0xC2>").startswith("truncated or invalid bytes")
    assert malformed_value_reason("x " * 800).startswith("runaway value")
    assert malformed_value_reason("FIG. 3. (color online) Summary of the main results. " * 14) is None  # a long caption is not a loop
    assert malformed_value_reason("BT1, NOMAD, HB-2A, BT-7, M20 beamline, Quantum Design PPMS") is None
    assert malformed_value_reason("BaFe 2 S 1 . 5 Se 1 . 5") is None  # spaced-out formula from the PDF text, not model noise
    assert malformed_value_reason("∆ E = 1 . 0 and 7.0 meV") is None


def test_revalidate_existing_catalog_keeps_ids_and_upgrades_ocr_variants(tmp_path: Path) -> None:
    cat = tmp_path / "catalog"; cat.mkdir()
    text = "We used beam collimations of open-80'-S80'-120' with fixed final energy of 14.7 meV and two pyrolytic graphite (PG) filters after the sample to reduce higher-order neutrons at BT 7 and one PG filter at HB-1."
    _jsonl(cat / "source_chunks.jsonl", [{"chunk_id": "c1", "paper_id": "p", "text": text, "source_hash": "h"}])
    _jsonl(cat / "papers.jsonl", [{"paper_id": "p", "title": "A paper"}])
    old = {
        "fact_id": "paper-fact-keepme", "paper_id": "p", "document_id": "d", "fact_type": "filter", "answer_status": "ambiguous",
        "value": {"text": "pyrolytic graphite (PG) filters"}, "confidence": None, "extractor": "bounded_ollama", "model": "gemma4:26b",
        "evidence": [{"chunk_id": "c1", "quote": "two pyrolytic graphite (PG) filters after the sample to reduce higher-order neutrons at BT-7", "support_status": "proposed", "warnings": ["Evidence quote was not found in any packet chunk; treat as fabricated evidence until traced."]}],
        "warnings": ["Downgraded from supported: no evidence quote could be located in any packet chunk."], "metadata": {"raw_fact_index": 0}, "provenance": [],
    }
    _jsonl(cat / "paper_facts.jsonl", [old])
    result = revalidate_paper_facts(cat, in_place=True)
    assert result["status"] == "ok" and result["changes"]["status_changed"] == 1
    new = json.loads((cat / "paper_facts.jsonl").read_text().splitlines()[0])
    assert new["fact_id"] == "paper-fact-keepme"
    assert new["answer_status"] == "supported" and new["metadata"]["previous_answer_status"] == "ambiguous"
    assert (cat / "paper_facts.pre_revalidate.jsonl").exists()


def test_instrument_second_pass_retypes_and_classifies():
    from tas_copilot_core.corpus.paper_facts import _sanitize_agentic_fact_semantics as sanitize

    def run(text):
        fact = sanitize({"fact_type": "instrument", "value": {"text": text}, "answer_status": "supported"})
        return fact["fact_type"], (fact.get("metadata") or {}).get("instrument_class"), (fact.get("metadata") or {}).get("retyped_from")

    assert run("BT-7 triple-axis spectrometer") == ("instrument", "neutron", None)
    assert run("Bruker X8 APEX CCD diffractometer") == ("instrument", "xray_or_synchrotron", None)
    assert run("SQUID magnetometer") == ("instrument", "laboratory", None)
    assert run("FullProf")[0] == "data_reduction"
    assert run("diamond anvil cell")[0] == "sample_environment"
    assert run("vertical Bridgman furnace")[0] == "sample_preparation"
    assert run("GdTiO 3")[0] == "material"
    assert run("XAS/XMCD")[0] == "measurement_kind"
    assert run("neutron diffraction")[0] == "measurement_kind"
    assert run("NCNR")[0] == "facility"
    assert run("Neural EKF") == ("instrument", "unclassified", None)


def test_integrity_flags_a_file_saved_under_another_papers_name(tmp_path: Path) -> None:
    """A wrong download with unique content passes the hash and title-body
    checks; the file name it was saved under (the download target) gives it
    away. A correct download whose file name is the title keeps `ok`."""
    from tas_copilot_core.corpus.catalog_integrity import catalog_integrity_report, download_name_overlap

    catalog = tmp_path / "catalog"; catalog.mkdir()
    papers = [
        {"paper_id": "p-good", "title": "Spin waves in a honeycomb antiferromagnet", "metadata": {"pdf_name": "Spin_waves_in_a_honeycomb_antiferromagnet.pdf"}},
        {"paper_id": "p-wrong", "title": "Multi-messenger observations of a binary neutron star merger", "metadata": {"pdf_name": "Evolution_of_the_magnetic_properties_in_the_antiferromagnet.pdf"}},
        {"paper_id": "p-noname", "title": "Magnetic order in a kagome metal", "metadata": {}},
    ]
    chunks = [
        {"chunk_id": "c1", "paper_id": "p-good", "text": "We measure spin waves in a honeycomb antiferromagnet with neutrons.", "source_hash": "h1"},
        {"chunk_id": "c2", "paper_id": "p-wrong", "text": "Multi-messenger observations of a binary neutron star merger were made by LIGO and Virgo.", "source_hash": "h2"},
        {"chunk_id": "c3", "paper_id": "p-noname", "text": "Magnetic order in a kagome metal studied by diffraction.", "source_hash": "h3"},
    ]
    (catalog / "papers.jsonl").write_text("".join(json.dumps(p) + "\n" for p in papers))
    (catalog / "source_chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks))
    assert download_name_overlap("Spin_waves_in_a_honeycomb_antiferromagnet.pdf", "Spin waves in a honeycomb antiferromagnet") == 1.0
    assert download_name_overlap("", "Anything") is None
    report = catalog_integrity_report(catalog)
    docs = report["documents"]
    assert docs["p-good"]["integrity_status"] == "ok"
    assert docs["p-noname"]["integrity_status"] == "ok" and docs["p-noname"]["download_name_overlap"] is None
    assert docs["p-wrong"]["integrity_status"] == "content_mismatch" and "wrong download" in docs["p-wrong"]["reason"]
    assert report["quarantine_paper_ids"] == ["p-wrong"]
