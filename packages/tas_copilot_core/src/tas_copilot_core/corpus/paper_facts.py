from __future__ import annotations

import hashlib
import html
import difflib
import json
import re
from pathlib import Path
from typing import Any

from tas_copilot_contracts import PaperFact, PaperFactEvidence, ProvenanceRecord
from tas_copilot_core.llm_router import ollama_chat, openai_compatible_chat
from tas_copilot_core.reporting import write_report_bundle


FACT_TYPES = [
    "instrument",
    "facility",
    "sample",
    "sample_form",
    "material",
    "material_alias",
    "composition",
    "crystal_structure",
    "magnetic_structure",
    "propagation_vector",
    "spin_direction",
    "ordered_moment",
    "exchange_interaction",
    "magnetic_anisotropy",
    "magnetization",
    "hysteresis_property",
    "sample_preparation",
    "measurement_kind",
    "detector_mode",
    "instrument_configuration",
    "monochromator",
    "analyzer",
    "collimation",
    "filter",
    "energy_setting",
    "temperature_range",
    "transition_temperature",
    "magnetic_field",
    "pressure",
    "sample_environment",
    "data_reduction",
    "calibration",
    "background",
    "normalization",
    "resolution",
    "figure_caption",
    "figure_panel_mapping",
    "source_data_clue",
    "uncertainty_note",
]
ANSWER_STATUSES = {"supported", "not_found", "ambiguous", "unsupported"}
PROMPT_VERSION = "paper_fact_extraction_v1"  # v1 (2026-10-17): citation required, verbatim short quotes, one fact per stated item, 60 facts max; temperature_range includes processing temperatures, ranges kept whole; k read from a stated magnetic space group


def extract_paper_facts(arguments: dict[str, Any]) -> dict[str, Any]:
    """Extract bounded, cited facts from one or more papers.

    This intentionally does not classify scientific sections by regex. Existing
    chunk labels are only input evidence spans; facts must come back as typed
    answers with chunk citations or as explicit not_found/ambiguous records.
    """

    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = str(arguments.get("provider") or "none")
    paper_id_filter = arguments.get("paper_id")
    max_chars_per_paper = int(arguments.get("max_chars_per_paper") or 45000)

    if not catalog_dir.exists():
        return _missing_report(output_dir, f"Corpus catalog does not exist: {catalog_dir}")

    chunks = _load_jsonl(catalog_dir / "source_chunks.jsonl")
    papers = _load_jsonl(catalog_dir / "papers.jsonl")
    documents = _load_jsonl(catalog_dir / "documents.jsonl")
    document_by_paper = {str(item.get("paper_id")): item for item in documents if item.get("paper_id")}
    title_by_paper = {str(item.get("paper_id")): item.get("title") for item in papers if item.get("paper_id")}
    paper_ids = sorted({str(chunk.get("paper_id")) for chunk in chunks if chunk.get("paper_id")})
    if paper_id_filter:
        paper_ids = [str(paper_id_filter)] if str(paper_id_filter) in paper_ids else []
    if not paper_ids:
        return _missing_report(output_dir, "No source chunks found for the requested paper.")

    all_facts: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for paper_id in paper_ids:
        paper_chunks = [chunk for chunk in chunks if str(chunk.get("paper_id")) == paper_id]
        packet = _paper_fact_packet(
            paper_id,
            paper_chunks,
            title=title_by_paper.get(paper_id) or document_by_paper.get(paper_id, {}).get("title"),
            max_chars=max_chars_per_paper,
        )
        packets.append(packet)
        if provider == "none":
            continue
        try:
            if provider == "json-file":
                raw = _load_model_output(Path(str(arguments.get("model_output") or "")).expanduser(), paper_id)
            elif provider == "ollama":
                model = str(arguments.get("model") or "")
                if not model:
                    errors.append({"paper_id": paper_id, "error": "provider=ollama requires --model"})
                    continue
                raw = _call_ollama(packet, base_url=str(arguments.get("ollama_url") or "http://127.0.0.1:11434"), model=model)
            elif provider == "vllm":
                model = str(arguments.get("model") or "")
                if not model:
                    errors.append({"paper_id": paper_id, "error": "provider=vllm requires --model"})
                    continue
                vllm_url = str(arguments.get("vllm_url") or "")
                if not vllm_url:
                    errors.append({"paper_id": paper_id, "error": "provider=vllm requires --vllm-url"})
                    continue
                api_key = str(arguments.get("api_key") or None)
                raw = openai_compatible_chat(
                    base_url=vllm_url,
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You extract bounded scientific paper facts for TAS Co-pilot. "
                                "Return only JSON. Do not create a graph. Do not infer missing facts. "
                                "Cite chunk_id and exact short quotes for every supported fact."
                            ),
                        },
                        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
                    ],
                    api_key=api_key,
                    response_format={"type": "json_object" },
                    timeout=300.0,
                )
                content = str(
                    (((raw.get("choices") or [{}])[0]).get("message") or {}).get("content")
                    or "{}"
                )
                raw = json.loads(content)
            else:
                errors.append({"paper_id": paper_id, "error": f"Unknown provider: {provider}"})
                continue
        except Exception as exc:
            errors.append({"paper_id": paper_id, "error": f"{provider} extraction failed: {exc}"})
            continue
        for error in raw.get("errors") or []:
            errors.append({"paper_id": paper_id, "error": str(error)})
        facts, validation_errors = normalize_paper_fact_response(raw, packet=packet, provider=provider, model=arguments.get("model"))
        all_facts.extend(facts)
        errors.extend({"paper_id": paper_id, "error": item} for item in validation_errors)

    _write_json(output_dir / "paper_fact_extraction_packets.json", {"packets": packets})
    if all_facts:
        existing = _load_jsonl(catalog_dir / "paper_facts.jsonl")
        _write_jsonl(catalog_dir / "paper_facts.jsonl", _dedupe(existing + all_facts, "fact_id"))

    report = {
        "status": "needs_model_or_review" if provider == "none" else ("ok" if not errors else "partial"),
        "provider": provider,
        "paper_count": len(paper_ids),
        "fact_count": len(all_facts),
        "prompt_version": PROMPT_VERSION,
        "facts": all_facts,
        "errors": errors,
        "warnings": [
            "Keyword-derived chunk labels are hints only and are not used as authoritative paper facts.",
            "Every supported fact must cite source chunks; not_found and ambiguous are valid outcomes.",
            "Generated facts from private papers are local derived artifacts and must not be committed.",
        ],
    }
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_paper_fact_markdown,
        json_filename="paper_fact_extraction_report.json",
        markdown_filename="paper_fact_extraction_report.md",
    )
    report["artifacts"]["packets"] = str(output_dir / "paper_fact_extraction_packets.json")
    return report


def paper_fact_review(arguments: dict[str, Any]) -> dict[str, Any]:
    """Write an interactive human-review packet for extracted paper facts."""

    catalog_dir = Path(arguments["catalog"]).expanduser()
    output_dir = Path(arguments["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_ids = _paper_id_filter(arguments.get("paper_id"), arguments.get("paper_ids_file"))
    if not catalog_dir.exists():
        return _missing_report(output_dir, f"Corpus catalog does not exist: {catalog_dir}")
    report = build_paper_fact_review(catalog_dir, paper_ids=paper_ids)
    _write_json(output_dir / "paper_fact_review.json", report)
    (output_dir / "paper_fact_review.html").write_text(_paper_fact_review_html(report), encoding="utf-8")
    report["artifacts"] = {
        "json": str(output_dir / "paper_fact_review.json"),
        "html": str(output_dir / "paper_fact_review.html"),
    }
    return report


def discover_fact_catalogs(runs_root: Path) -> list[dict[str, Any]]:
    """Corpus catalogs usable for fact review: corpus_catalogs plus any
    run directory carrying a synced catalog/ (e.g. Vista agentic-RAG runs)."""
    catalogs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    roots = [runs_root / "corpus_catalogs", runs_root]
    patterns = ["*/papers.jsonl", "*/catalog/papers.jsonl"]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for papers_path in sorted(root.glob(pattern)):
                catalog_dir = papers_path.parent.resolve()
                if catalog_dir in seen:
                    continue
                seen.add(catalog_dir)
                fact_path = catalog_dir / "paper_facts.jsonl"
                fact_count = sum(1 for line in fact_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()) if fact_path.exists() else 0
                label = catalog_dir.name if catalog_dir.name != "catalog" else catalog_dir.parent.name
                catalogs.append({"path": str(catalog_dir), "label": label, "fact_count": fact_count})
    catalogs.sort(key=lambda item: (-item["fact_count"], item["label"]))
    return catalogs


def paper_fact_table(catalog_dir: Path) -> dict[str, Any]:
    """Per-paper fact counts for the workbench fact-review page."""
    papers = _load_jsonl(catalog_dir / "papers.jsonl")
    facts = _load_jsonl(catalog_dir / "paper_facts.jsonl")
    counts: dict[str, dict[str, Any]] = {}
    for fact in facts:
        entry = counts.setdefault(str(fact.get("paper_id")), {"fact_count": 0, "fact_types": set(), "supported": 0})
        entry["fact_count"] += 1
        entry["fact_types"].add(str(fact.get("fact_type")))
        if fact.get("answer_status") == "supported":
            entry["supported"] += 1
    rows = []
    for paper in papers:
        paper_id = str(paper.get("paper_id"))
        entry = counts.get(paper_id) or {"fact_count": 0, "fact_types": set(), "supported": 0}
        rows.append(
            {
                "paper_id": paper_id,
                "title": paper.get("title"),
                "fact_count": entry["fact_count"],
                "supported_count": entry["supported"],
                "fact_types": sorted(entry["fact_types"]),
            }
        )
    rows.sort(key=lambda row: (-row["fact_count"], str(row["title"] or row["paper_id"])))
    extract_hint = (
        "python -m tas_copilot_core.cli corpus extract-paper-facts "
        f"--catalog {catalog_dir} --paper-id [PAPER_ID] --provider ollama --output-dir local_runs/paper_facts_extraction"
    )
    return {"catalog": str(catalog_dir), "paper_count": len(rows), "papers": rows, "extract_command_hint": extract_hint}


def build_paper_fact_review(catalog_dir: Path, *, paper_ids: set[str] | None = None) -> dict[str, Any]:
    """Assemble the fact-review report (no artifacts written)."""

    papers = _load_jsonl(catalog_dir / "papers.jsonl")
    facts = _load_jsonl(catalog_dir / "paper_facts.jsonl")
    chunks = _load_jsonl(catalog_dir / "source_chunks.jsonl")
    title_by_paper = {str(item.get("paper_id")): item.get("title") for item in papers if item.get("paper_id")}
    chunk_by_id = {str(item.get("chunk_id")): item for item in chunks if item.get("chunk_id")}
    if paper_ids:
        facts = [fact for fact in facts if str(fact.get("paper_id")) in paper_ids]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        grouped.setdefault(str(fact.get("paper_id")), []).append(fact)
    records: list[dict[str, Any]] = []
    for paper_id in sorted(grouped):
        paper_facts = sorted(grouped[paper_id], key=lambda item: (str(item.get("fact_type")), str(item.get("fact_id"))))
        records.append(
            {
                "paper_id": paper_id,
                "title": title_by_paper.get(paper_id),
                "fact_count": len(paper_facts),
                "fact_types": sorted({str(fact.get("fact_type")) for fact in paper_facts}),
                "facts": [_fact_for_review(fact, chunk_by_id=chunk_by_id) for fact in paper_facts],
            }
        )

    report = {
        "status": "ok",
        "catalog": str(catalog_dir),
        "paper_count": len(records),
        "fact_count": sum(item["fact_count"] for item in records),
        "records": records,
        "review_instructions": [
            "Check whether each value is actually supported by the cited quote (expand the chunk text to see context).",
            "Set a verdict per fact and add notes where useful; selections persist in this browser.",
            "Export the labels JSONL, then ingest with: tas-copilot review-labels ingest --labels paper_fact_review_labels.jsonl --store <review_outcomes.sqlite>",
            "Do not treat supported PaperFact records as accepted paper-experiment links; they are evidence for later ranking.",
        ],
    }
    return report


def _fact_value_text(fact: dict[str, Any]) -> str:
    value = fact.get("value")
    if isinstance(value, dict):
        return " ".join(str(item) for item in value.values() if item is not None)
    return "" if value is None else str(value)


_FACILITY_WORDS = re.compile(r"\b(ncnr|nist|hfir|ornl|sns|spallation neutron source|high flux isotope reactor|center for neutron research|isis|ill\b|institut laue|j-parc|ansto|psi\b|frm ii|mlz|lansce|cern|oak ridge national lab|brookhaven|argonne|los alamos)", re.IGNORECASE)
_TRANSITION_WORDS = re.compile(r"(\bt\s*_?\s*[nc]\b|\bt\s*n\s*=|\bt\s*c\s*=|n[ée]el|curie|superconducting transition|structural transition|orders? (magnetically )?(below|at)|transition temperature|\btn\b|\btc\b|\bts\b)", re.IGNORECASE)
# Second pass over `instrument` values (2026-09-20): the flag pass and the rebuilt catalog showed
# software, techniques, sample environment, growth apparatus and even material names filed as
# instruments. Clear non-instruments are retyped; the rest are classified in metadata.
_NEUTRON_INSTRUMENT_WORDS = re.compile(r"\b(bt-?\d|hb-?\d[a-z]?|ng-?\d|cg-?\d[a-z]?|spins|macs|sequoia|arcs|hyspec|cncs|merlin|mari|maps|puma|taipan|wand\s?2?|dcs|nse|sans|reflectometer|triple[- ]axis|spectrometer|diffractometer|neutron powder|d2b|d20|in\d+|wish|let\b|vulcan|nomad|powgen|corelli|topaz|snap|hb-2a|bt-?1\b)", re.IGNORECASE)
_XRAY_INSTRUMENT_WORDS = re.compile(r"\b(beamline|beam line|bl\d+|synchrotron|aps\b|nsls|ssrl|als\b|spring-8|bruker|huber|rigaku|panalytical|image plate|ccd|x-?ray diffractometer|four-circle)", re.IGNORECASE)
_LAB_INSTRUMENT_WORDS = re.compile(r"\b(ppms|mpms|squid|magnetometer|vsm|physical propert(y|ies) measurement system|magnetic propert(y|ies) measurement system|quantum design|dilatometer|calorimeter|electron microscope|sem\b|tem\b|edx|eds\b|xps|nmr spectrometer)", re.IGNORECASE)
_SOFTWARE_WORDS = re.compile(r"\b(fullprof|shelx\w*|tobyfit|dave\b|mantid|gsas\w*|spinw|sarah|jana\d*|vesta|reslib|resolution program|program|software|package|code)\b", re.IGNORECASE)
_ENVIRONMENT_WORDS = re.compile(r"\b(cryostat|refrigerator|dilution|diamond[- ]?anvil|pressure cell|indenter cell|displex|closed[- ]cycle|ccr\b|cryomagnet|superconducting magnet|orange cryostat|he-?\s?cryostat|\d\s?he[- ]cryostat|cryofurnace)\b", re.IGNORECASE)
_GROWTH_WORDS = re.compile(r"\b(furnace|floating[- ]zone|bridgman|czochralski|flux growth|flux method|arc[- ]melt\w*|image furnace|solid[- ]state reaction)\b", re.IGNORECASE)
_TECHNIQUE_WORDS = re.compile(r"\b(scattering|diffraction|spectroscop\w*|magnetometry|dichroism|xas|xmcd|arpes|\wpes\b|transport|susceptibility|heat capacity|specific heat|resistivity|analysis|measurements?|reflectometry|imaging|tomography|dft|calculations?|simulations?|monte carlo)\b", re.IGNORECASE)
_FORMULA_LIKE = re.compile(r"^(?:\(?[A-Z][a-z]?\)?\s?\d*\.?\d*\s?)+$")


def _instrument_second_pass(fact: dict[str, Any], text: str) -> dict[str, Any]:
    """Retype clear non-instruments; classify what stays as an instrument in metadata."""
    low = text.lower()
    meta = fact.setdefault("metadata", {})
    if _XRAY_INSTRUMENT_WORDS.search(text) and not re.search(r"\bneutron\b", low):
        meta["instrument_class"] = "xray_or_synchrotron"; return fact
    if _NEUTRON_INSTRUMENT_WORDS.search(text):
        meta["instrument_class"] = "neutron"; return fact
    if _LAB_INSTRUMENT_WORDS.search(text):
        meta["instrument_class"] = "laboratory"; return fact
    target, reason = "", ""
    if _SOFTWARE_WORDS.search(text):
        target, reason = "data_reduction", "software, not an apparatus"
    elif _ENVIRONMENT_WORDS.search(text):
        target, reason = "sample_environment", "sample environment, not the instrument"
    elif _GROWTH_WORDS.search(text):
        target, reason = "sample_preparation", "growth apparatus, not the instrument"
    elif _FORMULA_LIKE.fullmatch(text.strip()) and re.search(r"[a-z]", text) and len(re.findall(r"[A-Z][a-z]?", text)) >= 2:
        target, reason = "material", "a chemical formula, not an instrument"
    elif _TECHNIQUE_WORDS.search(text) and not re.search(r"\binstrument\b", low):
        target, reason = "measurement_kind", "a technique, not an apparatus"
    if target:
        fact["fact_type"] = target
        meta["retyped_from"] = "instrument"
        fact["warnings"] = list(fact.get("warnings") or []) + [f"Retyped instrument -> {target}: the value is {reason}."]
    else:
        meta["instrument_class"] = "unclassified"
    return fact


_MEASUREMENT_WORDS = re.compile(r"\b(measured|measurements?|scans?|collected|data|from|down to|up to|between|range|to)\b", re.IGNORECASE)


def malformed_value_reason(text: str) -> str | None:
    """Model-output artifacts that are not paper content: serialized evidence
    records, the model's own reasoning, degenerate repetition, letter-spaced text."""
    t = str(text or "")
    if not t.strip():
        return None
    low = t.lower()
    if re.search(r"(chunk_id\s*=|\bquote\s*=|\bevidence\s*=|\bconfidence\s*=\s*0\.0|answer_status\s*:|\}\s*,\s*//)", low):
        return "serialized record fields inside the value"
    if re.search(r"(\bwait,|the prompt asks|i should return|the user wants|as an ai|let me |\[thought\]|```json)", low):
        return "model reasoning text inside the value"
    if re.search(r"\b(\w{3,}[.:]?)(?:\s+\1){4,}", low):
        return "degenerate repetition (model loop)"
    if re.search(r"\b(\w{3,})(?:[\s\-_/.,]+\1\b){4,}", low):
        return "degenerate repetition (model loop)"  # "details-details-details-..." joined by punctuation
    if "\ufffd" in t or re.search(r"<0x[0-9a-f]{2}>", low):
        return "truncated or invalid bytes in the value (generation cut off)"
    if len(t) > 1500:  # a long figure caption is 700-800 characters; a loop that hit the token limit is thousands
        return "runaway value (longer than any stated fact)"
    tokens = t.split()
    # Letter-spaced output ("B r u k e r  A P E X") is single LETTERS; a formula
    # the PDF text spaces out ("BaFe 2 S 1 . 5 Se 1 . 5", "E = 1 . 0 meV") is
    # single digits and dots around real words, and is paper content.
    if len(tokens) >= 8 and sum(1 for tok in tokens if len(tok) == 1 and tok.isalpha()) >= 0.6 * len(tokens):
        return "letter-spaced text"
    return None


def _value_key(value: Any) -> str:
    return _normalize_alnum(json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value))


def _sanitize_agentic_fact_semantics(fact: dict[str, Any]) -> dict[str, Any]:
    fact = dict(fact)
    fact_type = str(fact.get("fact_type") or "")
    text = _fact_value_text(fact)
    low = text.lower()
    warning: str | None = None
    # Deterministic retyping of two patterns that dominated the 2026-09 flag pass:
    # facilities filed as instruments, and transition temperatures filed as
    # measurement ranges. The fact id is already fixed, so retyping is safe.
    if fact_type == "instrument" and _FACILITY_WORDS.search(text) and not re.search(r"\b(bt-?\d|hb-?\d|ng-?\d|spins|macs|sequoia|arcs|hyspec|cncs|merlin|spectrometer|diffractometer|beamline|beam line|magnetometer|ppms|squid|mpms)\b", low):
        fact["fact_type"] = "facility"
        fact.setdefault("metadata", {})["retyped_from"] = "instrument"
        fact["warnings"] = list(fact.get("warnings") or []) + ["Retyped instrument -> facility: the value names a laboratory or source, not an apparatus."]
    elif fact_type == "instrument" and text.strip():
        fact = _instrument_second_pass(fact, text)
    elif fact_type == "temperature_range" and _TRANSITION_WORDS.search(text) and not _MEASUREMENT_WORDS.search(text):
        fact["fact_type"] = "transition_temperature"
        fact.setdefault("metadata", {})["retyped_from"] = "temperature_range"
        fact["warnings"] = list(fact.get("warnings") or []) + ["Retyped temperature_range -> transition_temperature: the value is a transition temperature, not a measurement range."]
    fact_type = str(fact.get("fact_type") or "")
    if fact.get("answer_status") == "supported":
        if fact_type == "energy_setting":
            has_energy = bool(re.search(r"\b(ei|ef|incident energy|final energy|mev|ev)\b", low))
            temperature_like = bool(re.search(r"\btemperature|temperatures|\d+\s*k\b", low))
            if temperature_like and not has_energy:
                warning = "Semantic sanity check: sample temperature is not a neutron energy setting."
        elif fact_type == "temperature_range":
            has_temperature = bool(re.search(r"\btemperature|temperatures|\d+\s*k\b", low))
            energy_like = bool(re.search(r"\bmev|incident energy|final energy|excitation energy\b", low))
            if energy_like and not has_temperature:
                warning = "Semantic sanity check: excitation/neutron energy is not a sample temperature range."
        elif fact_type == "magnetic_field":
            has_field_unit = bool(re.search(r"\b(applied field|magnetic field|h|koe|oe|tesla|t)\b", low))
            magnetic_order_like = bool(re.search(r"\bmagnetic moment|spin direction|magnetic order|exchange interaction|j\s*[123c]\b", low))
            if magnetic_order_like and not has_field_unit:
                warning = "Semantic sanity check: magnetic order/moment/exchange is not an externally applied magnetic field."
        elif fact_type == "sample_environment":
            if re.search(r"\b(powder neutron diffraction|inelastic neutron scattering|measurement|measurements)\b", low) and not re.search(
                r"\b(displex|cryostat|furnace|magnet|pressure cell|closed cycle|ccr|sample environment|helium)\b", low
            ):
                warning = "Semantic sanity check: measurement kind is not sample environment."
        elif fact_type == "facility":
            if re.fullmatch(r"\s*bt-?\d+\s*", low):
                warning = "Semantic sanity check: instrument name alone is not a facility."
    if warning:
        fact["answer_status"] = "ambiguous"
        fact["human_review_required"] = True
        warnings = list(fact.get("warnings") or [])
        warnings.append(warning)
        fact["warnings"] = warnings
    return fact


def normalize_paper_fact_response(
    raw: dict[str, Any],
    *,
    packet: dict[str, Any],
    provider: str,
    model: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    chunk_by_id = {str(chunk["chunk_id"]): chunk for chunk in packet.get("chunks", [])}
    facts_in = raw.get("facts") if isinstance(raw, dict) else None
    if not isinstance(facts_in, list):
        return [], ["Model output must be an object with a facts list."]
    facts: list[dict[str, Any]] = []
    errors: list[str] = []
    paper_id = str(packet["paper_id"])
    document_id = packet.get("document_id")
    norm_chunk_texts = {cid: _normalize_alnum(str(chunk.get("text") or "")) for cid, chunk in chunk_by_id.items()}
    paper_by_chunk = {cid: str(chunk.get("paper_id") or paper_id) for cid, chunk in chunk_by_id.items()}
    for index, item in enumerate(facts_in):
        if not isinstance(item, dict):
            errors.append(f"facts[{index}] is not an object")
            continue
        fact_type = str(item.get("fact_type") or "")
        status = str(item.get("answer_status") or "not_found")
        if fact_type not in FACT_TYPES:
            errors.append(f"facts[{index}] has unsupported fact_type={fact_type!r}")
            continue
        if status not in ANSWER_STATUSES:
            errors.append(f"facts[{index}] has unsupported answer_status={status!r}")
            continue
        evidence_records: list[PaperFactEvidence] = []
        warnings: list[str] = []
        quoted_evidence_count = 0
        located_evidence_count = 0
        for evidence in item.get("evidence") or []:
            if not isinstance(evidence, dict):
                warnings.append("Dropped non-object evidence record.")
                continue
            chunk_id = str(evidence.get("chunk_id") or "")
            if chunk_id.strip().lower() in _NULL_STRINGS:
                chunk_id = ""
            # Revalidation feeds stored evidence back in; start from the model's
            # own transcription where it was kept, so the result is the same on
            # every pass.
            model_quote = str(evidence.get("quote_as_extracted") or evidence.get("quote") or "").strip() or None
            if model_quote and model_quote.lower() in _NULL_STRINGS:
                model_quote = None  # the model wrote the word "None"/"null" where a quote belongs
            if not model_quote and chunk_id not in chunk_by_id:
                # Nothing to verify and nothing to cite: an empty record, not a citation.
                warnings.append("Dropped an evidence record with no quote and no known chunk id.")
                continue
            ev_warnings: list[str] = []
            quote = model_quote
            if quote:
                quote, leaked = _strip_schema_leak(quote)
                quote = quote.strip() or None
                if leaked:
                    ev_warnings.append("Extraction schema text had leaked into the quote and was removed.")
            located = _locate_quote(
                quote or "",
                cited_chunk_id=chunk_id,
                fact_paper_id=paper_id,
                norm_chunk_texts=norm_chunk_texts,
                paper_by_chunk=paper_by_chunk,
            )
            support_status = "supported"
            snap_in: str | None = None  # chunk whose text the quote should be snapped to
            if quote:
                quoted_evidence_count += 1
            if chunk_id not in chunk_by_id and located["status"] in ("no_quote", "not_found"):
                ev_warnings.append("Evidence chunk_id is not present in the source packet.")
                support_status = "proposed"
            if not quote and chunk_id:
                ev_warnings.append("Evidence record cites a chunk but provides no quote; the citation cannot be machine-verified.")
                support_status = "proposed"
            if located["status"] == "in_other_chunk" and len(located["found_in"]) == 1:
                # Mechanical provenance repair: the quote is real verbatim text,
                # only its bookkeeping was wrong. Keep the model-cited id in the
                # warning so the correction is auditable.
                new_chunk_id = located["found_in"][0]
                ev_warnings.append(
                    f"Citation re-homed from {chunk_id or '(missing chunk_id)'} to {new_chunk_id}: the quote is verbatim there, not in the model-cited chunk."
                )
                chunk_id = new_chunk_id
                snap_in = chunk_id
                located_evidence_count += 1
            elif located["status"] == "in_other_chunk":
                ev_warnings.append(
                    "Evidence quote is verbatim in multiple other chunks (" + ", ".join(located["found_in"]) + ") but not in the cited chunk; requires review."
                )
                support_status = "proposed"
                located_evidence_count += 1
            elif located["status"] == "in_other_paper":
                ev_warnings.append(
                    "Evidence quote is verbatim only in another paper's chunk (" + ", ".join(located["found_in"]) + ") - cross-paper leak; requires review."
                )
                support_status = "proposed"
            elif located["status"] == "stitched_fragments":
                pieces = _split_stitched_quote(
                    quote or "", cited_chunk_id=chunk_id, fact_paper_id=paper_id, chunk_by_id=chunk_by_id,
                    norm_chunk_texts=norm_chunk_texts, paper_by_chunk=paper_by_chunk,
                )
                if pieces:
                    # Each piece is real contiguous text; only the joining was
                    # the model's. Cite the pieces separately.
                    for piece_chunk_id, snapped in pieces:
                        piece_chunk = chunk_by_id.get(piece_chunk_id)
                        evidence_records.append(
                            PaperFactEvidence(
                                chunk_id=piece_chunk_id,
                                quote=" ".join(snapped["text"].split()),
                                quote_as_extracted=model_quote,
                                char_start=snapped["char_start"],
                                char_end=snapped["char_end"],
                                quote_match="split_fragment",
                                page=piece_chunk.get("page_start") if piece_chunk else None,
                                section=piece_chunk.get("section") or piece_chunk.get("chunk_type") if piece_chunk else None,
                                support_status="supported",
                                warnings=ev_warnings + [
                                    f"Split from a stitched quote: the model joined {len(pieces)} separate passages with an ellipsis; this passage is contiguous in {piece_chunk_id}."
                                ],
                                provenance=[ProvenanceRecord(source_type="source_chunk", source_id=piece_chunk_id, method="paper_fact_evidence_citation")],
                            )
                        )
                    located_evidence_count += 1
                    continue
                ev_warnings.append(
                    "Evidence quote is stitched from fragments of " + ", ".join(located["found_in"]) + "; it is not contiguous text in any chunk; requires review."
                )
                support_status = "proposed"
                located_evidence_count += 1
            elif located["status"] == "elided_passages":
                home = located["found_in"][0]
                home_chunk = chunk_by_id.get(home) or {}
                home_text = str(home_chunk.get("text") or "")
                _, offsets = _alnum_offsets(home_text)
                runs = located["runs"]
                if home != chunk_id:
                    ev_warnings.append(
                        f"Citation re-homed from {chunk_id or '(missing chunk_id)'} to {home}: the passages are there, not in the model-cited chunk."
                    )
                for start, end in runs:
                    char_start, char_end = _complete_span(home_text, offsets, start, end)
                    evidence_records.append(
                        PaperFactEvidence(
                            chunk_id=home,
                            quote=" ".join(home_text[char_start:char_end].split()),
                            quote_as_extracted=model_quote,
                            char_start=char_start,
                            char_end=char_end,
                            quote_match="split_fragment" if len(runs) > 1 else "snapped",
                            page=home_chunk.get("page_start"),
                            section=home_chunk.get("section") or home_chunk.get("chunk_type"),
                            support_status="supported",
                            warnings=ev_warnings + [
                                f"Split from a shortened quote: the model copied {len(runs)} passage(s) of {home} and left the text between them out; this passage is contiguous there."
                                if len(runs) > 1 else
                                f"Quote replaced by the exact source passage in {home}; the model's copy dropped part of it."
                            ],
                            provenance=[ProvenanceRecord(source_type="source_chunk", source_id=home, method="paper_fact_evidence_citation")],
                        )
                    )
                located_evidence_count += 1
                continue
            elif located["status"] == "not_found" and chunk_id in chunk_by_id:
                ev_warnings.append("Evidence quote was not found in any packet chunk; treat as fabricated evidence until traced.")
                support_status = "proposed"
            elif located["status"] == "near_verbatim_in_chunk":
                # The evidence is real, only the transcription is noisy. It is
                # replaced below by the exact source passage.
                found_chunk_id = located["found_in"][0]
                if found_chunk_id != chunk_id:
                    ev_warnings.append(
                        f"Citation re-homed from {chunk_id or '(missing chunk_id)'} to {found_chunk_id}: the passage is there, not in the model-cited chunk."
                    )
                    chunk_id = found_chunk_id
                snap_in = chunk_id
            elif located["status"] == "in_cited_chunk":
                snap_in = chunk_id
                located_evidence_count += 1
            chunk = chunk_by_id.get(chunk_id)
            quote_out, quote_as_extracted, char_start, char_end, quote_match = quote, None, None, None, None
            if quote and model_quote != quote:
                quote_as_extracted = model_quote
            if quote:
                quote_match = "unverified"
            if snap_in is not None and quote and chunk is not None:
                snapped = _snap_quote_to_source(quote, str(chunk.get("text") or ""))
                if snapped is not None:
                    source_text = " ".join(snapped["text"].split())
                    char_start, char_end = snapped["char_start"], snapped["char_end"]
                    quote_match = "verbatim" if snapped["exact"] else "snapped"
                    if source_text != " ".join((model_quote or "").split()):
                        quote_as_extracted = model_quote
                    quote_out = source_text
                    if located["status"] == "near_verbatim_in_chunk":
                        ev_warnings.append(
                            f"Quote replaced by the exact source passage in {chunk_id} ({snapped['coverage']:.0%} in-order match); "
                            "the model's transcription differed and is kept as quote_as_extracted."
                        )
                        located_evidence_count += 1
                elif located["status"] == "near_verbatim_in_chunk":
                    ev_warnings.append("Evidence quote resembles text in the chunk but could not be located as one passage; requires review.")
                    support_status = "proposed"
            elif located["status"] == "near_verbatim_in_chunk":
                ev_warnings.append("Evidence quote resembles text in a chunk that is not in the source packet; requires review.")
                support_status = "proposed"
            evidence_records.append(
                PaperFactEvidence(
                    chunk_id=chunk_id,
                    quote=quote_out,
                    quote_as_extracted=quote_as_extracted,
                    char_start=char_start,
                    char_end=char_end,
                    quote_match=quote_match,
                    page=chunk.get("page_start") if chunk else None,
                    section=chunk.get("section") or chunk.get("chunk_type") if chunk else None,
                    support_status=support_status,
                    warnings=ev_warnings,
                    provenance=[ProvenanceRecord(source_type="source_chunk", source_id=chunk_id, method="paper_fact_evidence_citation")],
                )
            )
        # The extractor often cites the same passage several times, sometimes
        # truncated differently. Keep the longest of overlapping passages.
        unique_records: list[PaperFactEvidence] = []
        seen_unlocated: set[tuple[str, str]] = set()
        for record in evidence_records:
            if record.quote and record.char_start is not None and record.char_end is not None:
                duplicate_of = None
                for position, kept in enumerate(unique_records):
                    if kept.chunk_id != record.chunk_id or kept.char_start is None or kept.char_end is None:
                        continue
                    overlap = min(kept.char_end, record.char_end) - max(kept.char_start, record.char_start)
                    shorter = min(kept.char_end - kept.char_start, record.char_end - record.char_start)
                    if shorter > 0 and overlap >= 0.8 * shorter:
                        duplicate_of = position
                        break
                if duplicate_of is not None:
                    kept = unique_records[duplicate_of]
                    if record.char_end - record.char_start > kept.char_end - kept.char_start:
                        unique_records[duplicate_of] = record
                    continue
            elif record.quote:
                key = (record.chunk_id, _normalize_alnum(record.quote))
                if key in seen_unlocated:
                    continue
                seen_unlocated.add(key)
            unique_records.append(record)
        if len(unique_records) < len(evidence_records):
            warnings.append(f"Removed {len(evidence_records) - len(unique_records)} duplicate evidence quote(s).")
            evidence_records = unique_records
        if status == "supported" and not evidence_records:
            warnings.append("Supported fact has no cited evidence; downgraded to ambiguous.")
            status = "ambiguous"
        elif status == "supported" and evidence_records and not located_evidence_count:
            warnings.append("Downgraded from supported: no evidence quote could be located in any packet chunk.")
            status = "ambiguous"
        value = item.get("value")
        if not isinstance(value, dict):
            value = {"text": "" if value is None else str(value)}
        if isinstance(value.get("text"), str) and value["text"].strip().lower() in _NULL_STRINGS:
            # the model wrote the word "None"/"null" where a value belongs: no value
            value = {k: v for k, v in value.items() if k != "text"}
            if status == "supported":
                warnings.append("Supported fact with no value (the model wrote 'None'); downgraded to ambiguous.")
                status = "ambiguous"
        if isinstance(value.get("text"), str):
            # "(for YbIG)" written into the text is a subject, not part of the value
            m_subject = re.search(r"\s*\((?:for|of|in)\s+([^()]{1,60})\)\s*$", value["text"])
            if m_subject and not value.get("subject"):
                value = dict(value, text=value["text"][: m_subject.start()].rstrip(), subject=m_subject.group(1).strip())
                warnings.append("Subject moved out of the value text into value.subject.")
            plain = plain_text_value(value["text"])
            if plain != value["text"]:
                value = dict(value, text=plain, text_as_extracted=value["text"])
                warnings.append("Value written with LaTeX markup by the model; stored as plain text as the paper prints it.")
        discarded_value = None
        if status == "not_found" and any(str(entry or "").strip() for entry in value.values()):
            # A not_found answer that still carries a value is a hallucination
            # magnet (e.g. facility "BT-7" on a SEQUOIA paper, cited to the
            # reference list) - clear it so nothing downstream can link on it.
            discarded_value = value
            value = {}
            warnings.append("not_found fact carried a value; value cleared to prevent false linkage (original preserved in metadata.discarded_value).")
        source_chunk_ids = sorted({evidence.chunk_id for evidence in evidence_records if evidence.chunk_id})
        fact_id = f"paper-fact-{_hash(paper_id + fact_type + json.dumps(value, sort_keys=True) + str(index))}"
        malformed = malformed_value_reason(" ".join(str(v) for v in value.values()))
        if malformed:
            # The id above was computed from the value as delivered, so labels on
            # earlier runs still resolve; the value itself is quarantined.
            discarded_value = value
            value = {}
            status = "ambiguous"
            warnings.append(f"Malformed value ({malformed}): model output artifact, not paper content; value cleared (original in metadata.discarded_value).")
        fact = PaperFact(
            fact_id=fact_id,
            paper_id=paper_id,
            document_id=document_id,
            fact_type=fact_type,
            value=value,
            answer_status=status,
            confidence=_coerce_confidence(item.get("confidence")),
            evidence=evidence_records,
            source_chunk_ids=source_chunk_ids,
            extractor=f"bounded_{provider}",
            model=model,
            prompt_version=PROMPT_VERSION,
            human_review_required=status != "supported" or any(ev.warnings for ev in evidence_records),
            warnings=warnings + [str(warning) for warning in item.get("warnings") or []],
            metadata={"raw_fact_index": index, **({"discarded_value": discarded_value} if discarded_value else {}), **({"malformed_value": malformed} if malformed else {})},
            provenance=[ProvenanceRecord(source_type="paper", source_id=paper_id, method="bounded_paper_fact_extraction")],
        )
        facts.append(_sanitize_agentic_fact_semantics(fact.model_dump(mode="json")))
    return _mark_duplicate_facts(facts), errors


def _mark_duplicate_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The extractor repeats a fact for every chunk that mentions it (up to
    twelve copies on one paper). Later copies keep their ids but are marked
    duplicate_of the first, with their evidence merged into it, so counts and
    linkage see one fact and existing labels still resolve."""
    first_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for fact in facts:
        key = (str(fact.get("fact_type")), _value_key(fact.get("value")))
        if not key[1]:
            continue
        first = first_by_key.get(key)
        if first is None:
            first_by_key[key] = fact
            continue
        seen = {(e.get("chunk_id"), _normalize_alnum(str(e.get("quote") or ""))) for e in first.get("evidence") or []}
        for ev in fact.get("evidence") or []:
            if (ev.get("chunk_id"), _normalize_alnum(str(ev.get("quote") or ""))) not in seen:
                first.setdefault("evidence", []).append(ev)
                seen.add((ev.get("chunk_id"), _normalize_alnum(str(ev.get("quote") or ""))))
        first["source_chunk_ids"] = sorted({str(e.get("chunk_id")) for e in first.get("evidence") or [] if e.get("chunk_id")})
        meta = dict(first.get("metadata") or {}); meta["merged_duplicates"] = int(meta.get("merged_duplicates") or 0) + 1; first["metadata"] = meta
        dmeta = dict(fact.get("metadata") or {}); dmeta["duplicate_of"] = first.get("fact_id"); fact["metadata"] = dmeta
        fact["human_review_required"] = False
        fact["warnings"] = list(fact.get("warnings") or []) + [f"Duplicate of {first.get('fact_id')}: same fact type and value; evidence merged into the first occurrence."]
    return facts


def revalidate_paper_facts(catalog_dir: str | Path, *, in_place: bool = False, output_name: str = "paper_facts.revalidated.jsonl", fact_file: str = "paper_facts.jsonl") -> dict[str, Any]:
    """Re-run evidence location, status logic, semantic retyping, malformed-value
    quarantine and duplicate marking over facts already in a catalog, using the
    current code. Fact ids are preserved (they derive from the delivered value
    and index), so labels made against the old run still apply. The model's own
    free-text warnings are not preserved; machine warnings are regenerated."""
    catalog = Path(catalog_dir).expanduser()
    facts = _load_jsonl(catalog / fact_file)
    chunks = _load_jsonl(catalog / "source_chunks.jsonl")
    if not facts:
        return {"status": "missing_input", "errors": [f"{fact_file} is absent or empty in {catalog}"]}
    chunks_by_paper: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        chunks_by_paper.setdefault(str(chunk.get("paper_id")), []).append(chunk)
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        by_paper.setdefault(str(fact.get("paper_id")), []).append(fact)
    out: list[dict[str, Any]] = []
    changes = {"status_changed": 0, "retyped": 0, "malformed": 0, "duplicates": 0, "id_mismatch": 0}
    for paper_id, paper_facts in by_paper.items():
        paper_facts.sort(key=lambda f: int((f.get("metadata") or {}).get("raw_fact_index", 0)))
        max_index = max(int((f.get("metadata") or {}).get("raw_fact_index", 0)) for f in paper_facts)
        raw_items: list[dict[str, Any] | None] = [None] * (max_index + 1)
        for fact in paper_facts:
            meta = fact.get("metadata") or {}
            value = meta.get("discarded_value") or fact.get("value") or {}
            status = str(fact.get("answer_status") or "not_found")
            # Recover the model's own status where a machine rule downgraded it;
            # the rules are re-applied below with the current code.
            machine_downgrades = ("Downgraded from supported", "downgraded to ambiguous", "Malformed value", "Semantic sanity check")
            if status == "ambiguous" and any(m in str(w) for w in fact.get("warnings") or [] for m in machine_downgrades):
                status = "supported"
            raw_items[int(meta.get("raw_fact_index", 0))] = {
                "fact_type": meta.get("retyped_from") or fact.get("fact_type"), "answer_status": status, "value": value,
                "confidence": fact.get("confidence"), "evidence": [{"chunk_id": e.get("chunk_id"), "quote": e.get("quote"), "quote_as_extracted": e.get("quote_as_extracted")} for e in fact.get("evidence") or []], "warnings": [],
                "_orig": fact,
            }
        packet = {"paper_id": paper_id, "document_id": next((f.get("document_id") for f in paper_facts), None), "chunks": chunks_by_paper.get(paper_id, [])}
        raw_list = [item if item is not None else {"fact_type": "instrument", "answer_status": "not_found", "value": {}, "evidence": [], "_placeholder": True} for item in raw_items]
        provider = str(paper_facts[0].get("extractor") or "bounded_x").replace("bounded_", "", 1)
        new_facts, _errors = normalize_paper_fact_response({"facts": raw_list}, packet=packet, provider=provider, model=paper_facts[0].get("model"))
        new_by_index = {int((f.get("metadata") or {}).get("raw_fact_index", -1)): f for f in new_facts}
        for item in raw_items:
            if not item:
                continue
            orig = item["_orig"]; idx = int((orig.get("metadata") or {}).get("raw_fact_index", 0))
            new = new_by_index.get(idx)
            if new is None:
                out.append(orig); continue
            if new.get("fact_id") != orig.get("fact_id"):
                changes["id_mismatch"] += 1
                new["fact_id"] = orig.get("fact_id")  # keep the original id; note the mismatch in metadata
                new.setdefault("metadata", {})["fact_id_recomputed_differently"] = True
            for key in ("provenance", "extractor", "model", "prompt_version"):
                if orig.get(key) is not None:
                    new[key] = orig[key]
            for key in ("integrity_status", "integrity_reason"):
                if (orig.get("metadata") or {}).get(key):
                    new.setdefault("metadata", {})[key] = orig["metadata"][key]
            if new.get("answer_status") != orig.get("answer_status"):
                changes["status_changed"] += 1
                new.setdefault("metadata", {})["previous_answer_status"] = orig.get("answer_status")
            if (new.get("metadata") or {}).get("retyped_from"):
                changes["retyped"] += 1
            if (new.get("metadata") or {}).get("malformed_value"):
                changes["malformed"] += 1
            if (new.get("metadata") or {}).get("duplicate_of"):
                changes["duplicates"] += 1
            out.append(new)
    target = catalog / (fact_file if in_place else output_name)
    if in_place:
        backup = catalog / f"{Path(fact_file).stem}.pre_revalidate{Path(fact_file).suffix}"
        if not backup.exists():
            backup.write_text((catalog / fact_file).read_text(encoding="utf-8"), encoding="utf-8")
    _write_jsonl(target, out)
    return {"status": "ok", "catalog": str(catalog), "fact_count": len(out), "changes": changes, "output": str(target)}


def _paper_fact_packet(paper_id: str, chunks: list[dict[str, Any]], *, title: str | None, max_chars: int) -> dict[str, Any]:
    selected = _select_chunks_for_packet(chunks, max_chars=max_chars)
    document_id = next((chunk.get("document_id") for chunk in selected if chunk.get("document_id")), None)
    return {
        "paper_id": paper_id,
        "document_id": document_id,
        "title": title,
        "prompt_version": PROMPT_VERSION,
        "instructions": [
            "Extract only typed facts needed for paper-to-experiment and figure-to-data linkage.",
            "Do not infer facts from keywords alone. Use not_found or ambiguous when evidence is missing or noisy.",
            "Every supported answer must cite one or more chunk_id values and include a short quote from that chunk. A fact you cannot cite is not supported: report it as ambiguous with an empty evidence list, never as supported without evidence.",
            "Quotes are copied, not written: take the quote character for character from the chunk text, as one contiguous passage. The shortest exact clause that contains the value is enough (ten words will do). Do not paraphrase, shorten a sentence from the inside, reorder, or join two passages; if the evidence spans two sentences, cite them as two evidence entries.",
            "Report every distinct fact the paper states for every fact type, each as its own fact with its own quote: three instruments are three facts, each measurement temperature is a fact, each filter, each energy setting. Do not merge distinct facts into one list and do not skip a fact because its quote would be long; a short exact clause is always available.",
            "Allowed fact_type values: " + ", ".join(FACT_TYPES),
            "Allowed answer_status values: " + ", ".join(sorted(ANSWER_STATUSES)),
            "instrument = a named beamline or apparatus (BT-7, HB-1, SQUID magnetometer); facility = the laboratory or source (NCNR, HFIR, ORNL, SNS). Never put a facility under instrument.",
            "material = the compound studied (one fact per compound of this paper's own work); composition = its stoichiometry or doping as a formula or x value; sample_form = the physical form measured: single crystal, powder, polycrystal, thin film, co-aligned crystals (one fact per form, with the count if stated); sample = the specimen's own details beyond those: mass, dimensions, orientation, mosaic, mount. Do not put the form or the formula under sample.",
            "material_alias = an abbreviation the paper defines for a material, as 'ABBREVIATION = full name' (\"YLIG = (Yb1-xLux)3Ir4Ge13\", \"LSMO = La0.7Sr0.3MnO3\"), one fact per defined abbreviation, cited to the sentence that defines it; the material fact itself carries the formula without the abbreviation. Never invent an abbreviation.",
            "A substitution or doping series is one material fact naming the series (\"(Yb1-xLux)3Ir4Ge13 substitution series\") plus one composition fact per studied member (\"x = 0.1\", \"x = 0.5\", with the formula), each cited where that x is stated; a range of x (\"0 <= x <= 1\") is a composition fact only when the paper states it as a range.",
            "temperature_range = temperatures at which measurements were made or the sample was processed (annealing, cooling schedules), as the paper states them: keep a stated range whole (\"700 C to room temperature\", not \"room temperature\"), and attach what the temperature belongs to. transition_temperature = Neel, Curie, superconducting or structural transition temperatures (TN, Tc, TS). Do not mix them.",
            "propagation_vector = the magnetic ordering wavevector k or q, as a vector. A stated magnetic space group fixes it: a type-III group that keeps the full nuclear lattice (e.g. Fd'd'2, Pn'ma) means k = 0; report k = (0,0,0) and cite the space-group sentence. Scan positions such as Q = (2+xi, 2-xi, 0) are measurement positions, not propagation vectors (use source_data_clue).",
            "uncertainty_note = the paper's convention for error bars only (e.g. one standard deviation); a measured number with its error is not an uncertainty note.",
            "When a paper studies more than one material or sample, say which one a fact is about in value.subject (a formula or an alias the paper defines: 'YbIG'), never inside value.text: write {\"text\": \"300 K to 0.1 K\", \"subject\": \"YbIG\"}, not \"300 K to 0.1 K (for YbIG)\". Facts about the whole paper (instrument, facility) need no subject.",
            "The value must contain only what the paper states: no reasoning, no commentary, no JSON, no chunk ids, no repeated words. Write it in plain text as the paper prints it (α-Cu2V2O7, 2.4 Å, 5 K), never in LaTeX or other markup, and without reference markers such as [10, 11] or superscript citation numbers. Report each distinct fact once; do not repeat a fact for every chunk that mentions it.",
        ],
        "output_schema": {
            "facts": [
                {
                    "fact_type": "instrument",
                    "answer_status": "supported|not_found|ambiguous|unsupported",
                    "value": {"text": "BT-7", "subject": "material, sample or defined alias this fact is about, when the paper studies more than one; omit otherwise"},
                    "confidence": 0.0,
                    "evidence": [{"chunk_id": "chunk-id", "quote": "short exact quote"}],
                    "warnings": [],
                }
            ]
        },
        "questions": [
            "What instrument and facility were used?",
            "What samples, materials, compositions, crystal structures, or sample preparation details are stated?",
            "What measurement kind, detector mode, instrument configuration, monochromator, analyzer, collimation, filter, and energy setting are stated?",
            "What temperature range, magnetic field, pressure, and sample environment are stated?",
            "What data reduction, calibration, background, normalization, or resolution information is stated?",
            "Which figure captions, figures, or panels are linked to measurements, samples, or source-data clues?",
            "What important information is ambiguous or not found?",
        ],
        "chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_type": chunk.get("chunk_type"),
                "title": chunk.get("title"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "section": chunk.get("section"),
                "text": str(chunk.get("text") or "")[:5000],
            }
            for chunk in selected
        ],
    }


def _select_chunks_for_packet(chunks: list[dict[str, Any]], *, max_chars: int) -> list[dict[str, Any]]:
    priority = {
        "title": 0,
        "abstract": 1,
        "experimental": 2,
        "methods": 2,
        "instrument_setup": 3,
        "sample_description": 4,
        "material_composition": 4,
        "data_reduction": 5,
        "structure_parameters": 6,
        "figure_caption": 7,
        "table": 8,
    }
    ordered = sorted(chunks, key=lambda chunk: (priority.get(str(chunk.get("chunk_type")), 20), str(chunk.get("chunk_id"))))
    selected: list[dict[str, Any]] = []
    used = 0
    for chunk in ordered:
        text = str(chunk.get("text") or "")
        if used and used + len(text) > max_chars:
            continue
        selected.append(chunk)
        used += len(text)
        if used >= max_chars:
            break
    return selected


def _call_ollama(packet: dict[str, Any], *, base_url: str, model: str) -> dict[str, Any]:
    schema = _paper_fact_response_schema(max_items=60)
    body = ollama_chat(
        base_url,
        model,
        [
            {
                "role": "system",
                "content": (
                    "You extract bounded scientific paper facts for TAS Co-pilot. "
                    "Return only JSON. Do not create a graph. Do not infer missing facts. "
                    "Cite chunk_id and exact short quotes for every supported fact."
                ),
            },
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
        ],
        format_schema=schema,
        think=False,
        options={"temperature": 0, "num_predict": 8192},
        timeout=300.0,
    )
    content = str((body.get("message") or {}).get("content") or "{}")
    parsed, parse_error = _parse_model_json(content)
    if parsed is not None:
        return parsed
    repaired = _repair_ollama_json(
        content,
        parse_error=parse_error,
        base_url=base_url,
        model=model,
        schema=schema,
    )
    repaired.setdefault("warnings", []).append("Initial model JSON was repaired before validation.")
    return repaired


def _paper_fact_response_schema(*, max_items: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "maxItems": max_items,
                "items": {
                    "type": "object",
                    "properties": {
                        "fact_type": {"type": "string", "enum": FACT_TYPES},
                        "answer_status": {"type": "string", "enum": sorted(ANSWER_STATUSES)},
                        "value": {"type": "object"},
                        "confidence": {"type": "number"},
                        "evidence": {
                            "type": "array",
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "chunk_id": {"type": "string"},
                                    "quote": {"type": "string"},
                                },
                                "required": ["chunk_id"],
                            },
                        },
                        "warnings": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["fact_type", "answer_status", "value"],
                },
            },
            "errors": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["facts"],
    }


def _parse_model_json(content: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse model JSON while tolerating common wrapper text.

    This does not repair scientific content. It only accepts a valid JSON object
    already present in the model response, such as a fenced JSON block or a
    larger response with explanatory text around the object.
    """

    candidates = [content.strip()]
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidates.append("\n".join(lines).strip())
    decoder = json.JSONDecoder()
    for start in [0] + [index for index, char in enumerate(content) if char == "{"]:
        try:
            obj, _end = decoder.raw_decode(content[start:])
        except json.JSONDecodeError:
            continue
        if _is_paper_fact_response(obj):
            return obj, None
    last_error: str | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if _is_paper_fact_response(obj):
            return obj, None
        last_error = "Model output JSON is not an object."
    # Lenient pass: models routinely emit literal newlines inside quoted
    # values, which strict JSON rejects outright. The content is otherwise fine.
    lenient = json.JSONDecoder(strict=False)
    for start in [0] + [index for index, char in enumerate(content) if char == "{"]:
        try:
            obj, _end = lenient.raw_decode(content, start)
        except json.JSONDecodeError:
            continue
        if _is_paper_fact_response(obj):
            return obj, None
    return None, last_error or "Model output was not valid JSON."


def _is_paper_fact_response(obj: Any) -> bool:
    return isinstance(obj, dict) and ("facts" in obj or "errors" in obj)


def _salvage_facts_from_partial(content: str) -> list[dict[str, Any]]:
    """Recover the complete fact objects from a malformed facts array.

    A single bad string (an unescaped quote, a raw newline) makes the whole
    response unparseable and costs the paper every fact it contained - 9 of 48
    papers in the 2026-07-14 run returned nothing this way. The objects before
    the damage are still intact, so pull them out rather than discarding all of
    them. Lenient decoding (strict=False) additionally tolerates literal control
    characters inside strings, which is the usual cause.
    """
    marker = content.find('"facts"')
    if marker < 0:
        return []
    start = content.find("[", marker)
    if start < 0:
        return []
    decoder = json.JSONDecoder(strict=False)
    facts: list[dict[str, Any]] = []
    index = start + 1
    length = len(content)
    while index < length:
        while index < length and content[index] in " \t\r\n,":
            index += 1
        if index >= length or content[index] != "{":
            break
        try:
            obj, end = decoder.raw_decode(content, index)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and obj.get("fact_type"):
            facts.append(obj)
        index = end
    return facts


def _repair_ollama_json(
    content: str,
    *,
    parse_error: str | None,
    base_url: str,
    model: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    body = ollama_chat(
        base_url,
        model,
        [
            {
                "role": "system",
                "content": (
                    "Repair malformed JSON from a previous bounded paper-fact extraction. "
                    "Return only valid JSON matching the schema. Do not add new facts. "
                    "If a fact cannot be repaired, omit it and include an errors entry."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "parse_error": parse_error,
                        "malformed_response": content[:24000],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        format_schema=schema,
        think=False,
        options={"temperature": 0, "num_predict": 8192},
        timeout=300.0,
    )
    repaired_content = str((body.get("message") or {}).get("content") or "{}")
    repaired, repair_error = _parse_model_json(repaired_content)
    if repaired is not None:
        warnings = [str(warning) for warning in repaired.get("warnings") or []]
        if parse_error:
            warnings.append(f"initial_json_parse_error_recovered: {parse_error}")
        repaired["warnings"] = warnings
        return repaired
    salvaged = _salvage_facts_from_partial(content)
    if salvaged:
        # Stamp the provenance on each fact, not just the response: a top-level
        # warning is dropped by normalization, which would leave salvaged facts
        # looking like ordinary clean extractions to a human reviewer.
        note = (
            f"salvaged_from_malformed_response: {len(salvaged)} complete fact objects were recovered after the "
            "response failed both parsing and repair; the damaged remainder was discarded."
        )
        for fact in salvaged:
            fact["warnings"] = [str(w) for w in (fact.get("warnings") or [])] + [note]
        return {
            "facts": salvaged,
            "errors": [
                f"initial_json_parse_error: {parse_error}",
                f"repair_json_parse_error: {repair_error}",
            ],
            "warnings": [note],
        }
    return {
        "facts": [],
        "errors": [
            f"initial_json_parse_error: {parse_error}",
            f"repair_json_parse_error: {repair_error}",
        ],
    }


def _load_model_output(path: Path, paper_id: str) -> dict[str, Any]:
    if not path.exists():
        return {"facts": [], "errors": [f"model output path missing: {path}"]}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "by_paper" in data:
        return data.get("by_paper", {}).get(paper_id, {"facts": []})
    return data if isinstance(data, dict) else {"facts": []}


def _paper_fact_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper Fact Extraction",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Provider: `{report.get('provider')}`",
        f"- Papers: {report.get('paper_count')}",
        f"- Facts: {report.get('fact_count')}",
        "",
        "## Policy",
        "",
        "- Keyword chunks are evidence hints, not authoritative paper facts.",
        "- Supported facts require cited chunks and short quotes.",
        "- `not_found` and `ambiguous` are valid extraction results.",
        "",
    ]
    if report.get("errors"):
        lines.extend(["## Errors", ""])
        lines.extend(f"- {item.get('paper_id')}: {item.get('error')}" for item in report["errors"])
        lines.append("")
    if report.get("facts"):
        lines.extend(["## Facts", ""])
        for fact in report["facts"]:
            evidence = ", ".join(fact.get("source_chunk_ids") or []) or "none"
            lines.append(f"- `{fact.get('fact_type')}` `{fact.get('answer_status')}` confidence={fact.get('confidence')} chunks={evidence}")
    return "\n".join(lines) + "\n"


def _paper_id_filter(paper_id: str | None, paper_ids_file: str | None) -> set[str]:
    paper_ids: set[str] = set()
    if paper_id:
        paper_ids.add(str(paper_id))
    if paper_ids_file:
        path = Path(paper_ids_file).expanduser()
        if path.exists():
            paper_ids.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return paper_ids


def _fact_for_review(fact: dict[str, Any], *, chunk_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence = []
    for item in fact.get("evidence") or []:
        chunk = chunk_by_id.get(str(item.get("chunk_id")))
        evidence.append(
            {
                "chunk_id": item.get("chunk_id"),
                "quote": item.get("quote"),
                "support_status": item.get("support_status"),
                "warnings": item.get("warnings") or [],
                "chunk_type": chunk.get("chunk_type") if chunk else None,
                "page": chunk.get("page_start") if chunk else item.get("page"),
                "chunk_excerpt": str(chunk.get("text") or "")[:700] if chunk else None,
            }
        )
    return {
        "fact_id": fact.get("fact_id"),
        "fact_type": fact.get("fact_type"),
        "answer_status": fact.get("answer_status"),
        "confidence": fact.get("confidence"),
        "value": fact.get("value"),
        "human_review_required": fact.get("human_review_required"),
        "warnings": fact.get("warnings") or [],
        "evidence": evidence,
    }


def _paper_fact_review_html(report: dict[str, Any]) -> str:
    rows = []
    for record in report.get("records") or []:
        paper_id = str(record.get("paper_id"))
        facts = []
        for fact in record.get("facts") or []:
            value = fact.get("value")
            value_text = value.get("text") if isinstance(value, dict) else value
            fact_id = str(fact.get("fact_id"))
            evidence_lines = []
            for evidence in fact.get("evidence") or []:
                quote = evidence.get("quote") or ""
                where = ", ".join(
                    part
                    for part in [
                        str(evidence.get("chunk_id") or ""),
                        str(evidence.get("chunk_type") or ""),
                        f"page {evidence.get('page')}" if evidence.get("page") is not None else "",
                    ]
                    if part
                )
                excerpt = evidence.get("chunk_excerpt")
                excerpt_html = (
                    f"<details><summary>chunk text</summary><blockquote class='chunk'>{html.escape(str(excerpt))}</blockquote></details>"
                    if excerpt
                    else ""
                )
                evidence_lines.append(
                    f"<li><code>{html.escape(where)}</code><br><blockquote>{html.escape(str(quote))}</blockquote>{excerpt_html}</li>"
                )
            verdict_select = (
                f"<select class='verdict' data-fact-id='{html.escape(fact_id)}' data-paper-id='{html.escape(paper_id)}'>"
                "<option value=''>unreviewed</option>"
                "<option value='correct'>correct</option>"
                "<option value='wrong_field'>wrong field</option>"
                "<option value='too_vague'>too vague</option>"
                "<option value='missing_important_fact'>missing important fact</option>"
                "<option value='ocr_chunking_problem'>OCR/chunking problem</option>"
                "<option value='not_relevant_to_linkage'>not relevant</option>"
                "</select>"
                f"<input class='notes' data-fact-id='{html.escape(fact_id)}' placeholder='notes'>"
            )
            facts.append(
                "<tr>"
                f"<td><code>{html.escape(str(fact.get('fact_type')))}</code></td>"
                f"<td>{html.escape(str(fact.get('answer_status')))}</td>"
                f"<td>{html.escape(str(fact.get('confidence')))}</td>"
                f"<td>{html.escape(str(value_text or ''))}</td>"
                f"<td><ul>{''.join(evidence_lines)}</ul></td>"
                f"<td>{verdict_select}</td>"
                "</tr>"
            )
        rows.append(
            "<section>"
            f"<h2>{html.escape(str(record.get('title') or record.get('paper_id')))}</h2>"
            f"<p><code>{html.escape(paper_id)}</code> - {record.get('fact_count')} facts</p>"
            "<table><thead><tr><th>Type</th><th>Status</th><th>Confidence</th><th>Value</th><th>Evidence</th><th>Review</th></tr></thead>"
            f"<tbody>{''.join(facts)}</tbody></table>"
            "</section>"
        )
    instructions = "".join(f"<li>{html.escape(str(item))}</li>" for item in report.get("review_instructions") or [])
    catalog = str(report.get("catalog"))
    script = """
<script>
const STORE_KEY = 'paper-fact-review:' + document.querySelector('p code').textContent;
function loadState() { try { return JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); } catch (e) { return {}; } }
function saveState(state) { localStorage.setItem(STORE_KEY, JSON.stringify(state)); updateProgress(); }
function updateProgress() {
  const selects = [...document.querySelectorAll('select.verdict')];
  const done = selects.filter(s => s.value).length;
  document.getElementById('progress').textContent = done + ' / ' + selects.length + ' reviewed';
}
document.addEventListener('DOMContentLoaded', () => {
  const state = loadState();
  document.querySelectorAll('select.verdict').forEach(select => {
    const saved = state[select.dataset.factId];
    if (saved && saved.verdict) select.value = saved.verdict;
    select.addEventListener('change', () => {
      const s = loadState();
      s[select.dataset.factId] = Object.assign(s[select.dataset.factId] || {}, {verdict: select.value, paperId: select.dataset.paperId});
      saveState(s);
    });
  });
  document.querySelectorAll('input.notes').forEach(input => {
    const saved = state[input.dataset.factId];
    if (saved && saved.notes) input.value = saved.notes;
    input.addEventListener('change', () => {
      const s = loadState();
      s[input.dataset.factId] = Object.assign(s[input.dataset.factId] || {}, {notes: input.value});
      saveState(s);
    });
  });
  updateProgress();
  document.getElementById('export-labels').addEventListener('click', () => {
    const verdictMap = {correct: 'positive', wrong_field: 'negative', too_vague: 'negative', ocr_chunking_problem: 'negative', not_relevant_to_linkage: 'negative', missing_important_fact: 'inconclusive'};
    const state = loadState();
    const lines = [];
    document.querySelectorAll('select.verdict').forEach(select => {
      const chosen = select.value;
      if (!chosen) return;
      const factId = select.dataset.factId;
      const notes = (state[factId] || {}).notes || '';
      lines.push(JSON.stringify({
        label_id: 'review-fact-' + factId,
        object_type: 'paper_fact',
        object_id: factId,
        object_secondary_id: select.dataset.paperId,
        source_workflow: 'paper-fact-review',
        source_artifact_path: window.location.pathname,
        verdict: verdictMap[chosen] || 'inconclusive',
        type_specific_status: chosen,
        review_notes: notes,
      }));
    });
    const blob = new Blob([lines.join('\n') + '\n'], {type: 'application/jsonl'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'paper_fact_review_labels.jsonl';
    link.click();
  });
});
</script>"""
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Paper Fact Review</title>"
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;line-height:1.35}"
        "table{border-collapse:collapse;width:100%;margin:12px 0 32px}th,td{border:1px solid #ccc;padding:6px;vertical-align:top}"
        "th{background:#f4f4f4}blockquote{margin:4px 0;padding-left:8px;border-left:3px solid #7c9;color:#333}"
        "blockquote.chunk{border-left-color:#bbb;color:#555;font-size:0.92em}"
        "code{background:#f7f7f7;padding:1px 3px}select.verdict{margin-bottom:4px;display:block}"
        "input.notes{width:100%;box-sizing:border-box}"
        ".toolbar{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:2px solid #eee;z-index:5}"
        "button{padding:6px 14px;font-size:1em}</style></head><body>"
        "<h1>Paper Fact Review</h1>"
        f"<p>Catalog: <code>{html.escape(catalog)}</code></p>"
        f"<div class='toolbar'><span id='progress'></span> &nbsp; <button id='export-labels'>Export labels JSONL</button></div>"
        f"<p>Papers: {report.get('paper_count')} &nbsp; Facts: {report.get('fact_count')}</p>"
        "<h2>How to review</h2><ul>"
        f"{instructions}</ul>"
        f"{''.join(rows)}"
        f"{script}"
        "</body></html>"
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _dedupe(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get(key):
            out[str(row[key])] = row
    return list(out.values())


def _missing_report(output_dir: Path, reason: str) -> dict[str, Any]:
    report = {"status": "missing_input", "fact_count": 0, "errors": [{"error": reason}], "facts": []}
    report["artifacts"] = write_report_bundle(
        output_dir,
        report,
        markdown_renderer=_paper_fact_markdown,
        json_filename="paper_fact_extraction_report.json",
        markdown_filename="paper_fact_extraction_report.md",
    )
    return report


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


_MIN_FRAGMENT_ALNUM_CHARS = 15
# Near-verbatim thresholds. The length floor matters: short strings are found
# "in order" inside almost any text by chance, so coverage alone would wave
# through genuinely fabricated short quotes.
_NEAR_VERBATIM_MIN_COVERAGE = 0.95
_NEAR_VERBATIM_MIN_ALNUM_CHARS = 40


# A near-verbatim match must also be compact: difflib will happily find a
# quote's characters "in order" scattered across a whole chunk, which is not a
# quotation of anything. The matched span may be somewhat longer than the quote
# (the model drops clauses) but not much.
_NEAR_VERBATIM_MAX_SPAN_RATIO = 1.6
_NEAR_VERBATIM_MIN_COMPACT_COVERAGE = 0.85
_MIN_ALIGNMENT_BLOCK = 3
# A shortened quote: the model copied a long passage (a caption, a paragraph)
# but left sentences out without marking the gaps. Its pieces are long exact
# runs of the chunk, in order, covering most of the quote; the span is too wide
# for one passage. Runs shorter than this are not evidence of anything.
_MIN_ELIDED_RUN_ALNUM_CHARS = 20
_ELIDED_MIN_COVERAGE = 0.85


def _compact_alignment(normalized_quote: str, normalized_text: str) -> tuple[int, int, float] | None:
    """Locate the quote as one compact passage of the (normalized) text.

    Returns (start, end, coverage) in normalized-text offsets, or None when the
    quote only matches as scattered pieces. An exact occurrence has coverage 1.
    """
    if not normalized_quote or not normalized_text:
        return None
    position = normalized_text.find(normalized_quote)
    if position >= 0:
        return position, position + len(normalized_quote), 1.0
    matcher = difflib.SequenceMatcher(None, normalized_quote, normalized_text, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= _MIN_ALIGNMENT_BLOCK]
    limit = _NEAR_VERBATIM_MAX_SPAN_RATIO * len(normalized_quote)
    # Trim stray edge blocks (appended debris, a repeated phrase far away) until
    # what remains is one passage. Drop the edge that sits farther from its
    # neighbour; on a tie drop the smaller block.
    while len(blocks) > 1 and (blocks[-1].b + blocks[-1].size - blocks[0].b) > limit:
        head_gap = blocks[1].b - (blocks[0].b + blocks[0].size)
        tail_gap = blocks[-1].b - (blocks[-2].b + blocks[-2].size)
        if (head_gap, -blocks[0].size) > (tail_gap, -blocks[-1].size):
            blocks = blocks[1:]
        else:
            blocks = blocks[:-1]
    if not blocks:
        return None
    start, end = blocks[0].b, blocks[-1].b + blocks[-1].size
    coverage = sum(block.size for block in blocks) / len(normalized_quote)
    if end - start > limit or coverage < _NEAR_VERBATIM_MIN_COMPACT_COVERAGE:
        return None
    return start, end, coverage


def _nearest_chunk_by_coverage(
    normalized_quote: str,
    norm_chunk_texts: dict[str, str],
) -> tuple[str, float, int, int] | None:
    """Best chunk containing the quote as one compact passage, ignoring stray
    characters. Returns (chunk_id, coverage, start, end) with offsets into that
    chunk's normalized text, or None when no chunk holds such a passage."""
    if len(normalized_quote) < _NEAR_VERBATIM_MIN_ALNUM_CHARS:
        return None
    best: tuple[str, float, int, int] | None = None
    for chunk_id, text in norm_chunk_texts.items():
        if not text:
            continue
        aligned = _compact_alignment(normalized_quote, text)
        if aligned is None:
            continue
        start, end, coverage = aligned
        if best is None or coverage > best[1]:
            best = (chunk_id, coverage, start, end)
    return best


def _alnum_offsets(text: str) -> tuple[str, list[int]]:
    """_normalize_alnum(text) plus, for each kept character, its index in text."""
    kept: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text.lower()):
        if ("a" <= char <= "z") or ("0" <= char <= "9"):
            kept.append(char)
            offsets.append(index)
    return "".join(kept), offsets


def _elided_runs(normalized_quote: str, normalized_text: str) -> list[tuple[int, int]]:
    """The in-order exact runs of normalized_text that a shortened quote was
    copied from, as (start, end) offsets into normalized_text; [] unless the
    runs are long, ordered, and together cover most of the quote."""
    if len(normalized_quote) < _NEAR_VERBATIM_MIN_ALNUM_CHARS or not normalized_text:
        return []
    matcher = difflib.SequenceMatcher(None, normalized_quote, normalized_text, autojunk=False)
    runs: list[tuple[int, int]] = []
    covered = 0
    for block in matcher.get_matching_blocks():
        if block.size < _MIN_ELIDED_RUN_ALNUM_CHARS:
            continue
        start, end = block.b, block.b + block.size
        if runs and start - runs[-1][1] <= 8:  # same passage split by a stray character
            runs[-1] = (runs[-1][0], end)
        else:
            runs.append((start, end))
        covered += block.size
    if not runs or covered / len(normalized_quote) < _ELIDED_MIN_COVERAGE:
        return []
    return runs


def _complete_span(chunk_text: str, offsets: list[int], start: int, end: int) -> tuple[int, int]:
    """Raw-text span for normalized offsets [start, end), widened to whole
    tokens and glued punctuation (the source says "[47]" and "400")."""
    char_start, char_end = offsets[start], offsets[end - 1] + 1
    while char_start > 0 and chunk_text[char_start - 1].isalnum():
        char_start -= 1
    while char_end < len(chunk_text) and chunk_text[char_end].isalnum():
        char_end += 1
    while char_start > 0 and not chunk_text[char_start - 1].isspace() and not chunk_text[char_start - 1].isalnum():
        char_start -= 1
    while char_end < len(chunk_text) and not chunk_text[char_end].isspace() and not chunk_text[char_end].isalnum():
        char_end += 1
    return char_start, char_end


def _snap_quote_to_source(quote: str, chunk_text: str) -> dict[str, Any] | None:
    """The exact passage of chunk_text that the quote transcribes.

    The model points at evidence; it is not trusted to retype it. Returns
    {"text", "char_start", "char_end", "coverage", "exact"} or None when the
    quote is not one compact passage of this chunk."""
    normalized_quote = _normalize_alnum(quote)
    normalized_text, offsets = _alnum_offsets(chunk_text)
    if not normalized_quote:
        return None
    exact_at = normalized_text.find(normalized_quote)
    if exact_at < 0 and len(normalized_quote) < _MIN_FRAGMENT_ALNUM_CHARS:
        # Too short to align fuzzily ("around TN = 5.7 K" is 11 characters); an
        # exact occurrence above is still a located quote, whatever its length.
        return None
    aligned = (exact_at, exact_at + len(normalized_quote), 1.0) if exact_at >= 0 else _compact_alignment(normalized_quote, normalized_text)
    if aligned is None:
        return None
    start, end, coverage = aligned
    exact = normalized_text[start:end] == normalized_quote
    if not exact and len(normalized_quote) < _NEAR_VERBATIM_MIN_ALNUM_CHARS:
        return None
    char_start, char_end = _complete_span(chunk_text, offsets, start, end)
    return {
        "text": chunk_text[char_start:char_end],
        "char_start": char_start,
        "char_end": char_end,
        "coverage": round(coverage, 3),
        "exact": exact,
    }


def _split_stitched_quote(
    quote: str,
    *,
    cited_chunk_id: str,
    fact_paper_id: str,
    chunk_by_id: dict[str, dict[str, Any]],
    norm_chunk_texts: dict[str, str],
    paper_by_chunk: dict[str, str],
) -> list[tuple[str, dict[str, Any]]]:
    """Resolve an ellipsis-joined quote into its real passages.

    Returns [(chunk_id, snapped passage)] in quote order, or [] when any
    substantial fragment cannot be placed in this paper's chunks."""
    pieces: list[tuple[str, dict[str, Any]]] = []
    for fragment in (part.strip() for part in quote.replace("\u2026", "...").split("...")):
        normalized = _normalize_alnum(fragment)
        if len(normalized) < _MIN_FRAGMENT_ALNUM_CHARS:
            continue
        homes = sorted(
            cid for cid, text in norm_chunk_texts.items()
            if normalized in text and (not fact_paper_id or paper_by_chunk.get(cid) == fact_paper_id)
        )
        if not homes:
            return []
        home = cited_chunk_id if cited_chunk_id in homes else homes[0]
        snapped = _snap_quote_to_source(fragment, str((chunk_by_id.get(home) or {}).get("text") or ""))
        if snapped is None:
            return []
        pieces.append((home, snapped))
    return pieces if len(pieces) > 1 else []


# Extraction schema that leaks into a quote ("...BT-7 [4type: instrument,
# answer_status: supported, value: {text:"). Everything from the leak onward is
# model output, not paper text.
_NULL_STRINGS = {"", "none", "null", "nan", "n/a", ","}
_LATEX_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε", "theta": "θ", "Theta": "Θ", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "Sigma": "Σ", "tau": "τ", "phi": "φ",
    "Phi": "Φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Omega": "Ω", "Gamma": "Γ", "eta": "η", "zeta": "ζ",
}
_LATEX_SYMBOLS = {r"\AA": "Å", r"\angstrom": "Å", r"\circ": "°", r"\degree": "°", r"\pm": "±", r"\approx": "≈", r"\sim": "~",
                  r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥", r"\times": "×", r"\cdot": "·", r"\parallel": "∥", r"\perp": "⊥",
                  r"\mu_B": "μB", r"\mu_\mathrm{B}": "μB", r"\,": " ", r"\;": " ", r"\!": "", r"\ ": " "}


def plain_text_value(text: str) -> str:
    """The model sometimes writes values in LaTeX ("$\\alpha$-Cu$_2$V$_2$O$_7$")
    where the paper prints plain text ("α-Cu2V2O7"). Values are stored as the
    paper prints them: math delimiters dropped, Greek letters and common
    symbols as Unicode, sub- and superscripts flattened. Unknown commands are
    left alone rather than guessed."""
    t = str(text or "")
    if "\\" not in t and "$" not in t and "_{" not in t and "^{" not in t:
        return t
    t = t.replace("$", "")
    for cmd, sym in _LATEX_SYMBOLS.items():
        t = t.replace(cmd, sym)
    t = re.sub(r"\\(mathrm|mathit|textrm|text|mathbf|rm|bf|it)\{([^{}]*)\}", r"\2", t)
    t = re.sub(r"\\(" + "|".join(sorted(_LATEX_GREEK, key=len, reverse=True)) + r")\b", lambda m: _LATEX_GREEK[m.group(1)], t)
    t = re.sub(r"[_^]\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"[_^]([A-Za-z0-9])", r"\1", t)
    t = t.replace("{", "").replace("}", "")
    return " ".join(t.split())
_SCHEMA_LEAK_RE = re.compile(
    r"(?:\"?(?:fact_)?type\"?\s*:\s*\"?\w+\"?\s*,\s*\"?answer_status\"?\s*:|\"?answer_status\"?\s*:\s*\"?(?:supported|ambiguous|not_found|unsupported)\b)"
)


def _strip_schema_leak(quote: str) -> tuple[str, bool]:
    match = _SCHEMA_LEAK_RE.search(quote)
    if not match:
        return quote, False
    return quote[: match.start()].rstrip(" ,;:[({\"'"), True


def _normalize_alnum(text: str) -> str:
    """Aggressive normalization for locating quotes: alphanumerics only.

    Tolerates OCR glyph variance (e.g. '1/4' rendered for '=', spaced-out
    formulas) that defeats whitespace-level comparison."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _locate_quote(
    quote: str,
    *,
    cited_chunk_id: str,
    fact_paper_id: str,
    norm_chunk_texts: dict[str, str],
    paper_by_chunk: dict[str, str],
) -> dict[str, Any]:
    """Where does this evidence quote actually occur?

    Statuses: in_cited_chunk, in_other_chunk, in_other_paper,
    stitched_fragments, not_found, no_quote."""
    if not quote or not quote.strip():
        return {"status": "no_quote", "found_in": [], "note": ""}
    normalized = _normalize_alnum(quote)
    if not normalized:
        return {"status": "no_quote", "found_in": [], "note": ""}
    # Fast path: most quotes sit in the chunk they cite. Checking that one chunk
    # first keeps the whole-corpus scan for the minority that need it (on the
    # 8.5k-fact full corpus the unconditional scan took hours).
    if normalized in (norm_chunk_texts.get(cited_chunk_id) or ""):
        return {"status": "in_cited_chunk", "found_in": [cited_chunk_id], "note": ""}
    homes = sorted(cid for cid, text in norm_chunk_texts.items() if normalized in text)
    if cited_chunk_id in homes:
        return {"status": "in_cited_chunk", "found_in": [cited_chunk_id], "note": ""}
    if homes:
        other_paper = [cid for cid in homes if fact_paper_id and paper_by_chunk.get(cid) != fact_paper_id]
        if other_paper and len(other_paper) == len(homes):
            return {
                "status": "in_other_paper",
                "found_in": homes,
                "note": "Quote is verbatim in another paper's chunk, not this paper's - cross-paper leak (model memory or packet contamination).",
            }
        return {
            "status": "in_other_chunk",
            "found_in": homes,
            "note": "Quote is verbatim in a different chunk of this paper - chunk-attribution error; the text itself is real.",
        }
    fragments = [frag.strip() for frag in quote.replace("\u2026", "...").split("...") if frag.strip()]
    if len(fragments) > 1:
        fragment_homes: list[list[str]] = []
        for fragment in fragments:
            norm_fragment = _normalize_alnum(fragment)
            if len(norm_fragment) < _MIN_FRAGMENT_ALNUM_CHARS:
                continue
            fragment_homes.append(sorted(cid for cid, text in norm_chunk_texts.items() if norm_fragment in text))
        if fragment_homes and all(fragment_homes):
            found = sorted({cid for homes_ in fragment_homes for cid in homes_})
            return {
                "status": "stitched_fragments",
                "found_in": found,
                "note": "Quote is stitched from real fragments (see chunks listed) but is not contiguous text anywhere.",
            }
    # Near-verbatim: the quote is the chunk's text with a few stray characters
    # (OCR debris, a duplicated word, a dropped space). Calling that "fabricated"
    # is wrong and expensive: on the 48-paper set it forced a reviewer to mark
    # substantively correct facts as OCR/chunking failures, which would have
    # reported the model far below its real accuracy.
    # Near-verbatim search is a difflib alignment per candidate chunk, so it is
    # limited to this paper's chunks (plus the cited chunk); a near-verbatim
    # match in another paper would be a cross-paper leak, not corrupted
    # transcription, and scanning every chunk in the catalog is unaffordable.
    candidate_ids = {cited_chunk_id} | {cid for cid, pid in paper_by_chunk.items() if fact_paper_id and pid == fact_paper_id}
    candidates = {cid: norm_chunk_texts[cid] for cid in candidate_ids if cid in norm_chunk_texts} or norm_chunk_texts
    near = _nearest_chunk_by_coverage(normalized, candidates)
    if near is not None:
        chunk_id, coverage, _start, _end = near
        return {
            "status": "near_verbatim_in_chunk",
            "found_in": [chunk_id],
            "coverage": round(coverage, 3),
            "note": (
                f"Quote is {coverage:.0%} contained, in order, in {chunk_id} but not character-exact - "
                "minor quote corruption (stray or dropped characters), not fabricated evidence."
            ),
        }
    for chunk_id_, text in sorted(candidates.items(), key=lambda kv: kv[0] != cited_chunk_id):
        runs = _elided_runs(normalized, text)
        if runs:
            covered = sum(e - b for b, e in runs) / len(normalized)
            return {
                "status": "elided_passages",
                "found_in": [chunk_id_],
                "runs": runs,
                "coverage": round(covered, 3),
                "note": (
                    f"Quote is a shortened copy of {chunk_id_}: {len(runs)} passage(s) are there verbatim, in order, "
                    f"covering {covered:.0%} of the quote, with text between them left out."
                ),
            }
    return {
        "status": "not_found",
        "found_in": [],
        "note": "Quote is not found in any catalog chunk - treat as fabricated evidence until traced.",
    }


def _coerce_confidence(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, number))
