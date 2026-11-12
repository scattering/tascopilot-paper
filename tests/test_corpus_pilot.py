from __future__ import annotations

import json
from pathlib import Path

from tas_copilot_contracts import DataFile, EvidenceGraphEdge, EvidenceGraphNode, Experiment, PaperFact, ReviewLabel, Scan, SourceChunk
from tas_copilot_core.catalogs.jsonl_catalog import JsonlCatalog
from tas_copilot_core.catalogs.review_outcomes_sqlite import ReviewOutcomeSQLiteStore
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.corpus.pilot import (
    agent_readiness,
    build_paper_experiment_review_dashboard,
    build_bt7_hydrated_index,
    corpus_pilot_report,
    corpus_search,
    extract_paper_constraints,
    figure_brief,
    ingest_extractor_run,
    ingest_paper_experiment_review_labels,
    ingest_pilot_corpus,
    link_paper_to_experiments,
    ontology_diagnostics,
    paper_brief,
    paper_candidate_shortlist,
    publication_data_lag_prior,
    run_corpus_pilot_eval,
)
from tas_copilot_core.corpus.extractors import extract_document, extractor_bakeoff, extractor_quality_report
from tas_copilot_core.corpus.fixtures import compile_figure_query, validate_source_support
from tas_copilot_core.corpus import paper_facts as paper_facts_module
from tas_copilot_core.corpus.paper_facts import extract_paper_facts, paper_fact_review
from tas_copilot_core.retrieval import GraphRetriever


FIXTURE_MANIFEST = "tests/fixtures/corpus_pilot/pilot_manifest.synthetic.yaml"


def test_corpus_pilot_ingest_builds_separate_document_catalog_and_graph(tmp_path: Path) -> None:
    catalog = tmp_path / "corpus_catalog"
    result = ingest_pilot_corpus({"manifest": FIXTURE_MANIFEST, "catalog": str(catalog), "output_dir": str(tmp_path / "ingest")})

    assert result["status"] == "ok"
    assert result["documents"] == 1
    assert result["graph"]["edge_count"] >= 6
    assert result["separation_policy"].startswith("document-side JSONL")
    assert (catalog / "source_chunks.jsonl").exists()
    assert not (catalog / "datafiles.jsonl").exists()
    chunks = JsonlCatalog(catalog / "source_chunks.jsonl", SourceChunk).load()
    assert {chunk.chunk_type for chunk in chunks} >= {"figure_caption", "instrument_setup", "sample_description"}
    edges = JsonlCatalog(catalog / "evidence_graph_edges.jsonl", EvidenceGraphEdge).load()
    assert {"HAS_CHUNK", "HAS_FIGURE", "HAS_PANEL", "SUPPORTS_CLAIM", "CANDIDATE_SOURCE_FOR_PANEL", "REJECTED_AS_SOURCE_FOR_PANEL"} <= {edge.edge_type for edge in edges}
    assert all(edge.provenance for edge in edges)
    nodes = JsonlCatalog(catalog / "evidence_graph_nodes.jsonl", EvidenceGraphNode).load()
    assert any(node.node_type == "FigurePanel" for node in nodes)

    graph_hits = GraphRetriever(catalog).search("BT-7 PSD elastic methods", limit=10)
    assert graph_hits
    assert any(hit.retrieval_method == "typed_graph_traversal" for hit in graph_hits)
    assert any((hit.structured_summary or {}).get("path") for hit in graph_hits)
    assert all(hit.provenance for hit in graph_hits)
    assert {"source_node_match", "target_node_match", "edge_match"} & set(
        reason
        for hit in graph_hits
        for reason in (hit.structured_summary or {}).get("match_reasons", [])
    )


def test_corpus_pilot_report_search_and_ontology_diagnostics(tmp_path: Path) -> None:
    catalog = tmp_path / "corpus_catalog"
    ingest_pilot_corpus({"manifest": FIXTURE_MANIFEST, "catalog": str(catalog), "output_dir": str(tmp_path / "ingest")})

    report = corpus_pilot_report(catalog, output_dir=tmp_path / "report")
    assert report["documents"] == 1
    assert report["figure_panels"] == 3
    assert report["evidence_graph_edges"] >= 6
    assert report["tas_concept_links"] >= 1
    assert Path(report["artifacts"]["json"]).exists()

    search = corpus_search({"catalog": str(catalog), "query": "BT-7 PSD elastic panel x_Mn 0.4", "top_k": 10})
    result_ids = {item["source_id"] for item in search["results"]}
    assert "chunk-lu-femn-o3-figure-caption" in result_ids
    assert any(item["code"] == "broad_query_term" for item in search["warnings"])
    assert all(item["provenance"] for item in search["results"])

    diagnostics = ontology_diagnostics({"catalog": str(catalog), "output_dir": str(tmp_path / "ontology")})
    assert diagnostics["status"] == "ok"
    assert diagnostics["concept_link_count"] >= 1
    assert Path(diagnostics["artifacts"]["markdown"]).exists()


def test_corpus_pilot_source_support_compile_and_briefs(tmp_path: Path) -> None:
    catalog = tmp_path / "corpus_catalog"
    ingest_pilot_corpus({"manifest": FIXTURE_MANIFEST, "catalog": str(catalog), "output_dir": str(tmp_path / "ingest")})

    compiled = compile_figure_query(catalog, paper_id="paper-lu-femn-o3-synthetic", figure_id="fig-lu-femn-o3-1", panel_id="k", user_sample="LuFe0.6Mn0.4O3")
    assert compiled["status"] == "ok"
    assert compiled["warnings"]
    assert compiled["query"]["target_composition"]["x_mn"] == 0.5

    constraints = extract_paper_constraints({"catalog": str(catalog), "output_dir": str(tmp_path / "constraints"), "paper_id": "paper-lu-femn-o3-synthetic"})
    formula_strings = set(constraints["constraints"][0]["formula_like_title_strings"])
    assert "LuFe" in formula_strings
    assert "BT" not in formula_strings

    support = validate_source_support(catalog)
    assert support["status"] == "ok"
    assert support["supported_claims"]

    paper = paper_brief({"corpus_catalog": str(catalog), "paper_id": "paper-lu-femn-o3-synthetic", "output_dir": str(tmp_path / "paper")})
    assert paper["status"] == "ok"
    assert paper["chunk_count"] >= 4
    assert Path(paper["artifacts"]["markdown"]).exists()

    figure = figure_brief({"corpus_catalog": str(catalog), "figure_id": "fig-lu-femn-o3-1", "output_dir": str(tmp_path / "figure")})
    assert figure["status"] == "ok"
    assert len(figure["panels"]) == 3
    assert figure["candidate_source_links"]

    links = link_paper_to_experiments({"corpus_catalog": str(catalog), "paper_id": "paper-lu-femn-o3-synthetic", "output_dir": str(tmp_path / "links")})
    assert links["status"] == "ok"
    assert links["candidate_links"][0]["status"] == "candidate"
    assert links["publication_data_lag_prior"]["publication_year"] == 2013
    assert links["publication_data_lag_prior"]["prior_center_year"] == 2010
    assert links["candidate_links"][0]["probability_score"] < 1.0
    assert links["candidate_links"][0]["publication_data_lag_date_score"] == 1.0
    assert links["review_label_diagnostics"]["status"] == "not_provided"

    private_prior = publication_data_lag_prior("private-2025", 2025)
    assert private_prior.ncnr_cap_applied is True
    assert private_prior.plausible_end_year == 2021


def test_link_paper_to_experiments_scores_data_catalog_candidates_probabilistically(tmp_path: Path) -> None:
    import pytest
    pytest.importorskip("bt7grouping", reason="header hydration requires the external BT-7 reader")
    import pytest
    pytest.importorskip("bt7grouping", reason="Header hydration requires the external BT-7 reader plugin")
    corpus = tmp_path / "corpus_catalog"
    corpus.mkdir()
    _write_jsonl(
        corpus / "papers.jsonl",
        [
            {
                "paper_id": "paper-cras",
                "title": "Frustrated Magnetic Interactions and Quenched Spin Fluctuations in CrAs",
                "year": 2020,
                "metadata": {"authors": ["A. Smith", "B. Jones"]},
                "provenance": [{"source_type": "fixture", "source_id": "paper-cras", "method": "test"}],
            }
        ],
    )
    _write_jsonl(
        corpus / "paper_constraints.jsonl",
        [
            {
                "constraint_id": "constraints-paper-cras",
                "paper_id": "paper-cras",
                "instrument": ["BT-7"],
                "detector_modes": [],
                "measurement_kinds": ["elastic", "inelastic", "scattering"],
                "sample_environment_terms": ["temperature"],
                "formula_like_title_strings": ["CrAs"],
                "constraint_evidence": [],
                "status": "proposed",
                "human_review_required": True,
            }
        ],
    )

    data_db = tmp_path / "bt7.sqlite"
    cras_dir = tmp_path / "bt7" / "201706"
    cras_dir.mkdir(parents=True)
    cras_file_1 = cras_dir / "CrAs_Tdep001.bt7"
    cras_file_2 = cras_dir / "CrAs_Tdep002.bt7"
    other_file = cras_dir / "OtherSample_Tdep003.bt7"
    cras_file_1.write_text(
        "\n".join(
            [
                "#ICE",
                "#Filename CrAs_Tdep001",
                "#Date 2017-06-01 10:00:00 EDT",
                "#ExptID 12345",
                "#ExptName Frustrated Magnetic Interactions in CrAs",
                "#ExptParticipants A. Smith, B. Jones",
                "#Npoints 5",
                "#ScanType ENV",
                "#AnalyzerDetectorMode 4 PSDDiff",
                "#ScanDescr Scan:SubID=12345:JType=ENV:Npts=5:Filename=CrAs_Tdep:Range=Temp=2 100 s:Title=elastic temperature scattering",
                "#Columns Temp A4 PSDC00 PSDC01 PSDC02",
                "2.0 29.5 10 11 12",
                "50.0 29.5 20 21 22",
                "100.0 29.5 30 31 32",
            ]
        ),
        encoding="utf-8",
    )
    cras_file_2.write_text(cras_file_1.read_text(encoding="utf-8").replace("CrAs_Tdep001", "CrAs_Tdep002"), encoding="utf-8")
    other_file.write_text(
        cras_file_1.read_text(encoding="utf-8")
        .replace("CrAs_Tdep001", "OtherSample_Tdep003")
        .replace("elastic temperature scattering", "other sample temperature scattering"),
        encoding="utf-8",
    )
    catalog = SQLiteDataCatalog(data_db)
    catalog.upsert_experiments(
        [
            Experiment(experiment_id="exp-cras", instrument="BT-7", title="BT-7 CrAs temperature scans", start_date="2017-06-01"),
            Experiment(experiment_id="exp-unrelated", instrument="BT-7", title="BT-7 unrelated sample", start_date="2011-01-01"),
            Experiment(experiment_id="exp-reviewed-positive", instrument="BT-7", title="BT-7 human reviewed paper link", start_date="2018-01-01"),
            Experiment(experiment_id="exp-unreviewed", instrument="BT-7", title="BT-7 CrAs weak unreviewed candidate", start_date="2018-02-01"),
        ]
    )
    catalog.upsert_scans(
        [
            Scan(scan_id="scan-cras-1", experiment_id="exp-cras", instrument="BT-7", scan_type="elastic temperature scattering", metadata={"sample_name": "CrAs"}),
            Scan(scan_id="scan-unrelated-1", experiment_id="exp-unrelated", instrument="BT-7", scan_type="alignment", metadata={"sample_name": "Other"}),
        ]
    )
    catalog.upsert_datafiles(
        [
            DataFile(datafile_id="df-cras-1", path=str(cras_file_1), scan_id="scan-cras-1", experiment_id="exp-cras", metadata={"instrument": "BT-7", "sample_name": "CrAs"}),
            DataFile(datafile_id="df-cras-2", path=str(cras_file_2), scan_id="scan-cras-1", experiment_id="exp-cras", metadata={"instrument": "BT-7", "sample_name": "CrAs"}),
            DataFile(datafile_id="df-cras-other-sample", path=str(other_file), scan_id="scan-cras-1", experiment_id="exp-cras", metadata={"instrument": "BT-7", "sample_name": "OtherSample"}),
            DataFile(datafile_id="df-unrelated-1", path="/bt7/201101/Other001.bt7", scan_id="scan-unrelated-1", experiment_id="exp-unrelated", metadata={"instrument": "BT-7", "sample_name": "Other"}),
        ]
    )

    result = link_paper_to_experiments(
        {
            "corpus_catalog": str(corpus),
            "data_catalog": str(data_db),
            "paper_id": "paper-cras",
            "output_dir": str(tmp_path / "links"),
            "top_k": 2,
            "hydrate_candidate_clues": True,
            "hydrate_candidate_limit": 2,
            "max_hydrated_files_per_candidate": 2,
        }
    )

    assert result["status"] == "ok"
    assert result["data_catalog"]["status"] == "scored"
    assert result["candidate_links"][0]["experiment_id"] == "exp-cras"
    assert result["candidate_links"][0]["status"] == "candidate"
    assert result["candidate_links"][0]["confidence_label"] in {"medium", "high"}
    assert result["candidate_links"][0]["probability_score"] < 1.0
    assert result["candidate_links"][0]["score_breakdown"]["material_title_score"] > 0
    assert result["candidate_links"][0]["score_breakdown"]["hydrated_header_score"] > 0
    assert result["candidate_links"][0]["metadata"]["human_review_required"] is True
    clues = result["candidate_links"][0]["historic_clue_summary"]
    assert 2017 in clues["data_years"]
    assert clues["top_filename_prefixes"][0]["value"] == "cras_tdep"
    assert result["candidate_links"][0]["metadata"]["historic_clue_summary"]["clue_policy"].startswith("historic_path")
    hydrated_clues = result["candidate_links"][0]["hydrated_clue_summary"]
    assert hydrated_clues["read_file_count"] == 2
    assert "expt_name" in hydrated_clues["reader_fields_seen"]
    assert "psdc00" in hydrated_clues["column_names_sample"]
    assert hydrated_clues["planned_npts_values"][0]["value"] == "5"
    assert hydrated_clues["actual_row_count_values"][0]["value"] == "3"
    assert hydrated_clues["reader_subgroups"][0]["expt_id"] == "12345"
    assert hydrated_clues["reader_subgroups"][0]["file_count"] == 2
    assert hydrated_clues["reader_subgroups"][0]["detector_modes"][0]["value"] == "4 PSDDiff"
    assert result["candidate_links"][0]["score_breakdown"]["hydrated_subgroup_mixed_penalty"] == 0.0
    assert hydrated_clues["source_tool"] == "tas.data_reader.read_bt7_file"
    assert SQLiteDataCatalog(data_db).list_paper_experiment_links()

    review_store = tmp_path / "review_outcomes.sqlite"
    prior_labels = [
        ReviewLabel(
            label_id=f"paper-link-prior-pos-{index}",
            object_type="paper_experiment_link",
            object_id=f"paper-prior-pos-{index}",
            object_secondary_id=f"exp-prior-pos-{index}",
            source_workflow="link-paper-to-experiments",
            verdict="positive",
            claim_state="accepted",
            evidence_channels_relied_on=["evidence_channel_paper_text"],
        )
        for index in range(10)
    ] + [
        ReviewLabel(
            label_id=f"paper-link-prior-neg-{index}",
            object_type="paper_experiment_link",
            object_id=f"paper-prior-neg-{index}",
            object_secondary_id=f"exp-prior-neg-{index}",
            source_workflow="link-paper-to-experiments",
            verdict="negative",
            claim_state="rejected",
            evidence_channels_relied_on=["evidence_channel_paper_text"],
        )
        for index in range(12)
    ]
    ReviewOutcomeSQLiteStore(review_store).upsert_labels(
        [
            *prior_labels,
            ReviewLabel(
                label_id="paper-link-exact-positive",
                object_type="paper_experiment_link",
                object_id="paper-cras",
                object_secondary_id="exp-reviewed-positive",
                source_workflow="link-paper-to-experiments",
                verdict="positive",
                claim_state="accepted",
                evidence_channels_relied_on=["evidence_channel_paper_text"],
                reviewer_id="Reviewer One",
            ),
            ReviewLabel(
                label_id="paper-link-exact-negative",
                object_type="paper_experiment_link",
                object_id="paper-cras",
                object_secondary_id="exp-cras",
                source_workflow="link-paper-to-experiments",
                verdict="negative",
                claim_state="rejected",
                evidence_channels_relied_on=["evidence_channel_hydrated_metadata"],
                reviewer_id="Reviewer One",
            ),
        ]
    )
    calibrated = link_paper_to_experiments(
        {
            "corpus_catalog": str(corpus),
            "data_catalog": str(data_db),
            "paper_id": "paper-cras",
            "output_dir": str(tmp_path / "calibrated_links"),
            "top_k": 4,
            "review_label_store": str(review_store),
        }
    )
    calibrated_by_id = {link["experiment_id"]: link for link in calibrated["candidate_links"]}
    assert calibrated["candidate_links"][0]["experiment_id"] == "exp-reviewed-positive"
    assert calibrated["candidate_links"][0]["reviewed_label_verdict"] == "positive"
    assert calibrated["candidate_links"][0]["status"] == "accepted"
    assert calibrated["candidate_links"][0]["label_calibrated_score"] == 1.0
    assert calibrated_by_id["exp-cras"]["reviewed_label_verdict"] == "negative"
    assert calibrated_by_id["exp-cras"]["status"] == "rejected"
    assert calibrated_by_id["exp-cras"]["label_calibrated_score"] == 0.0
    unreviewed = calibrated_by_id["exp-unreviewed"]
    assert unreviewed["review_label_status"] == "unreviewed"
    assert unreviewed["uncalibrated_probability_score"] == unreviewed["probability_score"]
    assert unreviewed["label_calibration_channels"] == ["evidence_channel_paper_text"]
    assert unreviewed["label_calibration_applied"] is True
    assert unreviewed["label_calibrated_score"] < unreviewed["uncalibrated_probability_score"]
    assert calibrated["label_informed_priors"]["behavior_change_readiness"]["status"] == "ready"

    index_path = tmp_path / "bt7_hydrated_index.jsonl"
    index = build_bt7_hydrated_index(
        {
            "data_catalog": str(data_db),
            "output": str(index_path),
            "output_dir": str(tmp_path / "bt7_index_report"),
            "path_contains": "201706",
        }
    )
    assert index["status"] == "ok"
    assert index["read_files"] == 3
    assert index["group_count"] == 1
    assert index["source_unit_count"] == 2
    assert index["candidate_count"] == 3
    indexed = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    assert {item["candidate_source"] for item in indexed} == {"bt7_hydrated_index", "bt7_hydrated_source_unit"}
    assert indexed[0]["hydrated_clue_summary"]["reader_subgroups"][0]["expt_id"] == "12345"
    source_units = [item for item in indexed if item["candidate_source"] == "bt7_hydrated_source_unit"]
    assert {item["hydrated_clue_summary"]["source_unit"]["prefix"] for item in source_units} == {"cras_tdep", "othersample_tdep"}

    indexed_result = link_paper_to_experiments(
        {
            "corpus_catalog": str(corpus),
            "data_catalog": str(data_db),
            "hydrated_index": str(index_path),
            "paper_id": "paper-cras",
            "output_dir": str(tmp_path / "indexed_links"),
            "top_k": 2,
        }
    )
    assert indexed_result["data_catalog"]["hydrated_index"]["status"] == "loaded"
    assert any(link["candidate_source"] == "bt7_hydrated_source_unit" for link in indexed_result["candidate_links"])
    indexed_link = next(link for link in indexed_result["candidate_links"] if link["candidate_source"] == "bt7_hydrated_source_unit" and link["hydrated_clue_summary"]["source_unit"]["prefix"] == "cras_tdep")
    assert indexed_link["metadata"]["candidate_source"] == "bt7_hydrated_source_unit"
    assert indexed_link["hydrated_clue_summary"]["read_file_count"] == 2


def test_paper_experiment_review_dashboard_and_label_ingest(tmp_path: Path) -> None:
    review_template = tmp_path / "paper_experiment_review_template.jsonl"
    assist = tmp_path / "paper_experiment_review_assist.json"
    source_report = tmp_path / "paper_experiment_links.json"
    scan_file = tmp_path / "scan1.bt7"
    scan_file.write_text(
        "\n".join(
            [
                "# Filename scan1",
                "# ExptID exp-1",
                "# Date 2020-01-02 03:04:05 EST",
                "# InstrName BT-7",
                "# ScanType VECTOR",
                "# AnalyzerDetectorMode 2 SingDetFlat",
                "# ScanDescr Scan:SubID=exp-1:JType=VECTOR:Npts=3:Title=scan1",
                "# Columns QX Detector Monitor E Temp",
                "0.0 10 100 0 5",
                "0.1 20 100 0 5",
                "0.2 12 100 0 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_report.write_text(
        json.dumps(
            {
                "paper_id": "paper-1",
                "extracted_constraints": [
                    {
                        "instrument": ["BT-7"],
                        "measurement_kinds": ["elastic"],
                        "detector_modes": ["single detector"],
                        "constraint_evidence": [{"value": "BT-7"}],
                    }
                ],
                "candidate_links": [
                    {
                        "experiment_id": "exp-1",
                        "score_breakdown": {"instrument": 1.0},
                        "metadata": {"sample_datafile_paths": [str(scan_file)]},
                        "hydrated_clue_summary": {
                            "planned_npts_values": [{"value": "3", "count": 1}],
                            "actual_row_count_values": [{"value": "3", "count": 1}],
                            "column_names_sample": ["QX", "Detector", "Monitor", "E", "Temp"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_report.with_suffix(".md").write_text("# Links\n", encoding="utf-8")
    _write_jsonl(
        review_template,
        [
            {
                "paper_id": "paper-1",
                "paper_title": "BT-7 paper",
                "candidate_experiment_id": "exp-1",
                "candidate_rank": 1,
                "candidate_status": "candidate",
                "confidence_label": "medium",
                "probability_score": 0.71,
                "human_review_status": "unreviewed",
                "allowed_human_review_statuses": ["accepted", "rejected", "needs_more_evidence"],
                "evidence_to_check": ["Matched instrument: BT-7."],
                "warnings": ["Candidate link is proposed."],
                "source_report": str(source_report),
                "provenance": [{"source_type": "fixture", "source_id": "paper-1", "method": "test"}],
            },
            {
                "paper_id": "paper-1",
                "paper_title": "BT-7 paper",
                "candidate_experiment_id": "exp-2",
                "candidate_rank": 2,
                "candidate_status": "candidate",
                "confidence_label": "low",
                "probability_score": 0.32,
                "human_review_status": "unreviewed",
                "allowed_human_review_statuses": ["accepted", "rejected", "needs_more_evidence"],
                "evidence_to_check": ["Matched only a broad term."],
                "warnings": ["Candidate link is proposed."],
                "source_report": str(source_report),
                "provenance": [{"source_type": "fixture", "source_id": "paper-1", "method": "test"}],
            },
        ],
    )
    assist.write_text(
        json.dumps(
            {
                "items": [
                    {"paper_id": "paper-1", "candidate_rank": 1, "review_priority": "high_review_priority", "cautions": ["candidate_not_accepted"], "review_questions": ["Does the proposal ID match?"]},
                    {"paper_id": "paper-1", "candidate_rank": 2, "review_priority": "low_review_priority", "cautions": ["weak_score"], "review_questions": ["Is this a false positive?"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    sidecar = tmp_path / "review_packet_scan_records.jsonl"
    _write_jsonl(
        sidecar,
        [
            {
                "paper_id": "paper-1",
                "candidate_experiment_id": "exp-2",
                "candidate_rank": 2,
                "scan_records": [
                    {
                        "scan_id": "precomputed-scan",
                        "path": "/private/raw/file.bt7",
                        "experiment_id": "exp-2",
                        "title": "Precomputed packet scan",
                        "scan_type": "A3",
                        "scan_variable": "A3",
                        "measurement_kind": "elastic",
                        "row_count": 3,
                        "column_names": ["A3", "Detector"],
                        "metadata": {"ScanDescr": "a3 scan"},
                        "plot_data": {"x": [1, 2, 3], "y": [10, 22, 9], "x_column": "A3", "y_column": "Detector"},
                    }
                ],
            }
        ],
    )

    dashboard = build_paper_experiment_review_dashboard(
        {
            "review_template": str(review_template),
            "review_assist": str(assist),
            "review_scan_records": str(sidecar),
            "output_dir": str(tmp_path / "dashboard"),
        }
    )

    assert dashboard["status"] == "ok"
    assert dashboard["review_row_count"] == 2
    assert Path(dashboard["artifacts"]["review_rows_csv"]).exists()
    assert Path(dashboard["artifacts"]["review_summary_json"]).exists()
    assert Path(dashboard["artifacts"]["review_outcomes_sqlite"]).exists()
    dashboard_store = ReviewOutcomeSQLiteStore(dashboard["artifacts"]["review_outcomes_sqlite"])
    assert dashboard_store.summary()["by_verdict"] == {"unreviewed": 2}
    html = Path(dashboard["artifacts"]["html"]).read_text(encoding="utf-8")
    assert "BT-7 Paper-to-Experiment Review" in html
    assert "high_review_priority" in html
    assert '<tr data-id="paper-1::rank-1::exp-1">' in html
    assert "Ignored corrupt saved browser edits" in html
    assert "Rows loaded" in html
    assert "Show all" in html
    assert "Review packet with plots" in html
    assert "Open paper from local cache/Vista" in html
    packet = dashboard["artifacts"].get("html") and tmp_path / "dashboard" / "review_packets"
    packet_htmls = list(packet.glob("*.html")) if packet else []
    assert packet_htmls
    packet_html = "\n".join(path.read_text(encoding="utf-8") for path in packet_htmls)
    assert "Paper Link Review Packet" in packet_html
    assert "scan1" in packet_html
    assert "precomputed-scan" in packet_html
    assert "scan plots loaded from review_packet_scan_records.jsonl sidecar" in packet_html
    assert "Detector" in packet_html
    assert "/papers/paper-1" in packet_html

    exported = tmp_path / "reviewed_rows.jsonl"
    _write_jsonl(
        exported,
        [
            {
                "paper_id": "paper-1",
                "paper_title": "BT-7 paper",
                "candidate_experiment_id": "exp-1",
                "candidate_rank": 1,
                "probability_score": 0.71,
                "human_review_status": "accepted",
                "reviewer": "tester",
                "evidence_channels_relied_on": ["evidence_channel_panel_label"],
            },
            {"paper_id": "paper-1", "paper_title": "BT-7 paper", "candidate_experiment_id": "exp-2", "candidate_rank": 2, "probability_score": 0.32, "human_review_status": "rejected", "reviewer": "tester"},
        ],
    )
    ingest = ingest_paper_experiment_review_labels({"labels": str(exported), "output_dir": str(tmp_path / "labels"), "min_each": 1})

    assert ingest["status"] == "ok"
    assert ingest["calibration_readiness"]["can_calibrate"] is True
    assert Path(ingest["generated_artifacts"]["validated_labels_jsonl"]).exists()
    assert Path(ingest["generated_artifacts"]["review_outcomes_sqlite"]).exists()
    ingest_store = ReviewOutcomeSQLiteStore(ingest["generated_artifacts"]["review_outcomes_sqlite"])
    assert ingest_store.summary()["by_verdict"] == {"negative": 1, "positive": 1}
    accepted = ingest_store.list_labels(object_type="paper_experiment_link", verdict="positive")
    assert accepted[0].evidence_channels_relied_on == ["evidence_channel_panel_label"]


def test_corpus_extractor_boundary_and_caption_chunking(tmp_path: Path) -> None:
    text_file = tmp_path / "synthetic_paper.txt"
    text_file.write_text(
        "\n".join(
            [
                "Synthetic BT-7 PSD Paper",
                "Abstract. This synthetic paper is for tests only.",
                "Experimental methods. Measurements used BT-7 with a PSD detector.",
                "Figure 2. Elastic PSD temperature map for LuFe_{1-x}Mn_xO3 panels.",
                "Data reduction used no private copyrighted text.",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "corpus_id: extractor_test",
                "documents:",
                "  - paper_id: synthetic-extractor-paper",
                "    type: paper",
                "    title: Synthetic Extractor Paper",
                f"    local_path: {text_file}",
                "    copyright_status: synthetic_fixture",
            ]
        ),
        encoding="utf-8",
    )

    extracted = extract_document({"path": str(text_file), "backend": "pdftotext_baseline", "max_chars": 5000})
    assert extracted["status"] == "success"
    assert extracted["quality_metrics"]["has_bt7_terms"] is True
    assert any(item["chunk_type"] == "figure_caption" for item in extracted["chunk_candidates"])

    auto = extract_document({"path": str(text_file), "backend": "auto", "max_chars": 5000})
    assert auto["status"] == "success"
    assert auto["backend"] == "pdftotext_baseline"

    fake_pdf = tmp_path / "synthetic_paper.pdf"
    fake_pdf.write_bytes(b"%PDF synthetic placeholder for unavailable backend tests")
    grobid = extract_document({"path": str(fake_pdf), "backend": "grobid"})
    assert grobid["status"] == "unavailable"
    assert "GROBID endpoint" in grobid["warnings"][0]

    docling = extract_document({"path": str(fake_pdf), "backend": "docling"})
    assert docling["status"] in {"unavailable", "error"}

    bakeoff = extractor_bakeoff({"manifest": str(manifest), "output_dir": str(tmp_path / "bakeoff"), "backends": ["pdftotext_baseline", "docling"]})
    assert bakeoff["status"] == "ok"
    assert bakeoff["reports"][0]["backends"][0]["status"] == "success"
    assert bakeoff["reports"][0]["backends"][1]["status"] in {"unavailable", "unsupported", "error"}

    fixture_bakeoff = extractor_bakeoff({"manifest": FIXTURE_MANIFEST, "output_dir": str(tmp_path / "fixture_bakeoff"), "backends": ["pdftotext_baseline"]})
    assert fixture_bakeoff["reports"][0]["backends"][0]["status"] == "success"

    catalog = tmp_path / "caption_catalog"
    ingest_pilot_corpus({"manifest": str(manifest), "catalog": str(catalog), "output_dir": str(tmp_path / "caption_ingest")})
    chunks = JsonlCatalog(catalog / "source_chunks.jsonl", SourceChunk).load()
    captions = [chunk for chunk in chunks if chunk.chunk_type == "figure_caption"]
    assert captions
    assert captions[0].metadata.get("figure_label_guess") == "Figure 2"


def test_chunking_derives_material_composition_and_structure_chunks(tmp_path: Path) -> None:
    text_file = tmp_path / "material_paper.txt"
    text_file.write_text(
        "\n".join(
            [
                "Synthetic Multiferroic Paper",
                "Abstract. We study hexamanganite physics in synthetic fixtures.",
                "Samples of LuFe0.5Mn0.5O3 were grown by floating zone. The parent compound",
                "LuFe0.5Mn0.5O3 retains the hexagonal crystal structure with space group P63cm",
                "and lattice parameter a = 6.038 angstrom. Neutron measurements used BT-7.",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "corpus_id: material_chunk_test",
                "documents:",
                "  - paper_id: synthetic-material-paper",
                "    type: paper",
                "    title: Synthetic Multiferroic Paper",
                f"    local_path: {text_file}",
                "    copyright_status: synthetic_fixture",
            ]
        ),
        encoding="utf-8",
    )

    catalog = tmp_path / "material_catalog"
    ingest_pilot_corpus({"manifest": str(manifest), "catalog": str(catalog), "output_dir": str(tmp_path / "material_ingest")})
    chunks = JsonlCatalog(catalog / "source_chunks.jsonl", SourceChunk).load()

    material = [chunk for chunk in chunks if chunk.chunk_type == "material_composition"]
    assert material
    candidates = material[0].metadata["formula_candidates"]
    assert candidates[0]["formula"] == "LuFe0.5Mn0.5O3"
    assert candidates[0]["occurrences"] == 2
    assert material[0].metadata["derivation"] == "regex_material_structure_v0"
    assert "LuFe0.5Mn0.5O3" in material[0].text
    # Canonical parsing goes through the material-identity boundary.
    assert "parser_backend" in candidates[0]
    if candidates[0]["parser_backend"] not in {"unavailable"}:
        assert candidates[0]["normalized_formula"]
        assert candidates[0]["parse_confidence"] > 0

    structure = [chunk for chunk in chunks if chunk.chunk_type == "structure_parameters"]
    assert structure
    assert "space group" in structure[0].metadata["structure_terms_found"]
    assert "hexagonal" in structure[0].metadata["structure_terms_found"]
    assert structure[0].metadata["space_group_guess"] == "P63cm"

    # Ordinary capitalized words and acronyms must not become formula candidates.
    all_formulas = [item["formula"] for item in candidates]
    assert "BT" not in all_formulas
    assert all("Abstract" not in formula for formula in all_formulas)


def test_ingest_extractor_run_builds_private_document_catalog(tmp_path: Path) -> None:
    run_dir = tmp_path / "vista_run"
    md_dir = run_dir / "outputs" / "docling" / "md"
    tei_dir = run_dir / "outputs" / "grobid" / "tei"
    md_dir.mkdir(parents=True)
    tei_dir.mkdir(parents=True)
    md_path = md_dir / "001_Synthetic_BT7_PSD.md"
    md_path.write_text(
        "\n".join(
            [
                "# Synthetic BT-7 PSD Article",
                "Abstract. Synthetic private-output fixture for importer tests.",
                "Experimental methods. Measurements used BT-7 PSD elastic scattering.",
                "Figure 1. Elastic PSD temperature map with panel j labelled x_Mn = 0.4.",
            ]
        ),
        encoding="utf-8",
    )
    tei_path = tei_dir / "001_Synthetic_BT7_PSD.tei.xml"
    tei_path.write_text(
        """
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <teiHeader>
            <fileDesc>
              <titleStmt><title level="a" type="main">Synthetic TEI BT-7 PSD Title</title></titleStmt>
              <sourceDesc>
                <biblStruct>
                  <analytic>
                    <author><persName><forename type="first">Ada</forename><surname>Lovelace</surname></persName></author>
                  </analytic>
                  <idno type="DOI">10.0000/synthetic.bt7</idno>
                  <note type="submission">Received: 1 January 2024; Accepted: 2 February 2024;</note>
                </biblStruct>
              </sourceDesc>
            </fileDesc>
          </teiHeader>
          <text><body><p>BT-7 PSD fallback text.</p></body></text>
        </TEI>
        """,
        encoding="utf-8",
    )
    (run_dir / "outputs" / "docling").mkdir(exist_ok=True)
    (run_dir / "outputs" / "grobid").mkdir(exist_ok=True)
    (run_dir / "outputs" / "docling" / "summary.json").write_text(
        json.dumps(
            [
                {"index": 1, "pdf_name": "001_Synthetic_BT7_PSD.pdf", "pdf": "/private/pdfs/001_Synthetic_BT7_PSD.pdf", "status": "success", "markdown_path": str(md_path), "bt7_terms": 1},
                {"index": 2, "pdf_name": "._001_Synthetic_BT7_PSD.pdf", "pdf": "/private/pdfs/._001_Synthetic_BT7_PSD.pdf", "status": "error"},
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "outputs" / "grobid" / "summary.json").write_text(
        json.dumps(
            [
                {"index": 1, "pdf_name": "001_Synthetic_BT7_PSD.pdf", "status": "success", "tei_path": str(tei_path), "bt7_terms": 1},
                {"index": 2, "pdf_name": "._001_Synthetic_BT7_PSD.pdf", "status": "error"},
            ]
        ),
        encoding="utf-8",
    )

    catalog = tmp_path / "extractor_catalog"
    result = ingest_extractor_run({"extractor_run_dir": str(run_dir), "catalog": str(catalog), "output_dir": str(tmp_path / "extractor_ingest")})

    assert result["status"] == "ok"
    assert result["documents"] == 1
    assert result["skipped_sidecar_rows"] == 2
    assert result["source_chunks"] >= 3
    assert result["separation_policy"].startswith("document-side JSONL")
    assert (catalog / "documents.jsonl").exists()
    assert not (catalog / "datafiles.jsonl").exists()
    document = json.loads((catalog / "documents.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert document["title"] == "Synthetic TEI BT-7 PSD Title"
    assert document["metadata"]["doi"] == "10.0000/synthetic.bt7"
    assert document["metadata"]["year"] == 2024
    assert document["metadata"]["year_source"].startswith("grobid_source_bibl_note")
    paper = json.loads((catalog / "papers.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert paper["year"] == 2024
    assert paper["authors"] == ["Ada Lovelace"]
    chunks = JsonlCatalog(catalog / "source_chunks.jsonl", SourceChunk).load()
    assert {chunk.chunk_type for chunk in chunks} >= {"title", "instrument_setup", "figure_caption"}
    assert all(chunk.metadata.get("private_local_only") for chunk in chunks)
    assert Path(result["artifacts"]["markdown"]).exists()

    quality = extractor_quality_report({"extractor_run_dir": str(run_dir), "output_dir": str(tmp_path / "extractor_quality")})
    assert quality["status"] == "ok"
    assert quality["summary"]["document_count"] == 1
    assert quality["skipped_sidecar_rows"] == 2
    assert quality["summary"]["bt7_hit_documents"] == 1
    assert quality["bt7_hit_documents"][0]["docling_figure_hits"] >= 1
    assert isinstance(quality["bt7_hit_documents"][0]["quality_flags"], list)
    assert Path(quality["artifacts"]["json"]).exists()

    shortlist = paper_candidate_shortlist(
        {
            "catalog": str(catalog),
            "quality_report": quality["artifacts"]["json"],
            "output_dir": str(tmp_path / "shortlist"),
            "top_k": 1,
        }
    )
    assert shortlist["status"] == "ok"
    assert shortlist["top_candidates"][0]["pdf_name"] == "001_Synthetic_BT7_PSD.pdf"
    assert shortlist["top_candidates"][0]["total_score"] > 0
    assert shortlist["top_candidates"][0]["private_local_only"] is True
    assert "figure_caption" in shortlist["top_candidates"][0]["chunk_type_counts"]
    assert Path(shortlist["artifacts"]["markdown"]).exists()

    paper_id = json.loads((catalog / "documents.jsonl").read_text(encoding="utf-8").splitlines()[0])["paper_id"]
    constraints = extract_paper_constraints({"catalog": str(catalog), "paper_id": paper_id, "output_dir": str(tmp_path / "constraints")})
    assert constraints["status"] == "ok"
    assert constraints["paper_count"] == 1
    record = constraints["constraints"][0]
    assert "BT-7" in record["instrument"]
    assert "PSD" in record["detector_modes"]
    assert "elastic" in record["measurement_kinds"]
    assert record["status"] == "proposed"
    assert record["human_review_required"] is True
    assert all("snippet" not in item for item in record["constraint_evidence"])
    assert Path(constraints["artifacts"]["json"]).exists()
    assert corpus_pilot_report(catalog)["paper_constraints"] == 1


def test_extract_paper_facts_requires_cited_typed_answers(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    paper_id = "paper-synthetic"
    _write_jsonl(
        catalog / "papers.jsonl",
        [{"paper_id": paper_id, "title": "Synthetic paper"}],
    )
    _write_jsonl(
        catalog / "documents.jsonl",
        [{"document_id": "doc-synthetic", "document_type": "paper", "paper_id": paper_id, "title": "Synthetic paper"}],
    )
    _write_jsonl(
        catalog / "source_chunks.jsonl",
        [
            {
                "chunk_id": "chunk-methods",
                "source_type": "document",
                "source_id": "doc-synthetic",
                "document_id": "doc-synthetic",
                "paper_id": paper_id,
                "chunk_type": "instrument_setup",
                "title": "Synthetic paper [instrument_setup]",
                "text": "Experimental methods. Measurements were performed on BT-7 using a PSD detector.",
            }
        ],
    )

    no_model = extract_paper_facts({"catalog": str(catalog), "paper_id": paper_id, "output_dir": str(tmp_path / "facts_packet")})
    assert no_model["status"] == "needs_model_or_review"
    assert no_model["fact_count"] == 0
    assert "packets" in no_model["artifacts"]
    assert not (catalog / "paper_facts.jsonl").exists()

    model_output = tmp_path / "model_output.json"
    model_output.write_text(
        json.dumps(
            {
                "facts": [
                    {
                        "fact_type": "instrument",
                        "answer_status": "supported",
                        "value": {"text": "BT-7"},
                        "confidence": 0.82,
                        "evidence": [{"chunk_id": "chunk-methods", "quote": "Measurements were performed on BT-7"}],
                    },
                    {
                        "fact_type": "facility",
                        "answer_status": "not_found",
                        "value": {},
                        "confidence": 0.7,
                        "evidence": [],
                    },
                    {
                        "fact_type": "composition",
                        "answer_status": "supported",
                        "value": {"text": "Synthetic composition"},
                        "confidence": 0.77,
                        "evidence": [{"chunk_id": "chunk-methods", "quote": "Measurements were performed on BT-7"}],
                    },
                    {
                        "fact_type": "magnetic_field",
                        "answer_status": "ambiguous",
                        "value": {"text": "field mentioned but not quantified"},
                        "confidence": 0.4,
                        "evidence": [{"chunk_id": "chunk-methods", "quote": "Experimental methods"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    with_model = extract_paper_facts(
        {
            "catalog": str(catalog),
            "paper_id": paper_id,
            "output_dir": str(tmp_path / "facts"),
            "provider": "json-file",
            "model_output": str(model_output),
        }
    )
    assert with_model["status"] == "ok"
    assert with_model["fact_count"] == 4
    facts = JsonlCatalog(catalog / "paper_facts.jsonl", PaperFact).load()
    assert {fact.fact_type for fact in facts} == {"composition", "facility", "instrument", "magnetic_field"}
    instrument = next(fact for fact in facts if fact.fact_type == "instrument")
    assert instrument.answer_status == "supported"
    assert instrument.source_chunk_ids == ["chunk-methods"]
    assert instrument.evidence[0].support_status == "supported"

    review = paper_fact_review({"catalog": str(catalog), "paper_id": paper_id, "output_dir": str(tmp_path / "fact_review")})
    assert review["status"] == "ok"
    assert review["paper_count"] == 1
    assert review["fact_count"] == 4
    assert Path(review["artifacts"]["html"]).exists()
    assert "Synthetic composition" in Path(review["artifacts"]["html"]).read_text(encoding="utf-8")


def test_extract_paper_facts_repairs_malformed_ollama_json(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    paper_id = "paper-synthetic"
    _write_jsonl(catalog / "papers.jsonl", [{"paper_id": paper_id, "title": "Synthetic paper"}])
    _write_jsonl(
        catalog / "documents.jsonl",
        [{"document_id": "doc-synthetic", "document_type": "paper", "paper_id": paper_id, "title": "Synthetic paper"}],
    )
    _write_jsonl(
        catalog / "source_chunks.jsonl",
        [
            {
                "chunk_id": "chunk-methods",
                "source_type": "document",
                "source_id": "doc-synthetic",
                "document_id": "doc-synthetic",
                "paper_id": paper_id,
                "chunk_type": "instrument_setup",
                "title": "Synthetic paper [instrument_setup]",
                "text": "Measurements were performed on BT-7 using fixed final energy.",
            }
        ],
    )
    repaired = {
        "facts": [
            {
                "fact_type": "instrument",
                "answer_status": "supported",
                "value": {"text": "BT-7"},
                "confidence": 0.9,
                "evidence": [{"chunk_id": "chunk-methods", "quote": "Measurements were performed on BT-7"}],
            }
        ]
    }
    calls: list[list[dict[str, str]]] = []

    def fake_ollama_chat(*_args, **_kwargs):
        messages = _args[2]
        calls.append(messages)
        if len(calls) == 1:
            return {
                "message": {
                    "content": (
                        '{"facts":[{"fact_type":"instrument","answer_status":"supported",'
                        '"value":{"text":"BT-7"} "evidence":[{"chunk_id":"chunk-methods"}]}]}'
                    )
                }
            }
        return {"message": {"content": json.dumps(repaired)}}

    monkeypatch.setattr(paper_facts_module, "ollama_chat", fake_ollama_chat)
    result = extract_paper_facts(
        {
            "catalog": str(catalog),
            "paper_id": paper_id,
            "provider": "ollama",
            "model": "mock-gemma",
            "output_dir": str(tmp_path / "facts"),
        }
    )

    assert len(calls) == 2
    assert "Repair malformed JSON" in calls[1][0]["content"]
    assert result["status"] == "ok"
    assert result["fact_count"] == 1
    assert result["errors"] == []
    facts = JsonlCatalog(catalog / "paper_facts.jsonl", PaperFact).load()
    assert facts[0].fact_type == "instrument"
    assert facts[0].model == "mock-gemma"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_corpus_pilot_eval_and_agent_readiness(tmp_path: Path) -> None:
    result = run_corpus_pilot_eval({"suite": "eval/corpus_pilot_retrieval.yaml", "output_dir": str(tmp_path / "eval")})
    assert result["status"] == "pass"
    assert result["check_count"] >= 4
    assert Path(result["artifacts"]["json"]).exists()

    metadata_suite = tmp_path / "metadata_suite.yaml"
    metadata_suite.write_text(
        "\n".join(
            [
                "suite_id: metadata_eval",
                "fixture_manifest: tests/fixtures/corpus_pilot/pilot_manifest.synthetic.yaml",
                "min_documents: 1",
                "min_source_chunks: 2",
                "min_evidence_graph_edges: 1",
                "min_tas_concept_links: 1",
                "tasks:",
                "  - name: bt7_warning_and_chunk_type",
                "    query: BT-7 PSD elastic methods",
                "    minimum_result_count: 1",
                "    required_warning_terms:",
                "      - bt-7",
                "    required_chunk_types:",
                "      - instrument_setup",
                "    forbidden_chunk_types:",
                "      - references",
                "    forbidden_title_terms:",
                "      - References",
            ]
        ),
        encoding="utf-8",
    )
    metadata_result = run_corpus_pilot_eval({"suite": str(metadata_suite), "output_dir": str(tmp_path / "metadata_eval")})
    assert metadata_result["status"] == "pass"
    assert any(check["name"] == "min_documents" for check in metadata_result["checks"])

    readiness = agent_readiness({"corpus_catalog": str(tmp_path / "eval" / "corpus_catalog"), "data_catalog": str(tmp_path / "data.sqlite"), "output_dir": str(tmp_path / "ready")})
    assert readiness["status"] == "pass"
    assert "bounded planner/router" in readiness["safe_conclusion"]


def test_paper_fact_review_html_is_an_interactive_review_interface(tmp_path):
    import json as _json

    from tas_copilot_core.corpus.paper_facts import paper_fact_review
    from tas_copilot_core.review_labels import ingest_review_labels

    catalog = tmp_path / "catalog"
    catalog.mkdir()
    (catalog / "papers.jsonl").write_text(_json.dumps({"paper_id": "doc-x", "title": "Test paper"}) + "\n", encoding="utf-8")
    (catalog / "source_chunks.jsonl").write_text(
        _json.dumps({"chunk_id": "chunk-x-0", "chunk_type": "instrument_setup", "text": "Measured on BT-7 with Ef=14.7 meV and PG filter."}) + "\n",
        encoding="utf-8",
    )
    (catalog / "paper_facts.jsonl").write_text(
        _json.dumps(
            {
                "fact_id": "paper-fact-abc",
                "paper_id": "doc-x",
                "fact_type": "instrument",
                "answer_status": "supported",
                "confidence": 0.9,
                "value": {"text": "BT-7"},
                "human_review_required": True,
                "evidence": [{"chunk_id": "chunk-x-0", "quote": "Measured on BT-7", "support_status": "proposed"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    review = paper_fact_review({"catalog": str(catalog), "output_dir": str(tmp_path / "review")})

    html = (tmp_path / "review/paper_fact_review.html").read_text(encoding="utf-8")
    # Real review controls, evidence content, and label export - not "review mentally".
    assert "select class='verdict'" in html and "data-fact-id='paper-fact-abc'" in html
    assert "Export labels JSONL" in html
    assert "Measured on BT-7 with Ef=14.7" in html  # chunk text, not just the id
    assert "mentally" not in html.lower()
    fact = review["records"][0]["facts"][0]
    assert fact["evidence"][0]["chunk_excerpt"].startswith("Measured on BT-7")

    # The exported row shape ingests into the unified review store.
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        _json.dumps(
            {
                "label_id": "review-fact-paper-fact-abc",
                "object_type": "paper_fact",
                "object_id": "paper-fact-abc",
                "object_secondary_id": "doc-x",
                "source_workflow": "paper-fact-review",
                "verdict": "positive",
                "type_specific_status": "correct",
                "review_notes": "checked against quote",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = ingest_review_labels({"labels": str(labels), "store": str(tmp_path / "store.sqlite")})
    assert result["status"] == "ok"
    assert result["ingested_label_count"] == 1
    assert result["by_object_type"] == {"paper_fact": 1}


def test_normalize_paper_fact_response_rehomes_and_demotes_citations() -> None:
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response

    packet = {
        "paper_id": "paper-x",
        "document_id": "doc-x",
        "chunks": [
            {"chunk_id": "chunk-x-000", "text": "The transition occurs below 85 K in the susceptibility. A refined moment of 3.25 muB was found."},
            {"chunk_id": "chunk-x-001", "text": "Samples were prepared from Sn flux and ground with an agate mortar."},
        ],
    }
    raw = {
        "facts": [
            # Quote verbatim, but model cited the wrong chunk -> re-homed, stays supported.
            {
                "fact_type": "sample",
                "answer_status": "supported",
                "value": {"text": "Sn flux growth"},
                "confidence": 0.9,
                "evidence": [{"chunk_id": "chunk-x-000", "quote": "prepared from Sn flux"}],
            },
            # Quote stitched from two real fragments -> split into two verbatim passages.
            {
                "fact_type": "ordered_moment",
                "answer_status": "supported",
                "value": {"text": "3.25 muB below 85 K"},
                "confidence": 0.9,
                "evidence": [{"chunk_id": "chunk-x-001", "quote": "transition occurs below 85 K...refined moment of 3.25 muB"}],
            },
            # Quote found nowhere -> fabricated evidence; supported demotes to ambiguous.
            {
                "fact_type": "propagation_vector",
                "answer_status": "supported",
                "value": {"text": "k = (0 0 0)"},
                "confidence": 0.9,
                "evidence": [{"chunk_id": "chunk-x-000", "quote": "propagation vector of one quarter along c"}],
            },
            # OCR glyph noise must not trip the check: alnum-normalized match.
            {
                "fact_type": "magnetic_structure",
                "answer_status": "supported",
                "value": {"text": "transition below 85 K"},
                "confidence": 0.9,
                "evidence": [{"chunk_id": "chunk-x-000", "quote": "transition occurs below 85 K, in the susceptibility"}],
            },
        ]
    }
    facts, errors = normalize_paper_fact_response(raw, packet=packet, provider="test", model="mock")
    assert not errors
    by_type = {fact["fact_type"]: fact for fact in facts}

    rehomed = by_type["sample"]
    assert rehomed["answer_status"] == "supported"
    assert rehomed["evidence"][0]["chunk_id"] == "chunk-x-001"
    assert rehomed["evidence"][0]["support_status"] == "supported"
    assert any("re-homed" in warning for warning in rehomed["evidence"][0]["warnings"])
    assert rehomed["human_review_required"] is True

    stitched = by_type["ordered_moment"]
    assert stitched["answer_status"] == "supported"
    assert [ev["quote"] for ev in stitched["evidence"]] == ["transition occurs below 85 K", "refined moment of 3.25 muB"]
    source_text = packet["chunks"][0]["text"]
    for ev in stitched["evidence"]:
        assert ev["chunk_id"] == "chunk-x-000"  # re-homed from the model-cited chunk-x-001
        assert ev["support_status"] == "supported"
        assert ev["quote_match"] == "split_fragment"
        assert source_text[ev["char_start"]:ev["char_end"]] == ev["quote"]
        assert ev["quote_as_extracted"] == "transition occurs below 85 K...refined moment of 3.25 muB"
        assert any("stitched" in warning for warning in ev["warnings"])
    assert stitched["human_review_required"] is True

    fabricated = by_type["propagation_vector"]
    assert fabricated["answer_status"] == "ambiguous"
    assert any("Downgraded from supported" in warning for warning in fabricated["warnings"])

    ocr_ok = by_type["magnetic_structure"]
    assert ocr_ok["answer_status"] == "supported"
    assert ocr_ok["evidence"][0]["support_status"] == "supported"
    assert not ocr_ok["evidence"][0]["warnings"]
    # The stored quote is the source text, with offsets; the model's punctuation
    # variant is kept beside it.
    assert ocr_ok["evidence"][0]["quote"] == "transition occurs below 85 K in the susceptibility."
    assert ocr_ok["evidence"][0]["quote_match"] == "verbatim"
    assert ocr_ok["evidence"][0]["quote_as_extracted"] == "transition occurs below 85 K, in the susceptibility"


def test_normalize_paper_fact_response_flags_quoteless_evidence() -> None:
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response

    packet = {
        "paper_id": "paper-x",
        "document_id": "doc-x",
        "chunks": [{"chunk_id": "chunk-x-000", "text": "The applied field was 1 kOe for susceptibility."}],
    }
    raw = {
        "facts": [
            # Supported fact whose only evidence is a bare chunk pointer, no quote:
            # unverifiable, so the fact demotes to ambiguous.
            {
                "fact_type": "magnetic_field",
                "answer_status": "supported",
                "value": {"text": "1 kOe"},
                "confidence": 0.9,
                "evidence": [{"chunk_id": "chunk-x-000"}],
            },
            # A verbatim quote alongside a quote-less pointer keeps the fact supported.
            {
                "fact_type": "measurement_kind",
                "answer_status": "supported",
                "value": {"text": "susceptibility"},
                "confidence": 0.9,
                "evidence": [
                    {"chunk_id": "chunk-x-000", "quote": "applied field was 1 kOe"},
                    {"chunk_id": "chunk-x-000"},
                ],
            },
        ]
    }
    facts, errors = normalize_paper_fact_response(raw, packet=packet, provider="test", model="mock")
    assert not errors
    by_type = {fact["fact_type"]: fact for fact in facts}

    quoteless = by_type["magnetic_field"]
    assert quoteless["answer_status"] == "ambiguous"
    assert quoteless["evidence"][0]["support_status"] == "proposed"
    assert any("no quote" in warning for warning in quoteless["evidence"][0]["warnings"])
    assert any("Downgraded from supported" in warning for warning in quoteless["warnings"])

    mixed = by_type["measurement_kind"]
    assert mixed["answer_status"] == "supported"
    assert mixed["evidence"][0]["support_status"] == "supported"
    assert mixed["evidence"][1]["support_status"] == "proposed"


def test_normalize_paper_fact_response_clears_not_found_values_and_lints_fields() -> None:
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response

    packet = {
        "paper_id": "paper-x",
        "document_id": "doc-x",
        "chunks": [{"chunk_id": "chunk-x-000", "text": "Measurements between 20 K and 300 K show magnetic order."}],
    }
    raw = {
        "facts": [
            # Hallucination magnet: not_found answer still carrying a value.
            {
                "fact_type": "facility",
                "answer_status": "not_found",
                "value": {"text": "BT-7"},
                "confidence": 0.3,
                "evidence": [],
            },
            # Clean not_found stays untouched.
            {
                "fact_type": "filter",
                "answer_status": "not_found",
                "value": {},
                "confidence": 0.6,
                "evidence": [],
            },
            # Field mixup: a temperature range filed under energy_setting is
            # caught by the semantic sanitizer on every extraction path now.
            {
                "fact_type": "energy_setting",
                "answer_status": "supported",
                "value": {"text": "temperature range of 20-300 K"},
                "confidence": 0.8,
                "evidence": [{"chunk_id": "chunk-x-000", "quote": "between 20 K and 300 K"}],
            },
        ]
    }
    facts, errors = normalize_paper_fact_response(raw, packet=packet, provider="test", model="mock")
    assert not errors
    by_type = {fact["fact_type"]: fact for fact in facts}

    cleared = by_type["facility"]
    assert cleared["value"] == {}
    assert cleared["metadata"]["discarded_value"] == {"text": "BT-7"}
    assert any("value cleared" in warning for warning in cleared["warnings"])

    clean = by_type["filter"]
    assert clean["value"] == {}
    assert "discarded_value" not in clean["metadata"]
    assert not clean["warnings"]

    mixup = by_type["energy_setting"]
    assert mixup["answer_status"] == "ambiguous"
    assert any("Semantic sanity check" in warning for warning in mixup["warnings"])


def test_parse_model_json_tolerates_control_chars_and_salvages_partial_facts() -> None:
    from tas_copilot_core.corpus.paper_facts import _parse_model_json, _salvage_facts_from_partial

    # A literal newline inside a quoted value is what actually broke 9 of 48
    # papers on Vista: strict JSON rejects it and the paper yielded nothing.
    with_newline = '{"facts": [{"fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7\nspectrometer"}}]}'
    obj, error = _parse_model_json(with_newline)
    assert error is None
    assert obj is not None and len(obj["facts"]) == 1

    # Truncated array: every complete object before the damage is recovered.
    truncated = (
        '{"facts": [{"fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}}, '
        '{"fact_type": "facility", "answer_status": "supported", "value": {"text": "NCN'
    )
    salvaged = _salvage_facts_from_partial(truncated)
    assert len(salvaged) == 1
    assert salvaged[0]["fact_type"] == "instrument"

    # Nothing to salvage when there is no facts array at all.
    assert _salvage_facts_from_partial('{"errors": ["boom"]}') == []

    # Well-formed responses are unaffected.
    good = '{"facts": [{"fact_type": "sample", "answer_status": "not_found", "value": {}}]}'
    obj, error = _parse_model_json(good)
    assert error is None and obj is not None and len(obj["facts"]) == 1


def test_salvaged_facts_carry_provenance_through_normalization() -> None:
    """A salvaged fact must not reach a reviewer looking like a clean extraction.

    The top-level response warning is dropped by normalization, so the marker
    has to ride on each fact. This is what made 44 recovered facts on Vista
    indistinguishable from ordinary ones in the artifact.
    """
    from tas_copilot_core.corpus.paper_facts import _salvage_facts_from_partial, normalize_paper_fact_response

    truncated = (
        '{"facts": [{"fact_type": "instrument", "answer_status": "supported", "value": {"text": "BT-7"}, '
        '"evidence": [{"chunk_id": "chunk-x-000", "quote": "measured on BT-7"}]}, '
        '{"fact_type": "facility", "answer_status": "supported", "value": {"text": "NCN'
    )
    salvaged = _salvage_facts_from_partial(truncated)
    assert len(salvaged) == 1
    note = "salvaged_from_malformed_response: 1 complete fact objects were recovered"
    for fact in salvaged:
        fact["warnings"] = [note]

    packet = {
        "paper_id": "paper-x",
        "document_id": "doc-x",
        "chunks": [{"chunk_id": "chunk-x-000", "text": "The sample was measured on BT-7 at NCNR."}],
    }
    facts, errors = normalize_paper_fact_response({"facts": salvaged}, packet=packet, provider="test", model="mock")
    assert not errors
    assert len(facts) == 1
    assert any("salvaged_from_malformed_response" in warning for warning in facts[0]["warnings"])


def test_near_verbatim_quote_is_not_treated_as_fabricated() -> None:
    """A quote with stray characters is corrupted transcription, not invention.

    On the 48-paper set this mislabeling was expensive: the reviewer was pushed
    to mark substantively correct facts as OCR/chunking failures because a
    duplicated word or dropped space tripped "not found in any catalog chunk",
    which would have reported the model far below its real accuracy.
    """
    from tas_copilot_core.corpus.paper_facts import _locate_quote, _normalize_alnum, normalize_paper_fact_response

    chunk_text = (
        "Neutron scattering experiments were carried out on the BT-7 thermal triple-axis spectrometer "
        "with two pyrolytic graphite filters after the sample position."
    )
    norm_chunks = {"chunk-a-000": _normalize_alnum(chunk_text)}
    papers = {"chunk-a-000": "paper-a"}

    # Stray doubled spaces and a dropped word: every character still appears in
    # order, so this is near-verbatim rather than absent.
    corrupted = "Neutron scattering experiments were carried   on the BT-7 thermal triple-axis spectrometer"
    located = _locate_quote(
        corrupted, cited_chunk_id="chunk-a-000", fact_paper_id="paper-a",
        norm_chunk_texts=norm_chunks, paper_by_chunk=papers,
    )
    assert located["status"] == "near_verbatim_in_chunk"
    assert located["found_in"] == ["chunk-a-000"]
    assert located["coverage"] >= 0.95

    # A genuinely fabricated quote of comparable length stays not_found.
    fabricated = "The sample was measured on the SEQUOIA chopper spectrometer at the Spallation Neutron Source"
    assert _locate_quote(
        fabricated, cited_chunk_id="chunk-a-000", fact_paper_id="paper-a",
        norm_chunk_texts=norm_chunks, paper_by_chunk=papers,
    )["status"] == "not_found"

    # Short strings are never promoted: they match "in order" by chance.
    assert _locate_quote(
        "BT-7", cited_chunk_id="chunk-a-000", fact_paper_id="paper-a",
        norm_chunk_texts=norm_chunks, paper_by_chunk=papers,
    )["status"] != "near_verbatim_in_chunk"

    # End to end: a near-verbatim quote must NOT demote the fact.
    packet = {"paper_id": "paper-a", "document_id": "doc-a", "chunks": [{"chunk_id": "chunk-a-000", "text": chunk_text}]}
    raw = {
        "facts": [
            {
                "fact_type": "instrument",
                "answer_status": "supported",
                "value": {"text": "BT-7"},
                "confidence": 0.9,
                "evidence": [{"chunk_id": "chunk-a-000", "quote": corrupted}],
            }
        ]
    }
    facts, errors = normalize_paper_fact_response(raw, packet=packet, provider="test", model="mock")
    assert not errors
    assert facts[0]["answer_status"] == "supported"
    evidence = facts[0]["evidence"][0]
    # The model's corrupted transcription is replaced by the exact source
    # passage, with offsets into the chunk; the original is kept for audit.
    assert evidence["quote"] == "Neutron scattering experiments were carried out on the BT-7 thermal triple-axis spectrometer"
    assert chunk_text[evidence["char_start"]:evidence["char_end"]] == evidence["quote"]
    assert evidence["quote_match"] == "snapped"
    assert evidence["quote_as_extracted"] == corrupted
    assert evidence["support_status"] == "supported"
    assert any("exact source passage" in warning for warning in evidence["warnings"])
    assert facts[0]["human_review_required"] is True


def test_scattered_character_match_is_not_a_quotation() -> None:
    """A quote whose characters merely occur in order across a whole chunk is
    not near-verbatim. On the gold set about a fifth of "near-verbatim" hits
    were of this kind (a caption fragment matched across an unrelated page)."""
    from tas_copilot_core.corpus.paper_facts import _compact_alignment, _locate_quote, _normalize_alnum

    filler = " ".join(f"Section {n} discusses lattice dynamics and transport in detail." for n in range(12))
    chunk_text = "Figure 1: spin " + filler + " wave velocities are " + filler + " reported as 555 meV A in the ab plane."
    quote = "Figure 1: spin wave velocities are reported as 555 meV A in the ab plane."
    assert _compact_alignment(_normalize_alnum(quote), _normalize_alnum(chunk_text)) is None
    located = _locate_quote(
        quote, cited_chunk_id="c0", fact_paper_id="p", norm_chunk_texts={"c0": _normalize_alnum(chunk_text)}, paper_by_chunk={"c0": "p"},
    )
    assert located["status"] == "not_found"

    # A dropped clause is still one passage: the span is longer than the quote, but compact.
    source = "two PG filters were used at BT-7 with collimations 48-80-80-120 and the same fixed final energy and one PG filter at HB-1."
    condensed = "two PG filters were used at BT-7 with collimations 48-80-80-120 and one PG filter at HB-1."
    aligned = _compact_alignment(_normalize_alnum(condensed), _normalize_alnum(source))
    assert aligned is not None and aligned[2] >= 0.95


def test_evidence_cleanup_strips_schema_leaks_and_duplicate_quotes() -> None:
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response

    text = "Neutron scattering experiments were carried out on the triple-axis spectrometers BT-7 [47] at NIST and HB-1 at HFIR."
    packet = {"paper_id": "paper-b", "document_id": "doc-b", "chunks": [{"chunk_id": "chunk-b-000", "text": text}]}
    leaked = 'carried out on the triple-axis spectrometers BT-7 [4type: instrument, answer_status: supported, value: {text:'
    raw = {
        "facts": [
            {
                "fact_type": "instrument",
                "answer_status": "supported",
                "value": {"text": "BT-7, HB-1"},
                "confidence": 0.9,
                "evidence": [
                    {"chunk_id": "chunk-b-000", "quote": "carried out on the triple-axis spectrometers BT-7"},
                    {"chunk_id": "chunk-b-000", "quote": "carried out on the  triple-axis spectrometers BT-7"},
                    {"chunk_id": "chunk-b-000", "quote": leaked},
                ],
            }
        ]
    }
    facts, errors = normalize_paper_fact_response(raw, packet=packet, provider="test", model="mock")
    assert not errors
    fact = facts[0]
    assert fact["answer_status"] == "supported"
    # Three citations of one passage collapse to one, and nothing of the leaked schema survives.
    assert len(fact["evidence"]) == 1
    assert any("duplicate evidence" in warning for warning in fact["warnings"])
    assert all("answer_status" not in (ev["quote"] or "") for ev in fact["evidence"])
    assert text[fact["evidence"][0]["char_start"]:fact["evidence"][0]["char_end"]] == fact["evidence"][0]["quote"]

    # Feeding the stored evidence back in (revalidation) gives the same evidence.
    again, _ = normalize_paper_fact_response(
        {"facts": [{**raw["facts"][0], "evidence": [dict(ev) for ev in fact["evidence"]]}]}, packet=packet, provider="test", model="mock",
    )
    keep = ("chunk_id", "quote", "char_start", "char_end", "quote_match")
    assert [{k: ev[k] for k in keep} for ev in again[0]["evidence"]] == [{k: ev[k] for k in keep} for ev in fact["evidence"]]


def test_empty_evidence_records_are_dropped_not_cited() -> None:
    """A record whose chunk id and quote are the literal words "None"/"," (the
    model filling slots it had nothing for) is dropped with a fact warning,
    and the real citation on the same fact is kept."""
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response

    chunk = {"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": "The composition determined from EDS is Ba1.00 Fe1.92 S1.38 Se1.37."}
    raw = {"facts": [{"fact_type": "composition", "answer_status": "supported", "value": {"text": "Ba1.00 Fe1.92 S1.38 Se1.37"}, "confidence": 0.9,
                      "evidence": [{"chunk_id": ",", "quote": "None"}, {"chunk_id": "chunk-a-000", "quote": "The composition determined from EDS is Ba1.00 Fe1.92 S1.38 Se1.37."}]}]}
    facts, _ = normalize_paper_fact_response(raw, packet={"paper_id": "paper-a", "document_id": "doc-a", "chunks": [chunk]}, provider="test", model="mock")
    assert len(facts) == 1
    fact = facts[0]
    assert [e["chunk_id"] for e in fact["evidence"]] == ["chunk-a-000"]
    assert fact["evidence"][0]["quote_match"] == "verbatim"
    assert any("Dropped an evidence record with no quote" in w for w in fact["warnings"])
    assert fact["answer_status"] == "supported"


def test_shortened_quote_is_split_into_its_verbatim_passages() -> None:
    """A caption copied with sentences left out (no ellipsis) is not fabricated
    evidence: each surviving sentence is cited as its own passage of the
    chunk, in order, with source text and offsets."""
    from tas_copilot_core.corpus.paper_facts import _locate_quote, _normalize_alnum, normalize_paper_fact_response

    caption = (
        "FIG. 3. Elastic scans along the (0.5, K, 0.5) direction across (a) K = 1 and (b) K = 0.5 at 5 and 90 K. "
        "(c, d) The temperature differences between 5 and 90 K for K = 1 and K = 0.5, respectively. "
        "The solid lines are Gaussian fits to the data. The bars in (a-d) represent the statistical errors that correspond to one standard deviation. "
        "The instrumental resolution at Q = (0.5, 1.0, 0.5) was determined by measuring the nuclear Bragg peak of Q = (1, 2, 1). "
        "(e) Resistivity measured as a function of temperature. (f) Magnetic susceptibility measured with a 1000 Oe magnetic field parallel to the b axis. "
        "The dashed lines mark kinks at 39 and 116 K."
    )
    shortened = (
        "FIG. 3. Elastic scans along the (0.5, K, 0.5) direction across (a) K = 1 and (b) K = 0.5 at 5 and 90 K. "
        "(c, d) The temperature differences between 5 and 90 K for K = 1 and K = 0.5, respectively. "
        "(e) Resistivity measured as a function of temperature. (f) Magnetic susceptibility measured with a 1000 Oe magnetic field parallel to the b axis."
    )
    chunks = [{"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": caption}]
    norm = {"chunk-a-000": _normalize_alnum(caption)}
    located = _locate_quote(shortened, cited_chunk_id="chunk-a-000", fact_paper_id="paper-a", norm_chunk_texts=norm, paper_by_chunk={"chunk-a-000": "paper-a"})
    assert located["status"] == "elided_passages" and len(located["runs"]) == 2 and located["coverage"] == 1.0

    raw = {"facts": [{"fact_type": "figure_caption", "answer_status": "supported", "value": {"text": "Figure 3: elastic scans; resistivity; susceptibility"}, "confidence": 0.9,
                      "evidence": [{"chunk_id": "chunk-a-000", "quote": shortened}]}]}
    facts, _ = normalize_paper_fact_response(raw, packet={"paper_id": "paper-a", "document_id": "doc-a", "chunks": chunks}, provider="test", model="mock")
    fact = facts[0]
    assert fact["answer_status"] == "supported"
    assert [e["quote_match"] for e in fact["evidence"]] == ["split_fragment", "split_fragment"]
    assert fact["evidence"][0]["quote"].startswith("FIG. 3. Elastic scans") and fact["evidence"][0]["quote"].endswith("respectively.")
    assert fact["evidence"][1]["quote"].startswith("(e) Resistivity") and fact["evidence"][1]["quote"].endswith("b axis.")
    for e in fact["evidence"]:
        assert caption[e["char_start"]:e["char_end"]] == e["quote"]
        assert e["quote_as_extracted"] == shortened and e["support_status"] == "supported"

    # Unrelated text is still not found.
    unrelated = "The specific heat of the sample shows a lambda anomaly at the ordering temperature of 42 K in zero field."
    assert _locate_quote(unrelated, cited_chunk_id="chunk-a-000", fact_paper_id="paper-a", norm_chunk_texts=norm, paper_by_chunk={"chunk-a-000": "paper-a"})["status"] == "not_found"


def test_short_exact_quote_is_located_with_offsets() -> None:
    """A quote shorter than the fuzzy-alignment minimum that occurs exactly in
    the cited chunk is still snapped to source text with offsets; it was left
    'unverified' and shown as not found."""
    from tas_copilot_core.corpus.paper_facts import _snap_quote_to_source, normalize_paper_fact_response

    chunk_text = "A long-range order occurs concomitantly with a structural distortion around TN = 5 : 7 K 14,35,36 . The order of the transition is unknown."
    snapped = _snap_quote_to_source("around TN = 5 : 7 K", chunk_text)
    assert snapped is not None and snapped["exact"] is True
    assert chunk_text[snapped["char_start"]:snapped["char_end"]] == "around TN = 5 : 7 K"
    assert _snap_quote_to_source("around TN = 9 K", chunk_text) is None  # short and not there: nothing to snap

    raw = {"facts": [{"fact_type": "transition_temperature", "answer_status": "supported", "value": {"text": "5.7 K"}, "confidence": 0.9,
                      "evidence": [{"chunk_id": "chunk-a-000", "quote": "around TN = 5 : 7 K"}]}]}
    facts, _ = normalize_paper_fact_response(raw, packet={"paper_id": "paper-a", "document_id": "doc-a", "chunks": [{"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": chunk_text}]}, provider="test", model="mock")
    ev = facts[0]["evidence"][0]
    assert ev["quote_match"] == "verbatim" and ev["char_start"] is not None and ev["support_status"] == "supported"


def test_values_written_in_latex_are_stored_as_plain_text() -> None:
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response, plain_text_value

    assert plain_text_value(r"$\alpha$-Cu$_2$V$_2$O$_7$") == "α-Cu2V2O7"
    assert plain_text_value(r"$\lambda = 2.4$ \AA") == "λ = 2.4 Å"
    assert plain_text_value(r"T_{N} = 33.4(1) K") == "TN = 33.4(1) K"
    assert plain_text_value(r"$\mu_B$ per Mn") == "μB per Mn"
    assert plain_text_value("BaFe 2 S 1 . 5 Se 1 . 5") == "BaFe 2 S 1 . 5 Se 1 . 5"  # plain text untouched
    chunk = {"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": "Spin dynamics of α -Cu 2 V 2 O 7 was investigated using single-crystal inelastic neutron scattering."}
    raw = {"facts": [{"fact_type": "material", "answer_status": "supported", "value": {"text": r"$\alpha$-Cu$_2$V$_2$O$_7$"}, "confidence": 0.9,
                      "evidence": [{"chunk_id": "chunk-a-000", "quote": "Spin dynamics of α -Cu 2 V 2 O 7 was investigated"}]}]}
    facts, _ = normalize_paper_fact_response(raw, packet={"paper_id": "paper-a", "document_id": "doc-a", "chunks": [chunk]}, provider="test", model="mock")
    fact = facts[0]
    assert fact["value"]["text"] == "α-Cu2V2O7" and fact["value"]["text_as_extracted"] == r"$\alpha$-Cu$_2$V$_2$O$_7$"
    assert any("LaTeX" in w for w in fact["warnings"]) and fact["answer_status"] == "supported"


def test_subject_written_into_the_value_text_moves_to_value_subject() -> None:
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response

    chunk = {"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": "YbIG exhibits dρ/dT < 0 in the whole measured temperature range from 300 to 0.1 K, including the region below TN."}
    raw = {"facts": [{"fact_type": "temperature_range", "answer_status": "supported", "value": {"text": "300 K to 0.1 K (for YbIG)"}, "confidence": 0.9,
                      "evidence": [{"chunk_id": "chunk-a-000", "quote": "measured temperature range from 300 to 0.1 K"}]}]}
    facts, _ = normalize_paper_fact_response(raw, packet={"paper_id": "paper-a", "document_id": "doc-a", "chunks": [chunk]}, provider="test", model="mock")
    assert facts[0]["value"]["text"] == "300 K to 0.1 K" and facts[0]["value"]["subject"] == "YbIG"
    assert any("value.subject" in w for w in facts[0]["warnings"])


def test_a_value_of_none_is_no_value() -> None:
    from tas_copilot_core.corpus.paper_facts import normalize_paper_fact_response

    chunk = {"chunk_id": "chunk-a-000", "paper_id": "paper-a", "text": "Samples with x = 0.1 and x = 0.2 were studied."}
    raw = {"facts": [{"fact_type": "composition", "answer_status": "supported", "value": {"text": "None"}, "confidence": 0.5,
                      "evidence": [{"chunk_id": "chunk-a-000", "quote": "Samples with x = 0.1 and x = 0.2 were studied."}]}]}
    facts, _ = normalize_paper_fact_response(raw, packet={"paper_id": "paper-a", "document_id": "doc-a", "chunks": [chunk]}, provider="test", model="mock")
    assert "text" not in facts[0]["value"] and facts[0]["answer_status"] == "ambiguous"
    assert any("wrote 'None'" in w for w in facts[0]["warnings"])
