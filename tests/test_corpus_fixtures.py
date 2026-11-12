import sqlite3

from tas_copilot_core.cli import main
from tas_copilot_core.corpus.fixtures import corpus_fixture_report, load_corpus_fixtures
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog


def test_load_corpus_fixtures_keeps_document_catalog_separate_from_sqlite_data_catalog(tmp_path):
    fixtures = "data/corpus_fixtures"
    corpus_catalog = tmp_path / "corpus"
    sqlite_catalog = SQLiteDataCatalog(tmp_path / "tas.sqlite")

    report = load_corpus_fixtures(fixtures, corpus_catalog)

    assert report["status"] == "ok"
    assert report["paper_chunks"] == 1
    assert report["textbook_chunks"] == 1
    assert report["textbook_sections"] == 1
    assert report["mappings"] == 1
    assert report["provenance_coverage"] == 1.0
    assert (corpus_catalog / "source_chunks.jsonl").exists()
    with sqlite3.connect(sqlite_catalog.path) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "source_chunks" not in tables
    assert "textbook_sections" not in tables


def test_corpus_fixture_cli_load_and_report(tmp_path, capsys):
    corpus_catalog = tmp_path / "corpus"

    assert main(["corpus", "load-fixtures", "--fixtures", "data/corpus_fixtures", "--catalog", str(corpus_catalog)]) == 0
    load_output = capsys.readouterr().out
    assert '"paper_chunks": 1' in load_output

    assert main(["corpus", "report", "--catalog", str(corpus_catalog)]) == 0
    report_output = capsys.readouterr().out
    assert '"source_support_validation": "ok"' in report_output

    report = corpus_fixture_report(corpus_catalog)
    assert report["separation_policy"].startswith("paper/textbook corpus fixtures")
