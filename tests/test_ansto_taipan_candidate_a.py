from __future__ import annotations

from pathlib import Path

import yaml

from tas_copilot_core.catalogs.import_data_catalog import import_jsonl_catalog_dir
from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog
from tas_copilot_core.workflows.retrieve_similar_scans import retrieve_similar_scans


FIXTURE = Path("tests/fixtures/ansto_taipan_exp1337_candidate_a")


def test_ansto_taipan_candidate_a_fixture_loads_as_data_catalog(tmp_path: Path) -> None:
    db_path = tmp_path / "ansto_candidate_a.sqlite"

    summary = import_jsonl_catalog_dir(FIXTURE, db_path)
    catalog = SQLiteDataCatalog(db_path)

    assert summary["experiments"] == 1
    assert summary["scans"] == 6
    assert summary["data_files"] == 6
    assert catalog.list_experiments()[0].instrument == "TAIPAN"
    assert {item.format for item in catalog.list_datafiles()} == {"ansto_taipan_nexus_hdf5"}
    assert all(item.path.startswith("sanitized://") for item in catalog.list_datafiles())


def test_ansto_taipan_candidate_a_preserves_material_counter_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "ansto_candidate_a.sqlite"
    import_jsonl_catalog_dir(FIXTURE, db_path)
    scans = SQLiteDataCatalog(db_path).list_scans()

    known_answer = [scan for scan in scans if scan.metadata.get("candidate_a_role") == "known_answer_member"]
    assert {scan.scan_id for scan in known_answer} == {
        "taipan-1337-115529",
        "taipan-1337-115544",
        "taipan-1337-115546",
        "taipan-1337-115552",
    }
    assert {scan.metadata["candidate_material"] for scan in known_answer} == {"Dy2Ti2O7"}
    assert {scan.metadata["folder_name_counter_evidence"] for scan in known_answer} == {"Ho2Ti2O7"}
    assert {scan.metadata["temperature_K"] for scan in known_answer} == {0.045, 0.8}
    assert all(scan.energy_transfer and scan.energy_transfer.value.value == 0.0 for scan in known_answer)


def test_ansto_taipan_candidate_a_expected_behavior_matches_fixture(tmp_path: Path) -> None:
    expected = yaml.safe_load((FIXTURE / "expected_behavior.yaml").read_text(encoding="utf-8"))["expected"]
    db_path = tmp_path / "ansto_candidate_a.sqlite"
    import_jsonl_catalog_dir(FIXTURE, db_path)
    scans = {scan.scan_id: scan for scan in SQLiteDataCatalog(db_path).list_scans()}

    assert expected["paper_linkage"]["accepted_link_expected"] is False
    assert set(expected["known_answer_scan_ids"]) <= set(scans)
    assert {scans[scan_id].metadata["temperature_K"] for scan_id in expected["known_answer_scan_ids"]} == set(expected["temperatures_K"])
    assert scans["taipan-1337-115553"].metadata["reflection_hkl"] == [0, 0, 2]
    assert scans["taipan-1337-115511"].metadata["reflection_hkl"] == [2, 2, 2]


def test_ansto_taipan_candidate_a_similar_scan_search_stays_with_003_family(tmp_path: Path) -> None:
    db_path = tmp_path / "ansto_candidate_a.sqlite"
    import_jsonl_catalog_dir(FIXTURE, db_path)

    results = retrieve_similar_scans("taipan-1337-115529", limit=3, catalog=db_path)
    result_ids = [item["scan_id"] for item in results]

    assert "taipan-1337-115544" in result_ids
    assert "taipan-1337-115546" in result_ids
    assert "taipan-1337-115553" not in result_ids
