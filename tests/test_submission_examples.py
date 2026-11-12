"""Portable submission examples must not become experimental evidence."""
import importlib.util
import json
from pathlib import Path

import yaml

from tas_copilot_contracts import DataFile, Experiment, PaperExperimentLink, Scan, SourceChunk

ROOT = Path(__file__).resolve().parents[1]


def test_core_catalog_is_explicitly_synthetic_and_schema_valid():
    models = {"datafiles": DataFile, "experiments": Experiment, "mappings": PaperExperimentLink,
              "scans": Scan, "source_chunks": SourceChunk}
    for name, model in models.items():
        records = [json.loads(line) for line in (ROOT / "data/catalogs" / f"{name}.jsonl").read_text().splitlines()]
        assert records
        for record in records:
            model.model_validate(record)
            assert record["provenance"][0]["source_type"] == "fixture"
            assert record["provenance"][0]["method"] == "authored_synthetic_example"
            assert record["provenance"][0]["metadata"]["not_experimental_data"] is True
            assert "/Users/" not in json.dumps(record)
            assert "Box-Box" not in json.dumps(record)


def test_synthetic_catalog_links_remain_candidates_and_files_are_unavailable():
    mappings = [json.loads(line) for line in (ROOT / "data/catalogs/mappings.jsonl").read_text().splitlines()]
    assert all(record["status"] == "candidate" for record in mappings)
    files = [json.loads(line) for line in (ROOT / "data/catalogs/datafiles.jsonl").read_text().splitlines()]
    assert all(record["metadata"]["raw_file_supplied"] is False for record in files)
    assert all(not (ROOT / record["path"]).exists() for record in files)


def test_reviewed_registry_contains_only_the_two_historical_records():
    records = yaml.safe_load((ROOT / "configs/psd_calibration/registry.yaml").read_text())["psd_calibrations"]
    assert [r["calibration_id"] for r in records] == [
        "bt7-psd-201305-lufemno-anatrans200-reviewed",
        "bt7-psd-201407-na2cu2teo6-anatrans200-minus-sign-policy",
    ]
    assert all(r["status"] == "accepted" and r["reviewed_by"] and r["provenance"] for r in records)
    assert [r["channel_mapping"]["sign_convention"] for r in records] == ["plus_relative_a4", "minus_relative_a4"]


def test_runtime_examples_do_not_select_a_host_or_enable_network():
    for path in (ROOT / "configs/tool_runtimes").glob("*.yaml"):
        record = yaml.safe_load(path.read_text())
        assert record["read_only"] is True
        assert record["network_access"] is False
        assert "working_directory" not in record
        assert "hpc_profile" not in record
        assert "/Users/" not in path.read_text()


def test_pyspinw_checkout_default_is_portable(monkeypatch):
    monkeypatch.delenv("PYSPINW_ROOT", raising=False)
    path = ROOT / "scripts/validate_srmn2sb2_pyspinw_pr110_migration.py"
    spec = importlib.util.spec_from_file_location("portable_migration_paths", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_PYSPINW_ROOT == ROOT.parent / "pySpinW"
