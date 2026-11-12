import yaml
import importlib

from tas_copilot_core.workflows.catalog_tas_batch import catalog_tas_batch


def test_catalog_tas_batch_runs_catalog_and_quality(tmp_path, monkeypatch):
    manifest = tmp_path / "matrix.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "defaults": {"max_files": 5},
                "directories": [
                    {"label": "bt7", "path": "/box/bt7", "db_path": str(tmp_path / "bt7.sqlite")},
                    {"label": "murr", "path": "/box/murr", "db_path": str(tmp_path / "murr.sqlite"), "max_files": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_catalog(arguments):
        calls.append(arguments)
        return {
            "status": "success",
            "db_path": arguments["db_path"],
            "imported": {"experiments": 1, "scans": arguments["max_files"], "data_files": arguments["max_files"]},
        }

    def fake_quality(db_path):
        return {"status": "ok", "db_path": str(db_path), "warnings": []}

    batch_module = importlib.import_module("tas_copilot_core.workflows.catalog_tas_batch")
    monkeypatch.setattr(batch_module, "catalog_tas_directory", fake_catalog)
    monkeypatch.setattr(batch_module, "data_catalog_quality_report", fake_quality)

    result = catalog_tas_batch(manifest)

    assert result["status"] == "ok"
    assert result["result_count"] == 2
    assert result["ok_count"] == 2
    assert calls[0]["max_files"] == 5
    assert calls[1]["max_files"] == 2
    assert result["results"][0]["quality_status"] == "ok"


def test_catalog_tas_batch_warns_on_quality_warning(tmp_path, monkeypatch):
    manifest = tmp_path / "matrix.yaml"
    manifest.write_text(
        yaml.safe_dump({"directories": [{"label": "bt4", "path": "/box/bt4", "db_path": str(tmp_path / "bt4.sqlite")}]}),
        encoding="utf-8",
    )

    batch_module = importlib.import_module("tas_copilot_core.workflows.catalog_tas_batch")
    monkeypatch.setattr(batch_module, "catalog_tas_directory", lambda arguments: {"status": "success", "db_path": arguments["db_path"]})
    monkeypatch.setattr(batch_module, "data_catalog_quality_report", lambda db_path: {"status": "warning", "warnings": ["null native scan ids"]})

    result = catalog_tas_batch(manifest)

    assert result["status"] == "warning"
    assert result["warning_count"] == 1


def test_catalog_tas_batch_requires_directories(tmp_path):
    manifest = tmp_path / "empty.yaml"
    manifest.write_text(yaml.safe_dump({"directories": []}), encoding="utf-8")

    result = catalog_tas_batch(manifest)

    assert result["status"] == "missing_input"
    assert result["errors"][0]["code"] == "missing_directories"
