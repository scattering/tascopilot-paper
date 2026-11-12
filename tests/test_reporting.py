import json

from tas_copilot_core.reporting import write_report_bundle


def test_write_report_bundle_writes_json_and_markdown(tmp_path):
    report = {"workflow_name": "fixture", "status": "ok", "value": 1}

    artifacts = write_report_bundle(
        tmp_path,
        report,
        markdown_renderer=lambda payload: f"# {payload['workflow_name']}\n\n{payload['status']}\n",
    )

    assert artifacts == {"json": str(tmp_path / "report.json"), "markdown": str(tmp_path / "report.md")}
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == report
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# fixture")


def test_write_report_bundle_supports_custom_filenames_and_excludes_artifacts_key(tmp_path):
    report = {"workflow_name": "fixture", "status": "ok", "artifacts": {"old": "path"}}

    artifacts = write_report_bundle(
        tmp_path,
        report,
        markdown_renderer=lambda payload: "# Report\n",
        json_filename="custom.json",
        markdown_filename="custom.md",
    )

    assert artifacts == {"json": str(tmp_path / "custom.json"), "markdown": str(tmp_path / "custom.md")}
    payload = json.loads((tmp_path / "custom.json").read_text(encoding="utf-8"))
    assert "artifacts" not in payload
