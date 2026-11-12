"""Regression checks for the portable review release."""

import sys

import pytest

from plugin_paths import plugin_path
from tas_copilot_core.workflows.andie_temperature_series import TemperatureSeriesCoordinator


def test_plugin_checkouts_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("TAS_COPILOT_TEST_PLUGIN_ROOT", str(tmp_path))
    assert plugin_path("rescalculator") == tmp_path / "rescalculator"
    monkeypatch.delenv("TAS_COPILOT_TEST_PLUGIN_ROOT")
    assert plugin_path("rescalculator").parent.name == "test_plugins"


def test_gsasii_does_not_guess_personal_checkouts(monkeypatch):
    from tas_gsasii_adapter import _configure_import_paths
    monkeypatch.delenv("TAS_GSASII_PATH", raising=False)
    monkeypatch.delenv("TAS_GSASII_BIN", raising=False)
    before = list(sys.path)
    assert _configure_import_paths({}) == (None, None)
    assert sys.path == before


def test_documentation_lists_only_existing_files(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from tas_copilot_core.web_app import create_app
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "reviewer_testing.md").write_text("# Reviewer Testing\n")
    client = TestClient(create_app(local_runs_dir=tmp_path / "runs", docs_dir=docs))
    page = client.get("/documentation")
    assert "/docs/reviewer_testing.md" in page.text
    assert client.get("/docs/reviewer_testing.md").status_code == 200
    assert "student_" not in page.text
    assert "QUICKSTART.html" not in page.text


def test_posterior_quality_gate_retains_generic_coverage():
    assess = TemperatureSeriesCoordinator.assess_posterior_quality
    good = {"bumps_gelman_r": {"Tc": 1.01, "J": 1.05}, "mean_acceptance_percent": 18.0}
    assert assess(good)["accepted"] is True
    for bad in (
        {"bumps_gelman_r": {"Tc": 1.4, "J": 1.0}, "mean_acceptance_percent": 18.0},
        {"bumps_gelman_r": {"Tc": None}, "mean_acceptance_percent": 18.0},
        {"bumps_gelman_r": {"Tc": 1.0}, "mean_acceptance_percent": 1.0},
        {"mean_acceptance_percent": 18.0},
    ):
        assert assess(bad)["accepted"] is False
