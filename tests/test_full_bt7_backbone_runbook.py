from __future__ import annotations

import subprocess
from pathlib import Path

import yaml
import pytest


def test_full_bt7_backbone_runner_writes_dry_run_bundle(tmp_path: Path) -> None:
    if not Path("scripts/run_bt7_backbone_cluster.sh").exists():
        pytest.skip("cluster deployment script is outside the paper release")
    output_root = tmp_path / "backbone"
    result = subprocess.run(
        [
            "bash",
            "scripts/run_bt7_backbone_cluster.sh",
            "--output-root",
            str(output_root),
            "--run-id",
            "pytest_backbone",
            "--bt7-data-root",
            "/tmp/not-real-bt7",
            "--rag-root",
            "/tmp/not-real-rag",
            "--psd-calibration-dir",
            "/tmp/not-real-psd",
            "--corpus-count",
            "1",
            "--max-files",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Dry run only" in result.stdout
    data_script = output_root / "commands" / "data_dashboard_spurion_psd.sh"
    corpus_script = output_root / "commands" / "corpus_extract_ingest_link.sh"
    readme = output_root / "README.md"
    assert data_script.exists()
    assert corpus_script.exists()
    assert readme.exists()
    data_text = data_script.read_text(encoding="utf-8")
    corpus_text = corpus_script.read_text(encoding="utf-8")
    assert "catalog-tas-directory" in data_text
    assert "build-bt7-hydrated-index" in data_text
    assert "survey-inelastic-spurion-evidence" in data_text
    assert "discover-psd-calibrations" in data_text
    assert "run_vista_corpus_extractors.sh" in corpus_text
    assert "COPYFILE_DISABLE=1" in corpus_text or "COPYFILE_DISABLE=1" in Path("scripts/run_bt7_backbone_cluster.sh").read_text(encoding="utf-8")
    assert "--exclude='._*'" in corpus_text
    assert "link-paper-to-experiments" in corpus_text
    subprocess.run(["bash", "-n", str(data_script)], check=True)
    subprocess.run(["bash", "-n", str(corpus_script)], check=True)


def test_full_bt7_backbone_readiness_eval_loads() -> None:
    suite = yaml.safe_load(Path("eval/full_bt7_backbone_readiness.yaml").read_text(encoding="utf-8"))
    assert suite["suite_id"] == "full_bt7_backbone_readiness"
    assert suite["fixture_manifest"] == "tests/fixtures/corpus_pilot/pilot_manifest.synthetic.yaml"
    assert any(task["name"] == "proposed_source_links_without_false_positive" for task in suite["tasks"])


def test_full_bt7_backbone_runbook_is_in_html_builder() -> None:
    if not Path("scripts/build_user_docs_html.py").exists():
        pytest.skip("development documentation builder is outside the paper release")
    builder = Path("scripts/build_user_docs_html.py").read_text(encoding="utf-8")
    assert "full_bt7_backbone_runbook.md" in builder


def test_vista_corpus_extractor_filters_macos_sidecars() -> None:
    if not Path("scripts/run_vista_corpus_extractors.sh").exists():
        pytest.skip("cluster corpus deployment is outside the paper release")
    script = Path("scripts/run_vista_corpus_extractors.sh").read_text(encoding="utf-8")
    assert "COPYFILE_DISABLE=1" in script
    assert "! -name '._*'" in script
    assert "--exclude='._*'" in script
    assert 'not pdf.name.startswith("._")' in script
