import json

from tas_copilot_core.cli import main


def test_ontology_vocabularies_list_json_cli(capsys):
    assert main(["ontology", "vocabularies", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert any(item["id"] == "panet" for item in payload["external_vocabularies"])


def test_ontology_mappings_validate_cli(capsys):
    assert main(["ontology", "mappings", "validate"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] in {"ok", "warning"}
    assert payload["error_count"] == 0


def test_ontology_mappings_report_cli(capsys):
    assert main(["ontology", "mappings", "report"]) == 0
    report = capsys.readouterr().out

    assert "TAS ontology external alignment report" in report
    assert "Mappings by vocabulary" in report


def test_ontology_concept_show_and_export_json_cli(capsys):
    assert main(["ontology", "concept", "show", "constant_q_scan"]) == 0
    concept_payload = json.loads(capsys.readouterr().out)
    assert concept_payload["concept"]["id"] == "constant_q_scan"
    assert concept_payload["external_alignment"]["local_concept_id"] == "constant_q_scan"

    assert main(["ontology", "export", "--format", "json"]) == 0
    export_payload = json.loads(capsys.readouterr().out)
    assert "external_vocabularies" in export_payload
    assert "alignment_report" in export_payload
