from __future__ import annotations

import json
from pathlib import Path

import pytest

from tas_copilot_core.cli import main
from tas_copilot_core.instrument_calibrations import (
    VELOCITY_SELECTOR_KIND,
    default_instrument_calibrations_registry_path,
    find_instrument_calibration,
    instrument_calibrations_report,
    load_instrument_calibrations_registry,
    validate_instrument_calibrations_registry,
)
from tas_copilot_core.instrument_state import harmonic_suppression_state
from tas_copilot_core.workflows.alignment_corpus_index import alignment_corpus_index
from tas_copilot_core.workflows.alignment_state import alignment_state_machine


def _candidate_calibration(**overrides) -> dict:
    record = {
        "id": "test-velocity-selector",
        "kind": VELOCITY_SELECTOR_KIND,
        "instrument": "BT-7",
        "scope": {"valid_from": "2018-01-01"},
        "payload": {"constant_velsel_units_angstrom": 29713, "passband_width_fwhm_fraction": None},
        "status": "candidate",
        "reviewed_by": None,
    }
    record.update(overrides)
    return record


def _write_registry(path: Path, records_yaml: str) -> Path:
    path.write_text(records_yaml, encoding="utf-8")
    return path


GOOD_REGISTRY = """
instrument_calibrations:
  - id: good-velsel
    kind: velocity_selector_speed_to_passband_center
    instrument: BT-7
    scope:
      valid_from: "2018-01-01"
      valid_to: null
    payload:
      constant_velsel_units_angstrom: 29713
      passband_width_fwhm_fraction: null
    status: candidate
    reviewed_by: null
    evidence:
      - "fixture evidence"
"""


def test_checked_in_registry_has_candidate_bt7_velocity_selector_seed() -> None:
    records = load_instrument_calibrations_registry()
    assert records
    seed = records[0]
    assert seed.kind == VELOCITY_SELECTOR_KIND
    assert seed.instrument == "BT-7"
    assert seed.status == "candidate"
    assert seed.reviewed_by is None
    assert seed.payload["constant_velsel_units_angstrom"] == 29713
    assert seed.payload["passband_width_fwhm_fraction"] is None
    assert seed.scope["valid_from"] == "2018-01-01"
    assert any("fpx840982" in item for item in seed.evidence)
    assert any("fpx841082" in item for item in seed.evidence)
    validation = validate_instrument_calibrations_registry()
    assert validation["status"] == "ok"
    assert validation["record_count"] >= 1
    assert default_instrument_calibrations_registry_path().exists()


def test_registry_load_and_find_good_record(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path / "registry.yaml", GOOD_REGISTRY)

    records = load_instrument_calibrations_registry(registry)
    assert len(records) == 1
    assert records[0].id == "good-velsel"

    validation = validate_instrument_calibrations_registry(registry)
    assert validation["status"] == "ok"
    assert validation["status_counts"] == {"candidate": 1}

    found = find_instrument_calibration(kind=VELOCITY_SELECTOR_KIND, instrument="BT-7", path=registry)
    assert found is not None and found.id == "good-velsel"
    assert find_instrument_calibration(kind="he3_cell_polarization", path=registry) is None


def test_registry_unknown_kind_fails_closed(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path / "registry.yaml",
        """
instrument_calibrations:
  - id: bad-kind
    kind: flux_capacitor_charge
    instrument: BT-7
    payload: {}
    status: candidate
""",
    )

    validation = validate_instrument_calibrations_registry(registry)
    assert validation["status"] == "error"
    assert any("unknown kind" in error for error in validation["errors"])
    with pytest.raises(ValueError, match="unknown kind"):
        load_instrument_calibrations_registry(registry)
    assert find_instrument_calibration(kind=VELOCITY_SELECTOR_KIND, path=registry) is None


def test_registry_malformed_payload_fails_closed(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path / "registry.yaml",
        """
instrument_calibrations:
  - id: bad-payload
    kind: velocity_selector_speed_to_passband_center
    instrument: BT-7
    payload:
      constant_velsel_units_angstrom: -5
    status: candidate
""",
    )

    validation = validate_instrument_calibrations_registry(registry)
    assert validation["status"] == "error"
    assert any("positive number" in error for error in validation["errors"])
    assert any("passband_width_fwhm_fraction is required" in error for error in validation["errors"])
    with pytest.raises(ValueError):
        load_instrument_calibrations_registry(registry)


def test_registry_accepted_requires_reviewed_by_and_valid_status(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path / "registry.yaml",
        """
instrument_calibrations:
  - id: unreviewed-accepted
    kind: velocity_selector_speed_to_passband_center
    instrument: BT-7
    payload:
      constant_velsel_units_angstrom: 29713
      passband_width_fwhm_fraction: null
    status: accepted
  - id: bad-status
    kind: velocity_selector_speed_to_passband_center
    instrument: BT-7
    payload:
      constant_velsel_units_angstrom: 29713
      passband_width_fwhm_fraction: null
    status: mostly_fine
""",
    )

    validation = validate_instrument_calibrations_registry(registry)
    assert validation["status"] == "error"
    assert any("requires reviewed_by" in error for error in validation["errors"])
    assert any("invalid status" in error for error in validation["errors"])


def test_instrument_calibrations_cli_validate_and_list(tmp_path: Path, capsys) -> None:
    registry = _write_registry(tmp_path / "registry.yaml", GOOD_REGISTRY)

    assert main(["instrument-calibrations", "validate", "--registry", str(registry)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"

    assert main(["instrument-calibrations", "list", "--registry", str(registry), "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["record_count"] == 1
    assert listing["records"][0]["kind"] == VELOCITY_SELECTOR_KIND

    bad = _write_registry(tmp_path / "bad.yaml", "instrument_calibrations:\n  - id: x\n    kind: nope\n    instrument: BT-7\n")
    assert main(["instrument-calibrations", "validate", "--registry", str(bad)]) == 1
    report = instrument_calibrations_report(bad)
    assert report["status"] == "error"
    assert report["records"] == []
    assert "fail-closed" in report["safe_conclusion"]


def test_harmonic_suppression_filter_in() -> None:
    state = harmonic_suppression_state("IN", 12594, 14.7, _candidate_calibration())

    assert state["verdict"] == "suppressed_by_filter"
    assert state["filter_state"] == "in"
    assert state["harmonic_order"] == 2


def test_harmonic_suppression_filter_out_tuned_candidate_is_width_unknown() -> None:
    state = harmonic_suppression_state("OUT", 12594, 14.7, _candidate_calibration())

    assert state["verdict"] == "selector_tuned_passband_width_unknown"
    assert state["filter_state"] == "out"
    assert state["nominal_incident_wavelength"] == pytest.approx(2.3590, abs=1e-3)
    assert state["selector_passband_center_wavelength"] == pytest.approx(29713 / 12594, abs=1e-6)
    assert state["passband_center_mismatch_fraction"] < 0.02
    assert state["calibration_status"] == "candidate"
    assert any("candidate" in item for item in state["rationale"])


def test_harmonic_suppression_accepted_calibration_with_width_is_suppressed() -> None:
    calibration = _candidate_calibration(
        status="accepted",
        reviewed_by="instrument-scientist",
        payload={"constant_velsel_units_angstrom": 29713, "passband_width_fwhm_fraction": 0.12},
    )

    state = harmonic_suppression_state("OUT", 12594, 14.7, calibration)

    assert state["verdict"] == "selector_tuned_suppressed"
    assert state["calibration_status"] == "accepted"


def test_harmonic_suppression_accepted_without_width_stays_width_unknown() -> None:
    calibration = _candidate_calibration(status="accepted", reviewed_by="instrument-scientist")

    state = harmonic_suppression_state("OUT", 12594, 14.7, calibration)

    assert state["verdict"] == "selector_tuned_passband_width_unknown"


def test_harmonic_suppression_filter_out_untuned_or_off_is_unsuppressed_candidate() -> None:
    far_off = harmonic_suppression_state("OUT", 17382, 14.7, _candidate_calibration())
    assert far_off["verdict"] == "unsuppressed_candidate"
    assert far_off["passband_center_mismatch_fraction"] >= 0.02

    selector_off = harmonic_suppression_state("OUT", 0.0, 14.7, _candidate_calibration())
    assert selector_off["verdict"] == "unsuppressed_candidate"


def test_harmonic_suppression_missing_inputs_are_unknown() -> None:
    assert harmonic_suppression_state("OUT", 12594, 14.7, None)["verdict"] == "unknown"
    assert harmonic_suppression_state(None, 12594, 14.7, _candidate_calibration())["verdict"] == "unknown"
    assert harmonic_suppression_state("OUT", None, 14.7, _candidate_calibration())["verdict"] == "unknown"
    assert harmonic_suppression_state("OUT", 12594, None, _candidate_calibration())["verdict"] == "unknown"


_BT7_STATE_FIXTURE = "\n".join(
    [
        "#ICE           0.4.0",
        "#Filename    fpx_filter_out_selector",
        "#Date        2018-05-23 12:39:20 EDT",
        "#ExptID      24858",
        "#ExptName    SrMn2Sb2 alignment fixture",
        "#ScanType   Find Peak Scan",
        "#Scan        12 A4",
        "#Lattice     4.58 4.58 7.71 90 90 120",
        "#Orient      0 0 2 1 1 0",
        "#AnalyzerDetectorMode 2 SingDetFlat",
        "#ScanDescr   Scan:Type=6:Fixed=0:FixedE=1:CountType=Time:Counts=1.0:Range=A4=35.5374 0.2:Npts=21:DetectorType=Detector:Filename=fpx:Title=FindPeak",
        "#Columns A4 Detector Monitor Time Filtran Velsel Ei",
        "33.5374 10 100 1 OUT 12594 14.7",
        "35.5374 100 100 1 OUT 12594 14.7",
        "37.5374 12 100 1 OUT 12594 14.7",
    ]
)


def test_alignment_corpus_index_extracts_filtran_and_velsel_columns(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "fpx_filter_out_selector.bt7").write_text(_BT7_STATE_FIXTURE, encoding="utf-8")

    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index"), "include_logs": False})

    assert index["status"] == "success"
    events = [json.loads(line) for line in Path(index["artifacts"]["events_jsonl"]).read_text(encoding="utf-8").splitlines()]
    assert len(events) == 1
    event = events[0]
    assert event["harmonic_filter_translation"] == "OUT"
    assert event["velocity_selector_speed"] == 12594.0
    assert event["incident_energy_mev"] == 14.7


def test_alignment_corpus_index_without_state_columns_stays_unknown(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "fpx_plain.bt7").write_text(
        "\n".join(
            [
                "#Filename    fpx_plain",
                "#ScanType   Find Peak Scan",
                "#ScanDescr   Scan:Type=6:Range=A4=40.0 0.5:Npts=5:Title=FindPeak",
                "#Columns A4 Detector Monitor Time",
                "39.0 10 100 1",
            ]
        ),
        encoding="utf-8",
    )

    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index"), "include_logs": False})

    events = [json.loads(line) for line in Path(index["artifacts"]["events_jsonl"]).read_text(encoding="utf-8").splitlines()]
    assert events[0]["harmonic_filter_translation"] == "unknown"
    assert events[0]["velocity_selector_speed"] is None


def test_state_machine_end_to_end_filter_out_selector_tuned_gets_candidate_prior(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "fpx_filter_out_selector.bt7").write_text(_BT7_STATE_FIXTURE, encoding="utf-8")
    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index"), "include_logs": False})

    result = alignment_state_machine(
        {
            "events_jsonl": index["artifacts"]["events_jsonl"],
            "instrument_calibrations": str(default_instrument_calibrations_registry_path()),
            "output_dir": str(tmp_path / "machine"),
        }
    )

    scores = result["lambda_half_and_background_scores"]
    assert scores["status"] == "scored"
    assert scores["filter_state_evidence"] == ["OUT"]
    assert "speed=12594" in scores["velocity_selector_evidence"]
    assert scores["velocity_selector_calibration"]["status"] == "candidate"
    assert scores["lambda_half_scores"]
    for row in scores["lambda_half_scores"]:
        assert row["status"] == "probability_scaffold"
        assert row["probability_prior"] == pytest.approx(0.10)
        assert row["calibration_status"] == "candidate"
        harmonic = row["harmonic_suppression"]
        assert harmonic["verdict"] == "selector_tuned_passband_width_unknown"
        assert harmonic["filter_state"] == "out"
        assert harmonic["passband_center_mismatch_fraction"] < 0.02
        assert any("candidate" in item for item in row["evidence"])


def test_state_machine_without_state_columns_keeps_legacy_prior(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "fpx_plain.bt7").write_text(
        "\n".join(
            [
                "#Filename    fpx_plain",
                "#ScanType   Find Peak Scan",
                "#ScanDescr   Scan:Type=6:Range=A4=40.0 0.5:Npts=5:Title=FindPeak",
                "#Columns A4 Detector Monitor Time",
                "39.0 10 100 1",
                "39.5 25 100 1",
                "40.0 100 100 1",
                "40.5 25 100 1",
                "41.0 10 100 1",
            ]
        ),
        encoding="utf-8",
    )
    index = alignment_corpus_index({"data_root": str(data_root), "output_dir": str(tmp_path / "index"), "include_logs": False})

    result = alignment_state_machine(
        {
            "events_jsonl": index["artifacts"]["events_jsonl"],
            "instrument_calibrations": str(default_instrument_calibrations_registry_path()),
            "output_dir": str(tmp_path / "machine"),
        }
    )

    scores = result["lambda_half_and_background_scores"]
    assert scores["status"] == "scored"
    assert scores["filter_state_evidence"] == []
    for row in scores["lambda_half_scores"]:
        assert row["probability_prior"] == pytest.approx(0.28)
        assert row["harmonic_suppression"] is None
        assert "base_lambda_half_prior_for_alignment_peak_review" in row["evidence"]


def test_spurion_reading_layer_extracts_typed_harmonic_state(tmp_path: Path) -> None:
    from tas_copilot_core.workflows.experiment_dashboard import _parse_scan_file

    path = tmp_path / "inelastic_filter_out.bt7"
    path.write_text(_BT7_STATE_FIXTURE, encoding="utf-8")

    record = _parse_scan_file(path, max_plot_points=100)

    assert record.metadata["harmonic_filter_translation"] == "OUT"
    assert record.metadata["velocity_selector_speed"] == 12594.0
    assert record.metadata["incident_energy_mev"] == pytest.approx(14.7)
    assert record.metadata["filter_state"]["state"] == "OUT"


def test_checked_in_registry_has_dave_bt7_monitor_correction() -> None:
    from tas_copilot_core.instrument_calibrations import (
        MONITOR_CORRECTION_KIND,
        find_instrument_calibration,
        monitor_correction_coefficients_in_energy_transfer,
        monitor_correction_factor,
    )

    record = find_instrument_calibration(kind=MONITOR_CORRECTION_KIND, instrument="BT-7")
    assert record is not None and record.status == "candidate"
    factors = monitor_correction_factor(record, [4.0, 13.5, 14.7, 34.0, 40.0])
    assert factors[0] is None  # DAVE refuses below 5 meV
    assert abs(factors[1] - 1.760) < 0.005
    assert abs(factors[2] - 1.624) < 0.005
    assert abs(factors[3] - 1.002) < 0.005
    assert factors[4] == 1.0  # DAVE sets 1.0 above 34 meV
    # re-expansion in energy transfer at fixed Ef reproduces CF(Ef + E)
    ef = 14.7
    shifted = monitor_correction_coefficients_in_energy_transfer(record, ef)
    for e in (-2.0, 0.0, 5.0, 12.0, 19.3):
        direct = monitor_correction_factor(record, ef + e)[0]
        via_e = sum(c * e**k for k, c in enumerate(shifted))
        assert abs(direct - via_e) < 1e-9


def test_monitor_correction_payload_validation(tmp_path: Path) -> None:
    from tas_copilot_core.instrument_calibrations import validate_instrument_calibrations_registry

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
instrument_calibrations:
  - id: bad
    kind: monitor_higher_order_correction_polynomial
    instrument: BT-7
    payload:
      variable: energy_transfer_mev
      coefficients: []
      valid_incident_energy_mev: [34, 5]
      operation: divide
      applies_when: {fixed_energy: Ei}
    status: candidate
""",
        encoding="utf-8",
    )
    report = validate_instrument_calibrations_registry(registry)
    assert report["status"] == "error"
    joined = " ".join(report["errors"])
    for fragment in ("variable", "coefficients", "valid_incident_energy_mev", "operation", "applies_when"):
        assert fragment in joined
