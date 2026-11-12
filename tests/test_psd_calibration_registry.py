from __future__ import annotations

from pathlib import Path

from tas_copilot_core.psd_calibration import accepted_psd_calibrations, compare_psd_calibration_matches, discover_psd_dat_calibrations, elastic_q_from_two_theta, load_psd_calibration_registry, match_psd_calibration, psd_calibration_report, psd_channel_q_axis, reviewed_psd_channel_to_a4_mapping, validate_psd_calibration_registry, wavelength_from_energy_mev
from tas_copilot_core.source_triage import q_temperature_reconstruction_blocked


def test_checked_in_psd_calibration_registry_has_reviewed_lu_femn_o3_policies() -> None:
    records = load_psd_calibration_registry()
    assert records
    accepted = accepted_psd_calibrations()
    assert [record.calibration_id for record in accepted] == [
        "bt7-psd-201305-lufemno-anatrans200-reviewed",
        "bt7-psd-201407-na2cu2teo6-anatrans200-minus-sign-policy",
    ]
    policy_2013, policy_2014 = accepted
    assert policy_2013.channel_mapping["mask_policy"]["masked_one_based_channels"] == [48]
    assert policy_2013.detector_geometry["analyzer_detector_translation"] == 200
    assert policy_2013.channel_mapping["sign_convention"] == "plus_relative_a4"
    # The 2014 explicit-composition policy uses the opposite sign convention and
    # has no accepted channel mask yet; the registry must keep both scoped.
    assert policy_2014.channel_mapping["sign_convention"] == "minus_relative_a4"
    assert policy_2014.channel_mapping["channel_to_a4_policy"] == "two_theta = A4 - relative_a4[channel]"
    assert policy_2014.channel_mapping["mask_policy"]["masked_one_based_channels"] == []
    assert policy_2014.detector_geometry["analyzer_detector_translation"] == 200
    report = psd_calibration_report()
    assert report["status"] == "ok"
    assert report["accepted_count"] == 2
    assert report["q_temperature_unlock_available"] is True


def test_checked_in_accepted_registry_still_requires_q_conversion_adapter() -> None:
    result = q_temperature_reconstruction_blocked("group-1")
    assert result.status == "unsupported"
    assert "reviewed_psd_channel_to_a4_calibration" not in result.missing_fields
    assert "reviewed_q_conversion_adapter" in result.missing_fields


def test_accepted_registry_passes_gate_but_adapter_remains_required(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
psd_calibrations:
  - calibration_id: accepted-test
    instrument: BT-7
    detector_mode: PSDDiff
    status: accepted
    reviewed_by: instrument-scientist
    reviewed_at: "2026-05-06"
    channel_mapping:
      type: fixture_linear_channel_angle
    detector_geometry:
      detector: fixture
    validation_evidence:
      - evidence_type: fixture
        summary: accepted fixture calibration
        score: 1.0
""",
        encoding="utf-8",
    )

    validation = validate_psd_calibration_registry(registry)
    assert validation["status"] == "ok"
    assert accepted_psd_calibrations(registry)
    result = q_temperature_reconstruction_blocked("group-1", calibration_registry=registry)
    assert result.status == "unsupported"
    assert "reviewed_psd_channel_to_a4_calibration" not in result.missing_fields
    assert "reviewed_q_conversion_adapter" in result.missing_fields


def test_accepted_registry_requires_review_and_mapping(tmp_path: Path) -> None:
    registry = tmp_path / "bad.yaml"
    registry.write_text(
        """
psd_calibrations:
  - calibration_id: bad-accepted
    instrument: BT-7
    detector_mode: PSDDiff
    status: accepted
""",
        encoding="utf-8",
    )

    validation = validate_psd_calibration_registry(registry)
    assert validation["status"] == "error"
    assert any("reviewed_by" in error for error in validation["errors"])
    assert any("channel_mapping" in error for error in validation["errors"])


def test_psd_dat_calibration_match_requires_analyzer_distance_review(tmp_path: Path) -> None:
    calibration_dir = tmp_path / "PSD"
    calibration_dir.mkdir()
    calibration = calibration_dir / "PSD_June2013_LuFeMnO_AnaTrans50.dat"
    calibration.write_text(
        "\n".join(
            [
                "#BT7 PSD calibration based on psdcali\\LuFe75Mn25O3_Correlations102248.bt7",
                "#Created on: Mon, 03 Jun 2013 17:44:00",
                "#Channels to be masked: 1,4-5",
                "#Efficiency  A4  Ef",
                "0.659837 -3.36057 0",
                "0.843937 -3.16331 0",
            ]
        ),
        encoding="utf-8",
    )
    data_file = tmp_path / "LuFe75Mn25O3_Correlations102248.bt7"
    data_file.write_text(
        "\n".join(
            [
                "#Filename    LuFe75Mn25O3_Correlations102248",
                "#Date        2013-06-01 18:26:10 EDT",
                "#AnalyzerDetectorMode 4 PSDDiff",
                "#Fixed       A3 A5 A6 ANACOLL ANADET AnalyzerRotation Ef PSDet",
                "#ScanDescr   Scan:SubID=18445:JType=ANGLE:Npts=29:Range=A4=25.0 32.0 s",
                "#Columns A4 Temp PSDC00 PSDC01 PSDet AnalyzerRotation Ef",
                "25.0 2.5 1 2 0 90.0 13.7",
            ]
        ),
        encoding="utf-8",
    )

    discovered = discover_psd_dat_calibrations(calibration_dir)
    assert discovered["record_count"] == 1
    assert discovered["records"][0]["analyzer_detector_translation"] == 50.0
    assert discovered["records"][0]["masked_channels"] == [1, 4, 5]

    match = match_psd_calibration(calibration_dir, data_file)
    assert match["status"] == "ok"
    assert match["best_match"]["filename"] == "PSD_June2013_LuFeMnO_AnaTrans50.dat"
    assert match["best_match"]["score"] >= 0.65
    assert "analyzer_detector_translation" in match["missing_fields"]
    assert match["best_match"]["safe_use"] is False
    assert "must be confirmed" in match["safe_conclusion"]
    assert match["source_file"]["psd_channel_count"] == 2
    assert match["source_file"]["two_theta_axis_source"] == "A4"
    assert match["source_file"]["two_theta_summary"]["first"] == 25.0
    assert match["source_file"]["q_axis"] is None
    assert "AnaTrans analyzer-to-detector translation distance" in match["source_file"]["reader_check"]

    comparison = compare_psd_calibration_matches(calibration_dir, data_file, top_k=1, output_dir=tmp_path / "comparison")
    assert comparison["status"] == "ok"
    assert comparison["best_match"]["filename"] == "PSD_June2013_LuFeMnO_AnaTrans50.dat"
    assert comparison["best_match"]["safe_use"] is False
    assert comparison["best_match"]["row_count"] == 2
    assert comparison["best_match"]["relative_a4_samples"]["first"] == -3.36057
    assert comparison["best_match"]["formula_preview"]["status"] == "computed_from_candidate_mapping_not_accepted"
    assert "reviewed" in " ".join(comparison["best_match"]["formula_preview"]["warnings"])
    assert Path(comparison["artifacts"]["json"]).exists()
    assert Path(comparison["artifacts"]["markdown"]).exists()
    assert Path(comparison["artifacts"]["csv"]).exists()


def test_psd_dat_path_only_match_uses_filename_date_and_material(tmp_path: Path) -> None:
    calibration_dir = tmp_path / "PSD"
    calibration_dir.mkdir()
    (calibration_dir / "PSD_May2013_LuFeMnO_AnaTrans200.dat").write_text("", encoding="utf-8")
    (calibration_dir / "PSD_June2013_LuFeMnO_AnaTrans50.dat").write_text("", encoding="utf-8")
    (calibration_dir / "PSD_Jan2015_Al2O3_AnaTrans200.dat").write_text("", encoding="utf-8")
    data_file = tmp_path / "LuMnFeO3_29pt5_TdepWarm102246.bt7"
    data_file.write_text(
        "\n".join(
            [
                "#Filename    LuMnFeO3_29pt5_TdepWarm102246",
                "#Date        2013-06-01 12:52:46 EDT",
                "#AnalyzerDetectorMode 4 PSDDiff",
                "#Fixed       A3 A5 A6 ANACOLL ANADET AnalyzerRotation Ef PSDet",
                "#ScanDescr   Scan:SubID=18445:JType=ENV:Npts=81:Range=Temp=2.5 202.5 s",
                "#Columns A4 Temp PSDC00 PSDC01 PSDet AnalyzerRotation Ef",
                "29.5 2.5 1 2 0 90.0 13.7",
            ]
        ),
        encoding="utf-8",
    )

    discovered = discover_psd_dat_calibrations(calibration_dir, path_only=True)
    assert discovered["record_count"] == 3
    assert {record["metadata_mode"] for record in discovered["records"]} == {"path_only"}

    match = match_psd_calibration(calibration_dir, data_file, path_only=True)
    assert match["status"] == "ok"
    assert match["matching_policy"]["path_only"] is True
    assert match["best_match"]["filename"] == "PSD_June2013_LuFeMnO_AnaTrans50.dat"
    assert "calibration filename month/year matches measurement month" in match["best_match"]["evidence"]
    assert "analyzer_detector_translation" in match["missing_fields"]
    anatrans200 = next(record for record in match["matches"] if record["filename"] == "PSD_May2013_LuFeMnO_AnaTrans200.dat")
    assert "historical ICE-era prior favors AnaTrans200" in " ".join(anatrans200["evidence"])


def test_elastic_powder_q_formula_is_explicit_but_review_gated() -> None:
    wavelength = wavelength_from_energy_mev(13.7)
    q = elastic_q_from_two_theta(29.5, wavelength_angstrom=wavelength)
    assert round(wavelength, 4) == 2.4436
    assert round(q, 4) == 1.3093

    axis = psd_channel_q_axis(29.5, [-0.5, 0.0, 0.5], energy_mev=13.7)
    assert axis["status"] == "computed_from_candidate_channel_to_a4_mapping"
    assert axis["two_theta_axis"] == [29.0, 29.5, 30.0]
    assert axis["q_axis"][0] < axis["q_axis"][1] < axis["q_axis"][2]
    assert "reviewed and accepted" in " ".join(axis["warnings"])


def test_reviewed_channel_to_a4_mapping_requires_accepted_registry_and_local_dat(tmp_path: Path) -> None:
    calibration_dir = tmp_path / "PSD"
    calibration_dir.mkdir()
    (calibration_dir / "accepted.dat").write_text(
        "\n".join(
            [
                "#Created on: Mon, 03 Jun 2013 17:44:00",
                "#Efficiency  A4  Ef",
                "1.0 -0.5 0",
                "0.5 0.0 0",
                "1.5 0.5 0",
            ]
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        """
psd_calibrations:
  - calibration_id: accepted-test
    instrument: BT-7
    detector_mode: PSDDiff
    date_range:
      start: "2013-06-01"
      end: "2013-06-08"
    status: accepted
    reviewed_by: instrument-scientist
    reviewed_at: "2026-05-24"
    channel_mapping:
      type: bt7_psd_dat_relative_a4
      source_filename: accepted.dat
      row_count: 3
      sign_convention: plus_relative_a4
      channel_to_a4_policy: "two_theta = A4 + relative_a4[channel]"
      mask_policy:
        masked_one_based_channels: [2]
    detector_geometry:
      analyzer_detector_translation: 200
    validation_evidence:
      - evidence_type: fixture
        summary: accepted fixture calibration
        score: 1.0
""",
        encoding="utf-8",
    )

    result = reviewed_psd_channel_to_a4_mapping(
        source_a4_degrees=29.5,
        detector_mode="4 PSDDiff",
        source_date="2013-06-02",
        registry_path=registry,
        calibration_dirs=[calibration_dir],
        energy_mev=13.7,
    )

    assert result["status"] == "success"
    assert result["safe_use"] is True
    assert result["two_theta_axis_degrees"] == [29.0, 29.5, 30.0]
    assert result["efficiency"] == [1.0, 0.5, 1.5]
    assert result["masked_one_based_channels"] == [2]
    assert result["q_preview"]["q_axis"][0] < result["q_preview"]["q_axis"][1] < result["q_preview"]["q_axis"][2]

    mismatch = reviewed_psd_channel_to_a4_mapping(
        source_a4_degrees=29.5,
        detector_mode="5 PSDFlat",
        source_date="2013-06-02",
        registry_path=registry,
        calibration_dirs=[calibration_dir],
    )
    assert mismatch["status"] == "missing_input"
    assert mismatch["safe_use"] is False
