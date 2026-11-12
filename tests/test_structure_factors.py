import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from tas_copilot_contracts import AlignmentPeakCandidate, AlignmentPeakSuggestionReport, StructureDatabaseCandidate
from tas_copilot_core.structure_factors import _background_peak_overlap_warnings, _plane_basis_hkls, _plane_status, _reciprocal_basis, _reciprocal_metric_from_basis, _metric_norm, _parse_formula, suggest_alignment_peaks


def _write_synthetic_icsd(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table ICSD (
            IDNUM integer,
            COLL_CODE integer,
            YEAR integer,
            SGR text,
            SGR_NUM integer,
            A_LEN real,
            B_LEN real,
            C_LEN real,
            ALPHA real,
            BETA real,
            GAMMA real,
            R_VAL real,
            SUM_FORM text,
            STRUCT_FORM text,
            CHEM_NAME text
        )
        """
    )
    conn.execute(
        """
        create table P_RECORD (
            IDNUM integer,
            P_SEQ integer,
            ATOM_SYMBOL text,
            W_MULT integer,
            W_LETT text,
            X_VAL real,
            Y_VAL real,
            Z_VAL real,
            SOF_VAL real
        )
        """
    )
    conn.execute(
        """
        create table SYMMETRY_RECORDS (
            SGR text,
            SYM_SEQ integer,
            T1 real,
            R11 integer,
            R12 integer,
            R13 integer,
            T2 real,
            R21 integer,
            R22 integer,
            R23 integer,
            T3 real,
            R31 integer,
            R32 integer,
            R33 integer
        )
        """
    )
    conn.execute(
        "insert into ICSD values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (1, 1, 2020, "P1", 1, 4.05, 4.05, 4.05, 90.0, 90.0, 90.0, 0.02, "Al1", "Al1", "synthetic aluminum"),
    )
    conn.execute("insert into P_RECORD values (?,?,?,?,?,?,?,?,?)", (1, 1, "Al", 1, "a", 0.0, 0.0, 0.0, 1.0))
    conn.execute(
        "insert into SYMMETRY_RECORDS values (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("P1", 1, 0.0, 1, 0, 0, 0.0, 0, 1, 0, 0.0, 0, 0, 1),
    )
    conn.commit()
    conn.close()


def test_alignment_peak_contracts_validate():
    structure = StructureDatabaseCandidate(candidate_id="icsd:1", source_database="ICSD", source_record_id="1", formula="Al")
    peak = AlignmentPeakCandidate(peak_id="hkl:1,0,0", h=1, k=0, l=0, d_spacing=4.05, q_inv_angstrom=1.551, two_theta_deg=35.0, scattering_plane_status="in_plane")
    report = AlignmentPeakSuggestionReport(status="success", formula="Al", selected_structure=structure, alignment_peaks=[peak])

    assert report.workflow_name == "suggest-alignment-peaks"
    assert report.alignment_peaks[0].h == 1


def test_simple_decimal_formula_fallback_handles_lsmo_and_icsd_sum_form():
    assert _parse_formula("La0.7Sr0.3MnO3") == {"La": 0.7, "Sr": 0.3, "Mn": 1.0, "O": 3.0}
    assert _parse_formula("La0.7 Mn1 O3 Sr0.3") == {"La": 0.7, "Mn": 1.0, "O": 3.0, "Sr": 0.3}


def test_explicit_scattering_plane_uses_reciprocal_metric_for_out_of_plane_angle():
    basis = _reciprocal_basis((3.0, 4.0, 5.0, 90.0, 100.0, 120.0))
    metric = _reciprocal_metric_from_basis(basis)
    plane = "[1,0,0];[0,1,0]"

    assert _plane_basis_hkls(plane) == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

    q_in = _metric_norm((1, 1, 0), metric)
    status_in, angle_in = _plane_status((1, 1, 0), basis, plane, q_in, 2.0, 1.0, 5.0)
    assert status_in == "in_plane"
    assert angle_in == pytest.approx(0.0, abs=1.0e-5)

    q_near = _metric_norm((6, 0, 1), metric)
    status_near, angle_near = _plane_status((6, 0, 1), basis, plane, q_near, 2.0, 1.0, 5.0)
    assert status_near == "near_plane_projection_candidate"
    assert 2.0 < angle_near <= 5.0

    q_out = _metric_norm((1, 0, 1), metric)
    status_out, angle_out = _plane_status((1, 0, 1), basis, plane, q_out, 2.0, 1.0, 5.0)
    assert status_out == "out_of_plane"
    assert angle_out > 5.0


def test_background_peak_overlap_warnings_use_gsasii_peak_tables():
    warnings = _background_peak_overlap_warnings(
        {"two_theta_deg": 39.2},
        [
            {
                "label": "Aluminum",
                "reflections": [
                    {"material_label": "Aluminum", "hkl": [1, 1, 1], "two_theta_deg": 39.0, "relative_structure_factor": 100.0},
                    {"material_label": "Copper", "hkl": [2, 0, 0], "two_theta_deg": 44.0, "relative_structure_factor": 80.0},
                ],
            }
        ],
        tolerance_deg=0.5,
    )

    assert warnings == ["GSAS-II background overlap candidate: Aluminum (1 1 1) at two-theta 39.000 deg, delta 0.200 deg."]


def test_suggest_alignment_peaks_missing_db_fails_closed(tmp_path):
    output_dir = tmp_path / "missing"
    result = suggest_alignment_peaks({"formula": "Al", "output_dir": str(output_dir)})

    assert result["status"] == "missing_input"
    assert "icsd_db" in result["missing_inputs"]
    assert (output_dir / "alignment_peak_report.json").exists()
    assert (output_dir / "alignment_peak_report.md").exists()


def test_suggest_alignment_peaks_requires_runnable_gsasii_backend(tmp_path):
    db_path = tmp_path / "synthetic_icsd.sqlite"
    output_dir = tmp_path / "peaks"
    _write_synthetic_icsd(db_path)

    result = suggest_alignment_peaks(
        {
            "formula": "Al",
            "icsd_db": str(db_path),
            "scattering_plane": "HK0",
            "energy_mev": 14.7,
            "top_k": 5,
            "gsasii_path": str(tmp_path / "missing_gsasii"),
            "gsasii_python": sys.executable,
            "output_dir": str(output_dir),
        }
    )

    assert result["status"] == "unavailable"
    assert result["selected_structure"]["source_record_id"] == "1"
    assert not result["alignment_peaks"]
    assert "gsasii_structure_factor_backend" in result["missing_inputs"]
    assert (output_dir / "alignment_peak_report.json").exists()
    assert (output_dir / "alignment_peak_report.md").exists()
    assert (output_dir / "candidate_structures.csv").exists()
    assert (output_dir / "alignment_peak_table.csv").exists()
    assert (output_dir / "background_peak_table.csv").exists()

    saved = json.loads((output_dir / "alignment_peak_report.json").read_text(encoding="utf-8"))
    assert saved["workflow_name"] == "suggest-alignment-peaks"
    assert saved["background_materials"][0]["label"] == "Aluminum"
    assert saved["artifact_paths"]["alignment_peak_table_csv"].endswith("alignment_peak_table.csv")
    assert any("gsasii_import_failed" in item or "No module named" in item for item in saved["warnings"])


def test_space_group_number_filter_is_authoritative_over_symbol_string(tmp_path):
    db_path = tmp_path / "synthetic_icsd.sqlite"
    output_dir = tmp_path / "peaks"
    _write_synthetic_icsd(db_path)

    result = suggest_alignment_peaks(
        {
            "formula": "Al",
            "icsd_db": str(db_path),
            "space_group": "P 1",
            "space_group_number": 1,
            "gsasii_path": str(tmp_path / "missing_gsasii"),
            "output_dir": str(output_dir),
        }
    )

    assert result["selected_structure"]["source_record_id"] == "1"
    assert result["selected_structure"]["space_group"] == "P1"
    assert result["selected_structure"]["space_group_number"] == 1


def test_suggest_alignment_peaks_uses_gsasii_runtime_when_available(tmp_path):
    if not os.environ.get("TAS_GSASII_RUNTIME"):
        pytest.skip("set TAS_GSASII_RUNTIME to opt in to the external GSAS-II runtime")
    db_path = tmp_path / "synthetic_icsd.sqlite"
    output_dir = tmp_path / "peaks"
    _write_synthetic_icsd(db_path)

    result = suggest_alignment_peaks(
        {
            "formula": "Al",
            "icsd_db": str(db_path),
            "scattering_plane": "HK0",
            "energy_mev": 14.7,
            "top_k": 5,
            "output_dir": str(output_dir),
        }
    )

    assert result["status"] == "success"
    assert result["alignment_peaks"]
    assert result["alignment_peaks"][0]["structure_factor_status"] == "calculated_gsasii_neutron_nuclear"
    assert result["alignment_peaks"][0]["relative_structure_factor"] is not None
    assert result["alignment_peaks"][0]["provenance"][0]["method"] == "GSASIIstrMath.StructureFactor2"
    assert result["background_peak_tables"]
    assert result["background_peak_tables"][0]["label"] == "Aluminum"
    assert any("GSAS-II background overlap candidate" in warning for peak in result["alignment_peaks"] for warning in peak["background_warnings"])


def test_parse_lattice_argument_and_matching() -> None:
    from tas_copilot_core.structure_factors import _lattice_matches, _parse_lattice_argument

    exp = _parse_lattice_argument("5.99022 5.99022 17.0073 90 90 120")
    assert exp == (5.99022, 5.99022, 17.0073, 90.0, 120.0) or exp[5] == 120.0
    # alpha-RuCl3 honeycomb cell matches; beta-RuCl3 (c~5.65) must not.
    assert _lattice_matches(exp, {"a": 5.96, "b": 5.96, "c": 17.2, "alpha": 90, "beta": 90, "gamma": 120})
    assert not _lattice_matches(exp, {"a": 6.02, "b": 6.02, "c": 5.65, "alpha": 90, "beta": 90, "gamma": 120})
    # Missing cell values fail closed.
    assert not _lattice_matches(exp, {"a": None, "b": 5.96, "c": 17.2, "alpha": 90, "beta": 90, "gamma": 120})
    assert _parse_lattice_argument("not a lattice") is None
    assert _parse_lattice_argument(None) is None
