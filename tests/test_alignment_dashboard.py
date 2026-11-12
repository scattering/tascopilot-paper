import json
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

from tas_copilot_contracts import ToolResult, ToolResultStatus

from tas_copilot_core.alignment_dashboard import alignment_dashboard_report, parse_cif_structure

NACL_CIF = """
data_NaCl
_symmetry_space_group_name_H-M   'Fm-3m'
_symmetry_Int_Tables_number      225
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
 Na1 Na 0 0 0 1
 Cl1 Cl 0.5 0.5 0.5 1
"""

FAKE_REFLECTIONS = [
    {"h": 1, "k": 1, "l": 1, "d_spacing": 3.2563, "q_inv_angstrom": 1.9296, "two_theta_deg": 31.8, "multiplicity": 8.0, "relative_structure_factor": 10.0, "structure_factor_status": "calculated_gsasii_neutron_nuclear"},
    {"h": 2, "k": 0, "l": 0, "d_spacing": 2.82, "q_inv_angstrom": 2.2281, "two_theta_deg": 36.9, "multiplicity": 6.0, "relative_structure_factor": 100.0, "structure_factor_status": "calculated_gsasii_neutron_nuclear"},
    {"h": 2, "k": 2, "l": 0, "d_spacing": 1.994, "q_inv_angstrom": 3.1511, "two_theta_deg": 53.0, "multiplicity": 12.0, "relative_structure_factor": 80.0, "structure_factor_status": "calculated_gsasii_neutron_nuclear"},
]


def _write_cif(tmp_path: Path) -> Path:
    path = tmp_path / "nacl.cif"
    path.write_text(NACL_CIF, encoding="utf-8")
    return path


def _fake_gsasii(arguments, record, *, energy_mev, a4_max_deg):
    return ToolResult(
        status=ToolResultStatus.success,
        output={"status": "success", "backend": {"name": "GSAS-II", "space_group": record.sgr}, "reflections": FAKE_REFLECTIONS, "warnings": []},
        tool_name="calculate_structure_factors",
        tool_namespace="tas.gsasii",
        tool_version="0.1.0",
    )


def test_parse_cif_structure_nacl(tmp_path: Path) -> None:
    structure = parse_cif_structure(_write_cif(tmp_path))

    assert structure["status"] == "success"
    assert structure["formula"] == "NaCl"
    assert structure["space_group_number"] == 225
    assert structure["crystal_system"] == "cubic"
    assert structure["lattice"]["a"] == pytest.approx(5.64, abs=1e-3)
    symbols = sorted(atom["symbol"] for atom in structure["atoms"])
    assert symbols == ["Cl", "Na"]


def test_parse_cif_structure_missing_file(tmp_path: Path) -> None:
    result = parse_cif_structure(tmp_path / "absent.cif")
    assert result["status"] == "missing_input"


def test_alignment_dashboard_report_with_runtime(tmp_path: Path, monkeypatch) -> None:
    import tas_copilot_core.alignment_dashboard as dashboard
    import tas_copilot_core.structure_factors as structure_factors

    monkeypatch.setattr(structure_factors, "_run_gsasii_reflections", _fake_gsasii)
    monkeypatch.setattr(dashboard, "_equivalent_reflections", _fake_equivalents)
    report = alignment_dashboard_report(
        {
            "cif_path": str(_write_cif(tmp_path)),
            "orient1": [1, 0, 0],
            "orient2": [0, 1, 0],
            "energy_mev": 14.7,
            "environment_materials": ["aluminum", "copper"],
            "output_dir": str(tmp_path / "report"),
        }
    )

    assert report["status"] == "success"
    assert report["scattering_plane"] == "[1,0,0];[0,1,0]"
    # Star representatives expand to the full set (fake star: hkl + Friedel mate).
    assert report["unique_star_count"] == len(FAKE_REFLECTIONS)
    assert report["reflection_count"] == 2 * len(FAKE_REFLECTIONS)
    expanded_hkls = {(r["h"], r["k"], r["l"]) for r in report["reflections"]}
    assert (-2, 0, 0) in expanded_hkls
    assert all("star_representative" in r for r in report["reflections"])
    suggested_ids = [peak["peak_id"] for peak in report["suggested_peaks"]]
    assert "hkl:2,0,0" in suggested_ids
    assert "hkl:2,2,0" in suggested_ids
    # (1,1,1) is out of the (100)/(010) plane and must not be suggested.
    assert "hkl:1,1,1" not in suggested_ids

    bz = report["bz_section"]
    assert len(bz["x"]) >= 3

    plane_lattice = report["plane_lattice"]
    axes = {axis["label"]: axis for axis in plane_lattice["reciprocal_axes"]}
    b_star = 2 * 3.141592653589793 / 5.64
    assert axes["a*"]["in_plane"] is True
    assert axes["a*"]["x"] == pytest.approx(b_star, rel=1e-3)
    assert abs(axes["a*"]["y"]) < 1e-6
    assert axes["b*"]["in_plane"] is True
    assert axes["b*"]["y"] == pytest.approx(b_star, rel=1e-3)
    assert axes["c*"]["out_of_plane_deg"] == pytest.approx(90.0, abs=0.1)
    tiling = plane_lattice["tiling"]
    assert tiling is not None
    t1_norm = (tiling["t1"]["x"] ** 2 + tiling["t1"]["y"] ** 2) ** 0.5
    assert t1_norm == pytest.approx(b_star, rel=1e-3)

    points = report["peak_points"]
    assert any(point["friedel_mate"] for point in points)
    strongest = max(points, key=lambda point: point["f2"])
    assert strongest["f2"] == pytest.approx(100.0)

    rings = {ring_set["material_id"]: ring_set for ring_set in report["environment_rings"]}
    assert rings["aluminum"]["status"] == "success"
    assert rings["copper"]["status"] == "success"
    assert rings["aluminum"]["rings"][0]["q_inv_angstrom"] > 0
    assert Path(report["artifacts"]["json"]).exists()


def test_alignment_dashboard_fails_closed_without_runtime(tmp_path: Path, monkeypatch) -> None:
    import tas_copilot_core.structure_factors as structure_factors

    def _unavailable(arguments, record, *, energy_mev, a4_max_deg):
        return ToolResult(
            status=ToolResultStatus.unavailable,
            errors=[],
            tool_name="calculate_structure_factors",
            tool_namespace="tas.gsasii",
            tool_version="0.1.0",
        )

    monkeypatch.setattr(structure_factors, "_run_gsasii_reflections", _unavailable)
    report = alignment_dashboard_report({"cif_path": str(_write_cif(tmp_path)), "orient1": [1, 0, 0], "orient2": [0, 1, 0]})

    assert report["status"] == "unavailable"
    assert "gsasii_structure_factor_runtime" in report["missing_inputs"]


MG_HEX_CIF = """
data_Mg
_symmetry_space_group_name_H-M   'P63/mmc'
_symmetry_Int_Tables_number      194
_cell_length_a 3.209
_cell_length_b 3.209
_cell_length_c 5.211
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
 Mg1 Mg 0.333333 0.666667 0.25 1
"""


def _fake_equivalents(arguments, structure, hkl_list, orient1, orient2):
    return {
        "status": "success",
        "backend": {"name": "GSAS-II", "method": "GSASIIspc.GenHKLf"},
        "results": [
            {
                "hkl": hkl,
                "systematically_absent": False,
                "multiplicity": 3,
                "equivalents": [
                    {"hkl": hkl, "in_plane": True, "friedel_mate": False},
                    {"hkl": [-value for value in hkl], "in_plane": True, "friedel_mate": True},
                ],
                "in_plane_equivalents": [hkl, [-value for value in hkl]],
            }
            for hkl in hkl_list
        ],
        "warnings": [],
    }


def test_vector_tools_report_hexagonal(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("rescalculator")
    import tas_copilot_core.alignment_dashboard as dashboard

    monkeypatch.setattr(dashboard, "_equivalent_reflections", _fake_equivalents)
    cif = tmp_path / "mg.cif"
    cif.write_text(MG_HEX_CIF, encoding="utf-8")

    report = dashboard.vector_tools_report(
        {
            "cif_path": str(cif),
            "orient1": [1, 0, 0],
            "orient2": [0, 1, 0],
            "v1": [1, 0, 0],
            "v2": [0, 1, 0],
        }
    )

    assert report["status"] == "success"
    # The lattice-calculator eps fix makes the hexagonal a*^b* angle exact.
    assert report["angle"]["angle_deg"] == pytest.approx(60.0, abs=1e-6)
    ortho = report["inplane_orthogonal"]
    assert ortho["orthogonal_hkl_rational"] in ([-1, 2, 0], [1, -2, 0])
    assert ortho["orthogonality_check_deg"] == pytest.approx(90.0, abs=1e-6)
    equivalents = report["equivalents"]
    assert equivalents["status"] == "success"
    # Both v1 and the rationalized orthogonal direction get equivalents.
    assert [entry["hkl"] for entry in equivalents["results"]] == [[1, 0, 0], ortho["orthogonal_hkl_rational"]]


def test_web_alignment_vector_tools_endpoint(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("rescalculator")
    from fastapi.testclient import TestClient

    import tas_copilot_core.alignment_dashboard as dashboard
    from tas_copilot_core.web_app import create_app

    monkeypatch.setattr(dashboard, "_equivalent_reflections", _fake_equivalents)
    runs = tmp_path / "runs"
    app = create_app(local_runs_dir=runs, docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    uploaded = client.post("/api/alignment/upload-cif", json={"filename": "mg.cif", "content": MG_HEX_CIF}).json()
    assert uploaded["structure"]["status"] == "success"
    report = client.post(
        "/api/alignment/vector-tools",
        json={"cif_content": MG_HEX_CIF, "orient1": [1, 0, 0], "orient2": [0, 1, 0], "v1": [1, 0, 0], "v2": [0, 0, 1]},
    ).json()

    assert report["status"] == "success"
    assert report["angle"]["angle_deg"] == pytest.approx(90.0, abs=1e-6)
    assert report["inplane_orthogonal"]["orthogonal_hkl_rational"] in ([-1, 2, 0], [1, -2, 0])

    missing = client.post("/api/alignment/vector-tools", json={"v1": [1, 0, 0]})
    assert missing.status_code == 422


def test_alignment_only_app_exposes_no_private_surfaces(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import tas_copilot_core.alignment_dashboard as dashboard
    import tas_copilot_core.structure_factors as structure_factors
    from tas_copilot_core.web_app import create_alignment_app

    monkeypatch.setattr(structure_factors, "_run_gsasii_reflections", _fake_gsasii)
    monkeypatch.setattr(dashboard, "_equivalent_reflections", _fake_equivalents)
    runs = tmp_path / "runs"
    app = create_alignment_app(local_runs_dir=runs)
    client = TestClient(app)

    page = client.get("/")
    assert page.status_code == 200
    assert "Alignment Dashboard" in page.text
    assert "planning evidence" in page.text.lower()
    # No workbench navigation leaks into the public page.
    assert 'href="/chunks"' not in page.text
    assert 'href="/review"' not in page.text
    assert client.get("/healthz").text == "ok"

    # The private workbench surfaces must not exist on this app.
    secret = runs / "secret_report.html"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("<h1>private</h1>", encoding="utf-8")
    for route in ["/chunks", "/review", "/labels", "/runs", "/ask", "/papers/paper-1", "/api/chunks/sources", "/local-runs/secret_report.html", "/api-docs", "/openapi.json"]:
        assert client.get(route).status_code == 404, route
    assert client.post("/api/ask", json={"question": "x"}).status_code == 404
    assert client.post("/api/review-labels/ingest", json={"labels": "x", "store": "y"}).status_code == 404

    # The alignment flow itself works end to end.
    uploaded = client.post("/api/alignment/upload-cif", json={"filename": "nacl.cif", "content": NACL_CIF}).json()
    assert uploaded["structure"]["status"] == "success"
    assert "cif_path" not in uploaded, "public app must not write uploads to disk"
    report = client.post(
        "/api/alignment/report",
        json={"cif_content": NACL_CIF, "orient1": [1, 0, 0], "orient2": [0, 1, 0]},
    ).json()
    assert report["status"] == "success"
    assert report["suggested_peaks"]
    leftovers = list((runs / "alignment_dashboard").rglob("*")) if (runs / "alignment_dashboard").exists() else []
    assert not leftovers, f"no user files should be written: {leftovers}"


def test_web_alignment_endpoints(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import tas_copilot_core.alignment_dashboard as dashboard
    import tas_copilot_core.structure_factors as structure_factors
    from tas_copilot_core.web_app import create_app

    monkeypatch.setattr(structure_factors, "_run_gsasii_reflections", _fake_gsasii)
    monkeypatch.setattr(dashboard, "_equivalent_reflections", _fake_equivalents)
    runs = tmp_path / "runs"
    app = create_app(local_runs_dir=runs, docs_dir=tmp_path / "missing_docs")
    client = TestClient(app)

    page = client.get("/alignment")
    assert page.status_code == 200
    assert "Brillouin Zone" in page.text

    uploaded = client.post("/api/alignment/upload-cif", json={"filename": "nacl.cif", "content": NACL_CIF}).json()
    assert uploaded["structure"]["status"] == "success"

    report = client.post(
        "/api/alignment/report",
        json={"cif_content": NACL_CIF, "orient1": [1, 0, 0], "orient2": [0, 1, 0], "environment_materials": ["aluminum"]},
    ).json()
    assert report["status"] == "success"
    assert report["suggested_peaks"]
    assert report["environment_rings"][0]["material_id"] == "aluminum"

    invalid = client.post("/api/alignment/report", json={"cif_content": "not a cif"})
    assert invalid.json()["status"] in {"error", "unavailable"}
