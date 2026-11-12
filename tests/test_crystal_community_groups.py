from __future__ import annotations

from pathlib import Path

from tas_copilot_core.corpus.crystal_community_groups import (
    build_paper_groups,
    canonical_formula,
    load_community_index,
    paper_primary_formulas,
)


def _write_index(tmp_path: Path) -> tuple[Path, Path]:
    assignments = tmp_path / "first_report_formulas.csv"
    assignments.write_text(
        "cif_id,reduced_formula,year,assigned_community,nearest_centroid_distance,is_in_basin,A_i\n"
        # Two Fe-chalcogenides in one community: one core member, one fringe.
        "1,FeTe,1990,4894,0.5,1,1.0\n"
        "2,FeSe,1991,4894,9.9,0,1.0\n"
        # A pnictide in a different community, whose top_neighbor is 4894.
        "3,LaFeAsO,2008,5450,1.0,1,1.0\n"
        # An unrelated oxide, no adjacency to the others.
        "4,Ho2Ti2O7,1999,7777,1.0,1,1.0\n",
        encoding="utf-8",
    )
    layout = tmp_path / "community_layout.csv"
    layout.write_text(
        "community,x,y,size,intercommunity_edge_count,top_neighbor\n"
        "4894,0.0,0.0,10,5,5450\n"
        "5450,1.0,1.0,8,5,4894\n"
        "7777,9.0,9.0,4,0,\n",
        encoding="utf-8",
    )
    return assignments, layout


def test_canonical_formula_normalizes_and_rejects_non_compositions() -> None:
    # Ordering and spacing are canonicalized, which is why this beats string
    # matching against the released reduced formulas.
    # Different orderings of the same composition must agree. The exact string
    # is pymatgen's convention (it factors repeated units, e.g. "Ca(MnSb)2"),
    # so we assert consistency, not a literal form - both the released
    # assignments and paper text are canonicalized through this same function.
    assert canonical_formula("Mn2Sb2Ca") == canonical_formula("CaMn2Sb2")
    assert canonical_formula("CaMn2Sb2") is not None
    assert canonical_formula(canonical_formula("CaMn2Sb2")) == canonical_formula("CaMn2Sb2")
    # Single elements are too generic to group papers on.
    assert canonical_formula("Mn") is None
    assert canonical_formula("Fe") is None
    # Ordinary words and punctuation are not compositions.
    assert canonical_formula("Magnetic") is None
    assert canonical_formula("the") is None
    assert canonical_formula("") is None


def test_primary_formula_prefers_the_title_over_mentioned_precursors() -> None:
    """A precursor named in the methods must not become the paper's subject.

    Taking every formula a paper mentions produced groups joining papers that
    merely shared MnO or CuO as a reagent - the exact noise subject-scoping is
    meant to remove.
    """
    facts = [
        {"paper_id": "paper-a", "fact_type": "sample", "value": {"text": "prepared from CuO and MnO powders"}},
    ]
    titles = {"paper-a": "Spin waves in CaMn2Sb2 single crystals"}
    primary = paper_primary_formulas(facts, titles)
    assert primary["paper-a"] == canonical_formula("CaMn2Sb2")

    # With no title formula, material facts are the fallback rather than nothing.
    primary_fallback = paper_primary_formulas(facts, {"paper-a": "Spin waves in a honeycomb antiferromagnet"})
    assert primary_fallback["paper-a"] in {canonical_formula("CuO"), canonical_formula("MnO")}


def test_community_mode_groups_same_community_only(tmp_path: Path) -> None:
    assignments, layout = _write_index(tmp_path)
    index = load_community_index(assignments, layout_csv=layout)
    assert index["status"] == "ok"
    assert len(index["by_formula"]) == 4
    assert index["neighbors"]["4894"] == "5450"

    titles = {
        "paper-fete": "Superconductivity in FeTe",
        "paper-fese": "Magnetic order in FeSe",
        "paper-lafeaso": "Spin density wave in LaFeAsO",
        "paper-hto": "Spin ice in Ho2Ti2O7",
    }
    groups, report = build_paper_groups([], titles=titles, index=index, mode="community")
    assert report["papers_matched_to_community"] == 4
    # FeTe and FeSe share community 4894; nothing else pairs.
    assert groups["paper-fete"] & groups["paper-fese"]
    assert not (groups["paper-fete"] & groups["paper-lafeaso"])
    assert not (groups["paper-fete"] & groups["paper-hto"])
    assert report["groups_spanning_multiple_papers"] == 1
    assert report["grouped_paper_count"] == 2


def test_basin_mode_excludes_fringe_members(tmp_path: Path) -> None:
    """is_in_basin=0 means the community assignment is weak; strict mode drops it."""
    assignments, layout = _write_index(tmp_path)
    index = load_community_index(assignments, layout_csv=layout)
    titles = {"paper-fete": "FeTe study", "paper-fese": "FeSe study"}
    groups, report = build_paper_groups([], titles=titles, index=index, mode="basin")
    assert "paper-fete" in groups  # in basin
    assert "paper-fese" not in groups  # out of basin
    assert report["skipped_out_of_basin"] == 1
    assert report["groups_spanning_multiple_papers"] == 0


def test_neighbor_mode_bridges_adjacent_communities(tmp_path: Path) -> None:
    """Graded closeness uses the authors' own inter-community adjacency."""
    assignments, layout = _write_index(tmp_path)
    index = load_community_index(assignments, layout_csv=layout)
    titles = {
        "paper-fete": "FeTe study",
        "paper-lafeaso": "LaFeAsO study",
        "paper-hto": "Ho2Ti2O7 study",
    }
    groups, report = build_paper_groups([], titles=titles, index=index, mode="neighbor")
    # 4894 and 5450 are mutual top neighbors, so these now pair...
    assert groups["paper-fete"] & groups["paper-lafeaso"]
    # ...while the unconnected community still does not.
    assert not (groups["paper-fete"] & groups["paper-hto"])
    assert report["mode"] == "neighbor"


def test_centroid_distance_cutoff_and_missing_input(tmp_path: Path) -> None:
    assignments, layout = _write_index(tmp_path)
    index = load_community_index(assignments, layout_csv=layout)
    titles = {"paper-fete": "FeTe study", "paper-fese": "FeSe study"}
    groups, report = build_paper_groups([], titles=titles, index=index, mode="community", max_centroid_distance=1.0)
    assert "paper-fese" not in groups  # distance 9.9 > 1.0
    assert report["skipped_far_from_centroid"] == 1

    # A formula absent from ICSD is reported, not silently dropped.
    _, report_unmatched = build_paper_groups([], titles={"paper-x": "Order in Sr2IrO4"}, index=index, mode="community")
    assert report_unmatched["skipped_formula_not_in_icsd"] == 1

    missing = load_community_index(tmp_path / "nope.csv")
    assert missing["status"] == "missing_input"
    groups_missing, report_missing = build_paper_groups([], titles=titles, index=missing, mode="community")
    assert groups_missing == {}
    assert report_missing["status"] == "missing_input"
    assert "error" in report_missing
