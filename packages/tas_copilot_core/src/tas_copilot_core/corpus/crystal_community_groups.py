"""Group papers by structural community, using crystal-communities artifacts.

Cross-paper relation judgments are only meaningful between papers about the
same subject. Exact formula matching is too brittle for that: on the 48-paper
BT-7 set it grouped 5 of 46 papers, because published formulas are written
inconsistently and doped compositions never match a catalogued entry.

This module reads the released crystal-communities outputs (an ICSD
formula -> structural community assignment, plus the community layout carrying
each community's strongest inter-community neighbor) and turns them into the
`paper_groups` mapping the evidence-graph proposer already accepts.

The crystal-communities repository is a separate, under-review codebase and is
NOT modified or imported here: only its published CSV outputs are read.
Raw ICSD data is not touched - the artifacts contain formulas, integer ids,
community labels, and distances only.

Grouping modes, from strict to permissive:

- ``basin``: same community AND both formulas inside the community basin
  (``is_in_basin``). Highest confidence, lowest recall.
- ``community`` (default): same assigned community.
- ``neighbor``: same community, or the two communities are structural
  neighbors according to the layout's ``top_neighbor`` column. Recovers
  near-misses across a community boundary using the authors' own graph
  adjacency rather than a distance metric invented here.

``max_centroid_distance`` optionally drops fringe members, whose community
assignment is weakest (the released distances run 0 to ~140, median ~1.1).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

_MATERIAL_FACT_TYPES = {"material", "composition", "sample"}
_GROUP_MODES = {"basin", "community", "neighbor"}


def canonical_formula(text: Any) -> str | None:
    """Canonical reduced formula for a candidate token, or None.

    Normalization matters more than it looks: naive string comparison matched
    5 of 46 papers where canonicalization matched 20, because the released
    assignments store pymatgen reduced formulas ("TbAlPd", "CaMn2Sb2") while
    papers write compositions in arbitrary order and spacing.

    Single-element tokens are rejected: "Mn" or "Fe" would group unrelated
    papers. Tokens that are not parseable compositions (ordinary words) are
    rejected too.
    """
    token = str(text or "").strip().rstrip(".,;:()")
    if len(token) < 2 or not token[0].isupper():
        return None
    if not re.fullmatch(r"[A-Za-z0-9.()]+", token):
        return None
    try:
        from pymatgen.core import Composition
    except Exception:
        # No pymatgen: fall back to a conservative case-preserving key so the
        # caller still works, and let the report say matching was degraded.
        return token if any(ch.isdigit() for ch in token) else None
    try:
        composition = Composition(token)
        if len(composition.elements) < 2:
            return None
        return str(composition.reduced_formula)
    except Exception:
        return None


def load_community_index(
    assignments_csv: str | Path,
    *,
    layout_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Read the released assignment (and optional layout) CSVs.

    Expects the crystal-communities columns: ``reduced_formula``,
    ``assigned_community``, ``is_in_basin``, ``nearest_centroid_distance``;
    and for the layout, ``community`` and ``top_neighbor``.
    """
    assignments_path = Path(assignments_csv).expanduser()
    if not assignments_path.exists():
        return {"status": "missing_input", "error": f"community assignments not found: {assignments_path}", "by_formula": {}, "neighbors": {}}
    by_formula: dict[str, dict[str, Any]] = {}
    with assignments_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_formula = row.get("reduced_formula") or ""
            key = canonical_formula(raw_formula) or str(raw_formula).strip()
            if not key or key in by_formula:
                continue
            try:
                distance = float(row.get("nearest_centroid_distance") or "nan")
            except ValueError:
                distance = float("nan")
            by_formula[key] = {
                "community": str(row.get("assigned_community") or ""),
                "in_basin": str(row.get("is_in_basin") or "0").strip() in {"1", "true", "True"},
                "centroid_distance": distance,
            }
    neighbors: dict[str, str] = {}
    if layout_csv:
        layout_path = Path(layout_csv).expanduser()
        if layout_path.exists():
            with layout_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    community = str(row.get("community") or "").strip()
                    neighbor = str(row.get("top_neighbor") or "").strip()
                    if community and neighbor:
                        neighbors[community] = neighbor
    return {"status": "ok", "by_formula": by_formula, "neighbors": neighbors, "assignments_csv": str(assignments_path)}


def paper_primary_formulas(
    facts: list[dict[str, Any]],
    titles: dict[str, Any] | None = None,
) -> dict[str, str]:
    """The compound each paper is *about*, one per paper.

    The title is preferred over extracted material facts. That ordering is the
    whole point: taking every formula a paper mentions pulled in precursors and
    binary oxides (MnO, CuO) and produced groups joining papers that merely
    share a reagent. Titles name the subject.

    Among several candidates the most specific wins (most elements, then
    longest), so "CaMn2Sb2" beats "Sb" from the same title.
    """
    candidates: dict[str, list[str]] = {}
    for paper_id, title in (titles or {}).items():
        found = [f for f in (canonical_formula(tok) for tok in re.findall(r"[A-Z][A-Za-z0-9.()]*", str(title or ""))) if f]
        if found:
            candidates[str(paper_id)] = found
    for fact in facts:
        paper_id = str(fact.get("paper_id") or "")
        if not paper_id or paper_id in candidates:
            continue
        if str(fact.get("fact_type") or "") not in _MATERIAL_FACT_TYPES:
            continue
        value = fact.get("value")
        text = value.get("text") if isinstance(value, dict) else value
        found = [f for f in (canonical_formula(tok) for tok in re.findall(r"[A-Z][A-Za-z0-9.()]*", str(text or ""))) if f]
        if found:
            candidates[paper_id] = found

    primary: dict[str, str] = {}
    for paper_id, found in candidates.items():
        primary[paper_id] = max(found, key=lambda formula: (_element_count(formula), len(formula)))
    return primary


def _element_count(formula: str) -> int:
    return len(re.findall(r"[A-Z][a-z]?", formula))


def build_paper_groups(
    facts: list[dict[str, Any]],
    *,
    titles: dict[str, Any] | None = None,
    index: dict[str, Any],
    mode: str = "community",
    max_centroid_distance: float | None = None,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Map paper_id -> structural group keys, plus a report of what happened.

    Returns groups suitable for the proposer's ``paper_groups`` argument. Two
    papers pair when their group keys intersect.
    """
    if mode not in _GROUP_MODES:
        raise ValueError(f"mode must be one of {sorted(_GROUP_MODES)}, got {mode!r}")
    by_formula = index.get("by_formula") or {}
    neighbors = index.get("neighbors") or {}
    primary = paper_primary_formulas(facts, titles)

    groups: dict[str, set[str]] = {}
    matched: dict[str, dict[str, Any]] = {}
    skipped_no_match = 0
    skipped_out_of_basin = 0
    skipped_far_from_centroid = 0
    for paper_id, formula in primary.items():
        record = by_formula.get(formula)
        if record is None:
            skipped_no_match += 1
            continue
        if mode == "basin" and not record["in_basin"]:
            skipped_out_of_basin += 1
            continue
        distance = record["centroid_distance"]
        if max_centroid_distance is not None and distance == distance and distance > max_centroid_distance:
            skipped_far_from_centroid += 1
            continue
        community = record["community"]
        keys = {f"community-{community}"}
        if mode == "neighbor":
            # Adjacency comes from the authors' own inter-community edges, not
            # from a similarity measure defined here. ONLY the unordered pair key
            # is emitted: a per-community "adjacent" key would also join two
            # papers whose separate communities both neighbor a third, which is a
            # two-hop link, not adjacency.
            neighbor = neighbors.get(community)
            if neighbor:
                keys.add(f"community-{min(community, neighbor)}~{max(community, neighbor)}")
        groups[paper_id] = keys
        matched[paper_id] = {"formula": formula, **record}

    shared = _shared_group_summary(groups)
    report = {
        "status": index.get("status", "ok"),
        "mode": mode,
        "max_centroid_distance": max_centroid_distance,
        "papers_with_primary_formula": len(primary),
        "papers_matched_to_community": len(matched),
        "skipped_formula_not_in_icsd": skipped_no_match,
        "skipped_out_of_basin": skipped_out_of_basin,
        "skipped_far_from_centroid": skipped_far_from_centroid,
        "groups_spanning_multiple_papers": len(shared),
        "grouped_paper_count": len({paper for papers in shared.values() for paper in papers}),
        "group_members": {key: sorted(papers) for key, papers in sorted(shared.items())},
        "matched_papers": {paper_id: matched[paper_id]["formula"] for paper_id in sorted(matched)},
        "icsd_formula_count": len(by_formula),
        "layout_neighbor_count": len(neighbors),
        "source": "crystal-communities released assignments (read-only; repo not modified)",
    }
    if index.get("error"):
        report["error"] = index["error"]
    return groups, report


def _shared_group_summary(groups: dict[str, set[str]]) -> dict[str, set[str]]:
    by_key: dict[str, set[str]] = {}
    for paper_id, keys in groups.items():
        for key in keys:
            by_key.setdefault(key, set()).add(paper_id)
    return {key: papers for key, papers in by_key.items() if len(papers) > 1}
