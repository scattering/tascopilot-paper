#!/usr/bin/env python
"""Generate TASCopilot ontology concepts from the vendored MagMO ontology.

MagMO (Magnetic Materials Ontology, EMMO domain ontology, CC BY 4.0;
Hortschitz et al., arXiv:2609.11464) names the magnetic-material properties a
triple-axis paper on a magnet uses: magnetization, anisotropy constants,
exchange stiffness, coercivity conventions, susceptibilities, material and
microstructure classes. This script turns its classes into local concepts so
extracted paper facts can be tagged with them, keeping the EMMO IRI as the
cross-reference. The architecture is unchanged: these are ordinary concepts in
a generated family, not an imported reasoner graph.

Input:   configs/ontology/vendor/magmo/magnetic-materials-<version>.ttl
Overlay: configs/ontology/magmo_overlay.yaml   (local annotations that survive regeneration)
Output:  configs/ontology/magmo_generated.yaml (do not edit by hand)

    python scripts/generate_magmo_concepts.py            # regenerate
    python scripts/generate_magmo_concepts.py --check    # exit 1 if the checked-in file is stale
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef
from rdflib.namespace import SKOS

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "configs/ontology/vendor/magmo"
OVERLAY = REPO / "configs/ontology/magmo_overlay.yaml"
OUTPUT = REPO / "configs/ontology/magmo_generated.yaml"

EMMO = Namespace("https://w3id.org/emmo#")
MAGMO = Namespace("https://w3id.org/emmo/domain/magnetic-materials#")
ELUCIDATION = EMMO["EMMO_967080e5_2f42_4eb2_a3a9_c58143e835f9"]
WIKIPEDIA_REF = MAGMO["EMMO_405ae501-8898-59c0-b2db-0afa09d561f9"]
IEV_REF = MAGMO["EMMO_eb748e96-0aee-5b55-b6be-56b8c6bdaed6"]

# Classes that describe characterization data records or units, not material
# properties: those belong with instrument schemas (NeXus, CIF) and QUDT.
EXCLUDE_PREFIXES = ("Edx", "Moke", "Profil", "Xrd", "XrayDiffraction", "LocalEdx", "LocalMoke", "LocalProfilometry", "LocalXray")
EXCLUDE_EXACT = {"IECEntry", "wikidataReference", "wikipediaReference", "Reflectivity", "LocalReflectivity",
                 "AmpereSquareMeterPerKilogram", "EnergyDensityUnit", "LineEnergyUnit", "MassMagnetizationUnit"}
MATERIAL_CLASSES = {"Magnet", "MagneticMaterial", "NonMagneticMaterial", "BulkMagnet", "ThinFilmMagnet", "MultilayerMagnet",
                    "CrystallineMagneticMaterial", "AmorphousMagneticMaterial", "MainMagneticPhase", "SecondaryPhase",
                    "GrainBoundaryPhase", "SpacerLayer", "StackingSequence", "Grain", "GranularMicrostructure", "GranularStructure",
                    "SampleGeometry", "GeometricShape", "RectangularCuboid", "GeometricalSize"}
CRYSTALLOGRAPHIC = {"CrystalStructure", "SpaceGroup", "CellVolume", "CrystallographicOrientation", "EulerAngles",
                    "LatticeConstantA", "LatticeConstantB", "LatticeConstantC", "LatticeConstantAlpha", "LatticeConstantBeta",
                    "LatticeConstantGamma", "LocalLatticeConstantA", "LocalLatticeConstantC"}
# Labels so generic that substring matching would tag everything; kept as
# concepts but flagged, and their bare word is not a keyword.
BROAD = {"Magnet", "MagneticMaterial", "Grain", "GeometricShape", "GeometricalSize", "SampleGeometry", "CrystalStructure"}


def spaced(label: str) -> str:
    """CamelCase -> words: 'SpontaneousMagnetization' -> 'spontaneous magnetization'; keeps K1c-style tokens."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", label)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return text.lower()


def snake(label: str) -> str:
    return spaced(label).replace(" ", "_")


def keyword_ok(word: str) -> bool:
    """Substring matching in the linker means short or symbolic labels (Ms, K1, BHc) would fire on unrelated text."""
    w = word.strip()
    return len(w) >= 6 and " " in w or (len(w) >= 9 and w.isalpha() and w.islower())


def build(ttl: Path) -> tuple[list[dict], dict]:
    g = Graph(); g.parse(ttl, format="turtle")
    version = re.search(r"(\d+\.\d+\.\d+)", ttl.name).group(1)
    concepts = []
    stats = {"classes_in_ttl": 0, "excluded": [], "generated": 0}
    for cls in sorted(set(g.subjects(RDF.type, OWL.Class)), key=str):
        pref = next((str(o) for o in g.objects(cls, SKOS.prefLabel)), None)
        if not pref:
            continue
        stats["classes_in_ttl"] += 1
        if pref in EXCLUDE_EXACT or pref.startswith(EXCLUDE_PREFIXES):
            stats["excluded"].append(pref); continue
        alts = [str(o) for o in g.objects(cls, SKOS.altLabel)]
        eluc = " ".join(str(o).split() for o in g.objects(cls, ELUCIDATION)) if False else " ".join(" ".join(str(o).split()) for o in g.objects(cls, ELUCIDATION))
        comment = " ".join(" ".join(str(o).split()) for o in g.objects(cls, RDFS.comment))
        wiki = [str(o) for o in g.objects(cls, WIKIPEDIA_REF)]
        iev = [str(o) for o in g.objects(cls, IEV_REF)]
        if pref in MATERIAL_CLASSES:
            family = "magnetic_material_class"
        elif pref in CRYSTALLOGRAPHIC:
            family = "crystallographic_metadata"
        else:
            family = "magnetic_material_property"
        words = spaced(pref)
        keywords = [k for k in dict.fromkeys([words] + [spaced(a) if re.match(r"^[A-Z][a-z]+[A-Z]", a) else a.lower() for a in alts]) if keyword_ok(k)]
        if pref in BROAD:
            keywords = [k for k in keywords if " " in k]
        symbols = [a for a in alts if not keyword_ok(spaced(a)) and len(a) <= 8]
        description = eluc or comment or f"MagMO class {pref}."
        if len(description) > 400:
            description = description[:397].rsplit(" ", 1)[0] + "..."
        concept = {
            "id": f"magmo_{snake(pref)}", "label": words, "family": family, "description": description,
            "keywords": keywords, "expected_modalities": ["text"], "examples": [pref] + symbols,
            "avoid_if_matches_too_broadly": pref in BROAD, "retrieval_policy": "metadata_filter_only",
            "external_mappings": [{"vocabulary_id": "magmo", "mapping_type": "exactMatch", "external_label": pref, "external_iri": str(cls),
                                   "mapping_status": "reviewed", "source": f"generate_magmo_concepts.py from MagMO {version}",
                                   "reviewed_by": "generator (source class, not a judgement)", "retrieval_policy": "metadata_filter_only",
                                   "evidence": [f"MagMO {version} class {pref}"] + [f"IEV {u.split('ievref=')[-1]}" for u in iev] + wiki}],
            "xrefs": {"exactMatch": [str(cls)]},
        }
        concepts.append(concept); stats["generated"] += 1
    stats["version"] = version
    return concepts, stats


def apply_overlay(concepts: list[dict], overlay_path: Path) -> list[str]:
    if not overlay_path.exists():
        return []
    overlay = yaml.safe_load(overlay_path.read_text()) or {}
    by_id = {c["id"]: c for c in concepts}
    applied = []
    for cid, patch in (overlay.get("concepts") or {}).items():
        if cid not in by_id:
            print(f"overlay: unknown concept {cid} (ignored)", file=sys.stderr); continue
        for key, value in (patch or {}).items():
            if key in {"keywords", "negative_keywords", "examples"} and isinstance(value, dict):
                cur = list(by_id[cid].get(key, []))
                cur = [v for v in cur if v not in set(value.get("remove", []))] + [v for v in value.get("add", []) if v not in cur]
                by_id[cid][key] = cur
            else:
                by_id[cid][key] = value
        applied.append(cid)
    return applied


def render(concepts: list[dict], stats: dict, applied: list[str]) -> str:
    header = (
        "# GENERATED by scripts/generate_magmo_concepts.py -- do not edit by hand.\n"
        f"# Source: MagMO {stats['version']} (configs/ontology/vendor/magmo, CC BY 4.0; Hortschitz et al., arXiv:2609.11464).\n"
        f"# {stats['generated']} concepts from {stats['classes_in_ttl']} classes; excluded {len(stats['excluded'])} data/unit classes:\n"
        f"#   {', '.join(stats['excluded'])}\n"
        f"# Overlay applied to: {', '.join(applied) or 'none'} (configs/ontology/magmo_overlay.yaml).\n"
        "# Loaded through `includes:` in tas_bt7_seed.yaml. Retrieval policy metadata_filter_only: these tag facts, they do not expand queries.\n"
    )
    return header + yaml.safe_dump({"concepts": concepts}, sort_keys=False, allow_unicode=True, width=200)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ttl", type=Path, default=None); ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    ttl = a.ttl or sorted(VENDOR.glob("magnetic-materials-*.ttl"))[-1]
    concepts, stats = build(ttl)
    applied = apply_overlay(concepts, OVERLAY)
    text = render(concepts, stats, applied)
    if a.check:
        if OUTPUT.exists() and OUTPUT.read_text() == text:
            print("magmo_generated.yaml is current"); return 0
        print("magmo_generated.yaml is stale; rerun scripts/generate_magmo_concepts.py"); return 1
    OUTPUT.write_text(text)
    print(f"{stats['generated']} concepts from MagMO {stats['version']} -> {OUTPUT} (excluded {len(stats['excluded'])}, overlay on {len(applied)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
