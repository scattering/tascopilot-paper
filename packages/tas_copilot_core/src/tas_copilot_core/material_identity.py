from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from tas_copilot_contracts import CompositionParseResult, MaterialIdentityRecord, ProvenanceRecord
from tas_copilot_core.config import CoreConfig


def family_patterns_path() -> Path:
    return CoreConfig().root / "configs" / "material_identity" / "family_patterns.yaml"


def load_family_patterns(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path else family_patterns_path()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {"families": {}}


def parse_material_text(text: str, family: str = "lu_femn_o3") -> CompositionParseResult:
    """Interpret a material mention using curated TAS filename/sample-log grammar.

    This is deliberately not a general chemistry parser. General formula parsing
    belongs behind the material-identity plugin boundary.
    """

    patterns = load_family_patterns()
    family_config = (patterns.get("families") or {}).get(family)
    if not family_config:
        return CompositionParseResult(
            raw_string=text,
            parser_backend="tas_filename_interpreter",
            interpretation_method="unknown_family",
            confidence=0.0,
            warnings=[f"Unknown material family: {family}"],
        )
    raw = str(text)
    compact = re.sub(r"[^A-Za-z0-9.]", "", raw)
    for alias in family_config.get("ambiguous_aliases", []):
        if alias.lower() in compact.lower() or alias.lower() in raw.lower():
            return CompositionParseResult(
                raw_string=text,
                normalized_formula=None,
                parser_backend="tas_filename_interpreter",
                interpretation_method="ambiguous_alias",
                confidence=0.2,
                ambiguity_reason=f"{alias} does not encode the Fe/Mn ratio.",
                warnings=[
                    "Ambiguous filename/sample shorthand; need paper panel label, sample log, proposal metadata, paper-to-experiment mapping, or human review."
                ],
                provenance=[ProvenanceRecord(source_type="material_text", source_id=raw, method="family_ambiguous_alias")],
            )
    for pattern in family_config.get("filename_patterns", []):
        match = re.search(pattern["regex"], compact, flags=re.IGNORECASE)
        if not match:
            continue
        groups = match.groupdict()
        if pattern["interpretation"] == "percent_occupancy":
            fe = int(groups["fe_pct"]) / 100.0
            mn = int(groups["mn_pct"]) / 100.0
        elif pattern["interpretation"] == "fractional_occupancy":
            fe = float(groups["fe_frac"])
            mn = float(groups["mn_frac"])
        else:
            continue
        warnings: list[str] = []
        if abs((fe + mn) - 1.0) > 0.03:
            warnings.append(f"Fe/Mn site occupancy sums to {fe + mn:g}, expected about 1.")
        normalized = f"LuFe{fe:.2f}Mn{mn:.2f}O3"
        return CompositionParseResult(
            raw_string=text,
            normalized_formula=normalized,
            reduced_formula=normalized,
            anonymous_formula="ABC3",
            element_amounts={"Lu": 1.0, "Fe": fe, "Mn": mn, "O": 3.0},
            fractional_composition={"Lu": 1.0 / 4.0, "Fe": fe / 4.0, "Mn": mn / 4.0, "O": 3.0 / 4.0},
            variable_composition={"family": family_config.get("label"), "x_Mn": mn},
            x_variables={pattern.get("x_variable", "x_Mn"): mn},
            parser_backend="tas_filename_interpreter",
            interpretation_method=pattern["interpretation"],
            confidence=0.9 if not warnings else 0.75,
            warnings=warnings,
            provenance=[ProvenanceRecord(source_type="material_text", source_id=raw, method="family_filename_pattern")],
        )
    return CompositionParseResult(
        raw_string=text,
        parser_backend="tas_filename_interpreter",
        interpretation_method="no_family_match",
        confidence=0.0,
        warnings=["No curated filename/material-family pattern matched."],
        provenance=[ProvenanceRecord(source_type="material_text", source_id=raw, method="family_filename_pattern")],
    )


def parse_formula(text: str, *, preferred_backend: str = "auto") -> CompositionParseResult:
    """Parse a standard chemical formula through optional chemistry backends.

    pymatgen is the preferred canonical parser for inorganic materials. ASE is
    used as a lightweight fallback/cross-check. The final decimal fallback is
    intentionally narrow and exists only to keep common solid-solution input
    usable when pymatgen is absent and ASE rejects decimal stoichiometry.
    """

    raw = str(text or "").strip()
    compact = re.sub(r"\s+", "", raw)
    if not compact:
        return CompositionParseResult(
            raw_string=raw,
            parser_backend="unavailable",
            interpretation_method="empty_formula",
            confidence=0.0,
            warnings=["No formula text was supplied."],
        )

    warnings: list[str] = []
    if preferred_backend in {"auto", "pymatgen"}:
        parsed = _parse_formula_pymatgen(compact)
        if parsed is not None:
            return parsed
        warnings.append("pymatgen is not installed or could not parse this formula.")
        if preferred_backend == "pymatgen":
            return _unavailable_formula_result(raw, warnings)

    if preferred_backend in {"auto", "ase"}:
        parsed = _parse_formula_ase(compact)
        if parsed is not None:
            parsed.warnings.extend(warnings)
            return parsed
        warnings.append("ASE is not installed or could not parse this formula.")
        if preferred_backend == "ase":
            return _unavailable_formula_result(raw, warnings)

    simple = _parse_simple_decimal_stoichiometry(compact)
    if simple:
        total = sum(simple.values())
        order = _element_order(compact)
        return CompositionParseResult(
            raw_string=raw,
            normalized_formula=_format_formula(simple, order=order),
            reduced_formula=_format_formula(_reduce_amounts(simple), order=order),
            anonymous_formula=_anonymous_formula(simple),
            element_amounts=simple,
            fractional_composition={element: amount / total for element, amount in simple.items()} if total else {},
            parser_backend="tas_simple_decimal_fallback",
            interpretation_method="limited_decimal_stoichiometry_fallback",
            confidence=0.62,
            warnings=warnings
            + [
                "Used a narrow decimal stoichiometry fallback because canonical chemistry parsers were unavailable or rejected the string.",
                "This fallback is not a general chemistry parser.",
            ],
            provenance=[ProvenanceRecord(source_type="material_text", source_id=raw, method="limited_decimal_formula_fallback")],
        )

    return _unavailable_formula_result(raw, warnings)


def material_identity_backend_status() -> dict[str, Any]:
    return {
        "preferred_backend": "pymatgen",
        "fallback_backend": "ase",
        "pymatgen_available": _module_available("pymatgen.core"),
        "ase_available": _module_available("ase.formula"),
        "chemdataextractor_available": _module_available("chemdataextractor"),
        "filename_interpreter": "available",
        "policy": "Use pymatgen Composition when installed, ASE Formula as fallback, and narrow TAS filename grammars for beamline shorthand. Missing optional parsers fail closed or degrade with warnings.",
    }


def material_identity_record(text: str, family: str = "lu_femn_o3", source_field: str = "text") -> MaterialIdentityRecord:
    result = parse_material_text(text, family=family)
    if result.confidence <= 0.0 and not result.ambiguity_reason:
        result = parse_formula(text)
    token = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:80] or "material"
    return MaterialIdentityRecord(
        material_id=f"material-{token}",
        raw_string=text,
        source_field=source_field,
        normalized_formula=result.normalized_formula,
        reduced_formula=result.reduced_formula,
        anonymous_formula=result.anonymous_formula,
        element_amounts=result.element_amounts,
        fractional_composition=result.fractional_composition,
        variable_composition=result.variable_composition,
        x_variables=result.x_variables,
        parser_backend=result.parser_backend,
        parser_version=result.parser_version,
        interpretation_method=result.interpretation_method,
        confidence=result.confidence,
        ambiguity_reason=result.ambiguity_reason,
        warnings=result.warnings,
        provenance=result.provenance,
    )


def compare_material_identity(left: str, right: str, family: str = "lu_femn_o3") -> dict[str, Any]:
    left_record = material_identity_record(left, family=family)
    right_record = material_identity_record(right, family=family)
    left_result = left_record.model_dump(mode="json")
    right_result = right_record.model_dump(mode="json")
    left_x = left_record.x_variables.get("x_Mn")
    right_x = right_record.x_variables.get("x_Mn")
    if left_x is None or right_x is None:
        composition_match, composition_reason = _compare_element_amounts(left_result.get("element_amounts") or {}, right_result.get("element_amounts") or {})
        if composition_match is None:
            status = "ambiguous"
            score = 0.0
            reason = "At least one material string lacks a resolved composition."
        else:
            status = "match" if composition_match else "mismatch"
            score = 1.0 if composition_match else 0.0
            reason = composition_reason
    else:
        diff = abs(float(left_x) - float(right_x))
        score = max(0.0, 1.0 - diff / 0.1)
        status = "match" if diff <= 0.025 else "mismatch"
        reason = f"x_Mn difference is {diff:g}."
    return {
        "status": status,
        "score": round(score, 4),
        "reason": reason,
        "left": left_result,
        "right": right_result,
    }


def _parse_formula_pymatgen(compact: str) -> CompositionParseResult | None:
    try:
        from pymatgen.core import Composition
        import pymatgen
    except Exception:
        return None
    try:
        composition = Composition(compact)
    except Exception:
        return None
    amounts = {element.symbol: float(amount) for element, amount in composition.items()}
    total = sum(amounts.values())
    return CompositionParseResult(
        raw_string=compact,
        normalized_formula=composition.formula.replace(" ", ""),
        reduced_formula=composition.reduced_formula,
        anonymous_formula=composition.anonymized_formula,
        element_amounts=amounts,
        fractional_composition={element: amount / total for element, amount in amounts.items()} if total else {},
        parser_backend="pymatgen",
        parser_version=getattr(pymatgen, "__version__", None),
        interpretation_method="pymatgen_composition",
        confidence=0.95,
        provenance=[ProvenanceRecord(source_type="material_text", source_id=compact, method="pymatgen_composition")],
    )


def _parse_formula_ase(compact: str) -> CompositionParseResult | None:
    try:
        from ase.formula import Formula
        import ase
    except Exception:
        return None
    try:
        amounts = {element: float(amount) for element, amount in Formula(compact).count().items()}
    except Exception:
        return None
    total = sum(amounts.values())
    order = _element_order(compact)
    return CompositionParseResult(
        raw_string=compact,
        normalized_formula=_format_formula(amounts, order=order),
        reduced_formula=_format_formula(_reduce_amounts(amounts), order=order),
        anonymous_formula=_anonymous_formula(amounts),
        element_amounts=amounts,
        fractional_composition={element: amount / total for element, amount in amounts.items()} if total else {},
        parser_backend="ase",
        parser_version=getattr(ase, "__version__", None),
        interpretation_method="ase_formula",
        confidence=0.86,
        provenance=[ProvenanceRecord(source_type="material_text", source_id=compact, method="ase_formula")],
    )


def _parse_simple_decimal_stoichiometry(formula: str) -> dict[str, float]:
    if not formula or any(ch in formula for ch in "()[]{}+-"):
        return {}
    token_re = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?|\.\d+)?")
    pos = 0
    amounts: dict[str, float] = {}
    for match in token_re.finditer(formula):
        if match.start() != pos:
            return {}
        element = match.group(1)
        raw_amount = match.group(2)
        amount = float(raw_amount) if raw_amount else 1.0
        amounts[element] = amounts.get(element, 0.0) + amount
        pos = match.end()
    return amounts if pos == len(formula) else {}


def _format_formula(amounts: dict[str, float], *, order: list[str] | None = None) -> str:
    parts = []
    ordered = [element for element in (order or []) if element in amounts]
    ordered.extend(element for element in sorted(amounts) if element not in ordered)
    for element in ordered:
        amount = amounts[element]
        parts.append(element if abs(amount - 1.0) < 1.0e-12 else f"{element}{amount:g}")
    return "".join(parts)


def _element_order(formula: str) -> list[str]:
    order: list[str] = []
    for element in re.findall(r"[A-Z][a-z]?", formula):
        if element not in order:
            order.append(element)
    return order


def _reduce_amounts(amounts: dict[str, float]) -> dict[str, float]:
    positive = [amount for amount in amounts.values() if amount > 0]
    if not positive:
        return amounts
    scale = min(positive)
    return {element: amount / scale for element, amount in amounts.items()}


def _anonymous_formula(amounts: dict[str, float]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pieces = []
    for index, (_element, amount) in enumerate(sorted(amounts.items(), key=lambda item: (item[1], item[0]))):
        label = labels[index] if index < len(labels) else f"X{index}"
        pieces.append(label if abs(amount - 1.0) < 1.0e-12 else f"{label}{amount:g}")
    return "".join(pieces)


def _compare_element_amounts(left: dict[str, Any], right: dict[str, Any], *, tolerance: float = 1.0e-6) -> tuple[bool | None, str]:
    if not left or not right:
        return None, "Missing parsed element amounts."
    if set(left) != set(right):
        return False, "Element sets differ."
    ratios = []
    for element in left:
        left_value = float(left[element])
        right_value = float(right[element])
        if abs(right_value) < tolerance:
            return False, f"Element {element} has zero amount in the right composition."
        ratios.append(left_value / right_value)
    if max(ratios) - min(ratios) <= tolerance:
        return True, "Element amounts match up to an overall scale."
    return False, "Element stoichiometric ratios differ."


def _unavailable_formula_result(raw: str, warnings: list[str]) -> CompositionParseResult:
    return CompositionParseResult(
        raw_string=raw,
        parser_backend="unavailable",
        interpretation_method="optional_formula_parser_unavailable_or_failed",
        confidence=0.0,
        warnings=warnings or ["No optional formula parser was available."],
        provenance=[ProvenanceRecord(source_type="material_text", source_id=raw, method="formula_parser_unavailable")],
    )


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False
