"""Curated instrument and facility aliases.

Papers name the same apparatus many ways: "BT-7", "BT7", "the BT-7 thermal
triple-axis spectrometer", "BT-7 at the NCNR". For linkage and for comparing
two extraction runs those must be one thing. The registry
(`configs/instruments/aliases.yaml`) is hand maintained; this module resolves
free text to a canonical id and never guesses: a name that is not in the
registry resolves to nothing, and the caller keeps the paper's wording.

Matching: the text and every surface form are reduced to lowercase tokens
(letters/digits only, so "BT-7" == "BT 7" == "BT7"); generic words listed
under `ignore_words` are dropped from the text; a form matches when its
token sequence occurs in the text's token sequence. The longest matching
form wins; ties go to the first listed instrument.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REGISTRY_PATH = Path(__file__).resolve().parents[4] / "configs" / "instruments" / "aliases.yaml"


@dataclass(frozen=True)
class AliasMatch:
    kind: str  # "instrument" | "facility"
    id: str
    label: str
    matched_form: str
    facility: str | None = None
    instrument_kind: str | None = None


@dataclass
class AliasRegistry:
    path: Path
    ignore_words: set[str]
    instruments: list[dict[str, Any]]
    facilities: list[dict[str, Any]]
    _forms: list[tuple[tuple[str, ...], str, dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for kind, entries in (("instrument", self.instruments), ("facility", self.facilities)):
            for entry in entries:
                for form in entry.get("forms") or []:
                    # forms are tokenized with the same ignore list as the text, so
                    # "NIST Center for Neutron Research" and the text agree on "neutron"
                    toks = _tokens(str(form), ignore=self.ignore_words)
                    if toks:
                        self._forms.append((toks, kind, entry))
        self._forms.sort(key=lambda item: -len(item[0]))

    def validate(self) -> list[str]:
        problems: list[str] = []
        seen: dict[tuple[str, ...], str] = {}
        for toks, kind, entry in self._forms:
            owner = f"{kind}:{entry['id']}"
            if toks in seen and seen[toks] != owner:
                problems.append(f"form {' '.join(toks)!r} is claimed by {seen[toks]} and {owner}")
            seen.setdefault(toks, owner)
        ids = [e["id"] for e in self.instruments] + [e["id"] for e in self.facilities]
        for dup in {i for i in ids if ids.count(i) > 1}:
            problems.append(f"duplicate id {dup!r}")
        facility_ids = {e["id"] for e in self.facilities}
        for e in self.instruments:
            if e.get("facility") and e["facility"] not in facility_ids:
                problems.append(f"instrument {e['id']} names unknown facility {e['facility']!r}")
        return problems

    def _match_all(self, text: str, kinds: tuple[str, ...]) -> list[AliasMatch]:
        toks = _tokens(text, ignore=self.ignore_words)
        if not toks:
            return []
        found: list[tuple[int, AliasMatch]] = []
        taken: set[int] = set()
        for form, kind, entry in self._forms:  # longest forms first
            if kind not in kinds:
                continue
            n = len(form)
            for i in range(len(toks) - n + 1):
                if any(j in taken for j in range(i, i + n)):
                    continue
                if tuple(toks[i:i + n]) == form:
                    taken.update(range(i, i + n))
                    found.append((i, AliasMatch(kind=kind, id=str(entry["id"]), label=str(entry["label"]), matched_form=" ".join(form),
                                                facility=entry.get("facility"), instrument_kind=entry.get("kind"))))
        # text order, one match per id
        out: list[AliasMatch] = []
        for _, m in sorted(found, key=lambda item: item[0]):
            if all(m.id != o.id for o in out):
                out.append(m)
        return out

    def instruments_in(self, text: str) -> list[AliasMatch]:
        return self._match_all(text, ("instrument",))

    def facilities_in(self, text: str) -> list[AliasMatch]:
        return self._match_all(text, ("facility",))

    def canonical_instrument(self, text: str) -> AliasMatch | None:
        hits = self.instruments_in(text)
        return hits[0] if hits else None

    def canonical_facility(self, text: str) -> AliasMatch | None:
        hits = self.facilities_in(text)
        return hits[0] if hits else None


def _tokens(text: str, *, ignore: set[str]) -> tuple[str, ...]:
    low = str(text or "").lower()
    low = low.replace("²", "2")
    words = re.findall(r"[a-z0-9]+", low)
    ignore_tokens = {w for phrase in ignore for w in re.findall(r"[a-z0-9]+", phrase.lower())} if ignore else set()
    return tuple(w for w in words if w not in ignore_tokens)


@lru_cache(maxsize=4)
def load_registry(path: str | Path | None = None) -> AliasRegistry:
    p = Path(path) if path else _REGISTRY_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AliasRegistry(
        path=p,
        ignore_words=set(str(w) for w in data.get("ignore_words") or []),
        instruments=list(data.get("instruments") or []),
        facilities=list(data.get("facilities") or []),
    )


def canonical_instrument(text: str) -> AliasMatch | None:
    return load_registry().canonical_instrument(text)


def canonical_facility(text: str) -> AliasMatch | None:
    return load_registry().canonical_facility(text)


def instruments_in(text: str) -> list[AliasMatch]:
    return load_registry().instruments_in(text)


__all__ = ["AliasMatch", "AliasRegistry", "canonical_facility", "canonical_instrument", "instruments_in", "load_registry"]
