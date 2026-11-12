from __future__ import annotations

from tas_copilot_contracts import TASConcept


def match_concepts(text: str, concepts: list[TASConcept]) -> list[TASConcept]:
    lowered = text.lower()
    matches = []
    for concept in concepts:
        if any(keyword.lower() in lowered for keyword in concept.keywords):
            if not any(keyword.lower() in lowered for keyword in concept.negative_keywords):
                matches.append(concept)
    return matches

