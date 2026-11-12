from __future__ import annotations


def analyze_dataset(identifier: str) -> dict:
    return {"identifier": identifier, "parsed_metadata": {}, "possible_scan_type": None, "available_next_actions": ["find-experiments", "retrieve-similar-scans"]}

