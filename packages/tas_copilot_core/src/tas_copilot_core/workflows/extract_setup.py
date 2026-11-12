from __future__ import annotations

import re


def extract_setup(text: str) -> dict:
    patterns = {
        "fixed_final_energy": r"(?:fixed final energy|Ef)\s*(?:=|of)?\s*([0-9.]+)\s*meV",
        "collimation": r"collimation\s*(?:=|was)?\s*([0-9A-Za-z' -]+)",
        "filter": r"\b(PG filter|Be filter)\b",
    }
    return {name: match.group(1) for name, pattern in patterns.items() if (match := re.search(pattern, text, re.IGNORECASE))}

