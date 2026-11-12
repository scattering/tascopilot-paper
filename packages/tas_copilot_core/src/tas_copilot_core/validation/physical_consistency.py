from __future__ import annotations

from tas_copilot_contracts import ResolutionInput


def validate_resolution_input(value: ResolutionInput) -> tuple[bool, str | None]:
    if value.energy_transfer.value.unit.lower() not in {"mev", "ev"}:
        return False, "energy transfer should use an energy unit"
    return True, None

