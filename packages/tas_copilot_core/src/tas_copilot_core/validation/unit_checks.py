from __future__ import annotations

from tas_copilot_contracts import UnitValue


def require_compatible_units(values: list[UnitValue]) -> tuple[bool, str | None]:
    units = {value.unit for value in values}
    if len(units) > 1:
        return False, f"incompatible units: {sorted(units)}"
    return True, None

