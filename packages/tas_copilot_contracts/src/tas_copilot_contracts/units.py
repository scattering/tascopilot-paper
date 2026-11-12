from __future__ import annotations

from pydantic import Field, field_validator

from .base import ContractModel


class UnitValue(ContractModel):
    value: float
    unit: str = Field(min_length=1)
    uncertainty: float | None = Field(default=None, ge=0)

    @field_validator("unit")
    @classmethod
    def unit_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("unit must not be blank")
        return cleaned

