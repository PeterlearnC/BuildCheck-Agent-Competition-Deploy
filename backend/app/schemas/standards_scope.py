"""Internal explicit standards scope contract for future compliance boundaries."""

from pydantic import BaseModel, field_validator


class StandardScope(BaseModel):
    standard_ids: list[str]
    disciplines: list[str] | None = None
    jurisdictions: list[str] | None = None

    @field_validator("standard_ids", "disciplines", "jurisdictions")
    @classmethod
    def normalize_values(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))
