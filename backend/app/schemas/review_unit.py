"""Source-grounded plan review units for deterministic compliance orchestration."""

import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.standards_scope import StandardScope


class ReviewUnit(BaseModel):
    """An exact, verified span from one physical page of a stored plan PDF."""

    model_config = ConfigDict(frozen=True)

    review_unit_id: str
    document_id: str
    page_number: int = Field(gt=0)
    source_text: str
    source_text_sha256: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    review_text: str
    retrieval_query: str
    standard_scope: StandardScope

    @field_validator("source_text", "review_text", "retrieval_query")
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Review-unit text fields must not be blank.")
        return value

    @model_validator(mode="after")
    def source_identity_is_consistent(self):
        if self.char_end != self.char_start + len(self.source_text):
            raise ValueError("Review-unit source span does not match source_text length.")
        expected_hash = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if self.source_text_sha256 != expected_hash:
            raise ValueError("Review-unit source_text_sha256 does not match source_text.")
        if self.review_text != self.source_text:
            raise ValueError("C.1 review_text must preserve the exact grounded source_text.")
        if not self.standard_scope.standard_ids:
            raise ValueError("Compliance review requires an explicit standard scope.")
        return self
