"""Typed, source-grounded structural evidence contracts."""

from enum import Enum
import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.ocr import SourceKind


class DocumentRegionType(str, Enum):
    FRONT_MATTER = "FRONT_MATTER"
    NORMATIVE_BODY = "NORMATIVE_BODY"
    EXPLANATION = "EXPLANATION"
    REPEALED_LIST = "REPEALED_LIST"
    APPENDIX = "APPENDIX"
    OTHER = "OTHER"


class StructuralEvidenceKind(str, Enum):
    CHAPTER = "CHAPTER"
    SECTION = "SECTION"
    APPENDIX = "APPENDIX"
    APPENDIX_SECTION = "APPENDIX_SECTION"


class StructuralEvidenceRole(str, Enum):
    STRUCTURAL_ONLY = "STRUCTURAL_ONLY"


class StructuralEvidenceProducer(str, Enum):
    OCR_TRANSITION = "OCR_TRANSITION"
    OCR_CONTEXT = "OCR_CONTEXT"
    PDF_TEXT = "PDF_TEXT"


class StructuralValidationBasis(str, Enum):
    REGION_ROLE = "REGION_ROLE"
    NUMBER_GRAMMAR = "NUMBER_GRAMMAR"
    SOURCE_ORDER = "SOURCE_ORDER"
    ARTICLE_PREFIX = "ARTICLE_PREFIX"
    PREFIX_TRANSITION = "PREFIX_TRANSITION"
    EXPLICIT_HEADING_SYNTAX = "EXPLICIT_HEADING_SYNTAX"
    OCR_TRANSITION = "OCR_TRANSITION"
    TYPOGRAPHIC_GEOMETRY = "TYPOGRAPHIC_GEOMETRY"


class StructuralSourceReference(BaseModel):
    page_number: int = Field(ge=1)
    source_order_index: int = Field(ge=0)
    raw_line_id: str | None = None
    parser_line_index: int | None = Field(default=None, ge=0)
    polygon: list[list[float]] | None = None

    @field_validator("raw_line_id")
    @classmethod
    def raw_line_id_is_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Structural raw_line_id must not be blank.")
        return value

    @field_validator("polygon")
    @classmethod
    def polygon_is_valid(
        cls, value: list[list[float]] | None
    ) -> list[list[float]] | None:
        if value is None:
            return None
        if len(value) != 4 or any(len(point) != 2 for point in value):
            raise ValueError("Structural polygons require four x/y points.")
        if any(
            not math.isfinite(float(coordinate))
            for point in value
            for coordinate in point
        ):
            raise ValueError("Structural polygon coordinates must be finite.")
        return value


class StructuralTransitionProvenance(BaseModel):
    issue_codes: list[str] = Field(default_factory=list)
    previous_article_heading_id: str | None = None
    previous_article_number: str | None = None
    next_article_heading_id: str | None = None
    next_article_number: str | None = None
    ambiguous: bool = False

    @field_validator("issue_codes")
    @classmethod
    def issue_codes_are_unique_and_not_blank(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Structural issue codes must not be blank.")
        if len(set(value)) != len(value):
            raise ValueError("Structural issue codes must be unique.")
        return value


class ValidatedStructuralEvidence(BaseModel):
    evidence_id: str = Field(pattern=r"^structure_[0-9a-f]{64}$")
    source_checksum: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    source_kind: SourceKind
    producer: StructuralEvidenceProducer
    structure_kind: StructuralEvidenceKind
    structure_number: str
    structure_title: str
    structure_level: int = Field(ge=1)
    region_role: DocumentRegionType
    evidence_role: Literal[StructuralEvidenceRole.STRUCTURAL_ONLY] = (
        StructuralEvidenceRole.STRUCTURAL_ONLY
    )
    accepted_as_normative: Literal[False] = False
    source_text: str
    normalized_text: str
    source_references: list[StructuralSourceReference]
    transition: StructuralTransitionProvenance | None = None
    validation_basis: list[StructuralValidationBasis]
    resolver_version: str

    @field_validator(
        "structure_number",
        "structure_title",
        "source_text",
        "normalized_text",
        "resolver_version",
    )
    @classmethod
    def required_text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Structural evidence text fields must not be blank.")
        return value

    @model_validator(mode="after")
    def evidence_is_source_grounded(self):
        if not self.source_references:
            raise ValueError("Validated structural evidence requires source references.")
        ordered = [
            (item.page_number, item.source_order_index)
            for item in self.source_references
        ]
        if ordered != sorted(ordered) or len(set(ordered)) != len(ordered):
            raise ValueError("Structural source references must be uniquely ordered.")
        if len({item.page_number for item in self.source_references}) != 1:
            raise ValueError("A structural heading cannot span physical pages.")
        if self.source_kind == SourceKind.OCR_TEXT and any(
            item.raw_line_id is None for item in self.source_references
        ):
            raise ValueError("OCR structural evidence requires raw line identities.")
        if self.source_kind == SourceKind.PDF_TEXT and any(
            item.parser_line_index is None for item in self.source_references
        ):
            raise ValueError("PDF-text structural evidence requires parser line indexes.")
        expected_level = {
            StructuralEvidenceKind.CHAPTER: 1,
            StructuralEvidenceKind.SECTION: 2,
            StructuralEvidenceKind.APPENDIX: 1,
            StructuralEvidenceKind.APPENDIX_SECTION: 2,
        }[self.structure_kind]
        if self.structure_level != expected_level:
            raise ValueError("Structural level does not match structural kind.")
        expected_region = (
            DocumentRegionType.APPENDIX
            if self.structure_kind
            in {
                StructuralEvidenceKind.APPENDIX,
                StructuralEvidenceKind.APPENDIX_SECTION,
            }
            else DocumentRegionType.NORMATIVE_BODY
        )
        if self.region_role != expected_region:
            raise ValueError("Structural evidence has an incompatible document region.")
        if not self.validation_basis or len(set(self.validation_basis)) != len(
            self.validation_basis
        ):
            raise ValueError("Structural validation bases must be non-empty and unique.")
        return self
