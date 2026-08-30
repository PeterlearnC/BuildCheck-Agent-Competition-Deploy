"""Structured models for deterministic standards-document parsing."""

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.ocr import (
    OCRProviderIdentity,
    OCRQualityState,
    OCRRenderMetadata,
    OCRSourceMapping,
    OfficialSourceBinding,
    SourceKind,
)
from app.schemas.structural import DocumentRegionType, ValidatedStructuralEvidence


class StandardParseStatus(str, Enum):
    UPLOADED = "UPLOADED"
    PARSED = "PARSED"
    PARSE_FAILED = "PARSE_FAILED"
    OCR_REQUIRED = "OCR_REQUIRED"


class DocumentCapabilityStatus(str, Enum):
    TEXT_READY = "TEXT_READY"
    TEXT_INSUFFICIENT = "TEXT_INSUFFICIENT"
    OCR_REQUIRED = "OCR_REQUIRED"
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"


class StandardIdentityStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    IDENTITY_UNCERTAIN = "IDENTITY_UNCERTAIN"


class ArticleType(str, Enum):
    NORMATIVE = "NORMATIVE"
    EXPLANATION = "EXPLANATION"
    APPENDIX = "APPENDIX"
    REPEALED = "REPEALED"
    NOTE = "NOTE"
    OTHER = "OTHER"


class MetadataConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class ArticleParseConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class StandardPage(BaseModel):
    page_number: int
    text: str
    image_count: int = 0
    image_coverage_ratio: float | None = None
    source_kind: SourceKind = SourceKind.PDF_TEXT
    ocr_run_id: str | None = None
    ocr_execution_id: str | None = None
    ocr_quality_state: OCRQualityState | None = None
    ocr_provider: OCRProviderIdentity | None = None
    ocr_render: OCRRenderMetadata | None = None
    source_mappings: list[OCRSourceMapping] = Field(default_factory=list)

    @field_validator("page_number")
    @classmethod
    def page_number_is_one_based(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Standard page numbers must be 1-based.")
        return value

    @field_validator("image_count")
    @classmethod
    def image_count_is_nonnegative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Page image_count cannot be negative.")
        return value

    @field_validator("image_coverage_ratio")
    @classmethod
    def image_coverage_is_bounded(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("Page image_coverage_ratio must be between 0 and 1.")
        return value

    @model_validator(mode="after")
    def ocr_pages_require_accepted_provenance(self):
        if self.source_kind == SourceKind.OCR_TEXT:
            if not self.ocr_run_id or not self.ocr_execution_id:
                raise ValueError("OCR_TEXT pages require OCR run and execution identities.")
            if self.ocr_quality_state != OCRQualityState.OCR_ACCEPTED:
                raise ValueError("Only OCR_ACCEPTED pages may become StandardPage input.")
            if self.ocr_provider is None or self.ocr_render is None:
                raise ValueError("OCR_TEXT pages require provider and render provenance.")
            if not self.source_mappings:
                raise ValueError("OCR_TEXT pages require raw source mappings.")
        return self


class StandardDocument(BaseModel):
    standard_id: str
    standard_code: str | None = None
    standard_name: str | None = None
    canonical_standard_code: str | None = None
    standard_display_code: str | None = None
    identity_status: StandardIdentityStatus = StandardIdentityStatus.IDENTITY_UNCERTAIN
    identity_confidence: MetadataConfidence = MetadataConfidence.UNKNOWN
    identity_reason: str | None = None
    edition: str | None = None
    publish_date: date | None = None
    effective_date: date | None = None
    source_filename: str
    source_checksum: str
    page_count: int
    parse_status: StandardParseStatus = StandardParseStatus.UPLOADED
    metadata_confidence: MetadataConfidence = MetadataConfidence.UNKNOWN
    parse_error: str | None = None
    parser_version: str | None = None
    corpus_semantics_version: str | None = None
    capability_status: DocumentCapabilityStatus | None = None
    capability_reason: str | None = None
    source_kind: SourceKind = SourceKind.PDF_TEXT
    ocr_run_id: str | None = None
    ocr_execution_id: str | None = None
    ocr_quality_state: OCRQualityState | None = None
    ocr_provider: OCRProviderIdentity | None = None
    official_source_binding: OfficialSourceBinding | None = None
    ocr_corpus_semantics_version: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("page_count")
    @classmethod
    def page_count_is_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Standard page_count must be positive.")
        return value


class StandardChapter(BaseModel):
    chapter_id: str
    standard_id: str
    chapter_number: str | None = None
    chapter_title: str
    level: int
    source_page_start: int | None = None
    source_page_end: int | None = None
    source_text: str | None = None
    parent_chapter_id: str | None = None
    structural_evidence: ValidatedStructuralEvidence | None = None

    @field_validator("level")
    @classmethod
    def level_is_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Chapter level must be positive.")
        return value

    @field_validator("source_page_start", "source_page_end")
    @classmethod
    def optional_pages_are_one_based(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("Chapter page numbers must be 1-based.")
        return value

    @model_validator(mode="after")
    def page_range_is_ordered(self):
        if (
            self.source_page_start is not None
            and self.source_page_end is not None
            and self.source_page_end < self.source_page_start
        ):
            raise ValueError("Chapter page range is reversed.")
        return self


class StandardArticle(BaseModel):
    article_id: str
    standard_id: str
    chapter_id: str | None = None
    chapter_number: str | None = None
    chapter_title: str | None = None
    article_number: str
    article_title: str | None = None
    content: str
    source_page_start: int
    source_page_end: int
    source_text: str
    parent_article_number: str | None = None
    sequence: int
    is_mandatory: bool | None = None
    parse_confidence: ArticleParseConfidence
    article_type: ArticleType = ArticleType.OTHER
    region_type: DocumentRegionType = DocumentRegionType.OTHER
    source_kind: SourceKind = SourceKind.PDF_TEXT
    ocr_run_id: str | None = None
    ocr_execution_id: str | None = None
    ocr_quality_state: OCRQualityState | None = None
    ocr_provider: OCRProviderIdentity | None = None
    ocr_render_metadata: list[OCRRenderMetadata] = Field(default_factory=list)
    source_mappings: list[OCRSourceMapping] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None

    @field_validator("article_number", "content", "source_text")
    @classmethod
    def required_text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Article number, content, and source_text are required.")
        return value

    @field_validator("source_page_start", "source_page_end", "sequence")
    @classmethod
    def article_numbers_are_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Article pages and sequence must be 1-based.")
        return value

    @model_validator(mode="after")
    def article_page_range_is_ordered(self):
        if self.source_page_end < self.source_page_start:
            raise ValueError("Article page range is reversed.")
        if self.source_kind == SourceKind.OCR_TEXT:
            if not self.ocr_run_id or not self.ocr_execution_id:
                raise ValueError("OCR_TEXT articles require OCR run provenance.")
            if self.ocr_quality_state != OCRQualityState.OCR_ACCEPTED:
                raise ValueError("Only OCR_ACCEPTED text may become a StandardArticle.")
            if not self.source_mappings:
                raise ValueError("OCR_TEXT articles require raw OCR source mappings.")
        return self


class StandardMetadataResult(BaseModel):
    standard_code: str | None = None
    standard_name: str | None = None
    canonical_standard_code: str | None = None
    display_standard_code: str | None = None
    identity_status: StandardIdentityStatus = StandardIdentityStatus.IDENTITY_UNCERTAIN
    identity_confidence: MetadataConfidence = MetadataConfidence.UNKNOWN
    identity_reason: str | None = None
    edition: str | None = None
    publish_date: date | None = None
    effective_date: date | None = None
    confidence: MetadataConfidence = MetadataConfidence.UNKNOWN


class DocumentCapabilityResult(BaseModel):
    page_count: int
    raw_text_char_count: int
    effective_text_char_count: int
    pages_with_meaningful_text: int
    pages_with_images: int
    meaningful_text_page_ratio: float
    image_page_ratio: float
    image_heavy_page_ratio: float = 0.0
    mean_image_coverage_ratio: float = 0.0
    repeated_text_ratio: float
    status: DocumentCapabilityStatus
    reason: str


class DocumentRegion(BaseModel):
    region_type: DocumentRegionType
    source_page_start: int
    source_page_end: int
    start_line_index: int
    end_line_index: int


class StandardParseResult(BaseModel):
    document: StandardDocument
    pages: list[StandardPage]
    chapters: list[StandardChapter]
    articles: list[StandardArticle]


class StandardUploadResponse(BaseModel):
    success: bool = True
    standard_id: str
    source_filename: str
    page_count: int
    parse_status: StandardParseStatus


class StandardParseSummary(BaseModel):
    chapters: int
    articles: int


class StandardParseResponse(BaseModel):
    success: bool = True
    standard_id: str
    cached: bool = False
    metadata: StandardDocument
    summary: StandardParseSummary


class StandardDetailResponse(BaseModel):
    success: bool = True
    standard: StandardDocument
    article_count: int


class StandardListResponse(BaseModel):
    success: bool = True
    standards: list[StandardDetailResponse]


class StandardRegistryStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    UNKNOWN = "UNKNOWN"


class StandardRegistryEntry(BaseModel):
    standard_code: str
    standard_name: str
    discipline: str | None = None
    status: StandardRegistryStatus = StandardRegistryStatus.UNKNOWN
    effective_date: date | None = None
    jurisdiction: str | None = None


class StandardRegistryResponse(BaseModel):
    success: bool = True
    entries: list[StandardRegistryEntry]


class StandardArticlesResponse(BaseModel):
    success: bool = True
    standard_id: str
    total: int
    limit: int
    offset: int
    articles: list[StandardArticle]
