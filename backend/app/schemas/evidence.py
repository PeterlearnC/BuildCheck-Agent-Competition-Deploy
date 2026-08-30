"""Immutable provenance returned with standards retrieval results."""

from pydantic import BaseModel, Field

from app.schemas.ocr import (
    OCRProviderIdentity,
    OCRQualityState,
    OCRRenderMetadata,
    OCRSourceMapping,
    OfficialSourceBinding,
    SourceKind,
)

from app.schemas.standards import (
    ArticleType,
    DocumentRegionType,
    MetadataConfidence,
    StandardIdentityStatus,
)


class EvidenceEnvelope(BaseModel):
    id: str
    standard_id: str
    standard_code: str | None = None
    canonical_standard_code: str | None = None
    standard_name: str | None = None
    identity_status: StandardIdentityStatus
    identity_confidence: MetadataConfidence
    identity_reason: str | None = None
    official_source_binding: OfficialSourceBinding | None = None
    source_checksum: str
    article_id: str
    chapter_id: str | None = None
    article_number: str
    source_page_start: int | None = None
    source_page_end: int | None = None
    source_text: str
    parser_version: str
    retrieval_version: str
    article_type: ArticleType
    region_type: DocumentRegionType
    source_kind: SourceKind = SourceKind.PDF_TEXT
    normalized_source_text: str | None = None
    ocr_run_id: str | None = None
    ocr_execution_id: str | None = None
    ocr_quality_state: OCRQualityState | None = None
    ocr_run_quality_state: OCRQualityState | None = None
    ocr_provider: OCRProviderIdentity | None = None
    ocr_render_metadata: list[OCRRenderMetadata] = Field(default_factory=list)
    raw_source_span_refs: list[OCRSourceMapping] = Field(default_factory=list)
    ocr_corpus_semantics_version: str | None = None
