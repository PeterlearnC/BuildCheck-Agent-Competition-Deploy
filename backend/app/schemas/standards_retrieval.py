"""Public schemas for standards retrieval without compliance decisions."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from app.schemas.evidence import EvidenceEnvelope
from app.schemas.ocr import OCRQualityState, OfficialSourceBinding, SourceKind
from app.schemas.standards import (
    ArticleParseConfidence,
    ArticleType,
    DocumentRegionType,
    MetadataConfidence,
    StandardIdentityStatus,
)


class RetrievalMethod(str, Enum):
    KEYWORD = "KEYWORD"
    VECTOR = "VECTOR"
    HYBRID = "HYBRID"


class RetrievalDecision(str, Enum):
    ACCEPT = "ACCEPT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_MATCH = "NO_MATCH"


class StandardRegistryFilter(BaseModel):
    discipline: str | None = None
    jurisdiction: str | None = None


class StandardSearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    standard_ids: list[str] | None = None
    chapter_numbers: list[str] | None = None
    registry_filter: StandardRegistryFilter | None = None
    article_types: list[ArticleType] | None = None

    @field_validator("query")
    @classmethod
    def query_has_meaningful_content(cls, value: str) -> str:
        if len("".join(character for character in value if character.isalnum())) < 2:
            raise ValueError("Search query must contain at least two meaningful characters.")
        return value


class StandardSearchHit(BaseModel):
    standard_id: str
    standard_code: str | None
    canonical_standard_code: str | None
    standard_name: str | None
    identity_status: StandardIdentityStatus
    identity_confidence: MetadataConfidence
    identity_reason: str | None = None
    official_source_binding: OfficialSourceBinding | None = None
    article_id: str
    article_number: str
    chapter_number: str | None
    chapter_title: str | None
    content: str
    source_page_start: int
    source_page_end: int
    keyword_score: float | None = None
    vector_score: float | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None
    retrieval_methods: list[RetrievalMethod]
    rank: int
    parse_confidence: ArticleParseConfidence
    article_type: ArticleType
    region_type: DocumentRegionType
    source_kind: SourceKind = SourceKind.PDF_TEXT
    ocr_run_id: str | None = None
    ocr_quality_state: OCRQualityState | None = None
    ocr_run_quality_state: OCRQualityState | None = None
    evidence: EvidenceEnvelope


class StandardSearchResponse(BaseModel):
    query: str
    normalized_query: str
    total_candidates: int
    returned: int
    hits: list[StandardSearchHit]
    corpus_fingerprint: str
    retrieval_version: str
    manifest_version: str
    scope_warning: str | None = None
    retrieval_decision: RetrievalDecision
    acceptance_reason: str
    query_coverage: float = Field(ge=0.0, le=1.0)
