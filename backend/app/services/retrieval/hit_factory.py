"""Create search hits exclusively from repository-backed article records."""

from app.schemas.evidence import EvidenceEnvelope
from app.schemas.standards_retrieval import RetrievalMethod, StandardSearchHit
from app.services.retrieval.models import RetrievalRecord
from app.services.retrieval.retrieval_manifest_service import RETRIEVAL_VERSION
from app.services.standards.stable_identity_service import (
    PARSER_VERSION,
    StableIdentityService,
)
from app.services.ocr.run_identity import OCR_CORPUS_SEMANTICS_VERSION


def make_hit(
    record: RetrievalRecord,
    *,
    rank: int,
    methods: list[RetrievalMethod],
    keyword_score: float | None = None,
    vector_score: float | None = None,
    hybrid_score: float | None = None,
    rerank_score: float | None = None,
) -> StandardSearchHit:
    article = record.article
    document = record.document
    pages_are_valid = (
        article.source_page_start <= article.source_page_end <= document.page_count
    )
    evidence = EvidenceEnvelope(
        id=StableIdentityService().chunk_id(
            article_id=article.article_id,
            chunk_index=0,
            chunk_text=article.source_text,
        ),
        standard_id=document.standard_id,
        standard_code=document.standard_code,
        canonical_standard_code=document.canonical_standard_code,
        standard_name=document.standard_name,
        identity_status=document.identity_status,
        identity_confidence=document.identity_confidence,
        identity_reason=document.identity_reason,
        official_source_binding=document.official_source_binding,
        source_checksum=document.source_checksum,
        article_id=article.article_id,
        chapter_id=article.chapter_id,
        article_number=article.article_number,
        source_page_start=article.source_page_start if pages_are_valid else None,
        source_page_end=article.source_page_end if pages_are_valid else None,
        source_text=article.source_text,
        parser_version=document.parser_version or PARSER_VERSION,
        retrieval_version=RETRIEVAL_VERSION,
        article_type=article.article_type,
        region_type=article.region_type,
        source_kind=article.source_kind,
        normalized_source_text=(
            article.content if article.source_kind.value == "OCR_TEXT" else None
        ),
        ocr_run_id=article.ocr_run_id,
        ocr_execution_id=article.ocr_execution_id,
        ocr_quality_state=article.ocr_quality_state,
        ocr_run_quality_state=document.ocr_quality_state,
        ocr_provider=article.ocr_provider,
        ocr_render_metadata=article.ocr_render_metadata,
        raw_source_span_refs=article.source_mappings,
        ocr_corpus_semantics_version=(
            document.ocr_corpus_semantics_version
            or (OCR_CORPUS_SEMANTICS_VERSION if article.source_kind.value == "OCR_TEXT" else None)
        ),
    )
    return StandardSearchHit(
        standard_id=document.standard_id,
        standard_code=document.standard_code,
        canonical_standard_code=document.canonical_standard_code,
        standard_name=document.standard_name,
        identity_status=document.identity_status,
        identity_confidence=document.identity_confidence,
        identity_reason=document.identity_reason,
        official_source_binding=document.official_source_binding,
        article_id=article.article_id,
        article_number=article.article_number,
        chapter_number=article.chapter_number,
        chapter_title=article.chapter_title,
        content=article.content,
        source_page_start=article.source_page_start,
        source_page_end=article.source_page_end,
        keyword_score=keyword_score,
        vector_score=vector_score,
        hybrid_score=hybrid_score,
        rerank_score=rerank_score,
        retrieval_methods=methods,
        rank=rank,
        parse_confidence=article.parse_confidence,
        article_type=article.article_type,
        region_type=article.region_type,
        source_kind=article.source_kind,
        ocr_run_id=article.ocr_run_id,
        ocr_quality_state=article.ocr_quality_state,
        ocr_run_quality_state=document.ocr_quality_state,
        evidence=evidence,
    )
