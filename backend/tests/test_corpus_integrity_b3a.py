from app.schemas.standards import (
    ArticleParseConfidence,
    ArticleType,
    DocumentRegionType,
)
from app.services.retrieval.hit_factory import make_hit
from app.schemas.standards_retrieval import RetrievalMethod
from app.services.standards.stable_identity_service import StableIdentityService
from tests.retrieval_helpers import make_record


def test_evidence_exposes_article_type_and_region() -> None:
    hit = make_hit(make_record("s", "1.0.1", "正文"), rank=1, methods=[RetrievalMethod.KEYWORD])
    assert hit.evidence.article_type == ArticleType.NORMATIVE
    assert hit.evidence.region_type == DocumentRegionType.NORMATIVE_BODY


def test_search_hit_exposes_article_type_and_region() -> None:
    hit = make_hit(make_record("s", "1.0.1", "正文"), rank=1, methods=[RetrievalMethod.KEYWORD])
    assert hit.article_type == ArticleType.NORMATIVE
    assert hit.region_type == DocumentRegionType.NORMATIVE_BODY


def test_stable_id_changes_with_article_type() -> None:
    service = StableIdentityService()
    common = dict(
        standard_source_checksum="a" * 64, standard_code="GB55023-2022",
        chapter_number="4", article_number="4.4.6", article_text="相同文字",
        source_page_start=15, source_page_end=15,
    )
    assert service.article_id(**common, region_type="NORMATIVE_BODY", article_type="NORMATIVE") != service.article_id(
        **common, region_type="EXPLANATION", article_type="EXPLANATION"
    )


def test_stable_id_changes_with_source_span() -> None:
    service = StableIdentityService()
    common = dict(
        standard_source_checksum="a" * 64, standard_code="GB55023-2022",
        chapter_number="4", article_number="4.4.6", article_text="正文",
        source_page_start=15, region_type="NORMATIVE_BODY", article_type="NORMATIVE",
    )
    assert service.article_id(**common, source_page_end=15) != service.article_id(**common, source_page_end=16)


def test_stable_id_is_deterministic_with_region() -> None:
    service = StableIdentityService()
    values = dict(
        standard_source_checksum="a" * 64, standard_code="GB55023-2022",
        chapter_number="4", article_number="4.4.6", article_text="正文",
        source_page_start=15, source_page_end=15,
        region_type="NORMATIVE_BODY", article_type="NORMATIVE",
    )
    assert service.article_id(**values) == service.article_id(**values)


def test_legacy_article_defaults_are_conservative() -> None:
    record = make_record("s", "1.0.1", "正文")
    legacy = record.article.model_copy(update={
        "article_type": ArticleType.OTHER,
        "region_type": DocumentRegionType.OTHER,
        "parse_confidence": ArticleParseConfidence.LOW,
    })
    assert legacy.article_type == ArticleType.OTHER
