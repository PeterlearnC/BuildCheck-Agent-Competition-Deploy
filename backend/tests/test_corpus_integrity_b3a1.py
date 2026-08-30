import json
from datetime import datetime, timezone

from app.core.config import Settings
from app.schemas.standards import (
    ArticleType,
    DocumentCapabilityStatus,
    DocumentRegionType,
    MetadataConfidence,
    StandardDocument,
    StandardIdentityStatus,
    StandardPage,
    StandardParseStatus,
)
from app.schemas.standards_retrieval import RetrievalDecision, StandardSearchRequest
from app.services.retrieval.keyword_retriever import KeywordRetriever
from app.services.retrieval.retrieval_acceptance_service import RetrievalAcceptanceService
from app.services.retrieval.retrieval_manifest_service import RetrievalManifestService
from app.services.retrieval.standards_search_service import StandardsSearchService
from app.services.standards.document_inspector_service import DocumentInspectorService
from app.services.standards.document_region_service import DocumentRegionService
from app.services.standards.stable_identity_service import (
    PARSER_SEMANTICS_VERSION,
    PARSER_VERSION,
)
from app.services.standards.standard_document_service import StandardDocumentService
from app.services.standards.standard_identity_service import StandardIdentityService
from app.services.standards.standard_parser_service import StandardParserService
from app.services.standards.standard_repository import StandardRepository
from tests.retrieval_helpers import make_record


def _document(**updates) -> StandardDocument:
    now = datetime.now(timezone.utc)
    values = {
        "standard_id": "11111111-1111-1111-1111-111111111111",
        "source_filename": "synthetic.pdf",
        "source_checksum": "a" * 64,
        "page_count": 1,
        "created_at": now,
        "updated_at": now,
    }
    values.update(updates)
    return StandardDocument(**values)


def _identity(text: str):
    return StandardIdentityService().identify([StandardPage(page_number=1, text=text)])


def _accept(query: str, content: str):
    records = [make_record("s", "4.4.6", content)]
    hits = KeywordRetriever().retrieve(query, records, 5)
    return RetrievalAcceptanceService().evaluate(query, records, hits)


def test_old_b2_parsed_document_is_reparsed(tmp_path) -> None:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    document = _document(
        parse_status=StandardParseStatus.PARSED,
        parser_version="v0.4-b.2",
    )
    repository.save_document(document)
    repository.save_pages(document.standard_id, [StandardPage(page_number=1, text="1 总则\n1.0.1 构件应可靠连接。")])
    repository.save_articles(document.standard_id, [make_record(document.standard_id, "9.9.9", "旧正文").article])

    result, cached = StandardDocumentService(repository=repository).parse(document.standard_id)

    assert cached is False
    assert result.document.parser_version == PARSER_VERSION
    assert result.document.corpus_semantics_version == PARSER_SEMANTICS_VERSION
    assert result.articles[0].article_number == "1.0.1"
    assert result.articles[0].article_type == ArticleType.NORMATIVE
    assert result.articles[0].region_type == DocumentRegionType.NORMATIVE_BODY
    assert result.articles[0].article_id != f"{document.standard_id}-9.9.9"


def test_current_version_with_legacy_article_json_is_reparsed(tmp_path) -> None:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    document = _document(
        parse_status=StandardParseStatus.PARSED,
        parser_version=PARSER_VERSION,
        corpus_semantics_version=PARSER_SEMANTICS_VERSION,
    )
    repository.save_document(document)
    repository.save_pages(document.standard_id, [StandardPage(page_number=1, text="1 总则\n1.0.1 构件应可靠连接。")])
    article_path = repository.document_dir(document.standard_id) / "articles.json"
    article_path.write_text(json.dumps([{"article_id": "legacy"}]), encoding="utf-8")

    result, cached = StandardDocumentService(repository=repository).parse(document.standard_id)

    assert cached is False
    assert result.articles[0].article_type == ArticleType.NORMATIVE


def test_second_b3a1_parse_uses_compatible_cache(tmp_path) -> None:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    document = _document()
    repository.save_document(document)
    repository.save_pages(document.standard_id, [StandardPage(page_number=1, text="1 总则\n1.0.1 构件应可靠连接。")])
    service = StandardDocumentService(repository=repository)
    first, first_cached = service.parse(document.standard_id)
    second, second_cached = service.parse(document.standard_id)
    assert first_cached is False
    assert second_cached is True
    assert first.articles[0].article_id == second.articles[0].article_id


def test_old_parser_manifest_is_incompatible() -> None:
    current = make_record("s", "1.0.1", "构件应可靠连接")
    current = current.__class__(
        current.document.model_copy(update={
            "parser_version": PARSER_VERSION,
            "corpus_semantics_version": PARSER_SEMANTICS_VERSION,
        }),
        current.article,
    )
    old = current.__class__(
        current.document.model_copy(update={
            "parser_version": "v0.4-b.2",
            "corpus_semantics_version": None,
        }),
        current.article,
    )
    service = RetrievalManifestService()
    assert not service.compatible(service.build([current]), service.build([old]))


def test_toc_region_names_do_not_switch_real_regions() -> None:
    page = StandardPage(page_number=1, text=(
        "目录\n附录 A …… 20\n条文说明 …… 31\n1 总则 …… 1\n"
        "1 总则\n1.0.1 构件应可靠连接。"
    ))
    result = DocumentRegionService().segment([page])
    assert result.line_regions[(1, 1)] == DocumentRegionType.FRONT_MATTER
    assert result.line_regions[(1, 2)] == DocumentRegionType.FRONT_MATTER
    assert result.line_regions[(1, 4)] == DocumentRegionType.NORMATIVE_BODY
    assert result.line_regions[(1, 5)] == DocumentRegionType.NORMATIVE_BODY


def test_incidental_repealed_sentence_does_not_switch_region() -> None:
    page = StandardPage(page_number=1, text=(
        "1 总则\n1.0.1 本标准实施后相关规定同时废止。\n1.0.2 构件应可靠连接。"
    ))
    result = DocumentRegionService().segment([page])
    assert result.line_regions[(1, 2)] == DocumentRegionType.NORMATIVE_BODY


def test_split_explicit_repealed_intro_with_list_is_detected() -> None:
    page = StandardPage(page_number=1, text=(
        "规定与本规范不一致的，以本规范规定为准。同时废止下列工\n"
        "程建设标准相关强制性条文:\n"
        "一、《测试标准》GB 12345-2020第4.4.6条。"
    ))
    result = DocumentRegionService().segment([page])
    assert result.line_regions[(1, 0)] == DocumentRegionType.REPEALED_LIST
    assert result.line_regions[(1, 2)] == DocumentRegionType.REPEALED_LIST


def test_split_normative_root_recovers_after_repealed_list() -> None:
    page = StandardPage(page_number=1, text=(
        "废止条款清单\n4.4.6 已废止。\n1 总\n则\n1.0.1 正文。"
    ))
    result = DocumentRegionService().segment([page])
    assert result.line_regions[(1, 1)] == DocumentRegionType.REPEALED_LIST
    assert result.line_regions[(1, 2)] == DocumentRegionType.NORMATIVE_BODY
    assert result.line_regions[(1, 4)] == DocumentRegionType.NORMATIVE_BODY


def test_structurally_confirmed_explanation_switches_region() -> None:
    page = StandardPage(page_number=1, text="1 总则\n1.0.1 正文。\n条文说明\n1.0.1 说明。")
    result = DocumentRegionService().segment([page])
    assert result.line_regions[(1, 2)] == DocumentRegionType.EXPLANATION
    assert result.line_regions[(1, 3)] == DocumentRegionType.EXPLANATION


def test_structurally_confirmed_appendix_switches_region() -> None:
    page = StandardPage(page_number=1, text="1 总则\n1.0.1 正文。\n附录 A 资料\nA.1.1 附录内容。")
    result = DocumentRegionService().segment([page])
    assert result.line_regions[(1, 2)] == DocumentRegionType.APPENDIX
    assert result.line_regions[(1, 3)] == DocumentRegionType.APPENDIX


def test_explicit_normative_marker_allows_region_recovery() -> None:
    page = StandardPage(page_number=1, text=(
        "1 总则\n1.0.1 正文。\n附录 A 资料\nA.1.1 附录内容。\n"
        "正文\n1 总则\n1.0.2 恢复正文。"
    ))
    result = DocumentRegionService().segment([page])
    assert result.line_regions[(1, 5)] == DocumentRegionType.NORMATIVE_BODY
    assert result.line_regions[(1, 6)] == DocumentRegionType.NORMATIVE_BODY


def test_ambiguous_root_after_appendix_is_not_normative() -> None:
    page = StandardPage(page_number=1, text=(
        "1 总则\n1.0.1 正文。\n附录 A 资料\nA.1.1 附录内容。\n1 总则\n1.0.2 歧义内容。"
    ))
    result = DocumentRegionService().segment([page])
    assert result.line_regions[(1, 4)] == DocumentRegionType.OTHER
    assert result.line_regions[(1, 5)] == DocumentRegionType.OTHER


def test_split_number_counterexample_is_not_fabricated() -> None:
    pages = [StandardPage(page_number=1, text=(
        "1 总则\n4.4.1 原条文。\n条文说明\n4.4.1\n3 个连墙件应按下列规定设置"
    ))]
    result = StandardParserService().parse(_document(), pages)
    assert "4.4.13" not in [article.article_number for article in result.articles]
    assert any("3 个连墙件" in article.source_text for article in result.articles)
    ambiguous = next(article for article in result.articles if "3 个连墙件" in article.source_text)
    assert ambiguous.article_type == ArticleType.OTHER
    assert ambiguous.region_type == DocumentRegionType.OTHER
    assert ambiguous.metadata == {"source_integrity_status": "AMBIGUOUS_SPLIT_ARTICLE_NUMBER"}


def test_split_ten_counterexample_remains_unrepaired_without_layout() -> None:
    pages = [StandardPage(page_number=1, text="1 总则\n5.3.1\n0 条文正文应符合要求。")]
    result = StandardParserService().parse(_document(), pages)
    assert [article.article_number for article in result.articles] == ["5.3.1"]
    assert "0 条文正文" in result.articles[0].source_text


def test_evidence_source_text_uses_raw_not_normalized_line() -> None:
    pages = [StandardPage(page_number=1, text="1 总则\n1.0.1 构件应符合：安全要求。")]
    article = StandardParserService().parse(_document(), pages).articles[0]
    assert "：" in article.source_text
    assert ":" in article.content


def test_reference_code_before_real_title_does_not_win() -> None:
    result = _identity("依据 GB 50016-2014\nGB 55023-2022\n施工脚手架通用规范")
    assert result.canonical_code == "GB55023-2022"
    assert result.standard_name == "施工脚手架通用规范"
    assert result.status == StandardIdentityStatus.CONFIRMED


def test_same_line_code_and_name_is_high_confidence() -> None:
    result = _identity("GB55023-2022 施工脚手架通用规范")
    assert result.status == StandardIdentityStatus.CONFIRMED
    assert result.confidence == MetadataConfidence.HIGH
    assert "bound" in result.reason


def test_two_unbound_codes_are_uncertain() -> None:
    result = _identity("GB 50016-2014\nGB 55023-2022")
    assert result.canonical_code is None
    assert result.status == StandardIdentityStatus.IDENTITY_UNCERTAIN
    assert "multiple" in result.reason


def test_earlier_reference_loses_to_same_line_real_identity() -> None:
    result = _identity("参见 JGJ 130-2011\nGB55023-2022施工脚手架通用规范")
    assert result.canonical_code == "GB55023-2022"
    assert result.status == StandardIdentityStatus.CONFIRMED


def test_article_number_only_query_is_accepted() -> None:
    assert _accept("4.4.6", "连墙件应可靠设置").decision == RetrievalDecision.ACCEPT


def test_article_number_with_matching_semantics_is_accepted() -> None:
    result = _accept("4.4.6 连墙件水平间距", "连墙件的水平间距不得超过三跨")
    assert result.decision == RetrievalDecision.ACCEPT


def test_article_number_with_wrong_domain_is_no_match() -> None:
    result = _accept("4.4.6 建筑给水排水节水", "脚手架连墙件的水平间距不得超过三跨")
    assert result.decision == RetrievalDecision.NO_MATCH


def test_explicit_scope_wrong_domain_returns_no_hits(tmp_path) -> None:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    record = make_record("s", "4.4.6", "脚手架连墙件的水平间距不得超过三跨")
    repository.save_document(record.document)
    repository.save_articles("s", [record.article])
    response = StandardsSearchService(repository).search(StandardSearchRequest(
        query="4.4.6 建筑给水排水节水", standard_ids=["s"]
    ))
    assert response.retrieval_decision == RetrievalDecision.NO_MATCH
    assert response.hits == []


def test_identity_status_change_invalidates_manifest() -> None:
    record = make_record("s", "1.0.1", "构件应可靠连接")
    uncertain = record.__class__(
        record.document.model_copy(update={
            "canonical_standard_code": "GB55023-2022",
            "identity_status": StandardIdentityStatus.IDENTITY_UNCERTAIN,
            "identity_confidence": MetadataConfidence.LOW,
        }),
        record.article,
    )
    confirmed = uncertain.__class__(
        uncertain.document.model_copy(update={
            "identity_status": StandardIdentityStatus.CONFIRMED,
            "identity_confidence": MetadataConfidence.HIGH,
        }),
        uncertain.article,
    )
    service = RetrievalManifestService()
    assert service.corpus_hash([uncertain]) != service.corpus_hash([confirmed])


def test_evidence_exposes_repository_identity_fields(tmp_path) -> None:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    record = make_record("s", "4.4.6", "连墙件水平间距")
    document = record.document.model_copy(update={
        "canonical_standard_code": "GB55023-2022",
        "standard_code": "GB 55023-2022",
        "identity_status": StandardIdentityStatus.CONFIRMED,
        "identity_confidence": MetadataConfidence.HIGH,
    })
    repository.save_document(document)
    repository.save_articles("s", [record.article])
    hit = StandardsSearchService(repository).search(
        StandardSearchRequest(query="4.4.6")
    ).hits[0]
    assert hit.canonical_standard_code == "GB55023-2022"
    assert hit.identity_status == StandardIdentityStatus.CONFIRMED
    assert hit.evidence.canonical_standard_code == "GB55023-2022"
    assert hit.evidence.identity_confidence == MetadataConfidence.HIGH


def test_small_logo_with_short_structured_text_is_ready() -> None:
    pages = [
        StandardPage(
            page_number=index,
            text=f"{index}.0.1 构件应可靠连接。",
            image_count=1,
            image_coverage_ratio=0.02,
        )
        for index in range(1, 4)
    ]
    result = DocumentInspectorService().inspect(pages)
    assert result.status == DocumentCapabilityStatus.TEXT_READY
    assert result.image_heavy_page_ratio == 0.0


def test_full_page_scans_with_watermark_require_ocr() -> None:
    pages = [
        StandardPage(
            page_number=index,
            text=f"浏览专用\n{index}",
            image_count=1,
            image_coverage_ratio=0.96,
        )
        for index in range(1, 18)
    ]
    assert DocumentInspectorService().inspect(pages).status == DocumentCapabilityStatus.OCR_REQUIRED


def test_repeated_normative_body_sentence_is_preserved() -> None:
    pages = [
        StandardPage(page_number=index, text="应符合下列规定\n1.0.1 构件应可靠连接。")
        for index in range(1, 4)
    ]
    result = DocumentInspectorService().inspect(pages)
    assert result.effective_text_char_count >= 3 * len("应符合下列规定")


def test_mixed_document_is_not_rejected_due_to_decorative_images() -> None:
    pages = [
        StandardPage(page_number=1, text="1 总则\n1.0.1 构件应可靠连接。", image_count=1, image_coverage_ratio=0.02),
        StandardPage(page_number=2, text="2.0.1 支撑应稳定。", image_count=1, image_coverage_ratio=0.03),
        StandardPage(page_number=3, text="3", image_count=1, image_coverage_ratio=0.95),
        StandardPage(page_number=4, text="4", image_count=1, image_coverage_ratio=0.95),
    ]
    result = DocumentInspectorService().inspect(pages)
    assert result.status != DocumentCapabilityStatus.OCR_REQUIRED
    assert result.image_heavy_page_ratio == 0.5
