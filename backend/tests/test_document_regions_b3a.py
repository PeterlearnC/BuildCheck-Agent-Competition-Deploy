from datetime import datetime, timezone

from app.schemas.standards import ArticleType, DocumentRegionType, StandardDocument, StandardPage
from app.services.standards.document_region_service import DocumentRegionService
from app.services.standards.standard_parser_service import StandardParserService


def _document() -> StandardDocument:
    now = datetime.now(timezone.utc)
    return StandardDocument(
        standard_id="std", source_filename="std.pdf", source_checksum="a" * 64,
        page_count=3, created_at=now, updated_at=now,
    )


def test_front_matter_precedes_normative_body() -> None:
    pages = [StandardPage(page_number=1, text="前言\n1 总则\n1.0.1 正文。")]
    result = DocumentRegionService().segment(pages)
    assert result.regions[0].region_type == DocumentRegionType.FRONT_MATTER
    assert result.regions[-1].region_type == DocumentRegionType.NORMATIVE_BODY


def test_explanation_heading_switches_whole_tail() -> None:
    pages = [StandardPage(page_number=1, text="1.0.1 正文。\n条文说明\n1.0.1 说明。")]
    result = DocumentRegionService().segment(pages)
    assert result.line_regions[(1, 2)] == DocumentRegionType.EXPLANATION


def test_numbered_explanation_heading_is_detected() -> None:
    pages = [StandardPage(page_number=1, text="1.0.1 正文。\n四、条文说明\n1.0.1 说明。")]
    result = DocumentRegionService().segment(pages)
    assert result.line_regions[(1, 2)] == DocumentRegionType.EXPLANATION


def test_drafting_notes_are_other_region() -> None:
    pages = [StandardPage(page_number=1, text="1.0.1 正文。\n起草说明\n2.0.1 引用。")]
    result = DocumentRegionService().segment(pages)
    assert result.line_regions[(1, 2)] == DocumentRegionType.OTHER


def test_repealed_list_is_detected() -> None:
    pages = [StandardPage(page_number=1, text="现行工程建设标准相关强制性条文同时废止\n4.4.6 引用条文。")]
    result = DocumentRegionService().segment(pages)
    assert result.line_regions[(1, 1)] == DocumentRegionType.REPEALED_LIST


def test_repealed_region_ends_at_normative_start() -> None:
    pages = [StandardPage(page_number=1, text="废止的条文\n4.4.6 引用。\n1 总则\n1.0.1 正文。")]
    result = DocumentRegionService().segment(pages)
    assert result.line_regions[(1, 1)] == DocumentRegionType.REPEALED_LIST
    assert result.line_regions[(1, 3)] == DocumentRegionType.NORMATIVE_BODY


def test_appendix_is_distinct_region() -> None:
    pages = [StandardPage(page_number=1, text="1.0.1 正文。\n附录 A 资料\nA.1.1 附录内容。")]
    result = DocumentRegionService().segment(pages)
    assert result.line_regions[(1, 2)] == DocumentRegionType.APPENDIX


def test_normative_and_explanation_articles_have_distinct_types() -> None:
    pages = [StandardPage(page_number=1, text="1 总则\n1.0.1 正文。\n条文说明\n1.0.1 说明。")]
    result = StandardParserService().parse(_document(), pages)
    assert [article.article_type for article in result.articles] == [
        ArticleType.NORMATIVE, ArticleType.EXPLANATION
    ]


def test_same_number_across_regions_has_distinct_stable_id() -> None:
    pages = [StandardPage(page_number=1, text="1.0.1 正文。\n条文说明\n1.0.1 正文。")]
    articles = StandardParserService().parse(_document(), pages).articles
    assert len({article.article_id for article in articles}) == 2


def test_region_article_ids_are_stable_across_parses() -> None:
    pages = [StandardPage(page_number=1, text="1.0.1 正文。\n条文说明\n1.0.1 说明。")]
    first = StandardParserService().parse(_document(), pages)
    second = StandardParserService().parse(_document(), pages)
    assert [a.article_id for a in first.articles] == [a.article_id for a in second.articles]


def test_repealed_reference_is_not_normative() -> None:
    pages = [StandardPage(page_number=1, text="废止的条文\n4.4.6 已废止。")]
    articles = StandardParserService().parse(_document(), pages).articles
    assert articles[0].article_type == ArticleType.REPEALED


def test_unclassified_article_is_not_defaulted_to_normative() -> None:
    from app.schemas.standards import ArticleParseConfidence, StandardArticle
    article = StandardArticle(
        article_id="a", standard_id="s", article_number="1.0.1", content="正文",
        source_page_start=1, source_page_end=1, source_text="1.0.1 正文",
        sequence=1, parse_confidence=ArticleParseConfidence.HIGH,
    )
    assert article.article_type == ArticleType.OTHER


def test_split_article_number_is_not_guessed_without_layout_evidence() -> None:
    pages = [StandardPage(page_number=1, text="4.4.1 原条文。\n4.4.1\n3 新条文。")]
    result = StandardParserService().parse(_document(), pages)
    assert "4.4.13" not in [article.article_number for article in result.articles]


def test_article_reference_is_not_new_article() -> None:
    pages = [
        StandardPage(
            page_number=1,
            text="1.0.1 正文。\n条文说明\n5.3.4、5.3.5 说明正文。\n5.3.4条所列情况应检查。",
        )
    ]
    result = StandardParserService().parse(_document(), pages)
    explanation = [a for a in result.articles if a.article_type == ArticleType.EXPLANATION]
    assert [article.article_number for article in explanation] == ["5.3.4"]
    assert "所列情况" in explanation[0].content
