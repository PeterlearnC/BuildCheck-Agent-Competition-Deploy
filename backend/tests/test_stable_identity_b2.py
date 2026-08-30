from datetime import datetime, timezone

from app.schemas.standards import StandardDocument, StandardPage
from app.services.standards.stable_identity_service import StableIdentityService
from app.services.standards.standard_parser_service import StandardParserService


def _article_id(**updates) -> str:
    values = {
        "standard_source_checksum": "a" * 64,
        "standard_code": "JGJ 999-2024",
        "chapter_number": "6.1",
        "article_number": "6.1.1",
        "article_text": "构件应可靠连接。",
        "source_page_start": 3,
    }
    values.update(updates)
    return StableIdentityService().article_id(**values)


def test_article_identity_is_repeatable() -> None:
    assert _article_id() == _article_id()


def test_article_identity_changes_with_page() -> None:
    assert _article_id() != _article_id(source_page_start=4)


def test_article_identity_changes_with_text() -> None:
    assert _article_id() != _article_id(article_text="构件不得松动。")


def test_chapter_identity_changes_with_title() -> None:
    service = StableIdentityService()
    first = service.chapter_id(standard_checksum="a", chapter_number="6", chapter_title="构造要求")
    second = service.chapter_id(standard_checksum="a", chapter_number="6", chapter_title="安全要求")
    assert first != second


def test_identity_normalizes_unicode_and_whitespace() -> None:
    assert _article_id(article_text="Ａ  构件") == _article_id(article_text="A 构件")


def test_chunk_identity_changes_with_index() -> None:
    service = StableIdentityService()
    assert service.chunk_id(article_id="article", chunk_index=0, chunk_text="text") != service.chunk_id(
        article_id="article", chunk_index=1, chunk_text="text"
    )


def test_same_document_parsed_twice_has_same_article_ids() -> None:
    now = datetime.now(timezone.utc)
    document = StandardDocument(
        standard_id="std",
        source_filename="std.pdf",
        source_checksum="b" * 64,
        page_count=1,
        created_at=now,
        updated_at=now,
    )
    pages = [StandardPage(page_number=1, text="标准编号 JGJ 999-2024\n第6章 构造\n6.1.1 构件应可靠连接。")]
    parser = StandardParserService()
    assert [item.article_id for item in parser.parse(document, pages).articles] == [
        item.article_id for item in parser.parse(document, pages).articles
    ]


def test_article_identity_changes_with_source_checksum() -> None:
    assert _article_id() != _article_id(standard_source_checksum="b" * 64)
