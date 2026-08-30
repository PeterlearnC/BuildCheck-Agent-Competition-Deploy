from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings
from app.schemas.standards import StandardDocument, StandardPage, StandardParseStatus
from app.services.standards.standard_parser_service import StandardParserService
from app.services.standards.standard_repository import StandardRepository


def _document() -> StandardDocument:
    now = datetime.now(timezone.utc)
    return StandardDocument(
        standard_id="standard-1",
        source_filename="synthetic.pdf",
        source_checksum="a" * 64,
        page_count=2,
        created_at=now,
        updated_at=now,
    )


def _pages() -> list[StandardPage]:
    return [
        StandardPage(
            page_number=1,
            text="中华人民共和国行业标准\n建筑施工测试技术规范\n标准编号 JGJ 999-2024\n第1章 总则\n1.0.1 构件应可靠连接。",
        ),
        StandardPage(page_number=2, text="1.0.2 构件安装前应进行检查。"),
    ]


def test_parser_builds_metadata_chapters_and_articles() -> None:
    result = StandardParserService().parse(_document(), _pages())
    assert result.document.parse_status == StandardParseStatus.PARSED
    assert result.document.standard_code == "JGJ 999-2024"
    assert result.document.standard_name == "建筑施工测试技术规范"
    assert [item.article_number for item in result.articles] == ["1.0.1", "1.0.2"]


def test_parser_is_deterministic_for_core_fields() -> None:
    first = StandardParserService().parse(_document(), _pages())
    second = StandardParserService().parse(_document(), _pages())
    assert first.document.standard_code == second.document.standard_code
    assert [(c.chapter_number, c.chapter_title, c.level) for c in first.chapters] == [
        (c.chapter_number, c.chapter_title, c.level) for c in second.chapters
    ]
    assert [(a.article_number, a.content, a.source_page_start, a.source_page_end) for a in first.articles] == [
        (a.article_number, a.content, a.source_page_start, a.source_page_end) for a in second.articles
    ]


def test_repository_round_trip(tmp_path: Path) -> None:
    settings = Settings(standards_dir=tmp_path / "standards")
    repository = StandardRepository(settings)
    parsed = StandardParserService().parse(_document(), _pages())
    repository.save_document(parsed.document)
    repository.save_pages("standard-1", parsed.pages)
    repository.save_chapters("standard-1", parsed.chapters)
    repository.save_articles("standard-1", parsed.articles)
    assert repository.load_document("standard-1").standard_code == "JGJ 999-2024"
    assert [a.article_number for a in repository.load_articles("standard-1")] == ["1.0.1", "1.0.2"]


def test_no_articles_marks_parse_failed() -> None:
    result = StandardParserService().parse(
        _document(), [StandardPage(page_number=1, text="普通说明文字，没有条文编号。")]
    )
    assert result.document.parse_status == StandardParseStatus.PARSE_FAILED
    assert result.document.parse_error == "NO_ARTICLES_DETECTED"
