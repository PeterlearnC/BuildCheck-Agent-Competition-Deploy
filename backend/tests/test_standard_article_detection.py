import pytest

from app.services.standards.article_detection_service import ArticleDetectionService
from app.services.standards.standard_chapter_service import StandardChapterService


@pytest.mark.parametrize("line", ["6.3.4 条文正文", "6.3.4\n", "第3.2.4条 构件应可靠连接", "A.1.1 附录条文"])
def test_detects_supported_article_numbers(line: str) -> None:
    result = ArticleDetectionService().detect_start(line.strip())
    assert result is not None
    assert result.article_number in {"6.3.4", "3.2.4", "A.1.1"}


@pytest.mark.parametrize(
    "line",
    ["6.2 构造要求", "2023.10.01", "1.2×1.5", "5.2kN", "表6.3.4 参数", "图6.3.4 构造"],
)
def test_rejects_non_article_candidates(line: str) -> None:
    assert ArticleDetectionService().detect_start(line) is None


def test_detects_level_one_chapter() -> None:
    assert StandardChapterService().parse_heading("第6章 施工") == ("6", "施工", 1)


def test_detects_level_two_chapter_not_article() -> None:
    assert StandardChapterService().parse_heading("6.2 构造要求") == ("6.2", "构造要求", 2)
    assert ArticleDetectionService().detect_start("6.2 构造要求") is None


def test_detects_appendix_heading() -> None:
    assert StandardChapterService().parse_heading("附录 A 试验方法") == ("A", "试验方法", 1)
