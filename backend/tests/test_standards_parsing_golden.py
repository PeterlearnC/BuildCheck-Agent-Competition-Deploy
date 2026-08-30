import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.schemas.standards import StandardDocument, StandardPage
from app.services.standards.standard_parser_service import StandardParserService


ROOT = Path(__file__).parent


@pytest.mark.parametrize("case_id", ["basic_standard", "cross_page_standard"])
def test_standards_parsing_golden(case_id: str) -> None:
    fixture = json.loads(
        (ROOT / "fixtures" / "standards" / f"{case_id}_pages.json").read_text(encoding="utf-8")
    )
    golden = json.loads(
        (ROOT / "golden_standards" / f"{case_id}.json").read_text(encoding="utf-8")
    )
    pages = [StandardPage.model_validate(page) for page in fixture["pages"]]
    now = datetime.now(timezone.utc)
    document = StandardDocument(
        standard_id=fixture["standard_id"],
        source_filename=fixture["source_filename"],
        source_checksum="b" * 64,
        page_count=len(pages),
        created_at=now,
        updated_at=now,
    )
    result = StandardParserService().parse(document, pages)

    if "standard_code" in golden:
        assert result.document.standard_code == golden["standard_code"]
    if "standard_name" in golden:
        assert result.document.standard_name == golden["standard_name"]
    assert [chapter.chapter_number for chapter in result.chapters] == golden["chapter_numbers"]
    assert [article.article_number for article in result.articles] == golden["article_numbers"]
    by_number = {article.article_number: article for article in result.articles}
    for number, expected in golden["articles"].items():
        assert by_number[number].source_page_start == expected["source_page_start"]
        assert by_number[number].source_page_end == expected["source_page_end"]
    assert not set(golden["must_not_articles"]) & set(by_number)
