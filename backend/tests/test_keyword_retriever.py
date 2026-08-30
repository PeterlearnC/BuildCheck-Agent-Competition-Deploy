from app.services.retrieval.keyword_retriever import KeywordRetriever
from tests.retrieval_helpers import basic_records, make_record


def test_exact_technical_term_is_top_one() -> None:
    hits = KeywordRetriever().retrieve("连墙件设置", basic_records(), 3)
    assert hits[0].article_number == "6.1.2"
    assert hits[0].standard_id == "std-a"


def test_article_number_exact_query_is_boosted() -> None:
    hits = KeywordRetriever().retrieve("6.1.1", basic_records(), 3)
    assert hits[0].article_number == "6.1.1"


def test_duplicate_article_numbers_are_not_collapsed_across_standards() -> None:
    hits = KeywordRetriever().retrieve("6.1.2", basic_records(), 10)
    assert {(hit.standard_id, hit.article_number) for hit in hits} >= {("std-a", "6.1.2"), ("std-b", "6.1.2")}


def test_article_number_and_standard_code_select_correct_standard() -> None:
    hits = KeywordRetriever().retrieve("GB 888-2023 6.1.2", basic_records(), 3)
    assert hits[0].standard_id == "std-b"


def test_chapter_title_can_retrieve_article() -> None:
    records = [make_record("std", "3.1.1", "构件应检查。", chapter_title="材料管理")]
    assert KeywordRetriever().retrieve("材料管理", records, 1)[0].article_number == "3.1.1"


def test_weakly_related_text_does_not_beat_exact_term() -> None:
    records = basic_records() + [make_record("std-c", "2.1.1", "墙体连接部位应设置标识。")]
    assert KeywordRetriever().retrieve("连墙件设置", records, 2)[0].standard_id == "std-a"


def test_empty_corpus_returns_empty() -> None:
    assert KeywordRetriever().retrieve("连墙件设置", [], 5) == []


def test_keyword_hit_preserves_repository_text_and_pages() -> None:
    record = make_record("std", "3.2.1", "原始条文内容。", page_start=7, page_end=8)
    hit = KeywordRetriever().retrieve("原始条文内容", [record], 1)[0]
    assert hit.content == record.article.content
    assert (hit.source_page_start, hit.source_page_end) == (7, 8)
