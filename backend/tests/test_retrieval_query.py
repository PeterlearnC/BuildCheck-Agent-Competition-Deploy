import pytest

from app.schemas.standards_retrieval import StandardSearchRequest
from app.services.retrieval.query_normalization_service import QueryNormalizationService


def test_query_normalization_trims_and_collapses_spaces() -> None:
    assert QueryNormalizationService().normalize("  连墙件   设置要求  ") == "连墙件设置要求"


def test_query_normalization_converts_full_width_ascii() -> None:
    assert QueryNormalizationService().normalize("ＪＧＪ １３０－２０１１") == "jgj 130-2011"


def test_query_normalization_lowercases_latin() -> None:
    assert QueryNormalizationService().normalize("JGJ Standard") == "jgj standard"


def test_query_normalization_preserves_article_number() -> None:
    assert "6.3.4" in QueryNormalizationService().normalize("查看 6.3.4 条")


def test_query_normalization_joins_only_spaces_between_cjk_characters() -> None:
    assert QueryNormalizationService().normalize(" 连 墙 件   设置要求 JGJ 130 ") == "连墙件设置要求 jgj 130"


def test_cjk_tokenization_includes_full_and_ngrams() -> None:
    tokens = QueryNormalizationService().tokenize("连墙件设置")
    assert {"连墙件设置", "连墙件", "连墙", "墙件", "设置"}.issubset(tokens)


def test_mixed_tokenization_preserves_ascii_number_and_cjk() -> None:
    tokens = QueryNormalizationService().tokenize("JGJ 130-2011 连墙件")
    assert {"jgj", "130-2011", "连墙件", "连墙", "墙件"}.issubset(tokens)


@pytest.mark.parametrize("query", ["", " ", "!", "1"])
def test_empty_or_too_short_query_is_rejected(query: str) -> None:
    with pytest.raises(ValueError):
        StandardSearchRequest(query=query)
