from pathlib import Path

from app.core.config import Settings
from app.schemas.standards_retrieval import RetrievalDecision, StandardSearchRequest
from app.services.retrieval.keyword_retriever import KeywordRetriever
from app.services.retrieval.retrieval_acceptance_service import RetrievalAcceptanceService
from app.services.retrieval.standards_search_service import StandardsSearchService
from app.services.standards.standard_repository import StandardRepository
from tests.retrieval_helpers import make_record


def _evaluate(query, records):
    hits = KeywordRetriever().retrieve(query, records, 10)
    return RetrievalAcceptanceService().evaluate(query, records, hits)


def test_exact_article_number_is_accepted() -> None:
    result = _evaluate("4.4.6 连墙件", [make_record("s", "4.4.6", "连墙件应可靠设置。")])
    assert result.decision == RetrievalDecision.ACCEPT


def test_relevant_technical_query_is_accepted() -> None:
    result = _evaluate("连墙件 水平 间距", [make_record("s", "4.4.6", "连墙件的水平间距不得超过三跨。")])
    assert result.decision == RetrievalDecision.ACCEPT


def test_wrong_domain_query_is_no_match() -> None:
    records = [make_record("s", "4.4.6", "脚手架连墙件的水平间距不得超过三跨。")]
    result = _evaluate("建筑给水排水节水", records)
    assert result.decision == RetrievalDecision.NO_MATCH


def test_empty_corpus_is_no_match() -> None:
    assert _evaluate("连墙件", []).decision == RetrievalDecision.NO_MATCH


def test_partial_ascii_coverage_is_low_confidence() -> None:
    result = _evaluate("alpha beta", [make_record("s", "1.0.1", "alpha requirement")])
    assert result.decision == RetrievalDecision.LOW_CONFIDENCE


def _service(tmp_path: Path) -> StandardsSearchService:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    record = make_record("s", "4.4.6", "脚手架连墙件的水平间距不得超过三跨。")
    repository.save_document(record.document)
    repository.save_articles("s", [record.article])
    return StandardsSearchService(repository)


def test_no_match_response_removes_hits(tmp_path: Path) -> None:
    response = _service(tmp_path).search(StandardSearchRequest(query="建筑给水排水节水"))
    assert response.retrieval_decision == RetrievalDecision.NO_MATCH
    assert response.hits == []


def test_no_match_applies_inside_explicit_scope(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(query="建筑给水排水节水", standard_ids=["s"])
    )
    assert response.retrieval_decision == RetrievalDecision.NO_MATCH
    assert response.scope_warning is None


def test_accept_response_exposes_coverage_and_reason(tmp_path: Path) -> None:
    response = _service(tmp_path).search(StandardSearchRequest(query="4.4.6 连墙件水平间距"))
    assert response.retrieval_decision == RetrievalDecision.ACCEPT
    assert response.query_coverage > 0
    assert response.acceptance_reason


def test_search_article_type_filter_can_exclude_normative(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(query="4.4.6", article_types=["EXPLANATION"])
    )
    assert response.total_candidates == 0
    assert response.hits == []
