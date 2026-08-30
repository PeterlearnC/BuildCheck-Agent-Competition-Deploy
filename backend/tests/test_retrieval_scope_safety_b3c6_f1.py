from pathlib import Path

from app.core.config import Settings
from app.schemas.standards_retrieval import RetrievalDecision, StandardSearchRequest
from app.services.retrieval.query_scope_resolver import (
    QueryScopeResolver,
    QueryScopeStatus,
)
from app.services.retrieval.standards_search_service import StandardsSearchService
from app.services.standards.standard_repository import StandardRepository
from tests.retrieval_helpers import make_record


def _records():
    return [
        make_record(
            "standard-a",
            "2.0.1",
            "支撑结构应设置可靠连接。",
            sequence=1,
            standard_code="JGJ 111-2024",
            standard_name="甲类工程测试规范",
            chapter_number="2",
        ),
        make_record(
            "standard-a",
            "2.0.2",
            "支撑结构的相似要求应经过验算。",
            sequence=2,
            standard_code="JGJ 111-2024",
            standard_name="甲类工程测试规范",
            chapter_number="2",
        ),
        make_record(
            "standard-b",
            "9.9.9",
            "另一标准中的同号条文不得跨标准引用。",
            sequence=1,
            standard_code="GB 222-2023",
            standard_name="乙类工程测试规范",
            chapter_number="9.9",
        ),
    ]


def _service(tmp_path: Path, records=None) -> StandardsSearchService:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    selected = records or _records()
    for standard_id in {record.document.standard_id for record in selected}:
        matching = [record for record in selected if record.document.standard_id == standard_id]
        repository.save_document(matching[0].document)
        repository.save_articles(standard_id, [record.article for record in matching])
    return StandardsSearchService(repository)


def test_known_scope_with_absent_target_never_crosses_standard(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(query="JGJ 111-2024 9.9.9")
    )
    assert response.retrieval_decision == RetrievalDecision.NO_MATCH
    assert response.hits == []
    assert "absent" in response.acceptance_reason


def test_known_scope_with_valid_exact_target_accepts_correct_hit(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(query="JGJ111-2024 2.0.1")
    )
    assert response.retrieval_decision == RetrievalDecision.ACCEPT
    assert [(hit.standard_id, hit.article_number) for hit in response.hits] == [
        ("standard-a", "2.0.1")
    ]


def test_known_scope_with_content_query_only_returns_scoped_standard(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(query="JGJ 111-2024 支撑结构可靠连接")
    )
    assert response.retrieval_decision == RetrievalDecision.ACCEPT
    assert response.hits
    assert {hit.standard_id for hit in response.hits} == {"standard-a"}
    assert response.total_candidates == 2


def test_caller_standard_ids_remain_a_hard_filter(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(query="9.9.9", standard_ids=["standard-b"])
    )
    assert response.retrieval_decision == RetrievalDecision.ACCEPT
    assert {hit.standard_id for hit in response.hits} == {"standard-b"}


def test_query_scope_matching_caller_scope_is_allowed(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(
            query="JGJ 111-2024 2.0.1",
            standard_ids=["standard-a"],
        )
    )
    assert response.retrieval_decision == RetrievalDecision.ACCEPT
    assert response.hits[0].standard_id == "standard-a"


def test_query_scope_intersects_multiple_caller_ids_without_widening(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(
            query="JGJ 111-2024 支撑结构",
            standard_ids=["standard-a", "standard-b"],
        )
    )
    assert response.total_candidates == 2
    assert {hit.standard_id for hit in response.hits} == {"standard-a"}


def test_query_scope_conflicting_with_caller_scope_fails_closed(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(
            query="JGJ 111-2024 2.0.1",
            standard_ids=["standard-b"],
        )
    )
    assert response.retrieval_decision == RetrievalDecision.NO_MATCH
    assert response.hits == []
    assert "conflicts" in response.acceptance_reason


def test_unknown_standard_code_fails_closed(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(query="GB 777-2024 9.9.9")
    )
    assert response.retrieval_decision == RetrievalDecision.NO_MATCH
    assert response.hits == []
    assert "not available" in response.acceptance_reason


def test_two_conflicting_standard_codes_fail_closed(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(query="JGJ 111-2024 GB 222-2023 2.0.1")
    )
    assert response.retrieval_decision == RetrievalDecision.NO_MATCH
    assert response.hits == []
    assert "conflicting" in response.acceptance_reason


def test_aliases_for_same_code_resolve_to_one_scope(tmp_path: Path) -> None:
    service = _service(tmp_path)
    documents = service.repository.list_documents()
    resolution = QueryScopeResolver().resolve(
        "JGJ111-2024 JGJ 111—2024 2.0.1",
        documents,
        None,
    )
    assert resolution.status == QueryScopeStatus.RESOLVED_UNIQUE_SCOPE
    assert resolution.resolved_standard_ids == ("standard-a",)
    assert resolution.residual_query == "2.0.1"
    assert len(resolution.consumed_scope_spans) == 2


def test_malformed_standard_code_fails_closed(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(query="JGJ 111-24 2.0.1")
    )
    assert response.retrieval_decision == RetrievalDecision.NO_MATCH
    assert response.hits == []
    assert "malformed" in response.acceptance_reason


def test_duplicate_documents_for_one_code_are_ambiguous(tmp_path: Path) -> None:
    records = _records() + [
        make_record(
            "standard-a-copy",
            "2.0.1",
            "重复文档不得被静默选择。",
            standard_code="JGJ 111-2024",
            standard_name="甲类工程测试规范副本",
            chapter_number="2",
        )
    ]
    response = _service(tmp_path, records).search(
        StandardSearchRequest(query="JGJ 111-2024 2.0.1")
    )
    assert response.retrieval_decision == RetrievalDecision.NO_MATCH
    assert response.hits == []
    assert "multiple" in response.acceptance_reason


def test_scope_tokens_cannot_satisfy_absent_article_target(tmp_path: Path) -> None:
    response = _service(tmp_path).search(
        StandardSearchRequest(query="JGJ 111-2024 9.9.9 支撑结构")
    )
    assert response.retrieval_decision == RetrievalDecision.NO_MATCH
    assert response.query_coverage == 0.0
    assert response.hits == []


def test_no_query_code_preserves_cross_standard_discovery(tmp_path: Path) -> None:
    response = _service(tmp_path).search(StandardSearchRequest(query="9.9.9"))
    assert response.retrieval_decision == RetrievalDecision.ACCEPT
    assert response.hits[0].standard_id == "standard-b"
    assert response.scope_warning is not None


def test_low_confidence_does_not_expose_authoritative_hits(tmp_path: Path) -> None:
    record = make_record(
        "standard-alpha",
        "1.0.1",
        "alpha requirement",
        standard_code=None,
    )
    response = _service(tmp_path, [record]).search(
        StandardSearchRequest(query="alpha beta")
    )
    assert response.retrieval_decision == RetrievalDecision.LOW_CONFIDENCE
    assert response.hits == []
    assert response.returned == 0


def test_accept_still_exposes_hits_and_no_match_does_not(tmp_path: Path) -> None:
    service = _service(tmp_path)
    accepted = service.search(StandardSearchRequest(query="支撑结构可靠连接"))
    no_match = service.search(StandardSearchRequest(query="不存在的卫星导航要求"))
    assert accepted.retrieval_decision == RetrievalDecision.ACCEPT
    assert accepted.hits
    assert no_match.retrieval_decision == RetrievalDecision.NO_MATCH
    assert no_match.hits == []


def test_valid_hit_article_and_evidence_identity_are_scope_invariant(tmp_path: Path) -> None:
    service = _service(tmp_path)
    caller_scoped = service.search(
        StandardSearchRequest(query="2.0.1", standard_ids=["standard-a"])
    ).hits[0]
    query_scoped = service.search(
        StandardSearchRequest(query="JGJ 111-2024 2.0.1")
    ).hits[0]
    assert query_scoped.article_id == caller_scoped.article_id
    assert query_scoped.evidence.id == caller_scoped.evidence.id
    assert query_scoped.evidence == caller_scoped.evidence
