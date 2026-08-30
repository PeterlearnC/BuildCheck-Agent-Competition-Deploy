from app.schemas.standards_retrieval import RetrievalMethod
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.keyword_retriever import KeywordRetriever
from app.services.retrieval.retrieval_reranker import RetrievalReranker
from app.services.retrieval.vector_retriever import VectorRetriever
from tests.retrieval_helpers import MappingEmbeddingProvider, make_record


def _hybrid() -> HybridRetriever:
    provider = MappingEmbeddingProvider(
        [
            ("防止架体整体失稳", [1.0, 0.0, 0.0]),
            ("整体稳定", [1.0, 0.0, 0.0]),
            ("材料分类", [0.0, 1.0, 0.0]),
        ]
    )
    return HybridRetriever(KeywordRetriever(), VectorRetriever(provider))


def _records():
    return [
        make_record("std", "1.1.1", "支撑体系应保持整体稳定。"),
        make_record("std", "1.1.2", "施工现场材料分类堆放。", sequence=2),
        make_record("std", "1.1.3", "临时设施应定期检查。", sequence=3),
    ]


def test_hybrid_merges_keyword_and_vector_routes() -> None:
    hits = _hybrid().retrieve("防止架体整体失稳", _records(), 3)
    assert hits[0].article_number == "1.1.1"
    assert RetrievalMethod.HYBRID in hits[0].retrieval_methods


def test_same_article_from_two_routes_is_deduplicated() -> None:
    hits = _hybrid().retrieve("防止架体整体失稳", _records(), 3)
    identities = [(hit.standard_id, hit.article_id) for hit in hits]
    assert len(identities) == len(set(identities))


def test_rrf_rank_is_deterministic() -> None:
    first = _hybrid().retrieve("防止架体整体失稳", _records(), 3)
    second = _hybrid().retrieve("防止架体整体失稳", _records(), 3)
    assert [(hit.article_number, hit.hybrid_score) for hit in first] == [
        (hit.article_number, hit.hybrid_score) for hit in second
    ]


def test_keyword_only_article_can_enter_hybrid_results() -> None:
    hits = _hybrid().retrieve("临时设施检查", _records(), 3)
    assert any(hit.article_number == "1.1.3" for hit in hits)


def test_vector_only_article_can_enter_hybrid_results() -> None:
    hits = _hybrid().retrieve("防止架体整体失稳", _records(), 3)
    assert any(hit.article_number == "1.1.1" and hit.vector_score is not None for hit in hits)


def test_hybrid_respects_top_k() -> None:
    assert len(_hybrid().retrieve("施工检查", _records(), 1)) <= 1


def test_reranker_exact_phrase_boost() -> None:
    records = [make_record("std", "1.1.1", "连墙件应按规定设置。"), make_record("std", "1.1.2", "墙体连接位置应设置标识。", sequence=2)]
    hits = KeywordRetriever().retrieve("连墙件", records, 2)
    assert RetrievalReranker().rerank("连墙件", hits)[0].article_number == "1.1.1"


def test_reranker_exact_article_number_boost() -> None:
    records = [make_record("std", "1.1.1", "相同内容。"), make_record("std", "1.1.2", "相同内容。", sequence=2)]
    hits = KeywordRetriever().retrieve("1.1.2 相同内容", records, 2)
    assert RetrievalReranker().rerank("1.1.2 相同内容", hits)[0].article_number == "1.1.2"


def test_reranker_token_coverage_is_stable() -> None:
    records = [make_record("std", "1.1.1", "构件安装检查。"), make_record("std", "1.1.2", "构件检查。", sequence=2)]
    hits = KeywordRetriever().retrieve("构件安装检查", records, 2)
    first = RetrievalReranker().rerank("构件安装检查", hits)
    second = RetrievalReranker().rerank("构件安装检查", hits)
    assert [hit.article_number for hit in first] == [hit.article_number for hit in second]
