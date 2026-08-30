import json
from pathlib import Path

from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.keyword_retriever import KeywordRetriever
from app.services.retrieval.retrieval_evaluation import evaluate_rankings
from app.services.retrieval.vector_retriever import VectorRetriever
from tests.retrieval_helpers import MappingEmbeddingProvider, make_record


def test_keyword_vector_hybrid_synthetic_benchmark_has_no_hybrid_regression() -> None:
    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "standards" / "retrieval_corpus.json").read_text(encoding="utf-8")
    )
    records = [make_record(**article) for article in payload["articles"]]
    provider = MappingEmbeddingProvider(
        [
            ("防止架体整体失稳", [1.0, 0.0, 0.0]),
            ("整体稳定", [1.0, 0.0, 0.0]),
            ("连墙件设置", [0.0, 1.0, 0.0]),
            ("连墙件", [0.0, 1.0, 0.0]),
            ("材料分类堆放", [0.0, 0.0, 1.0]),
            ("材料", [0.0, 0.0, 1.0]),
        ]
    )
    keyword = KeywordRetriever()
    vector = VectorRetriever(provider)
    hybrid = HybridRetriever(keyword, vector)
    queries = ["连墙件设置", "防止架体整体失稳", "材料分类堆放"]
    relevant = [{"std-a:6.1.2"}, {"std-a:6.1.3"}, {"std-b:6.1.3"}]

    def ranking(retriever):
        return [
            [f"{hit.standard_id}:{hit.article_number}" for hit in retriever.retrieve(query, records, 10)]
            for query in queries
        ]

    keyword_metrics = evaluate_rankings(ranking(keyword), relevant)
    vector_metrics = evaluate_rankings(ranking(vector), relevant)
    hybrid_metrics = evaluate_rankings(ranking(hybrid), relevant)
    assert hybrid_metrics.recall_at_1 >= keyword_metrics.recall_at_1
    assert hybrid_metrics.recall_at_3 >= keyword_metrics.recall_at_3
    assert vector_metrics.recall_at_3 == 1.0
