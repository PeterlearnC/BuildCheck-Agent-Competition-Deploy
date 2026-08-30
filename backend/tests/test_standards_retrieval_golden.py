import json
from pathlib import Path

import pytest

from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.keyword_retriever import KeywordRetriever
from app.services.retrieval.vector_retriever import VectorRetriever
from tests.retrieval_helpers import MappingEmbeddingProvider, make_record


ROOT = Path(__file__).parent


def _records():
    payload = json.loads(
        (ROOT / "fixtures" / "standards" / "retrieval_corpus.json").read_text(encoding="utf-8")
    )
    return [make_record(**article) for article in payload["articles"]]


def _semantic_provider() -> MappingEmbeddingProvider:
    return MappingEmbeddingProvider(
        [
            ("防止架体整体失稳", [1.0, 0.0, 0.0]),
            ("整体稳定", [1.0, 0.0, 0.0]),
            ("连墙件设置", [0.0, 1.0, 0.0]),
            ("连墙件", [0.0, 1.0, 0.0]),
            ("材料", [0.0, 0.0, 1.0]),
        ]
    )


@pytest.mark.parametrize("case_id", ["exact_term", "semantic_route", "wrong_standard"])
def test_standards_retrieval_golden(case_id: str) -> None:
    golden = json.loads(
        (ROOT / "golden_retrieval" / f"{case_id}.json").read_text(encoding="utf-8")
    )
    records = _records()
    if golden.get("standard_ids") is not None:
        allowed = set(golden["standard_ids"])
        records = [record for record in records if record.document.standard_id in allowed]
    if golden["route"] == "hybrid":
        hits = HybridRetriever(
            KeywordRetriever(), VectorRetriever(_semantic_provider())
        ).retrieve(golden["query"], records, golden["top_k"])
    else:
        hits = KeywordRetriever().retrieve(golden["query"], records, golden["top_k"])
    identities = [f"{hit.standard_id}:{hit.article_number}" for hit in hits]
    assert set(golden.get("must_retrieve", [])).issubset(identities)
    assert set(golden.get("top_k_contains", [])).issubset(identities[: golden["top_k"]])
    if identities:
        assert identities[0] not in golden.get("must_not_top", [])
