"""Reciprocal-rank fusion without mixing BM25 and cosine score spaces."""

from app.schemas.standards_retrieval import RetrievalMethod, StandardSearchHit
from app.services.retrieval.keyword_retriever import KeywordRetriever
from app.services.retrieval.models import RetrievalRecord
from app.services.retrieval.retrieval_reranker import RetrievalReranker
from app.services.retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    RRF_K = 60
    CANDIDATE_MULTIPLIER = 4
    MIN_CANDIDATES_PER_ROUTE = 20

    def __init__(
        self,
        keyword_retriever: KeywordRetriever,
        vector_retriever: VectorRetriever,
        reranker: RetrievalReranker | None = None,
    ) -> None:
        self.keyword_retriever = keyword_retriever
        self.vector_retriever = vector_retriever
        self.reranker = reranker or RetrievalReranker()

    def retrieve(
        self, query: str, records: list[RetrievalRecord], top_k: int
    ) -> list[StandardSearchHit]:
        pool_size = max(top_k * self.CANDIDATE_MULTIPLIER, self.MIN_CANDIDATES_PER_ROUTE)
        keyword_hits = self.keyword_retriever.retrieve(query, records, pool_size)
        vector_hits = self.vector_retriever.retrieve(query, records, pool_size)
        fused: dict[tuple[str, str], StandardSearchHit] = {}
        scores: dict[tuple[str, str], float] = {}
        for route in (keyword_hits, vector_hits):
            for hit in route:
                key = (hit.standard_id, hit.article_id)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.RRF_K + hit.rank)
                if key not in fused:
                    fused[key] = hit
                else:
                    existing = fused[key]
                    fused[key] = existing.model_copy(
                        update={
                            "keyword_score": existing.keyword_score or hit.keyword_score,
                            "vector_score": existing.vector_score or hit.vector_score,
                        }
                    )
        candidates: list[StandardSearchHit] = []
        for key, hit in fused.items():
            methods = list(dict.fromkeys([*hit.retrieval_methods, RetrievalMethod.HYBRID]))
            if hit.keyword_score is not None and RetrievalMethod.KEYWORD not in methods:
                methods.insert(0, RetrievalMethod.KEYWORD)
            if hit.vector_score is not None and RetrievalMethod.VECTOR not in methods:
                methods.insert(0, RetrievalMethod.VECTOR)
            candidates.append(
                hit.model_copy(
                    update={
                        "hybrid_score": scores[key],
                        "retrieval_methods": methods,
                    }
                )
            )
        candidates.sort(
            key=lambda item: (-(item.hybrid_score or 0.0), item.standard_id, item.article_id)
        )
        candidates = [
            hit.model_copy(update={"rank": index}) for index, hit in enumerate(candidates, 1)
        ]
        return self.reranker.rerank(query, candidates)[:top_k]
