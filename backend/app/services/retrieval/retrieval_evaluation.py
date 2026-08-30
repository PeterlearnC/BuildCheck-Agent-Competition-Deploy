"""Simple deterministic retrieval metrics for synthetic and Golden benchmarks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float


def evaluate_rankings(rankings: list[list[str]], relevant: list[set[str]]) -> RetrievalMetrics:
    if len(rankings) != len(relevant):
        raise ValueError("Rankings and relevance labels must have the same length.")
    if not rankings:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    def recall_at(k: int) -> float:
        values = []
        for ranking, expected in zip(rankings, relevant, strict=True):
            values.append(len(set(ranking[:k]).intersection(expected)) / len(expected) if expected else 0.0)
        return sum(values) / len(values)

    reciprocal_ranks = []
    for ranking, expected in zip(rankings, relevant, strict=True):
        reciprocal_ranks.append(
            next((1.0 / index for index, value in enumerate(ranking, 1) if value in expected), 0.0)
        )
    return RetrievalMetrics(
        recall_at_1=recall_at(1),
        recall_at_3=recall_at(3),
        recall_at_5=recall_at(5),
        recall_at_10=recall_at(10),
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks),
    )
