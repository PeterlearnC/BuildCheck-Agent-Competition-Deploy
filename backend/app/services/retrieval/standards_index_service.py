"""Stable corpus fingerprints and lightweight index-manifest compatibility."""

from dataclasses import dataclass
from app.services.retrieval.models import RetrievalRecord
from app.services.retrieval.retrieval_manifest_service import (
    RETRIEVAL_VERSION,
    RetrievalManifestService,
)


RETRIEVAL_INDEX_VERSION = RETRIEVAL_VERSION


@dataclass(frozen=True)
class RetrievalIndexManifest:
    corpus_fingerprint: str
    retrieval_version: str
    provider_name: str
    provider_version: str
    embedding_dimension: int


class StandardsIndexService:
    def corpus_fingerprint(self, records: list[RetrievalRecord]) -> str:
        return RetrievalManifestService().corpus_hash(records)

    @staticmethod
    def compatible(expected: RetrievalIndexManifest, cached: RetrievalIndexManifest) -> bool:
        return expected == cached
