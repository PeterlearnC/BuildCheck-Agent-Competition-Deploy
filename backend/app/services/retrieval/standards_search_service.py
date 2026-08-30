"""Repository-backed standards search orchestration."""

from app.schemas.standards_retrieval import (
    StandardRegistryFilter,
    StandardSearchRequest,
    StandardSearchResponse,
)
from app.services.retrieval.hybrid_retriever import HybridRetriever
from app.services.retrieval.keyword_retriever import KeywordRetriever
from app.services.retrieval.models import RetrievalRecord
from app.services.retrieval.query_normalization_service import QueryNormalizationService
from app.services.retrieval.query_scope_resolver import QueryScopeResolver
from app.services.retrieval.retrieval_reranker import RetrievalReranker
from app.services.retrieval.retrieval_acceptance_service import (
    RetrievalAcceptanceService,
)
from app.schemas.standards_retrieval import RetrievalDecision
from app.services.retrieval.retrieval_manifest_service import MANIFEST_VERSION
from app.services.retrieval.standards_index_service import (
    RETRIEVAL_INDEX_VERSION,
    StandardsIndexService,
)
from app.services.retrieval.vector_retriever import VectorRetriever
from app.services.standards.standard_repository import StandardRepository
from app.services.standards.standard_registry_service import StandardRegistryService
from app.services.standards.standard_identity_service import StandardIdentityService
from app.schemas.standards import ArticleType, StandardDocument


class UnknownStandardsError(ValueError):
    pass


class StandardsSearchService:
    UNSCOPED_SEARCH_WARNING = (
        "Search without explicit standard scope may include multiple standards"
    )
    def __init__(
        self,
        repository: StandardRepository | None = None,
        vector_retriever: VectorRetriever | None = None,
        registry: StandardRegistryService | None = None,
    ) -> None:
        self.repository = repository or StandardRepository()
        self.normalizer = QueryNormalizationService()
        self.scope_resolver = QueryScopeResolver()
        self.keyword = KeywordRetriever(self.normalizer)
        self.reranker = RetrievalReranker(self.normalizer)
        self.vector = vector_retriever
        self.registry = registry or StandardRegistryService(self.repository.settings)
        self.index = StandardsIndexService()
        self.acceptance = RetrievalAcceptanceService(self.normalizer)

    def search(self, request: StandardSearchRequest) -> StandardSearchResponse:
        documents = self.repository.list_documents()
        self._validate_standard_ids(request.standard_ids, documents)
        scope = self.scope_resolver.resolve(
            request.query,
            documents,
            request.standard_ids,
        )
        if scope.failed:
            return self._empty_response(request, scope.reason)
        records = self._load_records(
            (
                list(scope.resolved_standard_ids)
                if scope.resolved_standard_ids is not None
                else None
            ),
            request.chapter_numbers,
            registry_filter=request.registry_filter,
            article_types=request.article_types,
            documents=documents,
        )
        if scope.resolved_standard_ids is not None and scope.article_numbers:
            available_numbers = {record.article.article_number for record in records}
            if not set(scope.article_numbers).issubset(available_numbers):
                return self._empty_response(
                    request,
                    "explicit article target is absent from the resolved standard scope",
                    records=records,
                )
        relevance_query = scope.residual_query
        if self.vector is None:
            candidates = self.reranker.rerank(
                relevance_query,
                self.keyword.retrieve(relevance_query, records, request.top_k * 4),
            )
        else:
            candidates = HybridRetriever(self.keyword, self.vector, self.reranker).retrieve(
                relevance_query, records, request.top_k * 4
            )
        candidates = self._prefer_normative(candidates)
        acceptance = self.acceptance.evaluate(relevance_query, records, candidates)
        hits = (
            candidates[: request.top_k]
            if acceptance.decision == RetrievalDecision.ACCEPT
            else []
        )
        return StandardSearchResponse(
            query=request.query,
            normalized_query=self.normalizer.normalize(request.query),
            total_candidates=len(records),
            returned=len(hits),
            hits=hits,
            corpus_fingerprint=self.index.corpus_fingerprint(records),
            retrieval_version=RETRIEVAL_INDEX_VERSION,
            manifest_version=MANIFEST_VERSION,
            scope_warning=(
                self.UNSCOPED_SEARCH_WARNING
                if scope.resolved_standard_ids is None
                else None
            ),
            retrieval_decision=acceptance.decision,
            acceptance_reason=acceptance.reason,
            query_coverage=acceptance.query_coverage,
        )

    def _load_records(
        self,
        standard_ids: list[str] | None,
        chapter_numbers: list[str] | None,
        registry_filter: StandardRegistryFilter | None = None,
        article_types: list[ArticleType] | None = None,
        documents: list[StandardDocument] | None = None,
    ) -> list[RetrievalRecord]:
        documents = (
            list(documents)
            if documents is not None
            else self.repository.list_documents()
        )
        by_id = {document.standard_id: document for document in documents}
        if standard_ids is not None:
            unknown = sorted(set(standard_ids).difference(by_id))
            if unknown:
                raise UnknownStandardsError(
                    "Unknown standard_id values: " + ", ".join(unknown)
                )
            documents = [by_id[standard_id] for standard_id in standard_ids]
        if registry_filter is not None:
            active_codes = self.registry.active_codes(
                discipline=registry_filter.discipline,
                jurisdiction=registry_filter.jurisdiction,
            )
            identity_service = StandardIdentityService()
            active_codes = {
                (identity_service.canonicalize(code) or code).casefold()
                for code in active_codes
            }
            documents = [
                document
                for document in documents
                if (document.canonical_standard_code or document.standard_code)
                and (
                    document.canonical_standard_code
                    or StandardIdentityService().canonicalize(document.standard_code)
                    or document.standard_code
                ).casefold() in active_codes
            ]
        chapter_filter = set(chapter_numbers) if chapter_numbers is not None else None
        type_filter = set(article_types) if article_types is not None else {
            ArticleType.NORMATIVE,
            ArticleType.APPENDIX,
            ArticleType.EXPLANATION,
        }
        records: list[RetrievalRecord] = []
        for document in documents:
            for article in self.repository.load_articles(document.standard_id):
                if article.article_type not in type_filter:
                    continue
                if chapter_filter is not None and article.chapter_number not in chapter_filter:
                    continue
                records.append(RetrievalRecord(document=document, article=article))
        return sorted(
            records,
            key=lambda item: (
                item.document.standard_id,
                item.article.sequence,
                item.article.article_number,
            ),
        )

    def _empty_response(
        self,
        request: StandardSearchRequest,
        reason: str,
        *,
        records: list[RetrievalRecord] | None = None,
    ) -> StandardSearchResponse:
        selected = records or []
        return StandardSearchResponse(
            query=request.query,
            normalized_query=self.normalizer.normalize(request.query),
            total_candidates=len(selected),
            returned=0,
            hits=[],
            corpus_fingerprint=self.index.corpus_fingerprint(selected),
            retrieval_version=RETRIEVAL_INDEX_VERSION,
            manifest_version=MANIFEST_VERSION,
            scope_warning=None,
            retrieval_decision=RetrievalDecision.NO_MATCH,
            acceptance_reason=reason,
            query_coverage=0.0,
        )

    @staticmethod
    def _validate_standard_ids(
        standard_ids: list[str] | None,
        documents: list[StandardDocument],
    ) -> None:
        if standard_ids is None:
            return
        known = {document.standard_id for document in documents}
        unknown = sorted(set(standard_ids).difference(known))
        if unknown:
            raise UnknownStandardsError(
                "Unknown standard_id values: " + ", ".join(unknown)
            )

    @staticmethod
    def _prefer_normative(hits):
        ordered = list(hits)
        for index, hit in enumerate(ordered):
            if hit.article_type == ArticleType.NORMATIVE:
                continue
            normative_index = next(
                (
                    candidate_index
                    for candidate_index in range(index + 1, len(ordered))
                    if ordered[candidate_index].article_number == hit.article_number
                    and ordered[candidate_index].standard_id == hit.standard_id
                    and ordered[candidate_index].article_type == ArticleType.NORMATIVE
                ),
                None,
            )
            if normative_index is not None:
                normative = ordered.pop(normative_index)
                ordered.insert(index, normative)
        return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(ordered, 1)]
