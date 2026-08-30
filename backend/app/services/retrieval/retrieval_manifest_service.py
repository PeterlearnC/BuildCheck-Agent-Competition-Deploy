"""Content-complete retrieval manifests used for safe index invalidation."""

from datetime import datetime, timezone
import hashlib
import json

from pydantic import BaseModel, Field, field_validator

from app.services.retrieval.models import RetrievalRecord
from app.services.standards.stable_identity_service import (
    PARSER_SEMANTICS_VERSION,
    PARSER_VERSION,
)


RETRIEVAL_VERSION = "v0.4-b.3a.1"
MANIFEST_VERSION = "v0.4-b.3a.1-manifest"


class RetrievalManifest(BaseModel):
    corpus_hash: str
    parser_version: str
    corpus_semantics_version: str = PARSER_SEMANTICS_VERSION
    retrieval_version: str
    embedding_provider: str
    embedding_model_id: str
    embedding_dimension: int
    article_count: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    manifest_version: str = MANIFEST_VERSION
    ocr_corpus_identities: list[str] = Field(default_factory=list)

    def compatibility_key(self) -> tuple[object, ...]:
        return (
            self.corpus_hash,
            self.parser_version,
            self.corpus_semantics_version,
            self.retrieval_version,
            self.embedding_provider,
            self.embedding_model_id,
            self.embedding_dimension,
            self.article_count,
            self.manifest_version,
            tuple(self.ocr_corpus_identities),
        )

    @field_validator("embedding_model_id")
    @classmethod
    def model_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("embedding_model_id must not be blank.")
        return value.strip()

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(
            self.compatibility_key(), ensure_ascii=False, separators=(",", ":"), default=str
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class RetrievalManifestService:
    def corpus_hash(
        self,
        records: list[RetrievalRecord],
        *,
        parser_version: str | None = None,
    ) -> str:
        effective_parser_version = parser_version or self._parser_version(records)
        payload = {
            "parser_version": effective_parser_version,
            "corpus_semantics_version": PARSER_SEMANTICS_VERSION,
            "standards": [self._record_payload(record) for record in self._sorted(records)],
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def build(
        self,
        records: list[RetrievalRecord],
        *,
        embedding_provider: str = "none",
        embedding_model_id: str = "none",
        embedding_dimension: int = 0,
        parser_version: str | None = None,
        created_at: datetime | None = None,
    ) -> RetrievalManifest:
        effective_parser_version = parser_version or self._parser_version(records)
        values = {
            "corpus_hash": self.corpus_hash(
                records, parser_version=effective_parser_version
            ),
            "parser_version": effective_parser_version,
            "corpus_semantics_version": PARSER_SEMANTICS_VERSION,
            "retrieval_version": RETRIEVAL_VERSION,
            "embedding_provider": embedding_provider,
            "embedding_model_id": embedding_model_id,
            "embedding_dimension": embedding_dimension,
            "article_count": len(records),
            "ocr_corpus_identities": self._ocr_corpus_identities(records),
        }
        if created_at is not None:
            values["created_at"] = created_at
        return RetrievalManifest(**values)

    @staticmethod
    def compatible(expected: RetrievalManifest, cached: RetrievalManifest | None) -> bool:
        return cached is not None and expected.compatibility_key() == cached.compatibility_key()

    @staticmethod
    def _parser_version(records: list[RetrievalRecord]) -> str:
        versions = sorted(
            {record.document.parser_version or PARSER_VERSION for record in records}
        )
        return "+".join(versions) if versions else PARSER_VERSION

    @staticmethod
    def _sorted(records: list[RetrievalRecord]) -> list[RetrievalRecord]:
        return sorted(
            records,
            key=lambda item: (
                item.document.standard_id,
                item.article.sequence,
                item.article.article_number,
                item.article.article_id,
            ),
        )

    @staticmethod
    def _record_payload(record: RetrievalRecord) -> dict[str, object]:
        document = record.document
        article = record.article
        payload = {
            "standard": {
                "standard_id": document.standard_id,
                "standard_code": document.standard_code,
                "canonical_standard_code": document.canonical_standard_code,
                "standard_display_code": document.standard_display_code,
                "standard_name": document.standard_name,
                "identity_status": document.identity_status.value,
                "identity_confidence": document.identity_confidence.value,
                "identity_reason": document.identity_reason,
                "official_source_binding": (
                    document.official_source_binding.model_dump(mode="json")
                    if document.official_source_binding
                    else None
                ),
                "edition": document.edition,
                "publish_date": document.publish_date.isoformat() if document.publish_date else None,
                "effective_date": document.effective_date.isoformat() if document.effective_date else None,
                "source_checksum": document.source_checksum,
                "parser_version": document.parser_version or PARSER_VERSION,
                "corpus_semantics_version": (
                    document.corpus_semantics_version or PARSER_SEMANTICS_VERSION
                ),
            },
            "chapter": {
                "chapter_id": article.chapter_id,
                "chapter_number": article.chapter_number,
                "chapter_title": article.chapter_title,
            },
            "article": {
                "article_id": article.article_id,
                "article_number": article.article_number,
                "content": article.content,
                "source_page_start": article.source_page_start,
                "source_page_end": article.source_page_end,
                "source_text": article.source_text,
                "parse_confidence": article.parse_confidence.value,
                "article_type": article.article_type.value,
                "region_type": article.region_type.value,
                "sequence": article.sequence,
            },
        }
        if article.source_kind.value == "OCR_TEXT":
            payload["ocr"] = {
                "source_kind": article.source_kind.value,
                "ocr_run_id": article.ocr_run_id,
                "ocr_execution_id": article.ocr_execution_id,
                "quality_state": (
                    article.ocr_quality_state.value if article.ocr_quality_state else None
                ),
                "run_quality_state": (
                    document.ocr_quality_state.value if document.ocr_quality_state else None
                ),
                "provider": (
                    article.ocr_provider.model_dump(mode="json")
                    if article.ocr_provider
                    else None
                ),
                "render_metadata": [
                    item.model_dump(mode="json")
                    for item in article.ocr_render_metadata
                ],
                "source_mappings": [
                    item.model_dump(mode="json") for item in article.source_mappings
                ],
                "ocr_corpus_semantics_version": document.ocr_corpus_semantics_version,
            }
        return payload

    @staticmethod
    def _ocr_corpus_identities(records: list[RetrievalRecord]) -> list[str]:
        return sorted(
            {
                ":".join(
                    value
                    for value in (
                        record.article.ocr_run_id,
                        record.document.ocr_corpus_semantics_version,
                    )
                    if value
                )
                for record in records
                if record.article.source_kind.value == "OCR_TEXT"
            }
        )
