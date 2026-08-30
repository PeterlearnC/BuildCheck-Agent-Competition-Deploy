from datetime import datetime, timezone

from app.services.retrieval.retrieval_manifest_service import RetrievalManifestService
from tests.retrieval_helpers import make_record


def _changed(record, *, document=None, article=None):
    return record.__class__(document or record.document, article or record.article)


def test_standard_name_change_invalidates_manifest() -> None:
    record = make_record("std", "1.1.1", "正文", standard_name="旧名称")
    changed = _changed(record, document=record.document.model_copy(update={"standard_name": "新名称"}))
    service = RetrievalManifestService()
    assert service.corpus_hash([record]) != service.corpus_hash([changed])


def test_article_change_invalidates_manifest() -> None:
    record = make_record("std", "1.1.1", "正文")
    changed = _changed(record, article=record.article.model_copy(update={"content": "新正文"}))
    assert RetrievalManifestService().corpus_hash([record]) != RetrievalManifestService().corpus_hash([changed])


def test_parser_version_change_invalidates_manifest() -> None:
    record = make_record("std", "1.1.1", "正文")
    changed = _changed(record, document=record.document.model_copy(update={"parser_version": "next"}))
    assert RetrievalManifestService().corpus_hash([record]) != RetrievalManifestService().corpus_hash([changed])


def test_source_checksum_change_invalidates_manifest() -> None:
    record = make_record("std", "1.1.1", "正文")
    changed = _changed(record, document=record.document.model_copy(update={"source_checksum": "f" * 64}))
    assert RetrievalManifestService().corpus_hash([record]) != RetrievalManifestService().corpus_hash([changed])


def test_source_text_and_chapter_metadata_invalidate_manifest() -> None:
    record = make_record("std", "1.1.1", "正文")
    changed = _changed(
        record,
        article=record.article.model_copy(update={"source_text": "不同原文", "chapter_title": "新章节"}),
    )
    assert RetrievalManifestService().corpus_hash([record]) != RetrievalManifestService().corpus_hash([changed])


def test_created_at_does_not_make_compatible_manifest_stale() -> None:
    service = RetrievalManifestService()
    records = [make_record("std", "1.1.1", "正文")]
    first = service.build(records, created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    second = service.build(records, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert service.compatible(first, second)
