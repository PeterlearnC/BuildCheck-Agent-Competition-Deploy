from app.services.retrieval.hit_factory import make_hit
from app.schemas.standards_retrieval import RetrievalMethod
from app.services.retrieval.retrieval_manifest_service import RETRIEVAL_VERSION
from app.services.standards.stable_identity_service import PARSER_VERSION
from tests.retrieval_helpers import make_record


def _hit(record=None):
    return make_hit(
        record or make_record("std", "6.1.1", "处理后的正文"),
        rank=1,
        methods=[RetrievalMethod.KEYWORD],
        keyword_score=1.0,
    )


def test_evidence_contains_all_required_identity_fields() -> None:
    evidence = _hit().evidence
    assert evidence.id and evidence.standard_id and evidence.article_id and evidence.article_number


def test_missing_source_page_returns_null() -> None:
    record = make_record("std", "6.1.1", "正文", page_start=11)
    evidence = _hit(record).evidence
    assert evidence.source_page_start is None
    assert evidence.source_page_end is None


def test_source_checksum_is_preserved() -> None:
    record = make_record("std", "6.1.1", "正文")
    assert _hit(record).evidence.source_checksum == record.document.source_checksum


def test_parser_version_is_preserved() -> None:
    record = make_record("std", "6.1.1", "正文")
    record = record.__class__(record.document.model_copy(update={"parser_version": "parser-x"}), record.article)
    assert _hit(record).evidence.parser_version == "parser-x"


def test_missing_legacy_parser_version_uses_current_version() -> None:
    assert _hit().evidence.parser_version == PARSER_VERSION


def test_evidence_uses_literal_source_text() -> None:
    hit = _hit()
    assert hit.evidence.source_text == "6.1.1 处理后的正文"
    assert hit.evidence.source_text != hit.content


def test_evidence_id_is_deterministic() -> None:
    assert _hit().evidence.id == _hit().evidence.id


def test_evidence_keeps_retrieval_version_and_legacy_fields() -> None:
    hit = _hit()
    assert hit.evidence.retrieval_version == RETRIEVAL_VERSION
    assert hit.content == "处理后的正文"
    assert hit.source_page_start == 3
