"""C.4 stateless human-workflow events preserve immutable C.3 authority."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.schemas.compliance_comparison import (
    ComparisonDecision,
    ComparisonDecisionScope,
    ComparisonReasonCode,
)
from app.schemas.finding_review import (
    MAX_REVIEWER_ID_LENGTH,
    MAX_REVIEWER_NOTE_LENGTH,
    FindingReviewCommand,
    FindingReviewDisposition,
    FindingReviewEvent,
    FindingReviewProjection,
    ReportInclusionStatus,
    ReviewerIdentityAssurance,
    deterministic_finding_authority_sha256,
    deterministic_review_event_id,
    deterministic_reviewer_note_sha256,
)
from app.services.review.finding_review_service import (
    FindingReviewAuthorityMismatchError,
    FindingReviewCommandError,
    FindingReviewService,
)
from app.services.review.review_finding_service import (
    ReviewFindingAuthorityUnavailableError,
)
from tests.test_review_finding_foundation_c3 import (
    DOCUMENT_ID,
    _finding,
    _preparation,
    _request,
)


FIXED_TIME = datetime(2026, 8, 29, 2, 30, 45, 123456, tzinfo=timezone.utc)


class _StaticC3Service:
    def __init__(self, *, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, document_id, request):
        self.calls.append((document_id, request))
        if self.error is not None:
            raise self.error
        return self.response


def _command(
    *,
    disposition=FindingReviewDisposition.ACCEPTED,
    report_inclusion=ReportInclusionStatus.UNDECIDED,
    reviewer_id="reviewer-one",
    reviewer_note="Reviewed locally.",
):
    return FindingReviewCommand(
        reviewer_id=reviewer_id,
        disposition=disposition,
        report_inclusion=report_inclusion,
        reviewer_note=reviewer_note,
    )


def _projection(
    tmp_path,
    *,
    plan_text="Platform width is 1.5 m.",
    fact_text="1.5 m",
    command=None,
    clock=lambda: FIXED_TIME,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    response = _finding(tmp_path, plan_text=plan_text, fact_text=fact_text)
    c3 = _StaticC3Service(response=response)
    service = FindingReviewService(review_finding_service=c3, clock=clock)
    request = _request(_preparation())
    projection = service.create(
        document_id=DOCUMENT_ID,
        finding_id=response.finding.finding_id,
        comparison_request=request,
        command=command or _command(),
    )
    return projection, response, c3, request


@pytest.mark.parametrize(
    ("plan_text", "fact_text", "disposition", "expected"),
    [
        (
            "Platform width is 1.5 m.",
            "1.5 m",
            FindingReviewDisposition.ACCEPTED,
            ComparisonDecision.COMPLIANT,
        ),
        (
            "Platform width is 1.0 m.",
            "1.0 m",
            FindingReviewDisposition.ACCEPTED,
            ComparisonDecision.NON_COMPLIANT,
        ),
        (
            "Platform width is not stated.",
            None,
            FindingReviewDisposition.NEEDS_INFORMATION,
            ComparisonDecision.INSUFFICIENT_INFORMATION,
        ),
    ],
)
def test_golden_style_workflow_preserves_each_c3_decision(
    tmp_path, plan_text, fact_text, disposition, expected
):
    projection, response, _, _ = _projection(
        tmp_path,
        plan_text=plan_text,
        fact_text=fact_text,
        command=_command(disposition=disposition),
    )
    assert projection.authoritative_finding == response.finding
    assert projection.authoritative_finding.decision == expected
    assert projection.review_event.disposition == disposition


def test_disposition_contract_excludes_unreviewed_and_unknown_values() -> None:
    assert {item.value for item in FindingReviewDisposition} == {
        "ACCEPTED",
        "REJECTED",
        "NEEDS_INFORMATION",
        "DEFERRED",
    }
    for invalid in ("UNREVIEWED", "APPROVED", ""):
        with pytest.raises(ValidationError):
            _command(disposition=invalid)

    with pytest.raises(ValidationError):
        _command(report_inclusion="PUBLISH")


@pytest.mark.parametrize("disposition", list(FindingReviewDisposition))
@pytest.mark.parametrize("inclusion", list(ReportInclusionStatus))
def test_report_inclusion_is_orthogonal_to_disposition(
    tmp_path, disposition, inclusion
) -> None:
    projection, response, _, _ = _projection(
        tmp_path,
        command=_command(disposition=disposition, report_inclusion=inclusion),
    )
    assert projection.review_event.report_inclusion == inclusion
    assert projection.review_event.disposition == disposition
    assert projection.authoritative_finding.decision == response.finding.decision


def test_only_unverified_reviewer_assurance_exists_and_metadata_cannot_upgrade_it(
    tmp_path,
) -> None:
    assert list(ReviewerIdentityAssurance) == [
        ReviewerIdentityAssurance.CALLER_ASSERTED_UNVERIFIED
    ]
    for field, value in (
        ("reviewer_identity_assurance", "VERIFIED"),
        ("display_name", "Licensed Reviewer"),
        ("role", "PROFESSIONAL_ENGINEER"),
    ):
        payload = _command().model_dump()
        payload[field] = value
        with pytest.raises(ValidationError):
            FindingReviewCommand.model_validate(payload)

    projection, _, _, _ = _projection(tmp_path)
    assert (
        projection.review_event.reviewer_identity_assurance
        == ReviewerIdentityAssurance.CALLER_ASSERTED_UNVERIFIED
    )


@pytest.mark.parametrize(
    "reviewer_id",
    ["", " ", " leading", "trailing ", "x" * (MAX_REVIEWER_ID_LENGTH + 1)],
)
def test_invalid_reviewer_ids_are_rejected(reviewer_id) -> None:
    with pytest.raises(ValidationError):
        _command(reviewer_id=reviewer_id)


def test_unicode_note_is_exact_and_note_hash_is_server_computed(tmp_path) -> None:
    note = "需要补充支撑节点资料。\n保持原文。"
    projection, _, _, _ = _projection(tmp_path, command=_command(reviewer_note=note))
    event = projection.review_event
    assert event.reviewer_note == note
    assert event.reviewer_note_sha256 == hashlib.sha256(note.encode()).hexdigest()
    assert event.reviewer_note != projection.authoritative_finding.summary


def test_oversized_note_and_caller_note_hash_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _command(reviewer_note="x" * (MAX_REVIEWER_NOTE_LENGTH + 1))
    payload = _command().model_dump()
    payload["reviewer_note_sha256"] = "0" * 64
    with pytest.raises(ValidationError):
        FindingReviewCommand.model_validate(payload)


@pytest.mark.parametrize(
    "forbidden",
    [
        "occurred_at",
        "review_event_id",
        "sequence",
        "previous_event_id",
        "previous_disposition",
        "final_adjudication",
        "history_snapshot",
        "authoritative_finding",
    ],
)
def test_caller_cannot_supply_event_or_history_authority(forbidden) -> None:
    payload = _command().model_dump()
    payload[forbidden] = "caller-controlled"
    with pytest.raises(ValidationError):
        FindingReviewCommand.model_validate(payload)


def test_occurred_at_is_server_issued_utc(tmp_path) -> None:
    offset_time = FIXED_TIME.astimezone(timezone(timedelta(hours=8)))
    projection, _, _, _ = _projection(tmp_path, clock=lambda: offset_time)
    assert projection.review_event.occurred_at == FIXED_TIME
    assert projection.review_event.occurred_at.utcoffset() == timedelta(0)

    with pytest.raises(FindingReviewCommandError, match="timezone-aware"):
        _projection(tmp_path / "naive", clock=lambda: FIXED_TIME.replace(tzinfo=None))


def test_event_identity_and_all_server_hashes_revalidate(tmp_path) -> None:
    projection, _, _, _ = _projection(tmp_path)
    event = projection.review_event
    assert event.reviewer_note_sha256 == deterministic_reviewer_note_sha256(
        event.reviewer_note
    )
    assert event.finding_authority_sha256 == deterministic_finding_authority_sha256(
        projection.authoritative_finding
    )
    assert event.review_event_id == deterministic_review_event_id(
        document_id=event.document_id,
        finding_id=event.finding_id,
        reviewer_id=event.reviewer_id,
        reviewer_identity_assurance=ReviewerIdentityAssurance(
            event.reviewer_identity_assurance
        ),
        disposition=event.disposition,
        report_inclusion=event.report_inclusion,
        reviewer_note_sha256=event.reviewer_note_sha256,
        occurred_at=event.occurred_at,
    )
    assert FindingReviewEvent.model_validate(event.model_dump()) == event
    assert FindingReviewProjection(
        authoritative_finding=projection.authoritative_finding,
        review_event=event,
    ) == projection
    with pytest.raises(ValidationError):
        event.disposition = FindingReviewDisposition.REJECTED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("review_event_id", "reviewevent_" + "0" * 64),
        ("reviewer_note_sha256", "0" * 64),
    ],
)
def test_frozen_event_rejects_tampered_identity_fields(tmp_path, field, value) -> None:
    projection, _, _, _ = _projection(tmp_path)
    payload = projection.review_event.model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        FindingReviewEvent.model_validate(payload)


def test_same_command_at_different_action_times_has_distinct_event_ids(tmp_path) -> None:
    first, response, _, request = _projection(tmp_path / "first")
    c3 = _StaticC3Service(response=response)
    second = FindingReviewService(
        review_finding_service=c3,
        clock=lambda: FIXED_TIME + timedelta(microseconds=1),
    ).create(
        document_id=DOCUMENT_ID,
        finding_id=response.finding.finding_id,
        comparison_request=request,
        command=_command(),
    )
    assert first.review_event.review_event_id != second.review_event.review_event_id
    assert first.authoritative_finding == second.authoritative_finding


def test_service_reconstructs_c3_with_exact_document_and_request(tmp_path) -> None:
    projection, response, c3, request = _projection(tmp_path)
    assert c3.calls == [(DOCUMENT_ID, request)]
    assert projection.authoritative_finding is response.finding


def test_fabricated_finding_id_and_wrong_document_are_rejected(tmp_path) -> None:
    response = _finding(tmp_path)
    request = _request(_preparation())
    service = FindingReviewService(
        review_finding_service=_StaticC3Service(response=response),
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(FindingReviewAuthorityMismatchError, match="finding_id"):
        service.create(
            document_id=DOCUMENT_ID,
            finding_id="finding_fabricated",
            comparison_request=request,
            command=_command(),
        )
    with pytest.raises(FindingReviewAuthorityMismatchError, match="document"):
        service.create(
            document_id="another-document",
            finding_id=response.finding.finding_id,
            comparison_request=request,
            command=_command(),
        )


def test_caller_supplied_finding_and_invalid_c3_response_are_rejected(tmp_path) -> None:
    response = _finding(tmp_path)
    service = FindingReviewService(
        review_finding_service=_StaticC3Service(response=response),
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(FindingReviewCommandError, match="not Finding authority"):
        service.create(
            document_id=DOCUMENT_ID,
            finding_id=response.finding.finding_id,
            comparison_request=response.finding,
            command=_command(),
        )
    invalid_service = FindingReviewService(
        review_finding_service=_StaticC3Service(response=response.finding),
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(FindingReviewAuthorityMismatchError, match="invalid"):
        invalid_service.create(
            document_id=DOCUMENT_ID,
            finding_id=response.finding.finding_id,
            comparison_request=_request(_preparation()),
            command=_command(),
        )


def test_c3_authority_unavailable_returns_no_event(tmp_path) -> None:
    error = ReviewFindingAuthorityUnavailableError("C.2 authority unavailable")
    service = FindingReviewService(
        review_finding_service=_StaticC3Service(error=error),
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(ReviewFindingAuthorityUnavailableError):
        service.create(
            document_id=DOCUMENT_ID,
            finding_id="finding_unknown",
            comparison_request=_request(_preparation()),
            command=_command(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("finding_id", "finding_tampered"),
        ("decision", ComparisonDecision.NON_COMPLIANT),
        ("reason_code", ComparisonReasonCode.NUMERIC_LIMIT_VIOLATED),
        ("decision_scope", "DOCUMENT"),
        ("summary", "The whole plan passed."),
    ],
)
def test_projection_rejects_tampered_c3_authority(tmp_path, field, value) -> None:
    projection, _, _, _ = _projection(tmp_path)
    tampered = projection.authoritative_finding.model_copy(update={field: value})
    with pytest.raises(ValidationError):
        FindingReviewProjection(
            authoritative_finding=tampered,
            review_event=projection.review_event,
        )


@pytest.mark.parametrize("citation_field", ["plan_citation", "standard_citation"])
def test_projection_rejects_tampered_citations(tmp_path, citation_field) -> None:
    projection, _, _, _ = _projection(tmp_path)
    citation = getattr(projection.authoritative_finding, citation_field)
    text_field = (
        "source_text" if citation_field == "plan_citation" else "requirement_text"
    )
    tampered_citation = citation.model_copy(update={text_field: "tampered"})
    tampered = projection.authoritative_finding.model_copy(
        update={citation_field: tampered_citation}
    )
    with pytest.raises(ValidationError):
        FindingReviewProjection(
            authoritative_finding=tampered,
            review_event=projection.review_event,
        )


@pytest.mark.parametrize(
    "disposition",
    [
        FindingReviewDisposition.ACCEPTED,
        FindingReviewDisposition.REJECTED,
        FindingReviewDisposition.NEEDS_INFORMATION,
        FindingReviewDisposition.DEFERRED,
    ],
)
def test_human_workflow_metadata_never_mutates_c3_authority(
    tmp_path, disposition
) -> None:
    projection, response, _, _ = _projection(
        tmp_path, command=_command(disposition=disposition)
    )
    finding = projection.authoritative_finding
    original = response.finding
    assert finding.finding_id == original.finding_id
    assert finding.decision == original.decision
    assert finding.reason_code == original.reason_code
    assert finding.decision_scope == ComparisonDecisionScope.REVIEW_UNIT_REQUIREMENT
    assert finding.plan_citation == original.plan_citation
    assert finding.standard_citation == original.standard_citation
    assert finding.summary == original.summary


def test_projection_is_only_authority_plus_event_and_has_no_report_authority(
    tmp_path,
) -> None:
    projection, _, _, _ = _projection(
        tmp_path,
        command=_command(
            disposition=FindingReviewDisposition.ACCEPTED,
            report_inclusion=ReportInclusionStatus.INCLUDE,
            reviewer_note="Human note: plan failed.",
        ),
    )
    assert set(FindingReviewProjection.model_fields) == {
        "authoritative_finding",
        "review_event",
    }
    assert "plan failed" not in projection.authoritative_finding.summary.lower()
    serialized = projection.model_dump()
    for forbidden in (
        "overall_status",
        "document_compliance",
        "report",
        "history",
        "conflict_winner",
    ):
        assert forbidden not in serialized
