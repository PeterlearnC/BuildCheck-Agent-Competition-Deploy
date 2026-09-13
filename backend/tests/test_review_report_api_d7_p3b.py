"""D.7-P3-B thin HTTP transport qualification tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
from typing import Any, get_args

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.routes.review_report import (
    _HTML_CONTENT_SECURITY_POLICY,
    _require_transportable_artifact,
    get_review_report_renderer,
    get_review_report_service,
)
from app.main import app
from app.schemas.compliance_comparison import ComparisonDecision
from app.schemas.finding_review import FindingReviewDisposition, ReportInclusionStatus
from app.schemas.findings_workspace import WorkspaceItemKind
from app.schemas.human_review import (
    InternalErrorHumanAction,
    ReviewGapHumanAction,
)
from app.schemas.review_report_artifact import (
    HTML_REPORT_MEDIA_TYPE,
    JSON_REPORT_MEDIA_TYPE,
    ReviewReportArtifactFormat,
)
from app.schemas.review_report_transport import (
    MAX_REPORT_ARTIFACT_BYTES,
    MAX_REVIEW_COMMANDS,
    ReviewReportDelivery,
    ReviewReportTransportRequest,
)
from app.services.pdf_service import InvalidPDFError, PDFNoExtractableTextError
from app.services.review.human_review_service import (
    HumanReviewStaleWorkspaceError,
    HumanReviewTargetError,
)
from app.services.review.review_report_renderer import (
    ReviewReportRenderError,
    ReviewReportRenderer,
)
from app.services.review.whole_plan_review_service import (
    WholePlanDocumentNotFoundError,
    WholePlanDocumentSourceChangedError,
)
from tests.test_human_review_d7 import _command, _human_env
from tests.test_review_report_renderer_d7_p3a import _report, _reviewed_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTE_PATH = PROJECT_ROOT / "backend" / "app" / "api" / "routes" / "review.py"
PROTECTED_REVIEW_ROUTE_PATH = ROUTE_PATH
ROUTE_PATH = (
    PROJECT_ROOT / "backend" / "app" / "api" / "routes" / "review_report.py"
)
SCHEMA_PATH = (
    PROJECT_ROOT / "backend" / "app" / "schemas" / "review_report_transport.py"
)
ENDPOINT = "/api/v1/documents/document-authority/review/report"


class _ReportService:
    def __init__(self, report=None, error: Exception | None = None) -> None:
        self.report = report
        self.error = error
        self.calls: list[tuple[str, tuple]] = []

    def build_report(self, document_id, commands):
        self.calls.append((document_id, commands))
        if self.error is not None:
            raise self.error
        return self.report


class _Renderer:
    def __init__(self, artifact=None, error: Exception | None = None) -> None:
        self.artifact = artifact
        self.error = error
        self.delegate = ReviewReportRenderer()
        self.json_calls = []
        self.html_calls = []

    def render_json(self, report):
        self.json_calls.append(report)
        if self.error is not None:
            raise self.error
        return self.artifact or self.delegate.render_json(report)

    def render_html(self, report):
        self.html_calls.append(report)
        if self.error is not None:
            raise self.error
        return self.artifact or self.delegate.render_html(report)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.pop(get_review_report_service, None)
    app.dependency_overrides.pop(get_review_report_renderer, None)


def _payload(*, format="JSON", delivery="INLINE", commands=None, **extra):
    return {
        "commands": [] if commands is None else commands,
        "format": format,
        "delivery": delivery,
        **extra,
    }


def _post(report, *, format="JSON", delivery="INLINE", commands=None, service=None, renderer=None):
    service = service or _ReportService(report)
    renderer = renderer or _Renderer()
    app.dependency_overrides[get_review_report_service] = lambda: service
    app.dependency_overrides[get_review_report_renderer] = lambda: renderer
    response = TestClient(app, raise_server_exceptions=False).post(
        ENDPOINT,
        json=_payload(format=format, delivery=delivery, commands=commands),
    )
    return response, service, renderer


def _one_command(tmp_path):
    source, _, _ = _human_env(tmp_path)
    return _command(source.workspace).model_dump(mode="json")


def test_schema_is_frozen_extra_forbid_and_immutable():
    assert ReviewReportTransportRequest.model_config["frozen"] is True
    assert ReviewReportTransportRequest.model_config["extra"] == "forbid"
    request = ReviewReportTransportRequest(
        commands=(), format=ReviewReportArtifactFormat.JSON, delivery=ReviewReportDelivery.INLINE
    )
    with pytest.raises(ValidationError):
        request.delivery = ReviewReportDelivery.ATTACHMENT


def test_schema_has_no_any_or_mutable_defaults():
    for field in ReviewReportTransportRequest.model_fields.values():
        assert field.annotation is not Any
        assert Any not in get_args(field.annotation)
        assert not isinstance(field.default, (list, dict, set))


@pytest.mark.parametrize(
    "field",
    [
        "report_model", "report_model_id", "artifact", "artifact_id",
        "content_sha256", "filename", "media_type", "machine_decision",
        "finding", "plan_fact", "counts", "coverage", "source",
        "overall_compliance", "filesystem_path",
    ],
)
def test_schema_rejects_caller_authority_fields(field):
    with pytest.raises(ValidationError):
        ReviewReportTransportRequest.model_validate(_payload(**{field: "forged"}))


@pytest.mark.parametrize("value", ["PDF", "DOCX", "XML", "json", ""])
def test_format_is_strict_and_unsupported_values_reject(value):
    with pytest.raises(ValidationError):
        ReviewReportTransportRequest.model_validate(_payload(format=value))


@pytest.mark.parametrize("value", ["DOWNLOAD", "PREVIEW", "inline", ""])
def test_delivery_is_strict_and_unknown_values_reject(value):
    with pytest.raises(ValidationError):
        ReviewReportTransportRequest.model_validate(_payload(delivery=value))


def test_zero_commands_are_valid():
    parsed = ReviewReportTransportRequest.model_validate(_payload())
    assert parsed.commands == ()


def test_exact_command_limit_is_valid(tmp_path):
    command = _one_command(tmp_path)
    parsed = ReviewReportTransportRequest.model_validate(
        _payload(commands=[command] * MAX_REVIEW_COMMANDS)
    )
    assert len(parsed.commands) == MAX_REVIEW_COMMANDS


def test_command_overflow_is_rejected_by_schema(tmp_path):
    command = _one_command(tmp_path)
    with pytest.raises(ValidationError):
        ReviewReportTransportRequest.model_validate(
            _payload(commands=[command] * (MAX_REVIEW_COMMANDS + 1))
        )


def test_command_overflow_rejects_before_service_invocation(tmp_path):
    service = _ReportService(object())
    renderer = _Renderer()
    app.dependency_overrides[get_review_report_service] = lambda: service
    app.dependency_overrides[get_review_report_renderer] = lambda: renderer
    command = _one_command(tmp_path)
    response = TestClient(app).post(
        ENDPOINT,
        json=_payload(commands=[command] * (MAX_REVIEW_COMMANDS + 1)),
    )
    assert response.status_code == 422
    assert service.calls == []
    assert renderer.json_calls == renderer.html_calls == []


def test_endpoint_is_registered_once():
    matches = [
        route
        for route in app.routes
        if getattr(route, "path", None) == "/api/v1/documents/{document_id}/review/report"
        and "POST" in getattr(route, "methods", set())
    ]
    assert len(matches) == 1


def test_protected_review_router_is_byte_identical_to_head():
    relative = "backend/app/api/routes/review.py"
    head_bytes = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    assert PROTECTED_REVIEW_ROUTE_PATH.read_bytes() == head_bytes


@pytest.mark.parametrize("format,selected", [("JSON", "json"), ("HTML", "html")])
def test_service_and_selected_renderer_are_each_called_once(tmp_path, format, selected):
    report = _report(tmp_path)
    response, service, renderer = _post(report, format=format)
    assert response.status_code == 200
    assert len(service.calls) == 1
    assert len(renderer.json_calls) == (selected == "json")
    assert len(renderer.html_calls) == (selected == "html")


@pytest.mark.parametrize("format", ["JSON", "HTML"])
def test_response_body_is_exact_frozen_artifact_content(tmp_path, format):
    report = _report(tmp_path)
    expected = getattr(ReviewReportRenderer(), f"render_{format.lower()}")(report)
    response, _, _ = _post(report, format=format)
    assert response.content == expected.content
    assert len(response.content) == expected.byte_length


@pytest.mark.parametrize(
    "format,media_type", [("JSON", JSON_REPORT_MEDIA_TYPE), ("HTML", HTML_REPORT_MEDIA_TYPE)]
)
def test_content_type_is_exact_artifact_media_type(tmp_path, format, media_type):
    response, _, _ = _post(_report(tmp_path), format=format)
    assert response.headers["content-type"] == media_type


@pytest.mark.parametrize(
    "format,delivery,prefix,extension",
    [
        ("JSON", "INLINE", "inline", ".json"),
        ("JSON", "ATTACHMENT", "attachment", ".json"),
        ("HTML", "INLINE", "inline", ".html"),
        ("HTML", "ATTACHMENT", "attachment", ".html"),
    ],
)
def test_content_disposition_uses_artifact_filename_only(
    tmp_path, format, delivery, prefix, extension
):
    report = _report(tmp_path)
    response, _, _ = _post(report, format=format, delivery=delivery)
    assert response.headers["content-disposition"] == (
        f'{prefix}; filename="{report.report_model_id}{extension}"'
    )


@pytest.mark.parametrize(
    "header,attribute",
    [
        ("x-report-model-id", "report_model_id"),
        ("x-report-artifact-id", "artifact_id"),
        ("x-content-sha256", "content_sha256"),
    ],
)
def test_identity_headers_come_from_artifact(tmp_path, header, attribute):
    report = _report(tmp_path)
    artifact = ReviewReportRenderer().render_json(report)
    response, _, _ = _post(report)
    assert response.headers[header] == getattr(artifact, attribute)


@pytest.mark.parametrize("format", ["JSON", "HTML"])
def test_all_formats_disable_cache_and_sniffing(tmp_path, format):
    response, _, _ = _post(_report(tmp_path), format=format)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("delivery", ["INLINE", "ATTACHMENT"])
def test_html_has_restrictive_csp_and_referrer_policy(tmp_path, delivery):
    response, _, _ = _post(_report(tmp_path), format="HTML", delivery=delivery)
    assert response.headers["content-security-policy"] == _HTML_CONTENT_SECURITY_POLICY
    assert response.headers["referrer-policy"] == "no-referrer"
    csp = response.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "style-src 'unsafe-inline'" in csp
    assert "base-uri 'none'" in csp
    assert "form-action 'none'" in csp
    assert "frame-ancestors 'none'" in csp


def test_json_omits_html_specific_headers(tmp_path):
    response, _, _ = _post(_report(tmp_path))
    assert "content-security-policy" not in response.headers
    assert "referrer-policy" not in response.headers


@pytest.mark.parametrize("format", ["JSON", "HTML"])
def test_delivery_changes_only_transport_headers_not_artifact_bytes_or_identity(tmp_path, format):
    report = _report(tmp_path)
    inline, _, _ = _post(report, format=format, delivery="INLINE")
    attachment, _, _ = _post(report, format=format, delivery="ATTACHMENT")
    assert inline.content == attachment.content
    assert inline.headers["x-content-sha256"] == attachment.headers["x-content-sha256"]
    assert inline.headers["x-report-artifact-id"] == attachment.headers["x-report-artifact-id"]
    assert inline.headers["content-disposition"] != attachment.headers["content-disposition"]


@pytest.mark.parametrize(
    "error,expected",
    [
        (WholePlanDocumentNotFoundError("private path"), 404),
        (HumanReviewStaleWorkspaceError("private path"), 409),
        (WholePlanDocumentSourceChangedError("private path"), 409),
        (HumanReviewTargetError("private path"), 422),
        (PDFNoExtractableTextError("private path"), 422),
        (InvalidPDFError("private path"), 400),
    ],
)
def test_expected_service_errors_map_narrowly_without_leakage(tmp_path, error, expected):
    service = _ReportService(error=error)
    response, _, renderer = _post(_report(tmp_path), service=service)
    assert response.status_code == expected
    assert "private path" not in response.text
    assert renderer.json_calls == renderer.html_calls == []


def test_renderer_failure_is_generic_500_without_partial_report(tmp_path):
    response, service, renderer = _post(
        _report(tmp_path), renderer=_Renderer(error=ReviewReportRenderError("C:/secret"))
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Report generation failed safely."}
    assert "secret" not in response.text.casefold()
    assert len(service.calls) == len(renderer.json_calls) == 1


def test_invalid_header_filename_fails_closed_before_response(tmp_path):
    report = _report(tmp_path)
    artifact = ReviewReportRenderer().render_json(report).model_copy(
        update={"filename": "safe.json\r\nX-Injected: yes"}
    )
    response, _, _ = _post(report, renderer=_Renderer(artifact=artifact))
    assert response.status_code == 500
    assert "x-injected" not in response.headers
    assert "safe.json" not in response.text


@pytest.mark.parametrize("filename", ["../x.json", "dir/x.json", "dir\\x.json", "x\r.json", "x\n.json"])
def test_transport_filename_guard_rejects_traversal_and_header_injection(tmp_path, filename):
    report = _report(tmp_path)
    artifact = ReviewReportRenderer().render_json(report).model_copy(
        update={"filename": filename}
    )
    with pytest.raises(Exception, match="unsafe"):
        _require_transportable_artifact(
            artifact,
            report_model_id=report.report_model_id,
            expected_format=ReviewReportArtifactFormat.JSON,
        )


def test_artifact_limit_exact_boundary_is_allowed_without_allocating_64_mib(tmp_path):
    report = _report(tmp_path)
    artifact = ReviewReportRenderer().render_json(report).model_copy(
        update={"byte_length": MAX_REPORT_ARTIFACT_BYTES}
    )
    _require_transportable_artifact(
        artifact,
        report_model_id=report.report_model_id,
        expected_format=ReviewReportArtifactFormat.JSON,
    )


def test_artifact_limit_plus_one_maps_to_413_without_truncation(tmp_path):
    report = _report(tmp_path)
    original = ReviewReportRenderer().render_json(report)
    oversized = original.model_copy(
        update={"byte_length": MAX_REPORT_ARTIFACT_BYTES + 1}
    )
    response, service, renderer = _post(report, renderer=_Renderer(artifact=oversized))
    assert response.status_code == 413
    assert response.json() == {"detail": "Rendered report exceeds the transport size limit."}
    assert len(service.calls) == len(renderer.json_calls) == 1
    assert response.content != original.content[:1]


def test_artifact_report_identity_mismatch_fails_closed(tmp_path):
    report = _report(tmp_path)
    artifact = ReviewReportRenderer().render_json(report).model_copy(
        update={"report_model_id": "reviewreport_" + "0" * 64}
    )
    response, _, _ = _post(report, renderer=_Renderer(artifact=artifact))
    assert response.status_code == 500


def test_artifact_format_mismatch_fails_closed(tmp_path):
    report = _report(tmp_path)
    artifact = ReviewReportRenderer().render_html(report)
    response, _, _ = _post(report, format="JSON", renderer=_Renderer(artifact=artifact))
    assert response.status_code == 500


@pytest.mark.parametrize(
    "decision,action",
    [
        (ComparisonDecision.COMPLIANT, FindingReviewDisposition.REJECTED),
        (ComparisonDecision.NON_COMPLIANT, FindingReviewDisposition.REJECTED),
        (ComparisonDecision.INSUFFICIENT_INFORMATION, FindingReviewDisposition.ACCEPTED),
    ],
)
def test_machine_and_human_axes_survive_http_exactly(tmp_path, decision, action):
    report = _reviewed_report(tmp_path, decision=decision, action=action)
    response, _, _ = _post(report)
    assert response.content == ReviewReportRenderer().render_json(report).content
    finding = response.json()["findings"][0]
    assert finding["machine_decision"] == decision.value
    assert finding["human_review"]["human_action"] == action.value
    assert "final_decision" not in finding


@pytest.mark.parametrize(
    "kind,action,section",
    [
        (WorkspaceItemKind.REVIEW_GAP, ReviewGapHumanAction.ACKNOWLEDGED, "review_gaps"),
        (WorkspaceItemKind.INTERNAL_ERROR, InternalErrorHumanAction.ACKNOWLEDGED, "internal_errors"),
    ],
)
def test_gap_and_error_sections_survive_http_without_conversion(tmp_path, kind, action, section):
    report = _reviewed_report(tmp_path, kind=kind, action=action)
    response, _, _ = _post(report)
    payload = response.json()
    assert payload[section]
    assert payload["findings"] == []
    assert response.content == ReviewReportRenderer().render_json(report).content


@pytest.mark.parametrize(
    "kind,action",
    [
        (WorkspaceItemKind.FINDING, FindingReviewDisposition.ACCEPTED),
        (WorkspaceItemKind.REVIEW_GAP, ReviewGapHumanAction.ACKNOWLEDGED),
        (WorkspaceItemKind.INTERNAL_ERROR, InternalErrorHumanAction.ACKNOWLEDGED),
    ],
)
def test_excluded_accounting_survives_http(tmp_path, kind, action):
    report = _reviewed_report(
        tmp_path, kind=kind, action=action, inclusion=ReportInclusionStatus.EXCLUDE
    )
    response, _, _ = _post(report)
    assert response.json()["excluded_items"] == report.model_dump(mode="json")["excluded_items"]


def test_chinese_utf8_bytes_are_not_decoded_or_reencoded(tmp_path):
    text = "施工方案 审查结论 人工复核 规范条文 间距不得大于3mm"
    report = _reviewed_report(tmp_path, note=text, reviewer_id="复核员甲")
    for format in ("JSON", "HTML"):
        expected = getattr(ReviewReportRenderer(), f"render_{format.lower()}")(report)
        response, _, _ = _post(report, format=format)
        assert response.content == expected.content
        assert text.encode("utf-8") in response.content


def test_case_a_exact_artifact_transport(tmp_path):
    report = _reviewed_report(
        tmp_path,
        decision=ComparisonDecision.COMPLIANT,
        action=FindingReviewDisposition.REJECTED,
    )
    response, _, _ = _post(report, format="HTML")
    assert response.content == ReviewReportRenderer().render_html(report).content


def test_case_b_has_no_route_numeric_recomparison_or_nearby_contamination(tmp_path):
    report = _reviewed_report(
        tmp_path,
        decision=ComparisonDecision.NON_COMPLIANT,
        action=FindingReviewDisposition.ACCEPTED,
    )
    response, _, _ = _post(report)
    assert response.content == ReviewReportRenderer().render_json(report).content
    route_source = ROUTE_PATH.read_text(encoding="utf-8")
    assert "200 mm" not in route_source
    assert "2.5 mm" not in route_source
    assert "3 mm" not in route_source


def test_case_c_gap_transport_success_is_not_compliance_status(tmp_path):
    report = _reviewed_report(
        tmp_path,
        kind=WorkspaceItemKind.REVIEW_GAP,
        action=ReviewGapHumanAction.NEEDS_FOLLOW_UP,
    )
    response, _, _ = _post(report)
    assert response.status_code == 200
    assert response.json()["review_gaps"]
    assert response.json()["findings"] == []


def test_http_200_and_headers_make_no_compliance_claim(tmp_path):
    response, _, _ = _post(_report(tmp_path))
    assert response.status_code == 200
    header_text = json.dumps(dict(response.headers)).casefold()
    for forbidden in ("plan compliant", "approved", "passed", "safe"):
        assert forbidden not in header_text


def test_public_route_has_exact_dependency_controlled_seam():
    from app.api.routes.review_report import render_review_report

    assert list(inspect.signature(render_review_report).parameters) == [
        "document_id", "request", "reports", "renderer"
    ]


@pytest.mark.parametrize(
    "forbidden",
    [
        "HumanReviewService(", "FindingsWorkspaceService(", "WholePlanReviewService(",
        "ComplianceComparisonService.compare", "json.dumps", "<html", "html.escape",
        "PlanFact(", "Finding(", "overall_compliance", "risk_score", "severity",
        "subprocess", "requests.", "httpx.", "Path.write", "open(", "FileResponse",
        "StreamingResponse", "reportlab", "docx", "ocr", "llm_service",
    ],
)
def test_p3b_route_contains_no_semantic_renderer_or_external_authority_logic(forbidden):
    source = ROUTE_PATH.read_text(encoding="utf-8")
    appended = source[source.index('@router.post("/{document_id}/review/report"') :]
    assert forbidden.casefold() not in appended.casefold()


def test_transport_schema_records_exact_provisional_limits():
    assert MAX_REVIEW_COMMANDS == 2000
    assert MAX_REPORT_ARTIFACT_BYTES == 64 * 1024 * 1024
    source = SCHEMA_PATH.read_text(encoding="utf-8")
    assert "MAX_REVIEW_COMMANDS = 2000" in source
    assert "64 * 1024 * 1024" in source


def test_raw_request_limit_is_not_falsely_claimed_in_application_code():
    source = (ROUTE_PATH.read_text(encoding="utf-8") + SCHEMA_PATH.read_text(encoding="utf-8"))
    assert "RAW_REQUEST_BODY_LIMIT" not in source
    assert "middleware" not in source.casefold()


def test_transport_has_no_authentication_or_persistence_claim():
    source = (ROUTE_PATH.read_text(encoding="utf-8") + SCHEMA_PATH.read_text(encoding="utf-8")).casefold()
    for forbidden in ("authenticated", "verified reviewer", "database", "redis", "repository.save", "write_bytes"):
        assert forbidden not in source
