"""D.7-P3-A deterministic in-memory JSON/HTML renderer tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from app.schemas.compliance_comparison import ComparisonDecision
from app.schemas.finding_review import FindingReviewDisposition, ReportInclusionStatus
from app.schemas.findings_workspace import WorkspaceItemKind
from app.schemas.review_report import ReviewReportModel
from app.schemas.review_report_artifact import (
    HTML_REPORT_MEDIA_TYPE,
    HTML_REPORT_RENDERER_VERSION,
    JSON_REPORT_MEDIA_TYPE,
    JSON_REPORT_RENDERER_VERSION,
    REPORT_ARTIFACT_IDENTITY_VERSION,
    ReviewReportArtifact,
    ReviewReportArtifactFormat,
    deterministic_report_artifact_id,
)
from app.services.review.human_review_service import HumanReviewService
from app.services.review.review_report_renderer import (
    ReviewReportRenderError,
    ReviewReportRenderer,
)
from app.services.review.review_report_service import ReviewReportService
from tests.test_findings_workspace_d6 import _workspace
from tests.test_human_review_d7 import FIXED_TIME, _WorkspaceService, _command, _human_env
from tests.test_review_report_d7_p2 import _report_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "backend" / "app" / "schemas" / "review_report_artifact.py"
RENDERER_PATH = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "services"
    / "review"
    / "review_report_renderer.py"
)


def _renderer() -> ReviewReportRenderer:
    return ReviewReportRenderer()


def _report(tmp_path, **kwargs) -> ReviewReportModel:
    return _report_env(tmp_path, **kwargs)[-1]


def _reviewed_report(
    tmp_path,
    *,
    kind=WorkspaceItemKind.FINDING,
    decision=None,
    action=None,
    reviewer_id="reviewer-one",
    note="Reviewed.",
    inclusion=ReportInclusionStatus.UNDECIDED,
):
    source, _, p1 = _human_env(tmp_path, kind=kind, decision=decision)
    command = _command(
        source.workspace,
        action=action,
        reviewer_id=reviewer_id,
        note=note,
        inclusion=inclusion,
    )
    human = p1.review_document(source.env.document_id, (command,))
    return ReviewReportService._project_report(human)


def _empty_report(tmp_path) -> ReviewReportModel:
    source = _workspace(tmp_path, candidate_texts=())
    human = HumanReviewService(
        findings_workspace_service=_WorkspaceService(source.workspace),
        clock=lambda: FIXED_TIME,
    ).review_document(source.env.document_id, ())
    return ReviewReportService._project_report(human)


def _html(artifact: ReviewReportArtifact) -> str:
    return artifact.content.decode("utf-8")


def test_artifact_schema_is_frozen_extra_forbid_and_runtime_immutable(tmp_path):
    assert ReviewReportArtifact.model_config["frozen"] is True
    assert ReviewReportArtifact.model_config["extra"] == "forbid"
    artifact = _renderer().render_json(_report(tmp_path))
    with pytest.raises(ValidationError):
        artifact.filename = "changed.json"
    payload = artifact.model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ReviewReportArtifact.model_validate(payload)


def test_artifact_schema_has_no_any_authority_or_mutable_defaults():
    for field in ReviewReportArtifact.model_fields.values():
        assert field.annotation is not Any
        assert Any not in get_args(field.annotation)
        assert not isinstance(field.default, (list, dict, set))


@pytest.mark.parametrize(
    "forbidden",
    [
        "filesystem_path",
        "output_path",
        "generated_at",
        "final_decision",
        "overall_compliance",
        "whole_plan_verdict",
        "severity",
        "risk_score",
        "compliance_percentage",
    ],
)
def test_artifact_schema_has_no_path_time_verdict_or_score_fields(forbidden):
    assert forbidden not in ReviewReportArtifact.model_fields


def test_public_renderer_seams_accept_only_review_report_model():
    assert list(inspect.signature(ReviewReportRenderer.render_json).parameters) == [
        "self",
        "report",
    ]
    assert list(inspect.signature(ReviewReportRenderer.render_html).parameters) == [
        "self",
        "report",
    ]


@pytest.mark.parametrize("method", ["render_json", "render_html"])
def test_renderer_rejects_non_report_input_with_typed_error(method):
    with pytest.raises(ReviewReportRenderError):
        getattr(_renderer(), method)(object())


def test_json_is_exact_canonical_frozen_model_serialization(tmp_path):
    report = _report(tmp_path)
    artifact = _renderer().render_json(report)
    expected = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert artifact.content == expected
    assert artifact.content.startswith(b'{"counts":')
    assert b" " not in artifact.content[:20]


def test_json_losslessly_roundtrips_complete_semantic_model(tmp_path):
    report = _report(tmp_path)
    artifact = _renderer().render_json(report)
    parsed = ReviewReportModel.model_validate(json.loads(artifact.content))
    assert parsed == report
    assert parsed.findings == report.findings
    assert parsed.review_gaps == report.review_gaps
    assert parsed.internal_errors == report.internal_errors
    assert parsed.excluded_items == report.excluded_items
    assert parsed.counts == report.counts
    assert parsed.coverage == report.coverage


def test_json_preserves_chinese_without_ascii_escaping(tmp_path):
    text = "施工方案 审查结论 人工复核 规范条文 间距不得大于3mm"
    report = _reviewed_report(tmp_path, note=text, reviewer_id="复核员甲")
    artifact = _renderer().render_json(report)
    assert text.encode("utf-8") in artifact.content
    assert "复核员甲".encode("utf-8") in artifact.content
    assert b"\\u65bd\\u5de5" not in artifact.content


def test_same_report_produces_identical_json_bytes_hash_and_id(tmp_path):
    report = _report(tmp_path)
    first = _renderer().render_json(report)
    second = _renderer().render_json(report)
    assert first == second
    assert first.content == second.content
    assert first.content_sha256 == second.content_sha256
    assert first.artifact_id == second.artifact_id


def test_html_is_deterministic_standalone_utf8_document(tmp_path):
    report = _report(tmp_path)
    first = _renderer().render_html(report)
    second = _renderer().render_html(report)
    html = _html(first)
    assert first == second
    assert html.startswith("<!DOCTYPE html>\n<html lang=\"zh-CN\">")
    assert '<meta charset="utf-8">' in html
    assert "<style>" in html and "</style>" in html
    assert "<script" not in html.casefold()
    assert "<iframe" not in html.casefold()
    assert "http://" not in html and "https://" not in html
    assert " src=" not in html and " href=" not in html
    assert "url(" not in html.casefold()


@pytest.mark.parametrize(
    "malicious",
    [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "\"&<>'",
    ],
)
def test_central_text_and_source_escaping_never_emits_executable_markup(malicious):
    escaped_text = ReviewReportRenderer._text(malicious)
    escaped_source = ReviewReportRenderer._json_block(
        {"source_text": malicious, "article_text": malicious, "document_id": malicious}
    )
    assert malicious not in escaped_text
    assert malicious not in escaped_source
    assert "&lt;" in escaped_text or "&quot;" in escaped_text
    assert "<script" not in escaped_source.casefold()
    assert "<img" not in escaped_source.casefold()


def test_malicious_reviewer_id_and_note_are_escaped(tmp_path):
    reviewer = '<img src=x onerror=alert(1)>'
    note = '<script>alert("review")</script> & "quoted"'
    report = _reviewed_report(tmp_path, reviewer_id=reviewer, note=note)
    html = _html(_renderer().render_html(report))
    assert reviewer not in html
    assert note not in html
    assert "&lt;img" in html
    assert "&lt;script&gt;" in html
    assert "onerror=alert(1)" in html
    assert "<script" not in html.casefold()


def test_all_required_html_sections_have_fixed_semantic_order(tmp_path):
    html = _html(_renderer().render_html(_report(tmp_path)))
    headings = [
        "1. Document Identity",
        "2. Processing / Coverage Summary",
        "3. Findings",
        "4. Review Gaps",
        "5. Internal Errors",
        "6. Human Review State",
        "7. Excluded Item Audit",
        "8. Coverage / Limitations",
        "9. Source References as Retained",
    ]
    positions = [html.index(value) for value in headings]
    assert positions == sorted(positions)


@pytest.mark.parametrize(
    ("decision", "action"),
    [
        (ComparisonDecision.COMPLIANT, FindingReviewDisposition.REJECTED),
        (ComparisonDecision.NON_COMPLIANT, FindingReviewDisposition.REJECTED),
        (
            ComparisonDecision.INSUFFICIENT_INFORMATION,
            FindingReviewDisposition.ACCEPTED,
        ),
    ],
)
def test_finding_machine_and_human_axes_remain_separate(tmp_path, decision, action):
    report = _reviewed_report(tmp_path, decision=decision, action=action)
    html = _html(_renderer().render_html(report))
    assert "Machine assessment" in html
    assert decision.value in html
    assert "Human review" in html
    assert action.value in html
    assert "Final result" not in html
    assert "Effective result" not in html
    assert "Final compliance" not in html
    assert report.findings[0].machine_decision == decision


def test_finding_displays_only_retained_identity_decision_and_sources(tmp_path):
    report = _report(tmp_path, decision=ComparisonDecision.NON_COMPLIANT)
    entry = report.findings[0]
    html = _html(_renderer().render_html(report))
    for value in (
        entry.finding_id,
        entry.comparison_id,
        entry.machine_decision.value,
        entry.reason_code.value,
        entry.plan_source.source_text,
        entry.standard_source.source_text,
    ):
        assert ReviewReportRenderer._text(value) in html
    for forbidden in ("Severity", "Risk level", "Recommendation"):
        assert forbidden not in html


def test_review_gap_is_separate_neutral_section(tmp_path):
    report = _report(tmp_path, kind=WorkspaceItemKind.REVIEW_GAP)
    gap = report.review_gaps[0]
    html = _html(_renderer().render_html(report))
    assert "<h2>4. Review Gaps</h2>" in html
    assert gap.review_gap_id in html
    assert gap.gap_source.value in html
    assert not report.findings
    assert "Gap severity" not in html


def test_internal_error_is_separate_operational_section_without_traceback(tmp_path):
    report = _report(tmp_path, kind=WorkspaceItemKind.INTERNAL_ERROR)
    error = report.internal_errors[0]
    html = _html(_renderer().render_html(report))
    assert "<h2>5. Internal Errors</h2>" in html
    assert error.internal_error_item_id in html
    assert error.error_stage.value in html
    assert error.error_code.value in html
    assert "Traceback" not in html
    assert not report.findings


@pytest.mark.parametrize("kind", list(WorkspaceItemKind))
def test_excluded_item_audit_is_never_erased(tmp_path, kind):
    report = _report(
        tmp_path,
        kind=kind,
        inclusion=ReportInclusionStatus.EXCLUDE,
    )
    excluded = report.excluded_items[0]
    html = _html(_renderer().render_html(report))
    assert "<h2>7. Excluded Item Audit</h2>" in html
    assert excluded.source_item_id in html
    assert excluded.item_kind.value in html
    assert excluded.review_record_id in html
    assert "EXCLUDE" in html
    status = excluded.machine_decision or excluded.gap_source or excluded.error_code
    assert status.value in html


@pytest.mark.parametrize("kind", list(WorkspaceItemKind))
def test_unreviewed_items_remain_visible_without_invented_metadata(tmp_path, kind):
    report = _report(tmp_path, kind=kind, reviewed=False)
    html = _html(_renderer().render_html(report))
    assert "Human review:</strong> Unreviewed" in html
    assert report.counts.human_unreviewed == 1
    assert "Reviewer ID</dt>" not in html
    assert "Reviewer note</dt>" not in html


def test_frozen_coverage_statement_is_rendered_exactly(tmp_path):
    report = _report(tmp_path)
    html = _html(_renderer().render_html(report))
    assert report.coverage.code in html
    assert report.coverage.statement in html


def test_zero_findings_with_gap_is_not_misleading(tmp_path):
    report = _report(tmp_path, kind=WorkspaceItemKind.REVIEW_GAP)
    html = _html(_renderer().render_html(report))
    assert "No detailed entries in this section." in html
    assert report.review_gaps[0].review_gap_id in html
    assert "No problems found" not in html
    assert "Plan compliant" not in html
    assert "Passed review" not in html
    assert "No violations" not in html


def test_all_primary_details_excluded_still_show_audit_authority(tmp_path):
    report = _report(tmp_path, inclusion=ReportInclusionStatus.EXCLUDE)
    html = _html(_renderer().render_html(report))
    assert not report.findings
    assert report.excluded_items[0].excluded_reference_id in html
    assert "EXCLUDE" in html
    assert "No problems found" not in html


def test_zero_workspace_items_render_empty_projection_without_verdict(tmp_path):
    report = _empty_report(tmp_path)
    json_artifact = _renderer().render_json(report)
    html = _html(_renderer().render_html(report))
    assert ReviewReportModel.model_validate_json(json_artifact.content) == report
    assert html.count("No detailed entries in this section.") >= 5
    for forbidden in ("No problems found", "Plan compliant", "Passed review", "Approved"):
        assert forbidden not in html


def test_html_preserves_chinese_utf8_and_uses_only_system_font_stack(tmp_path):
    text = "施工方案 审查结论 人工复核 规范条文 间距不得大于3mm"
    report = _reviewed_report(tmp_path, reviewer_id="复核员", note=text)
    artifact = _renderer().render_html(report)
    html = _html(artifact)
    assert text in html
    assert "复核员" in html
    assert '"Microsoft YaHei"' in html
    assert '"PingFang SC"' in html
    assert '"Noto Sans CJK SC"' in html
    assert "@font-face" not in html


@pytest.mark.parametrize("method", ["render_json", "render_html"])
def test_artifact_metadata_matches_exact_bytes(method, tmp_path):
    report = _report(tmp_path)
    artifact = getattr(_renderer(), method)(report)
    assert artifact.report_model_id == report.report_model_id
    assert artifact.byte_length == len(artifact.content)
    assert artifact.content_sha256 == hashlib.sha256(artifact.content).hexdigest()
    assert artifact.artifact_id == deterministic_report_artifact_id(
        report_model_id=artifact.report_model_id,
        artifact_format=artifact.format,
        renderer_version=artifact.renderer_version,
        content_sha256=artifact.content_sha256,
        media_type=artifact.media_type,
    )


def test_json_and_html_artifact_identities_differ_without_semantic_mutation(tmp_path):
    report = _report(tmp_path)
    original_dump = report.model_dump(mode="json")
    json_artifact = _renderer().render_json(report)
    html_artifact = _renderer().render_html(report)
    assert json_artifact.report_model_id == html_artifact.report_model_id
    assert json_artifact.report_model_id == report.report_model_id
    assert json_artifact.content_sha256 != html_artifact.content_sha256
    assert json_artifact.artifact_id != html_artifact.artifact_id
    assert report.model_dump(mode="json") == original_dump


def test_renderer_version_is_part_of_artifact_identity(tmp_path):
    artifact = _renderer().render_json(_report(tmp_path))
    changed = deterministic_report_artifact_id(
        report_model_id=artifact.report_model_id,
        artifact_format=artifact.format,
        renderer_version=artifact.renderer_version + "-changed",
        content_sha256=artifact.content_sha256,
        media_type=artifact.media_type,
    )
    assert changed != artifact.artifact_id


@pytest.mark.parametrize(
    ("method", "artifact_format", "version", "media_type", "extension"),
    [
        (
            "render_json",
            ReviewReportArtifactFormat.JSON,
            JSON_REPORT_RENDERER_VERSION,
            JSON_REPORT_MEDIA_TYPE,
            ".json",
        ),
        (
            "render_html",
            ReviewReportArtifactFormat.HTML,
            HTML_REPORT_RENDERER_VERSION,
            HTML_REPORT_MEDIA_TYPE,
            ".html",
        ),
    ],
)
def test_safe_filename_fixed_media_type_and_version(
    tmp_path, method, artifact_format, version, media_type, extension
):
    report = _report(tmp_path)
    artifact = getattr(_renderer(), method)(report)
    assert artifact.format == artifact_format
    assert artifact.renderer_version == version
    assert artifact.media_type == media_type
    assert artifact.filename == report.report_model_id + extension
    assert "/" not in artifact.filename and "\\" not in artifact.filename


@pytest.mark.parametrize(
    "field,value",
    [
        ("content_sha256", "0" * 64),
        ("byte_length", 999999),
        ("filename", "../report.json"),
        ("media_type", "application/octet-stream"),
        ("renderer_version", "caller-version"),
        ("artifact_id", "reportartifact_" + "0" * 64),
    ],
)
def test_artifact_schema_fails_closed_on_tampered_metadata(tmp_path, field, value):
    artifact = _renderer().render_json(_report(tmp_path))
    payload = artifact.model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        ReviewReportArtifact.model_validate(payload)


def test_identity_version_is_fixed_and_deterministic(tmp_path):
    artifact = _renderer().render_json(_report(tmp_path))
    assert artifact.identity_version == REPORT_ARTIFACT_IDENTITY_VERSION
    assert artifact.artifact_id.startswith("reportartifact_")
    assert len(artifact.artifact_id) == len("reportartifact_") + 64


def test_production_renderer_has_no_filesystem_network_subprocess_or_template_engine():
    source = RENDERER_PATH.read_text(encoding="utf-8")
    forbidden = (
        "Path(",
        ".write_text(",
        ".write_bytes(",
        "tempfile",
        "shutil",
        "subprocess",
        "requests",
        "httpx",
        "urllib",
        "Jinja",
        "weasyprint",
        "wkhtmltopdf",
        "reportlab",
        "python-docx",
        "playwright",
        "selenium",
    )
    assert all(term not in source for term in forbidden)


def test_production_renderer_has_no_upstream_service_or_source_repository_calls():
    source = RENDERER_PATH.read_text(encoding="utf-8")
    forbidden = (
        "HumanReviewService",
        "ReviewReportService",
        "FindingsWorkspaceService",
        "WholePlanReviewService",
        "ComplianceComparisonService",
        "ReviewFindingService",
        "FindingReviewService",
        "StandardRepository",
        "PDFService",
        ".review_document(",
        ".build_report(",
        ".build_workspace(",
        ".compare(",
        ".create(",
    )
    assert all(term not in source for term in forbidden)


def test_production_paths_have_no_api_frontend_pdf_docx_ocr_llm_or_network_imports():
    source = SCHEMA_PATH.read_text(encoding="utf-8") + RENDERER_PATH.read_text(
        encoding="utf-8"
    )
    forbidden = (
        "fastapi",
        "APIRouter",
        "HTMLResponse",
        "FileResponse",
        "React",
        "PDF",
        "DOCX",
        "OCR",
        "LLM",
        "socket",
    )
    assert all(term not in source for term in forbidden)


def test_production_has_no_report_file_open_or_partial_fallback():
    source = RENDERER_PATH.read_text(encoding="utf-8")
    assert "open(" not in source
    assert "except Exception" not in source
    assert "return b\"\"" not in source
    assert "ReviewReportRenderError" in source


@pytest.mark.parametrize(
    "forbidden",
    [
        "GB55023",
        "4.4.15",
        "4.4.16",
        "compliance percentage",
        "risk score",
        "whole-plan verdict",
        "No problems found",
        "Plan compliant",
        "Passed review",
    ],
)
def test_static_authority_isolation(forbidden):
    production = SCHEMA_PATH.read_text(encoding="utf-8") + RENDERER_PATH.read_text(
        encoding="utf-8"
    )
    assert forbidden not in production


def test_case_a_compliant_machine_rejected_human_remain_separate(tmp_path):
    report = _reviewed_report(
        tmp_path,
        decision=ComparisonDecision.COMPLIANT,
        action=FindingReviewDisposition.REJECTED,
    )
    html = _html(_renderer().render_html(report))
    parsed = json.loads(_renderer().render_json(report).content)
    assert "COMPLIANT" in html and "REJECTED" in html
    assert parsed["findings"][0]["machine_decision"] == "COMPLIANT"
    assert parsed["findings"][0]["human_review"]["human_action"] == "REJECTED"


def test_case_b_non_compliant_values_are_presented_without_recomparison(tmp_path):
    report = _report(tmp_path, decision=ComparisonDecision.NON_COMPLIANT)
    html = _html(_renderer().render_html(report))
    assert "NON_COMPLIANT" in html
    assert "3mm" in html
    source = RENDERER_PATH.read_text(encoding="utf-8")
    assert "200 mm" not in source and "2.5" not in source
    assert "compare(" not in source


def test_case_c_review_gap_does_not_fabricate_finding(tmp_path):
    report = _report(tmp_path, kind=WorkspaceItemKind.REVIEW_GAP)
    html = _html(_renderer().render_html(report))
    parsed = json.loads(_renderer().render_json(report).content)
    assert "Review Gap" in html
    assert parsed["findings"] == []
    assert len(parsed["review_gaps"]) == 1
    source = RENDERER_PATH.read_text(encoding="utf-8")
    assert "3.60 m" not in source and "4.5 m" not in source


def test_presentation_only_2000_item_characterization_is_linear(tmp_path):
    report = _report(tmp_path, reviewed=False)
    entry = report.findings[0]
    synthetic_entries = tuple(
        entry.model_copy(update={"source_ordinal": index}) for index in range(2000)
    )
    synthetic = report.model_copy(update={"findings": synthetic_entries})
    start = time.perf_counter()
    json_artifact = _renderer().render_json(synthetic)
    json_elapsed = time.perf_counter() - start
    start = time.perf_counter()
    html_artifact = _renderer().render_html(synthetic)
    html_elapsed = time.perf_counter() - start
    assert json_artifact.byte_length > 0 and html_artifact.byte_length > 0
    assert json_elapsed < 10.0
    assert html_elapsed < 10.0
