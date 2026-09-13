"""Pure deterministic JSON and HTML rendering for frozen semantic reports."""

from __future__ import annotations

import hashlib
import json
from html import escape
from typing import Iterable

from pydantic import BaseModel

from app.schemas.review_report import ReviewReportModel
from app.schemas.review_report_artifact import (
    HTML_REPORT_MEDIA_TYPE,
    HTML_REPORT_RENDERER_VERSION,
    JSON_REPORT_MEDIA_TYPE,
    JSON_REPORT_RENDERER_VERSION,
    ReviewReportArtifact,
    ReviewReportArtifactFormat,
    deterministic_report_artifact_id,
)


class ReviewReportRenderError(ValueError):
    """Raised when a complete deterministic artifact cannot be produced."""


class ReviewReportRenderer:
    """Render one already-qualified semantic model without upstream authority calls."""

    def render_json(self, report: ReviewReportModel) -> ReviewReportArtifact:
        self._require_report(report)
        try:
            content = json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, UnicodeError, ValueError) as exc:
            raise ReviewReportRenderError(
                "Semantic report could not be rendered as canonical JSON."
            ) from exc
        return self._artifact(
            report=report,
            artifact_format=ReviewReportArtifactFormat.JSON,
            renderer_version=JSON_REPORT_RENDERER_VERSION,
            media_type=JSON_REPORT_MEDIA_TYPE,
            extension=".json",
            content=content,
        )

    def render_html(self, report: ReviewReportModel) -> ReviewReportArtifact:
        self._require_report(report)
        try:
            content = self._html_document(report).encode("utf-8")
        except (TypeError, UnicodeError, ValueError) as exc:
            raise ReviewReportRenderError(
                "Semantic report could not be rendered as standalone HTML."
            ) from exc
        return self._artifact(
            report=report,
            artifact_format=ReviewReportArtifactFormat.HTML,
            renderer_version=HTML_REPORT_RENDERER_VERSION,
            media_type=HTML_REPORT_MEDIA_TYPE,
            extension=".html",
            content=content,
        )

    @staticmethod
    def _require_report(report: ReviewReportModel) -> None:
        if not isinstance(report, ReviewReportModel):
            raise ReviewReportRenderError(
                "Renderer input must be a qualified ReviewReportModel."
            )

    @staticmethod
    def _artifact(
        *,
        report: ReviewReportModel,
        artifact_format: ReviewReportArtifactFormat,
        renderer_version: str,
        media_type: str,
        extension: str,
        content: bytes,
    ) -> ReviewReportArtifact:
        content_sha256 = hashlib.sha256(content).hexdigest()
        artifact_id = deterministic_report_artifact_id(
            report_model_id=report.report_model_id,
            artifact_format=artifact_format,
            renderer_version=renderer_version,
            content_sha256=content_sha256,
            media_type=media_type,
        )
        return ReviewReportArtifact(
            artifact_id=artifact_id,
            report_model_id=report.report_model_id,
            format=artifact_format,
            renderer_version=renderer_version,
            media_type=media_type,
            filename=f"{report.report_model_id}{extension}",
            content_sha256=content_sha256,
            byte_length=len(content),
            content=content,
        )

    @staticmethod
    def _text(value: object) -> str:
        if value is None:
            return "Not retained"
        if isinstance(value, str):
            rendered = value
        elif hasattr(value, "value"):
            rendered = str(value.value)
        else:
            rendered = str(value)
        return escape(rendered, quote=True)

    @staticmethod
    def _json_block(value: BaseModel | dict[str, object]) -> str:
        payload = value.model_dump(mode="json", exclude_none=True) if isinstance(
            value, BaseModel
        ) else value
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f'<pre class="source-data">{escape(rendered, quote=True)}</pre>'

    @classmethod
    def _rows(cls, rows: Iterable[tuple[str, object]]) -> str:
        return "".join(
            f"<dt>{escape(label, quote=True)}</dt><dd>{cls._text(value)}</dd>"
            for label, value in rows
        )

    @classmethod
    def _human_review(cls, review) -> str:
        if review is None:
            return '<p class="human-axis"><strong>Human review:</strong> Unreviewed</p>'
        return (
            '<div class="human-axis"><h4>Human review</h4><dl>'
            + cls._rows(
                (
                    ("Action", review.human_action),
                    ("Report inclusion", review.report_inclusion),
                    ("Reviewer ID", review.reviewer_id),
                    ("Identity assurance", review.reviewer_identity_assurance),
                    ("Reviewer note", review.reviewer_note),
                    ("Reviewer note SHA-256", review.reviewer_note_sha256),
                    ("Review record ID", review.review_record_id),
                    ("Created at (audit metadata)", review.created_at.isoformat()),
                )
            )
            + "</dl></div>"
        )

    @classmethod
    def _finding_cards(cls, report: ReviewReportModel) -> str:
        if not report.findings:
            return cls._empty_section()
        cards = []
        for entry in report.findings:
            cards.append(
                '<article class="report-entry finding-entry">'
                '<h3>Finding</h3><dl>'
                + cls._rows(
                    (
                        ("Finding ID", entry.finding_id),
                        ("Comparison ID", entry.comparison_id),
                        ("Machine assessment", entry.machine_decision),
                        ("Reason code", entry.reason_code),
                        ("Candidate ID", entry.candidate_id),
                        ("Requirement ID", entry.requirement_id),
                        ("PlanFact ID", entry.plan_fact_id),
                        ("Report entry ID", entry.report_entry_id),
                        ("Inclusion basis", entry.inclusion_basis),
                    )
                )
                + "</dl>"
                + cls._human_review(entry.human_review)
                + '<h4>Plan evidence</h4>'
                + cls._json_block(entry.plan_source)
                + '<h4>Normative evidence</h4>'
                + cls._json_block(entry.standard_source)
                + "</article>"
            )
        return "".join(cards)

    @classmethod
    def _gap_cards(cls, report: ReviewReportModel) -> str:
        if not report.review_gaps:
            return cls._empty_section()
        cards = []
        for entry in report.review_gaps:
            cards.append(
                '<article class="report-entry gap-entry">'
                '<h3>Review Gap</h3><dl>'
                + cls._rows(
                    (
                        ("Review Gap ID", entry.review_gap_id),
                        ("Gap source", entry.gap_source),
                        ("Candidate terminal state", entry.candidate_terminal_state),
                        ("Requirement terminal state", entry.requirement_terminal_state),
                        ("Decomposition status", entry.decomposition_status),
                        ("Unresolved reason", entry.unresolved_reason),
                        ("Candidate ID", entry.candidate_id),
                        ("Requirement ID", entry.requirement_id),
                        ("Unresolved span ID", entry.unresolved_span_id),
                        ("Report entry ID", entry.report_entry_id),
                        ("Inclusion basis", entry.inclusion_basis),
                    )
                )
                + "</dl>"
                + cls._human_review(entry.human_review)
                + '<h4>Plan source</h4>'
                + cls._json_block(entry.plan_source)
                + (
                    '<h4>Standard source</h4>' + cls._json_block(entry.standard_source)
                    if entry.standard_source is not None
                    else ""
                )
                + "</article>"
            )
        return "".join(cards)

    @classmethod
    def _error_cards(cls, report: ReviewReportModel) -> str:
        if not report.internal_errors:
            return cls._empty_section()
        cards = []
        for entry in report.internal_errors:
            cards.append(
                '<article class="report-entry error-entry">'
                '<h3>Internal Error</h3><dl>'
                + cls._rows(
                    (
                        ("Internal error item ID", entry.internal_error_item_id),
                        ("Error stage", entry.error_stage),
                        ("Error code", entry.error_code),
                        ("Candidate ID", entry.candidate_id),
                        ("Requirement ID", entry.requirement_id),
                        ("Report entry ID", entry.report_entry_id),
                        ("Inclusion basis", entry.inclusion_basis),
                    )
                )
                + "</dl>"
                + cls._human_review(entry.human_review)
                + '<h4>Affected plan source</h4>'
                + cls._json_block(entry.plan_source)
                + "</article>"
            )
        return "".join(cards)

    @classmethod
    def _excluded_cards(cls, report: ReviewReportModel) -> str:
        if not report.excluded_items:
            return cls._empty_section()
        return "".join(
            '<article class="report-entry excluded-entry"><h3>Excluded Item</h3>'
            + cls._json_block(item)
            + "</article>"
            for item in report.excluded_items
        )

    @classmethod
    def _source_references(cls, report: ReviewReportModel) -> str:
        entries = (*report.findings, *report.review_gaps, *report.internal_errors)
        if not entries:
            return cls._empty_section()
        references = []
        for entry in sorted(entries, key=lambda item: item.source_ordinal):
            references.append(
                '<article class="source-reference"><h3>'
                + cls._text(entry.source_item_id)
                + "</h3><h4>Plan source</h4>"
                + cls._json_block(entry.plan_source)
                + (
                    '<h4>Standard source</h4>' + cls._json_block(entry.standard_source)
                    if getattr(entry, "standard_source", None) is not None
                    else ""
                )
                + "</article>"
            )
        return "".join(references)

    @staticmethod
    def _empty_section() -> str:
        return '<p class="empty-section">No detailed entries in this section.</p>'

    @classmethod
    def _html_document(cls, report: ReviewReportModel) -> str:
        document_rows = cls._rows(report.document.model_dump(mode="json").items())
        processing = cls._json_block(report.processing)
        counts = cls._json_block(report.counts)
        coverage_code = cls._text(report.coverage.code)
        coverage_statement = cls._text(report.coverage.statement)
        return (
            "<!DOCTYPE html>\n"
            '<html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>BuildCheck Semantic Report</title>"
            "<style>"
            ":root{color-scheme:light;--ink:#1f2d29;--muted:#60736c;--line:#ccd7d2;--paper:#fff;--wash:#f4f7f5}"
            "*{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font-family:system-ui,\"Microsoft YaHei\",\"PingFang SC\",\"Noto Sans CJK SC\",sans-serif;line-height:1.55}"
            "main{max-width:1100px;margin:auto;padding:32px}section{margin:18px 0;padding:22px;background:var(--paper);border:1px solid var(--line)}"
            "h1,h2,h3,h4{margin-top:0}h1{font-size:26px}h2{font-size:20px;border-bottom:1px solid var(--line);padding-bottom:8px}h3{font-size:16px}h4{font-size:14px;margin-top:16px}"
            "dl{display:grid;grid-template-columns:minmax(180px,max-content) 1fr;gap:5px 14px;margin:0}dt{color:var(--muted);font-weight:650}dd{margin:0;overflow-wrap:anywhere}"
            ".report-entry,.source-reference{margin:12px 0;padding:16px;border:1px solid var(--line);border-left:4px solid #70877e}.gap-entry{border-left-color:#9b885d}.error-entry{border-left-color:#807777}.excluded-entry{border-left-style:dashed}"
            ".human-axis{margin-top:14px;padding:12px;background:var(--wash)}.source-data{white-space:pre-wrap;overflow-wrap:anywhere;padding:12px;background:var(--wash);border:1px solid var(--line)}"
            ".empty-section{color:var(--muted);font-style:italic}.authority-note{color:var(--muted);font-size:13px}"
            "@media(max-width:700px){main{padding:12px}dl{grid-template-columns:1fr}dt{margin-top:8px}}"
            "</style></head><body><main>"
            "<header><h1>BuildCheck Semantic Report</h1>"
            '<p class="authority-note">Presentation of one frozen ReviewReportModel. Machine assessment and human review remain separate.</p></header>'
            '<section id="document-identity"><h2>1. Document Identity</h2><dl>'
            + document_rows
            + "</dl></section>"
            '<section id="processing-coverage-summary"><h2>2. Processing / Coverage Summary</h2><h3>Processing authority</h3>'
            + processing
            + "<h3>Factual counts</h3>"
            + counts
            + "</section>"
            '<section id="findings"><h2>3. Findings</h2>'
            + cls._finding_cards(report)
            + "</section>"
            '<section id="review-gaps"><h2>4. Review Gaps</h2>'
            + cls._gap_cards(report)
            + "</section>"
            '<section id="internal-errors"><h2>5. Internal Errors</h2>'
            + cls._error_cards(report)
            + "</section>"
            '<section id="human-review-state"><h2>6. Human Review State</h2>'
            + cls._json_block(
                {
                    "workflow_completeness": report.processing.human_review_workflow_completeness.value,
                    "counts": report.processing.human_review_counts.model_dump(mode="json"),
                }
            )
            + "</section>"
            '<section id="excluded-item-audit"><h2>7. Excluded Item Audit</h2>'
            + cls._excluded_cards(report)
            + "</section>"
            '<section id="coverage-limitations"><h2>8. Coverage / Limitations</h2><dl>'
            + cls._rows((("Coverage code", coverage_code),))
            + "</dl><p>"
            + coverage_statement
            + "</p></section>"
            '<section id="source-references"><h2>9. Source References as Retained</h2>'
            + cls._source_references(report)
            + "</section>"
            "</main></body></html>\n"
        )
