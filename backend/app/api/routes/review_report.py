"""Thin HTTP transport for frozen semantic review-report artifacts."""

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.schemas.review_report_artifact import (
    HTML_REPORT_MEDIA_TYPE,
    ReviewReportArtifact,
    ReviewReportArtifactFormat,
)
from app.schemas.review_report_transport import (
    MAX_REPORT_ARTIFACT_BYTES,
    ReviewReportDelivery,
    ReviewReportTransportRequest,
)
from app.services.pdf_service import (
    InvalidPDFError,
    PDFNoExtractableTextError,
    PDFTextExtractionError,
)
from app.services.review.findings_workspace_service import (
    FindingsWorkspaceAuthorityInvariantError,
    FindingsWorkspaceSourceNavigationError,
)
from app.services.review.human_review_service import (
    HumanReviewAuthorityInvariantError,
    HumanReviewCommandError,
    HumanReviewConflictError,
    HumanReviewStaleWorkspaceError,
    HumanReviewTargetError,
)
from app.services.review.review_report_renderer import (
    ReviewReportRenderError,
    ReviewReportRenderer,
)
from app.services.review.review_report_service import (
    ReviewReportAuthorityInvariantError,
    ReviewReportService,
)
from app.services.review.whole_plan_review_service import (
    WholePlanAuthorityInvariantError,
    WholePlanDocumentNotFoundError,
    WholePlanDocumentSourceChangedError,
)


router = APIRouter(prefix="/documents", tags=["review reports"])

_HTML_CONTENT_SECURITY_POLICY = (
    "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; "
    "form-action 'none'; frame-ancestors 'none'"
)


class ReviewReportTransportError(RuntimeError):
    """Base class for transport-only report failures."""


class ReviewReportArtifactTooLargeError(ReviewReportTransportError):
    """The complete artifact exceeds the explicit in-memory response limit."""


class ReviewReportTransportInvariantError(ReviewReportTransportError):
    """Frozen renderer metadata is unsafe or inconsistent for HTTP transport."""


def get_review_report_service() -> ReviewReportService:
    return ReviewReportService()


def get_review_report_renderer() -> ReviewReportRenderer:
    return ReviewReportRenderer()


def _require_transportable_artifact(
    artifact: ReviewReportArtifact,
    *,
    report_model_id: str,
    expected_format: ReviewReportArtifactFormat,
) -> None:
    if not isinstance(artifact, ReviewReportArtifact):
        raise ReviewReportTransportInvariantError(
            "Renderer returned an invalid report artifact."
        )
    if (
        artifact.report_model_id != report_model_id
        or artifact.format != expected_format
    ):
        raise ReviewReportTransportInvariantError(
            "Renderer artifact differs from requested report authority."
        )
    if any(token in artifact.filename for token in ("/", "\\", "..", "\r", "\n")):
        raise ReviewReportTransportInvariantError(
            "Renderer artifact filename is unsafe for HTTP transport."
        )
    if artifact.byte_length > MAX_REPORT_ARTIFACT_BYTES:
        raise ReviewReportArtifactTooLargeError(
            "Rendered report exceeds the transport size limit."
        )


def _report_artifact_response(
    artifact: ReviewReportArtifact,
    delivery: ReviewReportDelivery,
) -> Response:
    disposition = "inline" if delivery == ReviewReportDelivery.INLINE else "attachment"
    headers = {
        "Content-Type": artifact.media_type,
        "Content-Disposition": f'{disposition}; filename="{artifact.filename}"',
        "X-Report-Model-ID": artifact.report_model_id,
        "X-Report-Artifact-ID": artifact.artifact_id,
        "X-Content-SHA256": artifact.content_sha256,
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if artifact.media_type == HTML_REPORT_MEDIA_TYPE:
        headers["Content-Security-Policy"] = _HTML_CONTENT_SECURITY_POLICY
        headers["Referrer-Policy"] = "no-referrer"
    return Response(content=artifact.content, status_code=status.HTTP_200_OK, headers=headers)


@router.post("/{document_id}/review/report", response_class=Response)
def render_review_report(
    document_id: str,
    request: ReviewReportTransportRequest,
    reports: ReviewReportService = Depends(get_review_report_service),
    renderer: ReviewReportRenderer = Depends(get_review_report_renderer),
) -> Response:
    """Build one server-authoritative report and transport one frozen artifact."""

    try:
        report = reports.build_report(document_id, request.commands)
        artifact = (
            renderer.render_json(report)
            if request.format == ReviewReportArtifactFormat.JSON
            else renderer.render_html(report)
        )
        _require_transportable_artifact(
            artifact,
            report_model_id=report.report_model_id,
            expected_format=request.format,
        )
        return _report_artifact_response(artifact, request.delivery)
    except WholePlanDocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored plan document was not found.",
        ) from exc
    except (
        HumanReviewStaleWorkspaceError,
        HumanReviewConflictError,
        WholePlanDocumentSourceChangedError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Report authority is stale or conflicts with the stored document.",
        ) from exc
    except (
        HumanReviewCommandError,
        HumanReviewTargetError,
        PDFNoExtractableTextError,
        PDFTextExtractionError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Report request could not be applied to the stored document.",
        ) from exc
    except InvalidPDFError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Stored plan document is not a valid PDF.",
        ) from exc
    except ReviewReportArtifactTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Rendered report exceeds the transport size limit.",
        ) from exc
    except (
        ReviewReportRenderError,
        ReviewReportTransportInvariantError,
        ReviewReportAuthorityInvariantError,
        HumanReviewAuthorityInvariantError,
        FindingsWorkspaceAuthorityInvariantError,
        FindingsWorkspaceSourceNavigationError,
        WholePlanAuthorityInvariantError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed safely.",
        ) from exc
