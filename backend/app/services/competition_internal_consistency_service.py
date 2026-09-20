"""Competition adapter for frozen D9 authority and controlled Case D bytes."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.competition_internal_consistency import (
    CompetitionConsistencyCandidateView,
    CompetitionConsistencySourceView,
    CompetitionConsistencyValueGroupView,
    CompetitionInternalConsistencyResponse,
)
from app.schemas.internal_consistency import (
    ConsistencyParameterProfile,
    EngineeringObjectProfile,
    InternalConsistencyCandidate,
)
from app.services.pdf_service import PDFServiceError
from app.services.review.internal_consistency_service import (
    InternalConsistencyReviewError,
    InternalConsistencyReviewService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASE_D_ROOT = (
    PROJECT_ROOT / "backend/tests/fixtures/internal_consistency/case_d"
).resolve()
CASE_D_METADATA_PATH = CASE_D_ROOT / "controlled-case-d.json"
CASE_D_DOCUMENT_ID = "a38da41d-3bb1-5cbb-8402-8d76545dfca9"
CASE_D_ASSET_PATH = CASE_D_ROOT / f"{CASE_D_DOCUMENT_ID}.pdf"
CASE_D_ASSET_SHA256 = (
    "732c04c8d0ad282292ba0529af04bc0777c83e67d6a2c99cb606b7883d6764e9"
)
CASE_D_CLASSIFICATION = "CONTROLLED_SYNTHETIC"
CASE_D_DISPLAY_LABEL = "受控合成演示案例"
CASE_D_PURPOSE = "INTERNAL_CONSISTENCY_DEMONSTRATION"

_OBJECT_DISPLAY_NAMES = {
    EngineeringObjectProfile.SUPPORT_STRUCTURE: "支护结构",
}
_PARAMETER_DISPLAY_NAMES = {
    ConsistencyParameterProfile.HORIZONTAL_DISPLACEMENT_ALARM_VALUE: (
        "水平位移报警值"
    ),
}


class CompetitionInternalConsistencyError(RuntimeError):
    """Base class for bounded Case D adapter failures."""


class CompetitionInternalConsistencyAuthorityError(
    CompetitionInternalConsistencyError
):
    """The packaged controlled authority is missing or no longer exact."""


class CompetitionInternalConsistencyExecutionError(
    CompetitionInternalConsistencyError
):
    """Frozen D9 could not reconstruct a review from the qualified bytes."""


class CompetitionInternalConsistencyInternalError(
    CompetitionInternalConsistencyError
):
    """The live D9 result could not be projected into the fixed UI view."""


class CompetitionInternalConsistencyDemoService:
    """Expose Case D without accepting caller-provided review authority."""

    def run_case_d(self) -> CompetitionInternalConsistencyResponse:
        metadata = self._load_controlled_metadata()
        asset_sha256 = self._validate_asset()
        try:
            result = InternalConsistencyReviewService(
                settings=Settings(upload_dir=CASE_D_ROOT)
            ).review_document(CASE_D_DOCUMENT_ID)
        except (InternalConsistencyReviewError, PDFServiceError) as exc:
            raise CompetitionInternalConsistencyExecutionError(
                "Controlled Case D could not be reviewed."
            ) from exc

        if (
            result.document_id != CASE_D_DOCUMENT_ID
            or result.document_sha256 != asset_sha256
        ):
            raise CompetitionInternalConsistencyAuthorityError(
                "Controlled Case D result is not bound to the qualified asset."
            )

        try:
            candidates = tuple(self._candidate_view(item) for item in result.candidates)
            return CompetitionInternalConsistencyResponse(
                case_id=metadata["case_id"],
                classification=metadata["classification"],
                display_label=metadata["classification_zh"],
                purpose=metadata["purpose"],
                document_id=result.document_id,
                document_sha256=result.document_sha256,
                review_id=result.review_id,
                coverage_status=result.coverage_status,
                candidate_count=result.candidate_count,
                candidates=candidates,
            )
        except (KeyError, ValidationError, ValueError) as exc:
            raise CompetitionInternalConsistencyInternalError(
                "Controlled Case D result projection failed."
            ) from exc

    @staticmethod
    def _candidate_view(
        candidate: InternalConsistencyCandidate,
    ) -> CompetitionConsistencyCandidateView:
        object_display_name = _OBJECT_DISPLAY_NAMES.get(candidate.object_profile_name)
        parameter_display_name = _PARAMETER_DISPLAY_NAMES.get(
            candidate.parameter_profile_name
        )
        if object_display_name is None or parameter_display_name is None:
            raise ValueError("Case D returned an unqualified presentation profile.")

        value_groups = tuple(
            CompetitionConsistencyValueGroupView(
                value=group.numeric_value,
                unit=group.normalized_unit,
                sources=tuple(
                    CompetitionConsistencySourceView(
                        physical_page=fact.physical_page,
                        source_start=fact.source_start,
                        source_end=fact.source_end,
                        source_text=fact.source_text,
                        source_text_sha256=fact.source_text_sha256,
                        page_text_sha256=fact.page_text_sha256,
                        fact_id=fact.fact_id,
                    )
                    for fact in group.facts
                ),
            )
            for group in candidate.value_groups
        )
        return CompetitionConsistencyCandidateView(
            candidate_id=candidate.candidate_id,
            status=candidate.status,
            relation=candidate.relation,
            reason=candidate.reason,
            review_status=candidate.review_status,
            object_display_name=object_display_name,
            parameter_display_name=parameter_display_name,
            unit=candidate.normalized_unit,
            assertion_class=candidate.assertion_class,
            value_groups=value_groups,
        )

    @staticmethod
    def _load_controlled_metadata() -> dict[str, object]:
        try:
            raw = CASE_D_METADATA_PATH.read_text(encoding="utf-8")
            metadata = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CompetitionInternalConsistencyAuthorityError(
                "Controlled Case D metadata is unavailable."
            ) from exc
        if not isinstance(metadata, dict):
            raise CompetitionInternalConsistencyAuthorityError(
                "Controlled Case D metadata is invalid."
            )

        required = {
            "case_id": "CASE-D",
            "document_id": CASE_D_DOCUMENT_ID,
            "asset_filename": CASE_D_ASSET_PATH.name,
            "asset_sha256": CASE_D_ASSET_SHA256,
            "classification": CASE_D_CLASSIFICATION,
            "classification_zh": CASE_D_DISPLAY_LABEL,
            "purpose": CASE_D_PURPOSE,
            "requires_ocr": False,
            "requires_network": False,
        }
        if any(metadata.get(key) != value for key, value in required.items()):
            raise CompetitionInternalConsistencyAuthorityError(
                "Controlled Case D metadata does not match frozen authority."
            )
        try:
            parsed = UUID(str(metadata["document_id"]))
        except (TypeError, ValueError, AttributeError) as exc:
            raise CompetitionInternalConsistencyAuthorityError(
                "Controlled Case D document identity is invalid."
            ) from exc
        if str(parsed) != CASE_D_DOCUMENT_ID:
            raise CompetitionInternalConsistencyAuthorityError(
                "Controlled Case D document identity is not canonical."
            )
        return metadata

    @staticmethod
    def _validate_asset() -> str:
        root = CASE_D_ROOT.resolve()
        path = CASE_D_ASSET_PATH.resolve()
        if path.parent != root or not path.is_file():
            raise CompetitionInternalConsistencyAuthorityError(
                "Controlled Case D asset is unavailable."
            )
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise CompetitionInternalConsistencyAuthorityError(
                "Controlled Case D asset is unavailable."
            ) from exc
        asset_sha256 = digest.hexdigest()
        if asset_sha256 != CASE_D_ASSET_SHA256:
            raise CompetitionInternalConsistencyAuthorityError(
                "Controlled Case D asset does not match frozen authority."
            )
        return asset_sha256


@lru_cache
def get_competition_internal_consistency_service(
) -> CompetitionInternalConsistencyDemoService:
    return CompetitionInternalConsistencyDemoService()
