"""External-artifact authority contract for qualified OCR corpora."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.ocr import OCRProviderIdentity, OCRRenderConfig


SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
OCR_ID = Annotated[str, Field(pattern=r"^ocr_[0-9a-f]{64}$")]
EXECUTION_ID = Annotated[str, Field(pattern=r"^execution_[0-9a-f]{64}$")]


class OCRQualificationStatus(str, Enum):
    QUALIFIED = "QUALIFIED"
    REQUALIFICATION_REQUIRED = "REQUALIFICATION_REQUIRED"


class OCRQualificationManifest(BaseModel):
    """Bind a small committed authority record to controlled large artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ocr-qualification-manifest-v1"]
    official_source_checksum: SHA256
    ocr_run_id: OCR_ID
    ocr_execution_id: EXECUTION_ID
    execution_artifact_sha256: SHA256 | None = None
    raw_ocr_result_artifact_sha256: SHA256 | None = None
    quality_assessment_artifact_sha256: SHA256 | None = None
    accepted_boundary_artifact_sha256: SHA256 | None = None
    qualified_parse_result_artifact_sha256: SHA256
    provider: OCRProviderIdentity
    render_config: OCRRenderConfig
    quality_gate_version: str
    ocr_corpus_semantics_version: str
    qualification_status: OCRQualificationStatus
    qualification_reason: str

    @model_validator(mode="after")
    def qualified_state_requires_the_complete_chain(self):
        if not self.quality_gate_version.strip() or not self.ocr_corpus_semantics_version.strip():
            raise ValueError("OCR qualification semantic versions must not be blank.")
        if not self.qualification_reason.strip():
            raise ValueError("OCR qualification requires a finite reason.")
        if self.qualification_status == OCRQualificationStatus.QUALIFIED:
            required = (
                self.execution_artifact_sha256,
                self.raw_ocr_result_artifact_sha256,
                self.quality_assessment_artifact_sha256,
                self.accepted_boundary_artifact_sha256,
            )
            if any(value is None for value in required):
                raise ValueError("QUALIFIED OCR authority requires the complete artifact chain.")
        return self
