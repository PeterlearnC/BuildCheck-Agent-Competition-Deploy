"""Typed, provenance-first contracts for controlled OCR ingestion."""

from enum import Enum
import math
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


SHA256 = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]


class SourceKind(str, Enum):
    PDF_TEXT = "PDF_TEXT"
    OCR_TEXT = "OCR_TEXT"


class OCRQualityState(str, Enum):
    OCR_ACCEPTED = "OCR_ACCEPTED"
    OCR_LOW_CONFIDENCE = "OCR_LOW_CONFIDENCE"
    OCR_REJECTED = "OCR_REJECTED"


class OCRIssueCode(str, Enum):
    EMPTY_OR_FAILED_OCR = "EMPTY_OR_FAILED_OCR"
    ARTICLE_NUMBER_AMBIGUITY = "ARTICLE_NUMBER_AMBIGUITY"
    ARTICLE_SEQUENCE_GAP = "ARTICLE_SEQUENCE_GAP"
    SECTION_ARTICLE_START_GAP = "SECTION_ARTICLE_START_GAP"
    STRUCTURAL_TRANSITION = "STRUCTURAL_TRANSITION"
    STRUCTURAL_SEQUENCE_GAP = "STRUCTURAL_SEQUENCE_GAP"
    SUSPICIOUS_OVERLAY_GEOMETRY = "SUSPICIOUS_OVERLAY_GEOMETRY"
    INSUFFICIENT_RASTER_TEXT_SUPPORT = "INSUFFICIENT_RASTER_TEXT_SUPPORT"
    READING_ORDER_ANOMALY = "READING_ORDER_ANOMALY"
    LOW_TEXT_COVERAGE = "LOW_TEXT_COVERAGE"
    IDENTITY_CONFLICT = "IDENTITY_CONFLICT"
    SOURCE_MAPPING_FAILURE = "SOURCE_MAPPING_FAILURE"


class OCRBindingMethod(str, Enum):
    CURATED_OFFICIAL_SOURCE = "CURATED_OFFICIAL_SOURCE"


class OCRModelArtifact(BaseModel):
    model_id: str
    sha256: SHA256

    @field_validator("model_id")
    @classmethod
    def model_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("OCR model_id must not be blank.")
        return value.strip()


class OCRProviderIdentity(BaseModel):
    provider: str
    provider_version: str
    runtime: str
    runtime_version: str
    device: str
    detector: OCRModelArtifact
    recognizer: OCRModelArtifact
    classifier: OCRModelArtifact | None = None
    classification_enabled: bool = False


class OCRRenderConfig(BaseModel):
    renderer: str = "PyMuPDF"
    renderer_version: str
    dpi: int = Field(default=300, ge=72, le=1200)
    colorspace: str = "GRAY"
    alpha: bool = False
    config_version: str

    @field_validator("colorspace")
    @classmethod
    def colorspace_is_supported(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"GRAY", "RGB"}:
            raise ValueError("OCR renderer colorspace must be GRAY or RGB.")
        return normalized


class OCRRenderMetadata(OCRRenderConfig):
    page_number: int = Field(ge=1)
    pixel_width: int = Field(ge=1)
    pixel_height: int = Field(ge=1)
    image_sha256: SHA256


class OCRCharacterResult(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    polygon: list[list[float]]

    @field_validator("polygon")
    @classmethod
    def character_polygon_is_valid(cls, value):
        return _validated_polygon(value)


class OCRVisualSupportGeometrySource(str, Enum):
    CHARACTER_POLYGONS = "CHARACTER_POLYGONS"
    LINE_POLYGON = "LINE_POLYGON"


class OCRVisualSupportDatum(BaseModel):
    geometry_source: OCRVisualSupportGeometrySource
    dark_support: float = Field(ge=0.0, le=1.0)
    page_reference_support: float = Field(ge=0.0, le=1.0)
    relative_support_ratio: float = Field(ge=0.0)
    height_ratio: float = Field(gt=0.0)
    aspect_ratio: float = Field(ge=0.0)
    rule_version: str

    @field_validator("rule_version")
    @classmethod
    def rule_version_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("OCR visual-support rule version must not be blank.")
        return value.strip()


class OCRLineResult(BaseModel):
    line_id: str
    page_number: int = Field(ge=1)
    raw_text: str
    polygon: list[list[float]]
    confidence: float = Field(ge=0.0, le=1.0)
    reading_order_index: int = Field(ge=0)
    characters: list[OCRCharacterResult] = Field(default_factory=list)
    visual_support: OCRVisualSupportDatum | None = None

    @field_validator("line_id")
    @classmethod
    def line_id_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("OCR line_id must not be blank.")
        return value

    @field_validator("polygon")
    @classmethod
    def line_polygon_is_valid(cls, value):
        return _validated_polygon(value)


class OCRPageResult(BaseModel):
    page_number: int = Field(ge=1)
    lines: list[OCRLineResult]
    render: OCRRenderMetadata
    provider: OCRProviderIdentity
    elapsed_seconds: float | None = Field(default=None, ge=0.0)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def page_numbers_match(self):
        if self.render.page_number != self.page_number:
            raise ValueError("OCR render page does not match OCR result page.")
        if any(line.page_number != self.page_number for line in self.lines):
            raise ValueError("OCR line page does not match OCR result page.")
        if any(line.visual_support is None for line in self.lines):
            raise ValueError("OCR lines require deterministic raster-support evidence.")
        return self


class OCRRun(BaseModel):
    ocr_run_id: str
    execution_id: str
    official_source_checksum: SHA256
    provider: OCRProviderIdentity
    render_config: OCRRenderConfig
    adapter_version: str
    quality_gate_version: str
    corpus_semantics_version: str
    pages: list[OCRPageResult]

    @model_validator(mode="after")
    def pages_match_semantic_configuration(self):
        page_numbers = [page.page_number for page in self.pages]
        if len(set(page_numbers)) != len(page_numbers):
            raise ValueError("OCR run page numbers must be unique.")
        render_config = self.render_config.model_dump(mode="json")
        for page in self.pages:
            if page.provider != self.provider:
                raise ValueError("OCR page provider identity differs from its run.")
            page_config = page.render.model_dump(
                mode="json",
                exclude={"page_number", "pixel_width", "pixel_height", "image_sha256"},
            )
            if page_config != render_config:
                raise ValueError("OCR page render configuration differs from its run.")
        return self


class OCRQualityIssue(BaseModel):
    code: OCRIssueCode
    reason: str
    triggering_line_ids: list[str] = Field(default_factory=list)
    affected_article_heading_id: str | None = None
    affected_line_ids: list[str] = Field(default_factory=list)
    affected_polygons: list[list[list[float]]] = Field(default_factory=list)
    structural_numbers: list[str] = Field(default_factory=list)
    previous_article_heading_id: str | None = None
    next_article_heading_id: str | None = None
    visual_support: OCRVisualSupportDatum | None = None


class OCRPageQualityAssessment(BaseModel):
    page_number: int = Field(ge=1)
    state: OCRQualityState
    issues: list[OCRQualityIssue] = Field(default_factory=list)
    accepted_line_ids: list[str] = Field(default_factory=list)


class OCRRunQualityAssessment(BaseModel):
    ocr_run_id: str
    execution_id: str
    state: OCRQualityState
    pages: list[OCRPageQualityAssessment]
    quality_gate_version: str


class OCRSourceMapping(BaseModel):
    page_number: int = Field(ge=1)
    parser_line_index: int = Field(ge=0)
    normalized_start: int = Field(ge=0)
    normalized_end: int = Field(ge=0)
    raw_line_id: str
    raw_text: str
    raw_character_start: int = Field(default=0, ge=0)
    raw_character_end: int = Field(ge=0)
    raw_polygon: list[list[float]]

    @model_validator(mode="after")
    def ranges_are_ordered(self):
        if self.normalized_end < self.normalized_start:
            raise ValueError("Normalized OCR source range is reversed.")
        if self.raw_character_end < self.raw_character_start:
            raise ValueError("Raw OCR source range is reversed.")
        return self

    @field_validator("raw_polygon")
    @classmethod
    def source_polygon_is_valid(cls, value):
        return _validated_polygon(value)


class OCRAcceptedSpan(BaseModel):
    page_number: int = Field(ge=1)
    state: OCRQualityState = OCRQualityState.OCR_ACCEPTED
    raw_text: str
    normalized_text: str
    source_mappings: list[OCRSourceMapping]

    @model_validator(mode="after")
    def only_accepted_spans_cross_boundary(self):
        if self.state != OCRQualityState.OCR_ACCEPTED:
            raise ValueError("Only OCR_ACCEPTED spans may cross the parser boundary.")
        if not self.source_mappings:
            raise ValueError("Accepted OCR spans require raw source mappings.")
        return self


class OCRAcceptedPage(BaseModel):
    page_number: int = Field(ge=1)
    ocr_run_id: str
    execution_id: str
    page_quality_state: OCRQualityState
    spans: list[OCRAcceptedSpan]
    provider: OCRProviderIdentity
    render: OCRRenderMetadata

    @model_validator(mode="after")
    def span_pages_match(self):
        if self.render.page_number != self.page_number:
            raise ValueError("Accepted OCR render page does not match its page.")
        if any(span.page_number != self.page_number for span in self.spans):
            raise ValueError("Accepted OCR span belongs to a different page.")
        return self


class OfficialSourceBinding(BaseModel):
    binding_method: OCRBindingMethod = OCRBindingMethod.CURATED_OFFICIAL_SOURCE
    source_checksum: SHA256
    canonical_standard_code: str
    display_standard_code: str
    standard_name: str
    binding_reason: str
    binding_provenance: str
    confirmed: bool

    @field_validator(
        "canonical_standard_code",
        "display_standard_code",
        "standard_name",
        "binding_reason",
        "binding_provenance",
    )
    @classmethod
    def binding_text_is_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Official-source binding fields must not be blank.")
        return value.strip()


def _validated_polygon(value: list[list[float]]) -> list[list[float]]:
    if len(value) != 4 or any(len(point) != 2 for point in value):
        raise ValueError("OCR polygons must contain exactly four x/y points.")
    if any(not math.isfinite(float(coordinate)) for point in value for coordinate in point):
        raise ValueError("OCR polygon coordinates must be finite.")
    return value
