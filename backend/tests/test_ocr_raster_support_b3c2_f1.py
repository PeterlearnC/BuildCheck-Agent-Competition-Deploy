import hashlib

import fitz
import pytest
from pydantic import ValidationError

from app.schemas.ocr import (
    OCRCharacterResult,
    OCRIssueCode,
    OCRLineResult,
    OCRModelArtifact,
    OCRPageResult,
    OCRProviderIdentity,
    OCRRenderConfig,
    OCRRenderMetadata,
    OCRRun,
    OCRVisualSupportDatum,
    OCRVisualSupportGeometrySource,
)
from app.schemas.standards import DocumentRegionType
from app.services.ocr.accepted_boundary import OCRAcceptedBoundaryService
from app.services.ocr.quality_gate import OCRQualityGate
from app.services.ocr.raster_support import (
    OCR_RASTER_SUPPORT_RULE_VERSION,
    OCRRasterSupportAnalyzer,
)
from app.services.ocr.run_identity import (
    OCR_ADAPTER_VERSION,
    OCR_CORPUS_SEMANTICS_VERSION,
    OCR_QUALITY_GATE_VERSION,
    execution_id,
    semantic_ocr_run_id,
    stable_line_id,
)
from app.services.standards.article_merge_service import ArticleMergeService


SOURCE_CHECKSUM = "c" * 64
OLD_QUALITY_VERSION = "v0.4-b.3b.2-f2-final-r1-quality"
OLD_CORPUS_VERSION = "v0.4-b.3b.2-f2-final-r1-ocr-corpus"
PAGE_WIDTH = 500
PAGE_HEIGHT = 420


def _provider() -> OCRProviderIdentity:
    return OCRProviderIdentity(
        provider="rapidocr",
        provider_version="3.9.2",
        runtime="onnxruntime",
        runtime_version="1.29.0",
        device="CPUExecutionProvider",
        detector=OCRModelArtifact(model_id="det.onnx", sha256="1" * 64),
        recognizer=OCRModelArtifact(model_id="rec.onnx", sha256="2" * 64),
        classifier=OCRModelArtifact(model_id="cls.onnx", sha256="3" * 64),
    )


def _polygon(left: int, top: int, width: int, height: int):
    return [
        [left, top],
        [left + width, top],
        [left + width, top + height],
        [left, top + height],
    ]


def _line(
    text: str,
    index: int,
    polygon: list[list[float]],
    *,
    characters: bool = True,
) -> OCRLineResult:
    return OCRLineResult(
        line_id=stable_line_id(
            page_number=1,
            reading_order_index=index,
            raw_text=text,
            polygon=polygon,
        ),
        page_number=1,
        raw_text=text,
        polygon=polygon,
        confidence=0.99,
        reading_order_index=index,
        characters=(
            [
                OCRCharacterResult(
                    text=text,
                    confidence=0.99,
                    polygon=polygon,
                )
            ]
            if characters
            else []
        ),
    )


def _fill(pixmap: fitz.Pixmap, polygon: list[list[float]], value: int) -> None:
    left = int(min(point[0] for point in polygon))
    right = int(max(point[0] for point in polygon))
    top = int(min(point[1] for point in polygon))
    bottom = int(max(point[1] for point in polygon))
    for y in range(top, bottom):
        for x in range(left, right):
            pixmap.set_pixel(x, y, (value,))


def _annotated_fixture(
    candidate_text: str = "neutral artifact",
    *,
    candidate_luminance: int = 200,
    candidate_width: int = 60,
    candidate_height: int = 30,
    candidate_characters: bool = True,
    normal_luminance: int = 0,
):
    polygons = [
        _polygon(50, 40, 300, 20),
        _polygon(50, 100, 300, 20),
        _polygon(50, 160, candidate_width, candidate_height),
        _polygon(50, 240, 300, 20),
        _polygon(50, 300, 300, 20),
    ]
    texts = [
        "1.0.1 safe article",
        "safe normative body",
        candidate_text,
        "1.0.2 next article",
        "next safe normative body",
    ]
    lines = [
        _line(
            text,
            index,
            polygon,
            characters=(candidate_characters if index == 2 else True),
        )
        for index, (text, polygon) in enumerate(zip(texts, polygons, strict=True))
    ]
    pixmap = fitz.Pixmap(
        fitz.csGRAY, fitz.IRect(0, 0, PAGE_WIDTH, PAGE_HEIGHT), False
    )
    pixmap.clear_with(255)
    for index, polygon in enumerate(polygons):
        _fill(
            pixmap,
            polygon,
            candidate_luminance if index == 2 else normal_luminance,
        )
    png_bytes = pixmap.tobytes("png")
    annotated = OCRRasterSupportAnalyzer().annotate(png_bytes, lines)
    return lines, annotated, png_bytes


def _run(
    lines: list[OCRLineResult],
    png_bytes: bytes,
    *,
    quality_version: str = OCR_QUALITY_GATE_VERSION,
    corpus_version: str = OCR_CORPUS_SEMANTICS_VERSION,
) -> OCRRun:
    provider = _provider()
    render = OCRRenderMetadata(
        renderer="PyMuPDF",
        renderer_version=fitz.VersionBind,
        dpi=300,
        colorspace="GRAY",
        alpha=False,
        config_version="render-1",
        page_number=1,
        pixel_width=PAGE_WIDTH,
        pixel_height=PAGE_HEIGHT,
        image_sha256=hashlib.sha256(png_bytes).hexdigest(),
    )
    config = OCRRenderConfig(
        **render.model_dump(
            exclude={"page_number", "pixel_width", "pixel_height", "image_sha256"}
        )
    )
    page = OCRPageResult(
        page_number=1,
        lines=lines,
        render=render,
        provider=provider,
    )
    return OCRRun(
        ocr_run_id=semantic_ocr_run_id(
            official_source_checksum=SOURCE_CHECKSUM,
            provider=provider,
            render_config=config,
            adapter_version=OCR_ADAPTER_VERSION,
            quality_gate_version=quality_version,
            corpus_semantics_version=corpus_version,
            render_metadata=[render],
        ),
        execution_id=execution_id([page]),
        official_source_checksum=SOURCE_CHECKSUM,
        provider=provider,
        render_config=config,
        adapter_version=OCR_ADAPTER_VERSION,
        quality_gate_version=quality_version,
        corpus_semantics_version=corpus_version,
        pages=[page],
    )


def _parse(run: OCRRun, assessment):
    pages = OCRAcceptedBoundaryService().to_standard_pages(run, assessment)
    regions = {
        (page.page_number, index): DocumentRegionType.NORMATIVE_BODY
        for page in pages
        for index, _ in enumerate(page.text.splitlines())
    }
    return ArticleMergeService().merge(
        "standard-id",
        pages,
        [],
        standard_source_checksum=SOURCE_CHECKSUM,
        standard_code="TEST-2026",
        region_by_line=regions,
        raw_pages=pages,
    )


def test_pale_compact_height_outlier_is_excluded_without_mutating_raw_ocr() -> None:
    raw_lines, annotated, png_bytes = _annotated_fixture()
    candidate_before = raw_lines[2]
    candidate_after = annotated[2]
    run = _run(annotated, png_bytes)

    assessment = OCRQualityGate().assess(run)
    issue = next(
        issue
        for issue in assessment.pages[0].issues
        if issue.code == OCRIssueCode.INSUFFICIENT_RASTER_TEXT_SUPPORT
    )
    articles = {article.article_number: article for article in _parse(run, assessment)}

    assert candidate_after.raw_text == candidate_before.raw_text
    assert candidate_after.polygon == candidate_before.polygon
    assert candidate_after.visual_support is not None
    assert candidate_after.visual_support.dark_support == 0.0
    assert candidate_after.visual_support.relative_support_ratio == 0.0
    assert issue.triggering_line_ids == [candidate_after.line_id]
    assert issue.affected_line_ids == [candidate_after.line_id]
    assert issue.affected_polygons == [candidate_after.polygon]
    assert issue.visual_support == candidate_after.visual_support
    assert candidate_after.line_id not in assessment.pages[0].accepted_line_ids
    assert "neutral artifact" not in articles["1.0.1"].source_text
    assert all(
        mapping.raw_line_id != candidate_after.line_id
        for article in articles.values()
        for mapping in article.source_mappings
    )
    assert "next safe normative body" in articles["1.0.2"].source_text


@pytest.mark.parametrize(
    "text",
    ["l/400", "kN/m²", "mm", "m", "min", "VII", "ABC"],
)
def test_legitimate_dark_compact_technical_text_remains_accepted(text: str) -> None:
    _, annotated, png_bytes = _annotated_fixture(
        text, candidate_luminance=0
    )
    run = _run(annotated, png_bytes)
    assessment = OCRQualityGate().assess(run)
    candidate = annotated[2]

    assert candidate.visual_support is not None
    assert candidate.visual_support.height_ratio >= 1.35
    assert candidate.visual_support.aspect_ratio <= 3.0
    assert candidate.visual_support.relative_support_ratio == 1.0
    assert candidate.line_id in assessment.pages[0].accepted_line_ids
    assert not any(
        issue.code == OCRIssueCode.INSUFFICIENT_RASTER_TEXT_SUPPORT
        for issue in assessment.pages[0].issues
    )


def test_compact_large_dark_heading_remains_accepted() -> None:
    _, annotated, png_bytes = _annotated_fixture(
        "GENERAL HEADING",
        candidate_luminance=0,
        candidate_width=90,
        candidate_height=30,
    )
    run = _run(annotated, png_bytes)
    assessment = OCRQualityGate().assess(run)

    assert annotated[2].line_id in assessment.pages[0].accepted_line_ids
    assert not any(
        issue.code == OCRIssueCode.INSUFFICIENT_RASTER_TEXT_SUPPORT
        for issue in assessment.pages[0].issues
    )


def test_missing_page_reference_does_not_misclassify_dark_text_as_deficient() -> None:
    _, annotated, png_bytes = _annotated_fixture(
        "GENERAL HEADING",
        candidate_luminance=0,
        normal_luminance=200,
    )
    run = _run(annotated, png_bytes)
    assessment = OCRQualityGate().assess(run)

    assert annotated[2].visual_support.page_reference_support == 0.0
    assert annotated[2].visual_support.dark_support == 1.0
    assert annotated[2].line_id in assessment.pages[0].accepted_line_ids
    assert not any(
        issue.code == OCRIssueCode.INSUFFICIENT_RASTER_TEXT_SUPPORT
        for issue in assessment.pages[0].issues
    )


def test_character_geometry_is_preferred_and_line_polygon_is_fallback() -> None:
    _, character_annotated, _ = _annotated_fixture()
    _, fallback_annotated, _ = _annotated_fixture(candidate_characters=False)

    assert (
        character_annotated[2].visual_support.geometry_source
        == OCRVisualSupportGeometrySource.CHARACTER_POLYGONS
    )
    assert (
        fallback_annotated[2].visual_support.geometry_source
        == OCRVisualSupportGeometrySource.LINE_POLYGON
    )


def test_visual_support_and_quality_outcome_are_exactly_deterministic() -> None:
    raw_lines, first, png_bytes = _annotated_fixture()
    second = OCRRasterSupportAnalyzer().annotate(png_bytes, raw_lines)
    first_run = _run(first, png_bytes)
    second_run = _run(second, png_bytes)

    assert [line.model_dump(mode="json") for line in first] == [
        line.model_dump(mode="json") for line in second
    ]
    assert OCRQualityGate().assess(first_run).model_dump(mode="json") == (
        OCRQualityGate().assess(second_run).model_dump(mode="json")
    )


def test_visual_support_is_immutable_json_provenance() -> None:
    _, annotated, png_bytes = _annotated_fixture()
    run = _run(annotated, png_bytes)
    assessment = OCRQualityGate().assess(run)

    restored_run = OCRRun.model_validate_json(run.model_dump_json())
    restored_issue = next(
        issue
        for issue in type(assessment).model_validate_json(
            assessment.model_dump_json()
        ).pages[0].issues
        if issue.code == OCRIssueCode.INSUFFICIENT_RASTER_TEXT_SUPPORT
    )

    assert restored_run.pages[0].lines[2].visual_support == annotated[2].visual_support
    assert restored_issue.visual_support == annotated[2].visual_support
    assert restored_issue.affected_polygons == [annotated[2].polygon]


def test_page_with_raw_line_but_no_visual_support_fails_closed() -> None:
    raw_lines, _, png_bytes = _annotated_fixture()
    with pytest.raises(ValidationError, match="raster-support evidence"):
        _run(raw_lines, png_bytes)


def test_raster_semantics_change_run_identity_but_not_raw_execution_identity() -> None:
    _, annotated, png_bytes = _annotated_fixture()
    current = _run(annotated, png_bytes)
    legacy = _run(
        annotated,
        png_bytes,
        quality_version=OLD_QUALITY_VERSION,
        corpus_version=OLD_CORPUS_VERSION,
    )
    changed_datum = OCRVisualSupportDatum(
        **annotated[2].visual_support.model_dump(
            exclude={"relative_support_ratio"}
        ),
        relative_support_ratio=0.1,
    )
    changed_lines = list(annotated)
    changed_lines[2] = changed_lines[2].model_copy(
        update={"visual_support": changed_datum}
    )
    derived_evidence_changed = _run(changed_lines, png_bytes)

    assert OCR_ADAPTER_VERSION == "v0.4-b.3b.1b-adapter"
    assert OCR_QUALITY_GATE_VERSION == "v0.4-b.3b-q1-r1-quality"
    assert OCR_CORPUS_SEMANTICS_VERSION == "v0.4-b.3b-q1-r1-ocr-corpus"
    assert current.ocr_run_id != legacy.ocr_run_id
    assert current.execution_id == legacy.execution_id
    assert current.execution_id == derived_evidence_changed.execution_id


def test_existing_rotated_overlay_rule_remains_independent() -> None:
    _, annotated, png_bytes = _annotated_fixture(candidate_luminance=0)
    rotated_polygon = _polygon(50, 160, 60, 30)
    rotated_polygon[1][1] += 20
    rotated_polygon[2][1] += 20
    annotated[2] = annotated[2].model_copy(update={"polygon": rotated_polygon})
    run = _run(annotated, png_bytes)
    assessment = OCRQualityGate().assess(run)

    assert any(
        issue.code == OCRIssueCode.SUSPICIOUS_OVERLAY_GEOMETRY
        and annotated[2].line_id in issue.affected_line_ids
        for issue in assessment.pages[0].issues
    )
