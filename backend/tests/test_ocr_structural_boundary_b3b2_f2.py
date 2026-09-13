from app.schemas.ocr import (
    OCRIssueCode,
    OCRLineResult,
    OCRModelArtifact,
    OCRPageQualityAssessment,
    OCRPageResult,
    OCRProviderIdentity,
    OCRQualityState,
    OCRRenderConfig,
    OCRRenderMetadata,
    OCRRun,
    OCRRunQualityAssessment,
    SourceKind,
    OCRVisualSupportDatum,
    OCRVisualSupportGeometrySource,
)
from app.schemas.standards import DocumentRegionType, StandardDocument
from app.services.ocr.accepted_boundary import (
    OCRAcceptedBoundaryService,
    OCRBoundaryError,
)
from app.services.ocr.quality_gate import OCRQualityGate
from app.services.ocr.raster_support import OCR_RASTER_SUPPORT_RULE_VERSION
from app.services.ocr.run_identity import (
    OCR_ADAPTER_VERSION,
    OCR_CORPUS_SEMANTICS_VERSION,
    OCR_QUALITY_GATE_VERSION,
    execution_id,
    semantic_ocr_run_id,
    stable_line_id,
)
from app.services.standards.article_merge_service import ArticleMergeService
from app.services.standards.standard_parser_service import StandardParserService
import pytest


SOURCE_CHECKSUM = "b" * 64


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


def _line(text: str, index: int, page: int) -> OCRLineResult:
    top = 100 + index * 60
    polygon = [[100, top], [1200, top], [1200, top + 40], [100, top + 40]]
    return OCRLineResult(
        line_id=stable_line_id(
            page_number=page,
            reading_order_index=index,
            raw_text=text,
            polygon=polygon,
        ),
        page_number=page,
        raw_text=text,
        polygon=polygon,
        confidence=0.999,
        reading_order_index=index,
        visual_support=OCRVisualSupportDatum(
            geometry_source=OCRVisualSupportGeometrySource.LINE_POLYGON,
            dark_support=0.5,
            page_reference_support=0.5,
            relative_support_ratio=1.0,
            height_ratio=1.0,
            aspect_ratio=27.5,
            rule_version=OCR_RASTER_SUPPORT_RULE_VERSION,
        ),
    )


def _run_pages(
    page_texts: list[list[str]],
    *,
    quality: str = OCR_QUALITY_GATE_VERSION,
    corpus: str = OCR_CORPUS_SEMANTICS_VERSION,
) -> OCRRun:
    provider = _provider()
    renders = [
        OCRRenderMetadata(
            renderer="PyMuPDF",
            renderer_version="1",
            dpi=300,
            colorspace="GRAY",
            alpha=False,
            config_version="render-1",
            page_number=number,
            pixel_width=1655,
            pixel_height=2396,
            image_sha256=f"{number:x}" * 64,
        )
        for number in range(1, len(page_texts) + 1)
    ]
    config = OCRRenderConfig(
        **renders[0].model_dump(
            exclude={"page_number", "pixel_width", "pixel_height", "image_sha256"}
        )
    )
    pages = [
        OCRPageResult(
            page_number=page_number,
            lines=[
                _line(text, index, page_number)
                for index, text in enumerate(texts)
            ],
            render=renders[page_number - 1],
            provider=provider,
        )
        for page_number, texts in enumerate(page_texts, start=1)
    ]
    return OCRRun(
        ocr_run_id=semantic_ocr_run_id(
            official_source_checksum=SOURCE_CHECKSUM,
            provider=provider,
            render_config=config,
            quality_gate_version=quality,
            corpus_semantics_version=corpus,
            render_metadata=renders,
        ),
        execution_id=execution_id(pages),
        official_source_checksum=SOURCE_CHECKSUM,
        provider=provider,
        render_config=config,
        adapter_version=OCR_ADAPTER_VERSION,
        quality_gate_version=quality,
        corpus_semantics_version=corpus,
        pages=pages,
    )


def _run(texts: list[str], **versions) -> OCRRun:
    return _run_pages([texts], **versions)


def _parse(run: OCRRun, assessment: OCRRunQualityAssessment):
    pages = OCRAcceptedBoundaryService().to_standard_pages(run, assessment)
    regions = {
        (page.page_number, index): DocumentRegionType.NORMATIVE_BODY
        for page in pages
        for index, _line_text in enumerate(page.text.splitlines())
    }
    articles = ArticleMergeService().merge(
        "standard-id",
        pages,
        [],
        standard_source_checksum=SOURCE_CHECKSUM,
        standard_code="GB55023-2022",
        region_by_line=regions,
        raw_pages=pages,
    )
    return {article.article_number: article for article in articles}


def _all_accepted(run: OCRRun) -> OCRRunQualityAssessment:
    return OCRRunQualityAssessment(
        ocr_run_id=run.ocr_run_id,
        execution_id=run.execution_id,
        state=OCRQualityState.OCR_ACCEPTED,
        quality_gate_version=run.quality_gate_version,
        pages=[
            OCRPageQualityAssessment(
                page_number=page.page_number,
                state=OCRQualityState.OCR_ACCEPTED,
                accepted_line_ids=[line.line_id for line in page.lines],
            )
            for page in run.pages
        ],
    )


def _issues(assessment: OCRRunQualityAssessment):
    return [issue for page in assessment.pages for issue in page.issues]


def test_parser_boundary_preserves_affected_heading_as_structural_only() -> None:
    run = _run(
        [
            "1.0.1 previous grounded normative requirement",
            "2 New structural controls",
            "2.0.1 next grounded normative requirement",
        ]
    )
    assessment = OCRQualityGate().assess(run)
    boundary = OCRAcceptedBoundaryService().parser_boundary(run, assessment)
    heading = run.pages[0].lines[1]

    assert [item.structure_number for item in boundary.structural_evidence] == ["2"]
    evidence = boundary.structural_evidence[0]
    assert evidence.source_references[0].raw_line_id == heading.line_id
    assert evidence.accepted_as_normative is False
    assert heading.line_id not in {
        mapping.raw_line_id
        for page in boundary.standard_pages
        for mapping in page.source_mappings
    }

    document = StandardDocument(
        standard_id="structural-boundary",
        source_filename="synthetic.pdf",
        source_checksum=SOURCE_CHECKSUM,
        page_count=1,
        source_kind=SourceKind.OCR_TEXT,
        ocr_run_id=run.ocr_run_id,
        ocr_execution_id=run.execution_id,
        ocr_quality_state=assessment.state,
        ocr_provider=run.provider,
    )
    parsed = StandardParserService().parse(
        document,
        boundary.standard_pages,
        structural_evidence=boundary.structural_evidence,
    )
    articles = {item.article_number: item for item in parsed.articles}

    assert articles["2.0.1"].chapter_number == "2"
    assert all(
        "New structural controls" not in article.source_text
        for article in parsed.articles
    )


def test_parser_boundary_can_reclassify_quality_accepted_structure() -> None:
    run = _run(
        [
            "1.0.1 previous grounded normative requirement",
            "2 New structural controls",
            "2.0.1 next grounded normative requirement",
        ]
    )
    assessment = _all_accepted(run)

    boundary = OCRAcceptedBoundaryService().parser_boundary(run, assessment)
    heading = run.pages[0].lines[1]

    assert heading.line_id in assessment.pages[0].accepted_line_ids
    assert boundary.structural_evidence[0].source_references[0].raw_line_id == heading.line_id
    assert heading.line_id not in {
        mapping.raw_line_id
        for page in boundary.standard_pages
        for mapping in page.source_mappings
    }
    with pytest.raises(OCRBoundaryError, match="Structural transition"):
        OCRAcceptedBoundaryService().to_standard_pages(run, assessment)


def test_validated_chapter_heading_terminates_preceding_article() -> None:
    run = _run([
        "1.0.4 normative one marker",
        "2 基本规定",
        "2.0.1 normative two marker",
    ])
    assessment = OCRQualityGate().assess(run)
    articles = _parse(run, assessment)

    assert articles["1.0.4"].source_text == "1.0.4 normative one marker"
    assert articles["2.0.1"].source_text == "2.0.1 normative two marker"
    chapter_line_id = run.pages[0].lines[1].line_id
    assert chapter_line_id not in {
        mapping.raw_line_id for mapping in articles["1.0.4"].source_mappings
    }
    assert OCRIssueCode.STRUCTURAL_TRANSITION in {
        issue.code for issue in _issues(assessment)
    }


@pytest.mark.parametrize("heading", ["3 材料与构配件", "材料与构配件"])
def test_cross_page_chapter_or_unambiguous_leading_transition_is_not_body(
    heading: str,
) -> None:
    run = _run_pages([
        ["2.0.6 safe body", "safe continuation"],
        [heading, "3.0.1 independently safe"],
    ])
    articles = _parse(run, OCRQualityGate().assess(run))

    assert articles["2.0.6"].source_text == "2.0.6 safe body\nsafe continuation"
    assert heading not in articles["2.0.6"].source_text
    assert articles["3.0.1"].source_text == "3.0.1 independently safe"


def test_ocr_heading_fragment_is_quarantined_without_reconstruction() -> None:
    run = _run_pages(
        [
            ["3.0.5 safe normative text"],
            ["\u8ba1", "4 \u8bbe", "4.1.1 independently safe text"],
        ]
    )
    fragment = run.pages[1].lines[0]
    fragment_polygon = [[100, 150], [500, 150], [500, 210], [100, 210]]
    fragment = fragment.model_copy(
        update={
            "polygon": fragment_polygon,
            "line_id": stable_line_id(
                page_number=2,
                reading_order_index=fragment.reading_order_index,
                raw_text=fragment.raw_text,
                polygon=fragment_polygon,
            ),
        }
    )
    second_page = run.pages[1].model_copy(
        update={"lines": [fragment, *run.pages[1].lines[1:]]}
    )
    pages = [run.pages[0], second_page]
    run = run.model_copy(update={"pages": pages, "execution_id": execution_id(pages)})
    assessment = OCRQualityGate().assess(run)
    articles = _parse(run, assessment)

    assert articles["3.0.5"].source_text == "3.0.5 safe normative text"
    assert articles["4.1.1"].source_text == "4.1.1 independently safe text"
    assert all("4 设计" not in article.source_text for article in articles.values())
    fragment_id = run.pages[1].lines[0].line_id
    transition = next(
        issue
        for issue in _issues(assessment)
        if issue.code == OCRIssueCode.STRUCTURAL_TRANSITION
    )
    assert fragment_id in transition.affected_line_ids
    assert transition.affected_polygons == [
        line.polygon
        for page in run.pages
        for line in page.lines
        if line.line_id in transition.affected_line_ids
    ]


def test_cross_prefix_missing_section_start_is_explicit_and_fail_closed() -> None:
    run = _run([
        "4.1.4 safe previous article",
        "4.2 荷载",
        "unbounded 4.2.1 body marker",
        "unbounded 4.2.2 body marker",
        "4.2.3 independently recognized article",
    ])
    assessment = OCRQualityGate().assess(run)
    articles = _parse(run, assessment)
    issue = next(
        issue
        for issue in _issues(assessment)
        if issue.code == OCRIssueCode.SECTION_ARTICLE_START_GAP
    )

    assert set(articles) == {"4.1.4", "4.2.3"}
    assert articles["4.1.4"].source_text == "4.1.4 safe previous article"
    assert articles["4.2.3"].source_text == "4.2.3 independently recognized article"
    assert "4.2.1" not in articles
    assert "4.2.2" not in articles
    assert run.pages[0].lines[2].line_id in issue.affected_line_ids
    assert run.pages[0].lines[3].line_id in issue.affected_line_ids
    assert len(issue.affected_polygons) == len(issue.affected_line_ids)


def test_valid_cross_prefix_section_start_does_not_emit_gap() -> None:
    run = _run([
        "4.1.4 safe previous article",
        "4.2 荷载",
        "4.2.1 safe first article",
    ])
    assessment = OCRQualityGate().assess(run)
    articles = _parse(run, assessment)

    assert set(articles) == {"4.1.4", "4.2.1"}
    assert OCRIssueCode.SECTION_ARTICLE_START_GAP not in {
        issue.code for issue in _issues(assessment)
    }


def test_boundary_independently_rejects_accepted_transition_text() -> None:
    run = _run([
        "1.0.4 safe previous",
        "2 基本规定",
        "2.0.1 safe next",
    ])

    with pytest.raises(OCRBoundaryError, match="Structural transition"):
        OCRAcceptedBoundaryService().to_standard_pages(run, _all_accepted(run))


@pytest.mark.parametrize("structural_number", ["2", "3"])
def test_terminal_structural_heading_is_excluded_with_complete_provenance(
    structural_number: str,
) -> None:
    run = _run([
        "1.0.1 safe normative text",
        f"{structural_number} New Chapter",
    ])
    assessment = OCRQualityGate().assess(run)
    articles = _parse(run, assessment)
    issue = next(
        issue
        for issue in _issues(assessment)
        if issue.code == OCRIssueCode.STRUCTURAL_TRANSITION
    )

    article = articles["1.0.1"]
    article_heading = run.pages[0].lines[0]
    terminal_heading = run.pages[0].lines[1]
    assert article.source_text == "1.0.1 safe normative text"
    assert terminal_heading.line_id not in {
        mapping.raw_line_id for mapping in article.source_mappings
    }
    assert issue.structural_numbers == [structural_number]
    assert issue.previous_article_heading_id == article_heading.line_id
    assert issue.next_article_heading_id is None
    assert issue.triggering_line_ids == [terminal_heading.line_id]
    assert issue.affected_line_ids == [terminal_heading.line_id]
    assert issue.affected_polygons == [terminal_heading.polygon]


def test_cross_page_terminal_structural_heading_has_no_off_by_one() -> None:
    run = _run_pages([
        ["6.0.5 safe terminal article body"],
        ["7 Inspection"],
    ])
    assessment = OCRQualityGate().assess(run)
    articles = _parse(run, assessment)

    assert articles["6.0.5"].source_text == "6.0.5 safe terminal article body"
    assert all("7 Inspection" not in article.source_text for article in articles.values())
    transition = next(
        issue
        for issue in _issues(assessment)
        if issue.code == OCRIssueCode.STRUCTURAL_TRANSITION
    )
    assert transition.affected_line_ids == [run.pages[1].lines[0].line_id]
    assert transition.previous_article_heading_id == run.pages[0].lines[0].line_id
    assert transition.next_article_heading_id is None


def test_terminal_ordinary_normative_tail_is_not_truncated() -> None:
    run = _run([
        "6.0.5 normative body line 1",
        "normative body line 2",
        "normative body line 3",
    ])
    assessment = OCRQualityGate().assess(run)
    articles = _parse(run, assessment)

    assert articles["6.0.5"].source_text == (
        "6.0.5 normative body line 1\n"
        "normative body line 2\n"
        "normative body line 3"
    )
    assert OCRIssueCode.STRUCTURAL_TRANSITION not in {
        issue.code for issue in _issues(assessment)
    }


def test_terminal_fragment_band_is_quarantined_without_text_repair() -> None:
    run = _run_pages([
        ["6.0.5 safe final article"],
        ["fragment", "7 Inspection"],
    ])
    fragment = run.pages[1].lines[0]
    fragment_polygon = [[100, 150], [500, 150], [500, 210], [100, 210]]
    fragment = fragment.model_copy(
        update={
            "polygon": fragment_polygon,
            "line_id": stable_line_id(
                page_number=2,
                reading_order_index=fragment.reading_order_index,
                raw_text=fragment.raw_text,
                polygon=fragment_polygon,
            ),
        }
    )
    second_page = run.pages[1].model_copy(
        update={"lines": [fragment, *run.pages[1].lines[1:]]}
    )
    pages = [run.pages[0], second_page]
    run = run.model_copy(update={"pages": pages, "execution_id": execution_id(pages)})
    assessment = OCRQualityGate().assess(run)
    articles = _parse(run, assessment)
    transition = next(
        issue
        for issue in _issues(assessment)
        if issue.code == OCRIssueCode.STRUCTURAL_TRANSITION
    )

    assert articles["6.0.5"].source_text == "6.0.5 safe final article"
    assert transition.affected_line_ids == [
        run.pages[1].lines[0].line_id,
        run.pages[1].lines[1].line_id,
    ]
    assert transition.structural_numbers == ["7"]


def test_structural_transition_provenance_survives_json_round_trip() -> None:
    run = _run([
        "1.0.1 safe body",
        "2 New Chapter",
    ])
    assessment = OCRQualityGate().assess(run)
    restored = OCRRunQualityAssessment.model_validate_json(
        assessment.model_dump_json()
    )
    issue = next(
        issue
        for issue in _issues(restored)
        if issue.code == OCRIssueCode.STRUCTURAL_TRANSITION
    )

    assert issue.structural_numbers == ["2"]
    assert issue.previous_article_heading_id == run.pages[0].lines[0].line_id
    assert issue.next_article_heading_id is None
    assert issue.triggering_line_ids == [run.pages[0].lines[1].line_id]
    assert issue.affected_line_ids == [run.pages[0].lines[1].line_id]
    assert issue.affected_polygons == [run.pages[0].lines[1].polygon]


def test_boundary_rejects_accepted_terminal_transition_without_next_article() -> None:
    run = _run([
        "1.0.1 safe previous",
        "2 New Chapter",
    ])

    with pytest.raises(OCRBoundaryError, match="Structural transition"):
        OCRAcceptedBoundaryService().to_standard_pages(run, _all_accepted(run))


def test_f2_versions_change_semantic_run_but_not_raw_execution() -> None:
    current = _run(["1.0.1 stable body"])
    f1 = _run(
        ["1.0.1 stable body"],
        quality="v0.4-b.3b.2-f1-quality",
        corpus="v0.4-b.3b.2-f1-ocr-corpus",
    )

    assert current.ocr_run_id != f1.ocr_run_id
    assert current.execution_id == f1.execution_id
    assert current.pages == f1.pages
    draft_f2 = _run(
        ["1.0.1 stable body"],
        quality="v0.4-b.3b.2-f2-quality",
        corpus="v0.4-b.3b.2-f2-ocr-corpus",
    )
    assert current.ocr_run_id != draft_f2.ocr_run_id
    assert current.execution_id == draft_f2.execution_id
    assert OCR_QUALITY_GATE_VERSION == "v0.4-b.3b-q1-r1-quality"
    assert OCR_CORPUS_SEMANTICS_VERSION == "v0.4-b.3b-q1-r1-ocr-corpus"
