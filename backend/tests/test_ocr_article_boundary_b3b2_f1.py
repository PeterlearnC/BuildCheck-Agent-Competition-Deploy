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
    OCRVisualSupportDatum,
    OCRVisualSupportGeometrySource,
)
from app.schemas.standards import DocumentRegionType
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
import pytest


SOURCE_CHECKSUM = "a" * 64


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


def _line(text: str, index: int, *, page: int = 1) -> OCRLineResult:
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


def _run(texts: list[str], *, quality: str = OCR_QUALITY_GATE_VERSION,
         corpus: str = OCR_CORPUS_SEMANTICS_VERSION) -> OCRRun:
    return _run_pages([texts], quality=quality, corpus=corpus)


def _run_pages(
    page_texts: list[list[str]], *, quality: str = OCR_QUALITY_GATE_VERSION,
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
            page_number=page_number,
            pixel_width=1655,
            pixel_height=2396,
            image_sha256=f"{page_number:x}" * 64,
        )
        for page_number in range(1, len(page_texts) + 1)
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
                _line(text, index, page=page_number)
                for index, text in enumerate(texts)
            ],
            render=renders[page_number - 1],
            provider=provider,
        )
        for page_number, texts in enumerate(page_texts, start=1)
    ]
    run_id = semantic_ocr_run_id(
        official_source_checksum=SOURCE_CHECKSUM,
        provider=provider,
        render_config=config,
        quality_gate_version=quality,
        corpus_semantics_version=corpus,
        render_metadata=renders,
    )
    return OCRRun(
        ocr_run_id=run_id,
        execution_id=execution_id(pages),
        official_source_checksum=SOURCE_CHECKSUM,
        provider=provider,
        render_config=config,
        adapter_version=OCR_ADAPTER_VERSION,
        quality_gate_version=quality,
        corpus_semantics_version=corpus,
        pages=pages,
    )


def _parse(run: OCRRun, assessment: OCRRunQualityAssessment):
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
        standard_code="GB55023-2022",
        region_by_line=regions,
        raw_pages=pages,
    )


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


def _by_number(articles):
    return {article.article_number: article for article in articles}


def test_2_0_3_list_gap_quarantines_complete_article_and_preserves_neighbors() -> None:
    run = _run([
        "2.0.2 safe previous marker",
        "2.0.3 unsafe article marker",
        "unsafe body with high confidence",
        "1 first item",
        "2 second item",
        "6 sixth item",
        "8 eighth item",
        "2.0.4 safe next marker",
    ])
    baseline = _by_number(_parse(run, _all_accepted(run)))
    assessment = OCRQualityGate().assess(run)
    articles = _by_number(_parse(run, assessment))

    assert set(articles) == {"2.0.2", "2.0.4"}
    assert "unsafe article marker" not in articles["2.0.2"].source_text
    assert "eighth item" not in articles["2.0.2"].source_text
    assert articles["2.0.2"].article_id == baseline["2.0.2"].article_id
    assert articles["2.0.4"].article_id == baseline["2.0.4"].article_id
    assert [mapping.raw_line_id for mapping in articles["2.0.2"].source_mappings] == [
        mapping.raw_line_id for mapping in baseline["2.0.2"].source_mappings
    ]
    assert [mapping.raw_line_id for mapping in articles["2.0.4"].source_mappings] == [
        mapping.raw_line_id for mapping in baseline["2.0.4"].source_mappings
    ]
    assert "7" not in "\n".join(page.text for page in OCRAcceptedBoundaryService().to_standard_pages(run, assessment))


def test_6_0_4_gap_does_not_leak_checklist_into_neighbors() -> None:
    run = _run([
        "6.0.3 safe previous inspection",
        "6.0.4 unsafe checklist heading",
        "unsafe checklist body",
        "1 first stage",
        "2 second stage",
        "6 sixth stage",
        "8 eighth stage",
        "6.0.5 safe next acceptance",
    ])
    articles = _by_number(_parse(run, OCRQualityGate().assess(run)))

    assert set(articles) == {"6.0.3", "6.0.5"}
    assert "unsafe checklist" not in articles["6.0.3"].source_text
    assert "sixth stage" not in articles["6.0.3"].source_text
    assert articles["6.0.5"].source_text == "6.0.5 safe next acceptance"


def test_missing_4_4_8_and_4_4_9_surfaces_article_gap_without_reconstruction() -> None:
    run = _run([
        "4.4.6 safe previous article",
        "4.4.7 recognized heading",
        "valid first body",
        "unbounded later body marker",
        "1 detached later list body",
        "2 another later list body",
        "4.4.10 independently recognized next article",
        "safe next body",
    ])
    assessment = OCRQualityGate().assess(run)
    articles = _by_number(_parse(run, assessment))
    issues = [issue for page in assessment.pages for issue in page.issues]

    assert OCRIssueCode.ARTICLE_SEQUENCE_GAP in {issue.code for issue in issues}
    assert set(articles) == {"4.4.6", "4.4.10"}
    assert "4.4.8" not in articles
    assert "4.4.9" not in articles
    assert "unbounded later body marker" not in articles["4.4.6"].source_text
    assert "unbounded later body marker" not in articles["4.4.10"].source_text
    assert articles["4.4.10"].source_text.startswith("4.4.10")


def test_list_gap_and_article_heading_gap_are_distinct_issue_types() -> None:
    run = _run([
        "4.4.4 article with unsafe list",
        "1 first",
        "2 second",
        "6 sixth",
        "8 eighth",
        "4.4.5 next article",
    ])
    issues = [
        issue
        for page in OCRQualityGate().assess(run).pages
        for issue in page.issues
    ]

    assert OCRIssueCode.STRUCTURAL_SEQUENCE_GAP in {issue.code for issue in issues}
    assert OCRIssueCode.ARTICLE_SEQUENCE_GAP not in {issue.code for issue in issues}
    assert all("article heading" not in issue.reason.lower() for issue in issues if issue.code == OCRIssueCode.STRUCTURAL_SEQUENCE_GAP)


def test_boundary_rejects_partial_article_interval_even_for_high_confidence_body() -> None:
    run = _run([
        "2.0.2 safe previous",
        "2.0.3 unsafe heading",
        "high confidence orphan body",
        "1 first",
        "2 second",
        "6 sixth",
        "8 eighth",
        "2.0.4 safe next",
    ])
    assessment = OCRQualityGate().assess(run)
    page = assessment.pages[0]
    issue = next(
        issue for issue in page.issues
        if issue.code == OCRIssueCode.STRUCTURAL_SEQUENCE_GAP
    )
    heading_id = issue.affected_article_heading_id
    body_id = run.pages[0].lines[2].line_id
    tampered_issue = issue.model_copy(update={"affected_line_ids": [heading_id]})
    tampered_page = page.model_copy(update={
        "issues": [tampered_issue],
        "accepted_line_ids": [
            line.line_id for line in run.pages[0].lines if line.line_id != heading_id
        ],
    })
    tampered = assessment.model_copy(update={"pages": [tampered_page]})

    assert body_id in tampered_page.accepted_line_ids
    with pytest.raises(OCRBoundaryError, match="partially excluded"):
        OCRAcceptedBoundaryService().to_standard_pages(run, tampered)


def test_cross_page_article_interval_quarantine_stops_before_next_heading() -> None:
    run = _run_pages([
        [
            "2.0.2 safe previous",
            "2.0.3 unsafe cross-page heading",
            "cross-page unsafe body",
        ],
        [
            "1 first",
            "2 second",
            "6 sixth",
            "8 eighth",
            "2.0.4 safe next heading",
            "safe next body",
        ],
    ])
    assessment = OCRQualityGate().assess(run)
    articles = _by_number(_parse(run, assessment))

    assert set(articles) == {"2.0.2", "2.0.4"}
    assert articles["2.0.2"].source_text == "2.0.2 safe previous"
    assert articles["2.0.4"].source_text == (
        "2.0.4 safe next heading\nsafe next body"
    )
    issue = next(
        issue for page in assessment.pages for issue in page.issues
        if issue.code == OCRIssueCode.STRUCTURAL_SEQUENCE_GAP
    )
    assert run.pages[0].lines[1].line_id == issue.affected_article_heading_id
    assert run.pages[1].lines[4].line_id not in issue.affected_line_ids


def test_structural_issue_preserves_trigger_heading_and_full_interval_ids() -> None:
    run = _run([
        "2.0.3 unsafe heading",
        "body marker",
        "1 first",
        "2 second",
        "6 sixth",
        "8 eighth",
        "2.0.4 next heading",
    ])
    assessment = OCRQualityGate().assess(run)
    issue = next(
        issue for page in assessment.pages for issue in page.issues
        if issue.code == OCRIssueCode.STRUCTURAL_SEQUENCE_GAP
    )

    assert issue.affected_article_heading_id == run.pages[0].lines[0].line_id
    assert issue.triggering_line_ids == [
        run.pages[0].lines[3].line_id,
        run.pages[0].lines[4].line_id,
    ]
    assert issue.affected_line_ids == [
        line.line_id for line in run.pages[0].lines[:6]
    ]
    assert not set(issue.affected_line_ids) & set(assessment.pages[0].accepted_line_ids)


def test_quality_and_corpus_version_change_invalidates_semantic_run_only() -> None:
    current = _run(["1.0.1 stable body"])
    legacy = _run(
        ["1.0.1 stable body"],
        quality="v0.4-b.3b.1b-quality",
        corpus="v0.4-b.3b.1b-ocr-corpus",
    )

    assert current.ocr_run_id != legacy.ocr_run_id
    assert current.execution_id == legacy.execution_id
    assert current.pages == legacy.pages
    assert OCR_QUALITY_GATE_VERSION == "v0.4-b.3c.2-f1-r1-quality"
    assert OCR_CORPUS_SEMANTICS_VERSION == "v0.4-b.3c.3-s1-r1-ocr-corpus"

    reassessed_legacy = OCRQualityGate().assess(legacy)
    assert reassessed_legacy.quality_gate_version == OCR_QUALITY_GATE_VERSION
    with pytest.raises(OCRBoundaryError, match="version"):
        OCRAcceptedBoundaryService().to_standard_pages(legacy, reassessed_legacy)
