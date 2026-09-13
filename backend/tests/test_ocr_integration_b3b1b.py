import hashlib
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess

import fitz
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.ocr import (
    OCRAcceptedPage,
    OCRAcceptedSpan,
    OCRCapabilityDecision,
    OCRCapabilityState,
    OCRIssueCode,
    OCRModelArtifact,
    OCRPageQualityAssessment,
    OCRPageResult,
    OCRProviderIdentity,
    OCRQualityState,
    OCRRenderConfig,
    OCRRenderMetadata,
    OCRRun,
    OCRRunQualityAssessment,
    OCRSourceMapping,
    OCRVisualSupportDatum,
    OCRVisualSupportGeometrySource,
    OfficialSourceBinding,
    SourceKind,
    TrustedOfficialSourceRecord,
)
from app.schemas.standards import StandardDocument, StandardPage
from app.schemas.standards_retrieval import RetrievalMethod
from app.services.ocr.accepted_boundary import (
    OCRAcceptedBoundaryService,
    OCRBoundaryError,
)
from app.services.ocr.ocr_repository import OCRRepository, OCRRepositoryError
from app.services.ocr.page_renderer import OCRPageRenderer, RenderedOCRPage
from app.services.ocr.pilot_service import ControlledOCRPilotService
from app.services.ocr.provider import (
    OCRProviderError,
    RapidOCRSubprocessProvider,
    RapidOCRWorkerConfig,
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
from app.services.ocr.trusted_source_registry import TrustedOfficialSourceRegistry
from app.services.retrieval.hit_factory import make_hit
from app.services.retrieval.models import RetrievalRecord
from app.services.retrieval.retrieval_manifest_service import RetrievalManifestService
from app.services.standards.standard_document_service import (
    StandardDocumentService,
    StandardParseFailure,
)
from app.services.standards.standard_parser_service import StandardParserService
from app.services.standards.standard_repository import StandardRepository


SOURCE_CHECKSUM = "a" * 64


def _provider(*, recognizer_hash: str = "2" * 64) -> OCRProviderIdentity:
    return OCRProviderIdentity(
        provider="rapidocr",
        provider_version="3.9.2",
        runtime="onnxruntime",
        runtime_version="1.29.0",
        device="CPUExecutionProvider",
        detector=OCRModelArtifact(model_id="PP-OCRv6_det_small.onnx", sha256="1" * 64),
        recognizer=OCRModelArtifact(model_id="PP-OCRv6_rec_small.onnx", sha256=recognizer_hash),
        classifier=OCRModelArtifact(model_id="cls.onnx", sha256="3" * 64),
        classification_enabled=False,
    )


def _render(page_number: int = 1, *, dpi: int = 300) -> OCRRenderMetadata:
    return OCRRenderMetadata(
        renderer="PyMuPDF",
        renderer_version="1",
        dpi=dpi,
        colorspace="GRAY",
        alpha=False,
        config_version="render-1",
        page_number=page_number,
        pixel_width=1655,
        pixel_height=2396,
        image_sha256=(hex(page_number)[2:] * 64)[:64],
    )


def _line(text: str, index: int, *, page: int = 1, polygon=None):
    polygon = polygon or [[100, 100 + index * 60], [900, 100 + index * 60], [900, 140 + index * 60], [100, 140 + index * 60]]
    from app.schemas.ocr import OCRLineResult

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
        confidence=0.99,
        reading_order_index=index,
        visual_support=OCRVisualSupportDatum(
            geometry_source=OCRVisualSupportGeometrySource.LINE_POLYGON,
            dark_support=0.5,
            page_reference_support=0.5,
            relative_support_ratio=1.0,
            height_ratio=1.0,
            aspect_ratio=20.0,
            rule_version=OCR_RASTER_SUPPORT_RULE_VERSION,
        ),
    )


def _run(lines, *, provider=None, render=None, quality_version=OCR_QUALITY_GATE_VERSION):
    provider = provider or _provider()
    render = render or _render()
    config = OCRRenderConfig(**render.model_dump(exclude={"page_number", "pixel_width", "pixel_height", "image_sha256"}))
    page = OCRPageResult(
        page_number=render.page_number,
        lines=lines,
        render=render,
        provider=provider,
        elapsed_seconds=1.0,
    )
    run_id = semantic_ocr_run_id(
        official_source_checksum=SOURCE_CHECKSUM,
        provider=provider,
        render_config=config,
        quality_gate_version=quality_version,
        render_metadata=[render],
    )
    return OCRRun(
        ocr_run_id=run_id,
        execution_id=execution_id([page]),
        official_source_checksum=SOURCE_CHECKSUM,
        provider=provider,
        render_config=config,
        adapter_version=OCR_ADAPTER_VERSION,
        quality_gate_version=quality_version,
        corpus_semantics_version=OCR_CORPUS_SEMANTICS_VERSION,
        pages=[page],
    )


def _binding() -> OfficialSourceBinding:
    return OfficialSourceBinding(
        source_checksum=SOURCE_CHECKSUM,
        canonical_standard_code="GB55023-2022",
        display_standard_code="GB 55023-2022",
        standard_name="Construction scaffold standard",
        binding_reason="Official ministry artifact was manually verified.",
        binding_provenance="controlled test fixture",
        confirmed=True,
    )


def _document(**updates) -> StandardDocument:
    values = dict(
        standard_id="11111111-1111-1111-1111-111111111111",
        source_filename="official.pdf",
        source_checksum=SOURCE_CHECKSUM,
        page_count=17,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    values.update(updates)
    return StandardDocument(**values)


def test_pdf_text_defaults_remain_backward_compatible() -> None:
    page = StandardPage(page_number=1, text="1.0.1 Existing text")
    assert page.source_kind == SourceKind.PDF_TEXT
    assert page.source_mappings == []


def test_raw_ocr_run_serialization_retains_geometry() -> None:
    run = _run([_line("1.0.1 Text", 0)])
    restored = OCRRun.model_validate_json(run.model_dump_json())
    assert restored.pages[0].lines[0].polygon == run.pages[0].lines[0].polygon
    assert restored.execution_id == run.execution_id


def test_ocr_standard_page_requires_accepted_boundary_provenance() -> None:
    with pytest.raises(ValidationError):
        StandardPage(page_number=1, text="OCR", source_kind=SourceKind.OCR_TEXT)


def test_rejected_ocr_cannot_enter_parser_boundary() -> None:
    run = _run([])
    assessment = OCRQualityGate().assess(run)
    pages = OCRAcceptedBoundaryService().accepted_pages(run, assessment)
    assert assessment.state == OCRQualityState.OCR_REJECTED
    assert pages == []


def test_low_confidence_lines_are_excluded_but_safe_lines_remain() -> None:
    safe = _line("1.0.1 Safe article", 0)
    overlay = _line("overlay", 1, polygon=[[1000, 200], [1120, 150], [1180, 340], [1060, 390]])
    run = _run([safe, overlay])
    assessment = OCRQualityGate().assess(run)
    accepted = OCRAcceptedBoundaryService().accepted_pages(run, assessment)
    accepted_ids = [mapping.raw_line_id for span in accepted[0].spans for mapping in span.source_mappings]
    assert assessment.pages[0].state == OCRQualityState.OCR_LOW_CONFIDENCE
    assert safe.line_id in accepted_ids
    assert overlay.line_id not in accepted_ids


def test_article_number_ambiguity_is_flagged_and_not_repaired() -> None:
    ambiguous = _line("4.46 Wall tie requirements", 0)
    run = _run([ambiguous, _line("4.4.5 Valid neighboring article", 1)])
    assessment = OCRQualityGate().assess(run)
    assert OCRIssueCode.ARTICLE_NUMBER_AMBIGUITY in {
        issue.code for issue in assessment.pages[0].issues
    }
    serialized = run.model_dump_json()
    assert "4.4.6" not in serialized


def test_section_heading_is_not_misclassified_as_article_ambiguity() -> None:
    run = _run([_line("4.4 Construction requirements", 0), _line("4.4.1 Valid article", 1)])
    assessment = OCRQualityGate().assess(run)
    assert OCRIssueCode.ARTICLE_NUMBER_AMBIGUITY not in {
        issue.code for issue in assessment.pages[0].issues
    }


def test_missing_list_number_is_surfaced_without_reconstruction() -> None:
    lines = [_line(f"{number} list body", number) for number in (1, 2, 3, 5, 6)]
    run = _run(lines)
    assessment = OCRQualityGate().assess(run)
    assert OCRIssueCode.STRUCTURAL_SEQUENCE_GAP in {
        issue.code for issue in assessment.pages[0].issues
    }
    assert all(line.raw_text != "4" for line in run.pages[0].lines)


def test_list_gap_excludes_its_enclosing_article_but_not_next_article() -> None:
    lines = [
        _line("4.4.4 Article with list", 0),
        *[_line(f"{number} list body", number) for number in (1, 2, 3, 5)],
        _line("4.4.5 Independent article", 6),
    ]
    run = _run(lines)
    assessment = OCRQualityGate().assess(run)
    accepted = set(assessment.pages[0].accepted_line_ids)
    assert lines[0].line_id not in accepted
    assert lines[-1].line_id in accepted


def test_detached_number_reading_order_is_surfaced() -> None:
    body = _line("detached list body", 0, polygon=[[100, 300], [900, 300], [900, 350], [100, 350]])
    number = _line("6", 1, polygon=[[80, 305], [120, 305], [120, 345], [80, 345]])
    run = _run([body, number, _line("other useful text", 2)])
    assessment = OCRQualityGate().assess(run)
    assert OCRIssueCode.READING_ORDER_ANOMALY in {
        issue.code for issue in assessment.pages[0].issues
    }


def test_source_mapping_is_explicit_in_accepted_parser_page() -> None:
    run = _run([_line("1.0.1 Raw  text", 0)])
    assessment = OCRQualityGate().assess(run)
    accepted = OCRAcceptedBoundaryService().accepted_pages(run, assessment)
    pages = OCRAcceptedBoundaryService().to_standard_pages(run, assessment)
    mapping = pages[0].source_mappings[0]
    assert pages[0].source_kind == SourceKind.OCR_TEXT
    assert mapping.raw_text == "1.0.1 Raw  text"
    assert pages[0].text == "1.0.1 Raw text"
    assert mapping.raw_polygon


def test_assessment_for_another_run_is_rejected() -> None:
    run = _run([_line("useful body", 0)])
    assessment = OCRQualityGate().assess(run).model_copy(update={"execution_id": "other"})
    with pytest.raises(OCRBoundaryError):
        OCRAcceptedBoundaryService().accepted_pages(run, assessment)


def test_tampered_semantic_run_identity_is_rejected() -> None:
    run = _run([_line("safe body text", 0)])
    tampered = run.model_copy(update={"ocr_run_id": "ocr_tampered"})
    with pytest.raises(ValueError, match="semantic run identity"):
        OCRQualityGate().assess(tampered)


def test_assessment_cannot_accept_a_line_affected_by_an_issue() -> None:
    overlay = _line("overlay", 0, polygon=[[1000, 200], [1120, 150], [1180, 340], [1060, 390]])
    safe = _line("safe body text", 1)
    run = _run([overlay, safe])
    assessment = OCRQualityGate().assess(run)
    tampered_page = assessment.pages[0].model_copy(
        update={"accepted_line_ids": [overlay.line_id, safe.line_id]}
    )
    tampered = assessment.model_copy(update={"pages": [tampered_page]})
    with pytest.raises(OCRBoundaryError, match="affected"):
        OCRAcceptedBoundaryService().accepted_pages(run, tampered)


def test_accepted_ocr_reaches_parser_with_raw_line_mapping() -> None:
    run = _run([_line("1 General", 0), _line("1.0.1 Components shall be stable.", 1)])
    assessment = OCRQualityGate().assess(run)
    pages = OCRAcceptedBoundaryService().to_standard_pages(run, assessment)
    document = _document(
        source_kind=SourceKind.OCR_TEXT,
        ocr_run_id=run.ocr_run_id,
        ocr_execution_id=run.execution_id,
        ocr_quality_state=assessment.state,
        ocr_provider=run.provider,
        official_source_binding=_binding(),
        ocr_corpus_semantics_version=OCR_CORPUS_SEMANTICS_VERSION,
    )
    result = StandardParserService().parse(document, pages)
    assert result.articles
    article = result.articles[0]
    assert article.source_kind == SourceKind.OCR_TEXT
    assert article.source_mappings
    assert article.source_text == "1.0.1 Components shall be stable."


def test_ocr_evidence_exposes_official_source_and_raw_spans() -> None:
    run = _run([_line("1 General", 0), _line("1.0.1 Components shall be stable.", 1)])
    assessment = OCRQualityGate().assess(run)
    pages = OCRAcceptedBoundaryService().to_standard_pages(run, assessment)
    document = _document(
        source_kind=SourceKind.OCR_TEXT,
        ocr_run_id=run.ocr_run_id,
        ocr_execution_id=run.execution_id,
        ocr_quality_state=assessment.state,
        ocr_provider=run.provider,
        official_source_binding=_binding(),
        ocr_corpus_semantics_version=OCR_CORPUS_SEMANTICS_VERSION,
    )
    trusted = TrustedOfficialSourceRecord(
        source_checksum=SOURCE_CHECKSUM,
        canonical_standard_code="GB55023-2022",
        display_standard_code="GB 55023-2022",
        standard_name="Construction scaffold standard",
        authority_id="test-qualified-source",
        authority_provenance="controlled test fixture",
    )
    parsed = StandardParserService(
        trusted_sources=TrustedOfficialSourceRegistry(records=(trusted,))
    ).parse(document, pages)
    hit = make_hit(
        RetrievalRecord(document=parsed.document, article=parsed.articles[0]),
        rank=1,
        methods=[RetrievalMethod.KEYWORD],
    )
    assert hit.evidence.source_kind == SourceKind.OCR_TEXT
    assert hit.evidence.source_checksum == SOURCE_CHECKSUM
    assert hit.evidence.source_page_start == 1
    assert hit.evidence.ocr_run_id == run.ocr_run_id
    assert hit.evidence.ocr_run_quality_state == assessment.state
    assert hit.evidence.raw_source_span_refs[0].raw_polygon
    assert hit.evidence.official_source_binding.binding_method.value == "CURATED_OFFICIAL_SOURCE"
    assert "Trusted official-source authority" in hit.evidence.identity_reason


def test_same_accepted_ocr_input_has_stable_article_identity() -> None:
    service = StandardParserService()
    run = _run([_line("1 General", 0), _line("1.0.1 Stable input.", 1)])
    assessment = OCRQualityGate().assess(run)
    pages = OCRAcceptedBoundaryService().to_standard_pages(run, assessment)
    document = _document(source_kind=SourceKind.OCR_TEXT, ocr_run_id=run.ocr_run_id, ocr_execution_id=run.execution_id, ocr_quality_state=OCRQualityState.OCR_ACCEPTED, ocr_provider=run.provider, official_source_binding=_binding())
    assert service.parse(document, pages).articles[0].article_id == service.parse(document, pages).articles[0].article_id


def test_changed_accepted_ocr_input_changes_article_identity() -> None:
    def parse(text):
        run = _run([_line("1 General", 0), _line(text, 1)])
        assessment = OCRQualityGate().assess(run)
        pages = OCRAcceptedBoundaryService().to_standard_pages(run, assessment)
        document = _document(source_kind=SourceKind.OCR_TEXT, ocr_run_id=run.ocr_run_id, ocr_execution_id=run.execution_id, ocr_quality_state=OCRQualityState.OCR_ACCEPTED, ocr_provider=run.provider, official_source_binding=_binding())
        return StandardParserService().parse(document, pages).articles[0]
    assert parse("1.0.1 First text.").article_id != parse("1.0.1 Changed text.").article_id


def test_render_model_and_gate_identity_invalidate_ocr_run() -> None:
    baseline = _run([_line("body text", 0)])
    changed_render = _run([_line("body text", 0)], render=_render(dpi=400))
    changed_model = _run([_line("body text", 0)], provider=_provider(recognizer_hash="4" * 64))
    changed_gate = _run([_line("body text", 0)], quality_version="next-gate")
    assert len({baseline.ocr_run_id, changed_render.ocr_run_id, changed_model.ocr_run_id, changed_gate.ocr_run_id}) == 4


def test_ocr_manifest_changes_with_ocr_identity() -> None:
    article_run = _run([_line("1 General", 0), _line("1.0.1 Stable input.", 1)])
    assessment = OCRQualityGate().assess(article_run)
    pages = OCRAcceptedBoundaryService().to_standard_pages(article_run, assessment)
    document = _document(source_kind=SourceKind.OCR_TEXT, ocr_run_id=article_run.ocr_run_id, ocr_execution_id=article_run.execution_id, ocr_quality_state=OCRQualityState.OCR_ACCEPTED, ocr_provider=article_run.provider, official_source_binding=_binding(), ocr_corpus_semantics_version=OCR_CORPUS_SEMANTICS_VERSION)
    parsed = StandardParserService().parse(document, pages)
    record = RetrievalRecord(parsed.document, parsed.articles[0])
    changed_article = record.article.model_copy(update={"ocr_run_id": "ocr_changed"})
    assert RetrievalManifestService().corpus_hash([record]) != RetrievalManifestService().corpus_hash([RetrievalRecord(record.document, changed_article)])


def test_raw_ocr_repository_is_immutable_and_separate_from_pages_json(tmp_path) -> None:
    standards = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    repository = OCRRepository(standards)
    run = _run([_line("raw OCR", 0)])
    directory = repository.save_raw_run("std", run)
    assert repository.save_raw_run("std", run) == directory
    changed = run.model_copy(update={"pages": [run.pages[0].model_copy(update={"elapsed_seconds": 9.0})]})
    with pytest.raises(OCRRepositoryError):
        repository.save_raw_run("std", changed)
    assert (directory / "raw_pages.json").is_file()
    assert not (standards.document_dir("std") / "pages.json").exists()


def test_ocr_repository_persists_under_long_windows_parent_path(tmp_path) -> None:
    long_parent = (
        tmp_path
        / ("normal_windows_pytest_parent_" + "x" * 18)
        / ("nested_" + "y" * 8)
    )
    standards = StandardRepository(
        Settings(standards_dir=long_parent / "standards")
    )
    repository = OCRRepository(standards)
    run = _run([_line("raw OCR", 0)])

    directory = repository.save_raw_run("std", run)

    assert directory.parent.name == "runs"
    assert directory.name == repository.storage_key(run)
    assert len(directory.name) == 32
    assert (directory / "run.json").is_file()
    assert (directory / "raw_pages.json").is_file()
    stored_run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    assert stored_run["ocr_run_id"] == run.ocr_run_id
    assert stored_run["execution_id"] == run.execution_id


def test_ocr_repository_separates_different_execution_identities(tmp_path) -> None:
    repository = OCRRepository(
        StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    )
    first = _run([_line("first raw OCR", 0)])
    second = _run([_line("different raw OCR", 0)])
    assert first.ocr_run_id == second.ocr_run_id
    assert first.execution_id != second.execution_id

    first_dir = repository.save_raw_run("std", first)
    second_dir = repository.save_raw_run("std", second)

    assert first_dir != second_dir
    assert (first_dir / "run.json").is_file()
    assert (second_dir / "run.json").is_file()


def test_ocr_repository_storage_key_collision_fails_closed(tmp_path) -> None:
    standards = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    repository = OCRRepository(standards)
    requested = _run([_line("requested raw OCR", 0)])
    different = _run([_line("different raw OCR", 0)])
    collision_dir = (
        standards.document_dir("std")
        / "ocr"
        / "runs"
        / repository.storage_key(requested)
    )
    collision_dir.mkdir(parents=True)
    (collision_dir / "run.json").write_text(
        different.model_dump_json(exclude={"pages"}), encoding="utf-8"
    )

    with pytest.raises(OCRRepositoryError, match="collision"):
        repository.save_raw_run("std", requested)


def test_ocr_repository_rejects_path_like_standard_identity(tmp_path) -> None:
    repository = OCRRepository(StandardRepository(Settings(standards_dir=tmp_path / "standards")))
    with pytest.raises(OCRRepositoryError, match="Invalid standard identity"):
        repository.save_raw_run("../escape", _run([_line("raw OCR", 0)]))


def test_provider_model_hash_mismatch_fails_before_worker_execution(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"wrong")
    identity = _provider().model_copy(update={
        "detector": OCRModelArtifact(model_id="model.onnx", sha256="1" * 64),
        "recognizer": OCRModelArtifact(model_id="model.onnx", sha256="2" * 64),
        "classifier": OCRModelArtifact(model_id="model.onnx", sha256="3" * 64),
    })
    config = RapidOCRWorkerConfig(
        python_executable=Path("python"),
        worker_script=Path("worker.py"),
        detector_model_path=model,
        recognizer_model_path=model,
        classifier_model_path=model,
        identity=identity,
    )
    provider = RapidOCRSubprocessProvider(config, runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runner called")))
    page = RenderedOCRPage(b"png", _render())
    with pytest.raises(OCRProviderError, match="hash"):
        provider.recognize_page(page)


def test_provider_contract_preserves_character_geometry(tmp_path) -> None:
    artifacts = []
    for name, content in (("det", b"det"), ("rec", b"rec"), ("cls", b"cls")):
        path = tmp_path / f"{name}.onnx"
        path.write_bytes(content)
        artifacts.append((path, hashlib.sha256(content).hexdigest()))
    identity = OCRProviderIdentity(
        provider="rapidocr",
        provider_version="3.9.2",
        runtime="onnxruntime",
        runtime_version="1.29.0",
        device="CPUExecutionProvider",
        detector=OCRModelArtifact(model_id="det.onnx", sha256=artifacts[0][1]),
        recognizer=OCRModelArtifact(model_id="rec.onnx", sha256=artifacts[1][1]),
        classifier=OCRModelArtifact(model_id="cls.onnx", sha256=artifacts[2][1]),
    )
    payload = {
        "provider_identity": identity.model_dump(mode="json"),
        "lines": [{
            "text": "4.4.6",
            "confidence": 0.99,
            "polygon": [[1, 1], [10, 1], [10, 5], [1, 5]],
            "characters": [{
                "text": "4", "confidence": 0.98,
                "polygon": [[1, 1], [2, 1], [2, 5], [1, 5]],
            }],
        }],
        "elapsed_seconds": 1.0,
        "warnings": [],
        "error": None,
    }
    completed = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
    provider = RapidOCRSubprocessProvider(
        RapidOCRWorkerConfig(
            python_executable=Path("python"), worker_script=Path("worker.py"),
            detector_model_path=artifacts[0][0], recognizer_model_path=artifacts[1][0],
            classifier_model_path=artifacts[2][0], identity=identity,
        ),
        runner=lambda *args, **kwargs: completed,
    )
    document = fitz.open()
    raster_page = document.new_page(width=20, height=10)
    raster_page.draw_rect(fitz.Rect(1, 1, 10, 5), fill=(0, 0, 0))
    pixmap = raster_page.get_pixmap(colorspace=fitz.csGRAY, alpha=False)
    png_bytes = pixmap.tobytes("png")
    document.close()
    render = _render().model_copy(
        update={
            "pixel_width": pixmap.width,
            "pixel_height": pixmap.height,
            "image_sha256": hashlib.sha256(png_bytes).hexdigest(),
        }
    )
    result = provider.recognize_page(RenderedOCRPage(png_bytes, render))
    assert result.lines[0].characters[0].text == "4"
    assert result.lines[0].characters[0].polygon
    assert result.lines[0].visual_support is not None
    assert (
        result.lines[0].visual_support.geometry_source
        == OCRVisualSupportGeometrySource.CHARACTER_POLYGONS
    )


def test_page_renderer_is_deterministic_and_one_based(tmp_path) -> None:
    pdf_path = tmp_path / "one.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "test")
    document.save(pdf_path)
    document.close()
    renderer = OCRPageRenderer(OCRRenderConfig(renderer_version=fitz.VersionBind, config_version="test"))
    first = renderer.render(pdf_path, 1)
    second = renderer.render(pdf_path, 1)
    assert first.metadata.image_sha256 == second.metadata.image_sha256
    assert first.metadata.page_number == 1


def test_worker_failure_is_persisted_and_rejected_without_parser_entry(tmp_path) -> None:
    pdf_path = tmp_path / "official.pdf"
    pdf_path.write_bytes(b"controlled official bytes")
    checksum = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    class FailingProvider:
        identity = _provider()

        def recognize_page(self, page):
            raise OCRProviderError("controlled worker failure")

    class FakeRenderer:
        config = OCRRenderConfig(renderer_version="1", config_version="test")

        def render(self, pdf_path, page_number):
            metadata = _render(page_number).model_copy(update={
                "renderer_version": "1", "config_version": "test"
            })
            return RenderedOCRPage(b"png", metadata)

    class RequiredCapability:
        def decide(self, **kwargs):
            return OCRCapabilityDecision(
                state=OCRCapabilityState.REQUIRED,
                document_status="OCR_REQUIRED",
                authorized_page_numbers=(1,),
                reason="controlled failure-path fixture",
            )

    standards = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    ocr_repository = OCRRepository(standards)
    document = _document(source_checksum=checksum)
    result = ControlledOCRPilotService(
        renderer=FakeRenderer(),
        provider=FailingProvider(),
        repository=ocr_repository,
        capability_gate=RequiredCapability(),
        trusted_sources=TrustedOfficialSourceRegistry(records=()),
    ).run(
        document=document,
        pdf_path=pdf_path,
        page_numbers=[1],
        binding=_binding().model_copy(update={"source_checksum": checksum}),
    )
    assert result.assessment.state == OCRQualityState.OCR_REJECTED
    assert result.parse_result is None
    assert result.run.pages[0].error == "controlled worker failure"
    assert (
        ocr_repository.execution_dir(document.standard_id, result.run)
        / "raw_pages.json"
    ).is_file()


def test_identity_conflict_rejects_run() -> None:
    run = _run([_line("useful body text", 0)])
    bad_binding = _binding().model_copy(update={"source_checksum": "f" * 64})
    assessment = OCRQualityGate().assess(run, bad_binding)
    assert assessment.state == OCRQualityState.OCR_REJECTED
    assert OCRIssueCode.IDENTITY_CONFLICT in {issue.code for issue in assessment.pages[0].issues}


def test_legacy_document_service_cannot_bypass_controlled_ocr_boundary(tmp_path) -> None:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    document = _document(source_kind=SourceKind.OCR_TEXT)
    repository.save_document(document)
    repository.save_pages(document.standard_id, [StandardPage(page_number=1, text="legacy")])
    with pytest.raises(StandardParseFailure, match="OCR_CONTROLLED_BOUNDARY_REQUIRED"):
        StandardDocumentService(repository=repository).parse(document.standard_id)


def test_core_requirements_do_not_include_rapidocr() -> None:
    core_files = [Path("requirements.txt"), Path("backend/requirements.txt"), Path("pyproject.toml")]
    contents = "\n".join(path.read_text(encoding="utf-8") for path in core_files if path.exists()).lower()
    assert "rapidocr" not in contents
    assert "onnxruntime" not in contents
