import hashlib
import json
from pathlib import Path
import subprocess

import pytest
from pydantic import ValidationError

from app.schemas.ocr import (
    OCRCapabilityDecision,
    OCRCapabilityState,
    OCRIssueCode,
    OCRModelArtifact,
    OCRProviderAvailability,
    OCRProviderIdentity,
    OCRQualityState,
)
from app.schemas.ocr_qualification import (
    OCRQualificationManifest,
    OCRQualificationStatus,
)
from app.schemas.standards import StandardPage, StandardParseStatus
from app.services.ocr.accepted_boundary import OCRAcceptedBoundaryService
from app.services.ocr.capability_gate import OCRCapabilityGate
from app.services.ocr.capability_gate import OCRCapabilityError
from app.services.ocr.pilot_service import ControlledOCRPilotService
from app.services.ocr.provider import (
    OCRProviderError,
    RapidOCRSubprocessProvider,
    RapidOCRWorkerConfig,
)
from app.services.ocr.qualification_manifest import (
    OCRQualificationManifestError,
    load_qualification_manifest,
    sha256_bytes,
)
from app.services.ocr.quality_gate import OCRQualityGate
from app.services.ocr.run_identity import (
    OCR_CORPUS_SEMANTICS_VERSION,
    OCR_QUALITY_GATE_VERSION,
)
from app.services.ocr.trusted_source_registry import (
    GB55023_2022_SOURCE_SHA256,
    TrustedOfficialSourceConflictError,
    TrustedOfficialSourceRegistry,
)
from app.services.pdf_service import (
    InvalidPDFError,
    PDFExtractionResult,
    PDFPageText,
)
from app.services.standards.standard_parser_service import StandardParserService
from scripts.prepare_competition_demo import (
    CompetitionBootstrapError,
    load_corpus_package,
    load_manifest,
)
from tests.test_ocr_article_boundary_b3b2_f1 import _parse
from tests.test_ocr_integration_b3b1b import (
    SOURCE_CHECKSUM,
    _binding,
    _document,
    _line,
    _provider,
    _render,
    _run,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _AvailableProvider:
    identity = _provider()

    def __init__(self, available: bool = True) -> None:
        self.available = available

    def availability(self) -> OCRProviderAvailability:
        return OCRProviderAvailability(
            available=self.available,
            reason="available" if self.available else "unavailable",
        )


class _ExtractionService:
    def __init__(self, pages=None, error: Exception | None = None) -> None:
        self.pages = pages or []
        self.error = error
        self.calls = 0

    def extract_text(self, path, *, require_text):
        self.calls += 1
        if self.error:
            raise self.error
        return PDFExtractionResult(
            page_count=len(self.pages),
            char_count=sum(len(page.text) for page in self.pages),
            text="\n".join(page.text for page in self.pages),
            pages=self.pages,
        )


def _decision(pages, requested, *, available=True):
    extraction = _ExtractionService(pages)
    gate = OCRCapabilityGate(pdf_service=extraction)
    return gate.decide(
        pdf_path=Path("facts-only.pdf"),
        requested_page_numbers=requested,
        provider=_AvailableProvider(available),
    )


@pytest.mark.parametrize("confidence", [0.0, 0.10, 0.50, 0.799999])
def test_below_acceptance_confidence_never_crosses_normative_boundary(confidence) -> None:
    line = _line("1.0.1 sufficiently long normative-looking text", 0).model_copy(
        update={"confidence": confidence}
    )
    run = _run([line])
    assessment = OCRQualityGate().assess(run)
    assert assessment.pages[0].accepted_line_ids == []
    assert OCRAcceptedBoundaryService().to_standard_pages(run, assessment) == []
    expected = (
        OCRIssueCode.CONFIDENCE_REJECTED
        if confidence < OCRQualityGate.CONFIDENCE_REJECT_THRESHOLD
        else OCRIssueCode.CONFIDENCE_REVIEW_REQUIRED
    )
    assert expected in {issue.code for issue in assessment.pages[0].issues}


@pytest.mark.parametrize("confidence", [0.80, 0.800001, 1.0])
def test_acceptance_confidence_boundary_is_inclusive_and_deterministic(confidence) -> None:
    line = _line("1.0.1 sufficiently long normative-looking text", 0).model_copy(
        update={"confidence": confidence}
    )
    run = _run([line])
    assessment = OCRQualityGate().assess(run)
    assert assessment.state == OCRQualityState.OCR_ACCEPTED
    assert assessment.pages[0].accepted_line_ids == [line.line_id]


@pytest.mark.parametrize("confidence", [float("nan"), -0.1, 1.1])
def test_malformed_confidence_fails_schema_closed(confidence) -> None:
    with pytest.raises(ValidationError):
        _line("1.0.1 text", 0).model_copy(
            update={"confidence": confidence}
        ).model_validate(
            {**_line("1.0.1 text", 0).model_dump(), "confidence": confidence}
        )


def test_one_low_confidence_body_line_quarantines_complete_article_interval() -> None:
    heading = _line("1.0.1 first article heading", 0)
    low_body = _line("body that must never leak partially", 1).model_copy(
        update={"confidence": 0.79}
    )
    next_heading = _line("1.0.2 independently safe article", 2)
    run = _run([heading, low_body, next_heading])
    assessment = OCRQualityGate().assess(run)
    accepted = set(assessment.pages[0].accepted_line_ids)
    assert heading.line_id not in accepted
    assert low_body.line_id not in accepted
    assert next_heading.line_id in accepted
    articles = _parse(run, assessment)
    assert [article.article_number for article in articles] == ["1.0.2"]


def test_perfect_geometry_does_not_override_low_confidence() -> None:
    line = _line("1.0.1 visually supported but unreliable text", 0).model_copy(
        update={"confidence": 0.0}
    )
    assessment = OCRQualityGate().assess(_run([line]))
    assert assessment.state == OCRQualityState.OCR_REJECTED


def test_text_ready_document_is_not_authorized_for_ocr() -> None:
    decision = _decision(
        [PDFPageText(1, "1 总则\n1.0.1 构件应可靠连接。")], [1]
    )
    assert decision.state == OCRCapabilityState.NOT_REQUIRED
    assert decision.authorized_page_numbers == ()


def test_image_only_document_is_authorized_for_ocr() -> None:
    decision = _decision([PDFPageText(1, "", image_count=1)], [1])
    assert decision.state == OCRCapabilityState.REQUIRED
    assert decision.authorized_page_numbers == (1,)


def test_text_ready_decision_does_not_depend_on_provider_availability() -> None:
    pages = [
        PDFPageText(
            1,
            "1 General provisions\n1.0.1 Components shall remain reliably connected.",
        )
    ]
    assert _decision(pages, [1], available=False).state == OCRCapabilityState.NOT_REQUIRED


def test_mixed_document_authorizes_only_image_only_pages() -> None:
    pages = [
        PDFPageText(1, "1 总则\n1.0.1 构件应可靠连接。"),
        PDFPageText(2, "", image_count=1),
    ]
    decision = _decision(pages, [2])
    assert decision.state == OCRCapabilityState.MIXED
    assert decision.authorized_page_numbers == (2,)
    assert decision.native_text_page_numbers == (1,)
    assert _decision(pages, [1, 2]).state == OCRCapabilityState.BLOCKED


def test_empty_and_unsupported_documents_are_blocked() -> None:
    assert _decision([], [1]).state == OCRCapabilityState.BLOCKED
    service = _ExtractionService(error=InvalidPDFError("unsupported"))
    gate = OCRCapabilityGate(pdf_service=service)
    assert gate.decide(
        pdf_path=Path("bad.pdf"),
        requested_page_numbers=[1],
        provider=_AvailableProvider(),
    ).state == OCRCapabilityState.BLOCKED


def test_provider_unavailable_fails_after_inspection_and_before_page_render() -> None:
    service = _ExtractionService([PDFPageText(1, "", image_count=1)])
    decision = OCRCapabilityGate(pdf_service=service).decide(
        pdf_path=Path("scan.pdf"),
        requested_page_numbers=[1],
        provider=_AvailableProvider(False),
    )
    assert decision.state == OCRCapabilityState.UNAVAILABLE
    assert service.calls == 1


@pytest.mark.parametrize(
    "state",
    [
        OCRCapabilityState.NOT_REQUIRED,
        OCRCapabilityState.UNAVAILABLE,
        OCRCapabilityState.BLOCKED,
    ],
)
def test_pilot_enforces_non_authorizing_capability_before_render(
    tmp_path, state
) -> None:
    pdf_path = tmp_path / "facts.pdf"
    pdf_path.write_bytes(b"capability facts")
    checksum = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

    class Capability:
        def decide(self, **kwargs):
            return OCRCapabilityDecision(
                state=state,
                document_status=state.value,
                reason="not authorized by inspector facts",
            )

    class Renderer:
        calls = 0

        def render(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("render must not run")

    renderer = Renderer()
    service = ControlledOCRPilotService(
        renderer=renderer,
        provider=_AvailableProvider(),
        repository=object(),
        capability_gate=Capability(),
        trusted_sources=TrustedOfficialSourceRegistry(records=()),
    )
    with pytest.raises(OCRCapabilityError):
        service.run(
            document=_document(source_checksum=checksum),
            pdf_path=pdf_path,
            page_numbers=[1],
        )
    assert renderer.calls == 0


def test_trusted_registry_resolves_exact_gb55023_authority() -> None:
    registry = TrustedOfficialSourceRegistry()
    record = registry.resolve(
        source_checksum=GB55023_2022_SOURCE_SHA256,
        requested=registry.binding(registry.default_records()[0]),
    )
    assert record is not None
    assert record.canonical_standard_code == "GB55023-2022"
    assert record.standard_name == "施工脚手架通用规范"


@pytest.mark.parametrize("field,value", [
    ("canonical_standard_code", "GB50016-2014"),
    ("standard_name", "建筑设计防火规范"),
])
def test_caller_identity_cannot_override_trusted_checksum(field, value) -> None:
    registry = TrustedOfficialSourceRegistry()
    requested = registry.binding(registry.default_records()[0]).model_copy(
        update={field: value}
    )
    with pytest.raises(TrustedOfficialSourceConflictError):
        registry.resolve(
            source_checksum=GB55023_2022_SOURCE_SHA256,
            requested=requested,
        )


def test_unknown_checksum_has_no_authority_elevation() -> None:
    registry = TrustedOfficialSourceRegistry()
    assert registry.resolve(source_checksum="f" * 64) is None
    pages = [StandardPage(page_number=1, text="1 总则\n1.0.1 正文应可靠。")]
    document = _document(source_checksum="f" * 64, official_source_binding=_binding())
    parsed = StandardParserService().parse(document, pages)
    assert parsed.document.identity_status.value != "CONFIRMED"
    assert parsed.document.official_source_binding is None


def test_known_title_with_wrong_checksum_is_rejected() -> None:
    registry = TrustedOfficialSourceRegistry()
    requested = registry.binding(registry.default_records()[0]).model_copy(
        update={"source_checksum": "f" * 64}
    )
    with pytest.raises(TrustedOfficialSourceConflictError):
        registry.resolve(source_checksum="f" * 64, requested=requested)


def test_ordinary_candidate_scored_identity_remains_unchanged() -> None:
    pages = [StandardPage(
        page_number=1,
        text="依据 GB 50016-2014\nGB 55023-2022\n施工脚手架通用规范\n1 总则\n1.0.1 正文应可靠。",
    )]
    parsed = StandardParserService().parse(_document(), pages)
    assert parsed.document.canonical_standard_code == "GB55023-2022"
    assert parsed.document.identity_status.value == "CONFIRMED"


def _qualified_manifest(**updates) -> OCRQualificationManifest:
    provider = _provider()
    values = {
        "schema_version": "ocr-qualification-manifest-v1",
        "official_source_checksum": SOURCE_CHECKSUM,
        "ocr_run_id": "ocr_" + "1" * 64,
        "ocr_execution_id": "execution_" + "2" * 64,
        "execution_artifact_sha256": "3" * 64,
        "raw_ocr_result_artifact_sha256": "4" * 64,
        "quality_assessment_artifact_sha256": "5" * 64,
        "accepted_boundary_artifact_sha256": "6" * 64,
        "qualified_parse_result_artifact_sha256": "7" * 64,
        "provider": provider,
        "render_config": _run([_line("1.0.1 text", 0)]).render_config,
        "quality_gate_version": OCR_QUALITY_GATE_VERSION,
        "ocr_corpus_semantics_version": OCR_CORPUS_SEMANTICS_VERSION,
        "qualification_status": OCRQualificationStatus.QUALIFIED,
        "qualification_reason": "synthetic complete-chain qualification",
    }
    values.update(updates)
    return OCRQualificationManifest(**values)


def test_qualified_manifest_requires_complete_external_artifact_chain() -> None:
    with pytest.raises(ValidationError):
        _qualified_manifest(raw_ocr_result_artifact_sha256=None)


def test_qualified_manifest_loads_only_exact_current_authority(tmp_path) -> None:
    manifest = _qualified_manifest()
    path = tmp_path / "qualification.json"
    data = json.dumps(
        manifest.model_dump(mode="json"), ensure_ascii=False, indent=2
    ).encode("utf-8")
    path.write_bytes(data)
    loaded = load_qualification_manifest(
        path,
        expected_file_sha256=sha256_bytes(data),
        expected_source_checksum=SOURCE_CHECKSUM,
        expected_parse_result_sha256="7" * 64,
    )
    assert loaded == manifest
    with pytest.raises(OCRQualificationManifestError):
        load_qualification_manifest(
            path,
            expected_file_sha256="0" * 64,
            expected_source_checksum=SOURCE_CHECKSUM,
            expected_parse_result_sha256="7" * 64,
        )


def test_requalified_gb55023_requires_complete_current_authority(tmp_path) -> None:
    manifest = load_manifest(PROJECT_ROOT / "competition/demo-manifest.json")
    package = load_corpus_package(manifest, PROJECT_ROOT)
    assert package.authority.parse_result_canonical_sha256 == (
        "2b53980c75987710afd949466460010a622c6141881ee95c1d5fb7756e20e442"
    )
    assert len(package.parse_result.articles) == 62

    authority_path = (
        PROJECT_ROOT / "competition/corpus/gb55023-ocr-qualification-manifest.json"
    )
    raw = json.loads(authority_path.read_text(encoding="utf-8"))
    assert raw["qualification_status"] == OCRQualificationStatus.QUALIFIED.value
    required_chain = (
        "execution_artifact_sha256",
        "raw_ocr_result_artifact_sha256",
        "quality_assessment_artifact_sha256",
        "accepted_boundary_artifact_sha256",
        "qualified_parse_result_artifact_sha256",
    )
    assert all(raw[field] for field in required_chain)

    raw["raw_ocr_result_artifact_sha256"] = None
    incomplete_path = tmp_path / "incomplete-qualification.json"
    incomplete_bytes = json.dumps(raw, ensure_ascii=False, indent=2).encode("utf-8")
    incomplete_path.write_bytes(incomplete_bytes)
    with pytest.raises(CompetitionBootstrapError, match="stale"):
        load_corpus_package(
            manifest,
            PROJECT_ROOT,
            qualification_manifest_path=incomplete_path,
            qualification_manifest_sha256=sha256_bytes(incomplete_bytes),
        )


def test_ocr_article_number_fragments_are_not_fabricated() -> None:
    first = _run([
        _line("4.4.1 original article", 0),
        _line("3 个连墙件应按下列规定设置", 1),
    ])
    first_articles = _parse(first, OCRQualityGate().assess(first))
    assert [item.article_number for item in first_articles] == ["4.4.1"]
    assert "4.4.13" not in first.model_dump_json()

    second = _run([
        _line("5.3.1", 0),
        _line("0 条文正文应符合要求", 1),
    ])
    second_articles = _parse(second, OCRQualityGate().assess(second))
    assert [item.article_number for item in second_articles] == ["5.3.1"]
    assert "5.3.10" not in second.model_dump_json()


def _provider_with_runner(tmp_path, runner):
    artifacts = []
    for name in ("det", "rec", "cls"):
        path = tmp_path / f"{name}.onnx"
        path.write_bytes(name.encode())
        artifacts.append((path, hashlib.sha256(name.encode()).hexdigest()))
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
    return RapidOCRSubprocessProvider(
        RapidOCRWorkerConfig(
            python_executable=tmp_path / "python.exe",
            worker_script=tmp_path / "worker.py",
            detector_model_path=artifacts[0][0],
            recognizer_model_path=artifacts[1][0],
            classifier_model_path=artifacts[2][0],
            identity=identity,
            timeout_seconds=7.5,
        ),
        runner=runner,
    )


def test_provider_enforces_shell_false_and_timeout(tmp_path) -> None:
    captured = {}

    def runner(*args, **kwargs):
        captured.update(kwargs)
        payload = {
            "provider_identity": provider.identity.model_dump(mode="json"),
            "lines": [],
            "elapsed_seconds": 0.0,
            "warnings": [],
            "error": None,
        }
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    provider = _provider_with_runner(tmp_path, runner)
    provider.recognize_page(type("Page", (), {"png_bytes": b"png", "metadata": _render()})())
    assert captured["shell"] is False
    assert captured["timeout"] == 7.5


@pytest.mark.parametrize(
    "mode", ["invalid-json", "nonzero", "identity-mismatch", "timeout"]
)
def test_worker_protocol_failures_are_closed(tmp_path, mode) -> None:
    def runner(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(cmd="ocr-worker", timeout=7.5)
        if mode == "invalid-json":
            return subprocess.CompletedProcess([], 0, "not-json", "")
        if mode == "nonzero":
            return subprocess.CompletedProcess([], 2, "", "worker failed")
        return subprocess.CompletedProcess(
            [],
            0,
            json.dumps({"provider_identity": {}, "lines": []}),
            "",
        )

    provider = _provider_with_runner(tmp_path, runner)
    with pytest.raises(OCRProviderError):
        provider.recognize_page(
            type("Page", (), {"png_bytes": b"png", "metadata": _render()})()
        )


def test_missing_model_artifact_fails_before_worker_execution(tmp_path) -> None:
    provider = _provider_with_runner(
        tmp_path,
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("worker must not run")
        ),
    )
    provider.config.detector_model_path.unlink()
    page = type("Page", (), {"png_bytes": b"png", "metadata": _render()})()
    with pytest.raises(OCRProviderError, match="missing"):
        provider.recognize_page(page)
