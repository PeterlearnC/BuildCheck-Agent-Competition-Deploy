from app.schemas.ocr import (
    OCRModelArtifact,
    OCRLineResult,
    OCRPageResult,
    OCRProviderIdentity,
    OCRRenderConfig,
    OCRRenderMetadata,
    OCRVisualSupportDatum,
    OCRVisualSupportGeometrySource,
)
from app.services.ocr.raster_support import OCR_RASTER_SUPPORT_RULE_VERSION
from app.services.ocr.run_identity import (
    OCR_ADAPTER_VERSION,
    OCR_CORPUS_SEMANTICS_VERSION,
    OCR_QUALITY_GATE_VERSION,
    execution_id,
    semantic_ocr_run_id,
)
from app.services.retrieval.retrieval_manifest_service import (
    MANIFEST_VERSION,
    RETRIEVAL_VERSION,
)
from app.services.standards.stable_identity_service import (
    PARSER_SEMANTICS_VERSION,
    StableIdentityService,
)
from app.services.standards.structural_evidence_resolver import (
    STRUCTURAL_RESOLVER_VERSION,
)


PRE_S1_OCR_CORPUS_SEMANTICS = "v0.4-b.3c.2-f1-r1-ocr-corpus"


def _provider() -> OCRProviderIdentity:
    return OCRProviderIdentity(
        provider="synthetic",
        provider_version="1",
        runtime="synthetic-runtime",
        runtime_version="1",
        device="CPUExecutionProvider",
        detector=OCRModelArtifact(model_id="detector", sha256="1" * 64),
        recognizer=OCRModelArtifact(model_id="recognizer", sha256="2" * 64),
        classifier=OCRModelArtifact(model_id="classifier", sha256="3" * 64),
        classification_enabled=False,
    )


def _render() -> tuple[OCRRenderConfig, OCRRenderMetadata]:
    config = OCRRenderConfig(
        renderer_version="1",
        config_version="v0.4-b.3b.1b-render",
    )
    metadata = OCRRenderMetadata(
        **config.model_dump(),
        page_number=1,
        pixel_width=100,
        pixel_height=100,
        image_sha256="4" * 64,
    )
    return config, metadata


def _raw_page() -> OCRPageResult:
    _config, metadata = _render()
    polygon = [[1, 1], [90, 1], [90, 20], [1, 20]]
    return OCRPageResult(
        page_number=1,
        lines=[
            OCRLineResult(
                line_id="ocrline_synthetic",
                page_number=1,
                raw_text="Synthetic raw evidence",
                polygon=polygon,
                confidence=0.99,
                reading_order_index=0,
                visual_support=OCRVisualSupportDatum(
                    geometry_source=OCRVisualSupportGeometrySource.LINE_POLYGON,
                    dark_support=0.5,
                    page_reference_support=0.5,
                    relative_support_ratio=1.0,
                    height_ratio=1.0,
                    aspect_ratio=4.0,
                    rule_version=OCR_RASTER_SUPPORT_RULE_VERSION,
                ),
            )
        ],
        render=metadata,
        provider=_provider(),
    )


def test_s1_semantic_versions_are_frozen_without_unrelated_bumps() -> None:
    assert STRUCTURAL_RESOLVER_VERSION == "v0.4-b.3c.3-s1-r1-structural"
    assert PARSER_SEMANTICS_VERSION == "v0.4-b.3c.3-s1-r1-parser"
    assert OCR_CORPUS_SEMANTICS_VERSION == "v0.4-b.3b-q1-r1-ocr-corpus"
    assert OCR_ADAPTER_VERSION == "v0.4-b.3b.1b-adapter"
    assert OCR_QUALITY_GATE_VERSION == "v0.4-b.3b-q1-r1-quality"
    assert OCR_RASTER_SUPPORT_RULE_VERSION == "v0.4-b.3c.2-f1-r1-raster-support"
    assert RETRIEVAL_VERSION == "v0.4-b.3a.1"
    assert MANIFEST_VERSION == "v0.4-b.3a.1-manifest"


def test_raw_execution_identity_is_independent_of_corpus_semantics() -> None:
    page = _raw_page()
    assert execution_id([page]) == execution_id([page.model_copy(deep=True)])


def test_s1_corpus_semantics_changes_semantic_ocr_run_identity() -> None:
    config, metadata = _render()
    common = {
        "official_source_checksum": "a" * 64,
        "provider": _provider(),
        "render_config": config,
        "render_metadata": [metadata],
    }

    current = semantic_ocr_run_id(**common)
    historical = semantic_ocr_run_id(
        **common,
        corpus_semantics_version=PRE_S1_OCR_CORPUS_SEMANTICS,
    )

    assert current != historical


def test_corrected_structure_changes_derived_identities_not_hashing_rules() -> None:
    identities = StableIdentityService()
    common = {
        "standard_source_checksum": "a" * 64,
        "standard_code": "STD-2026",
        "article_number": "4.0.1",
        "article_text": "Grounded normative content.",
        "source_page_start": 7,
        "source_page_end": 7,
        "region_type": "NORMATIVE_BODY",
        "article_type": "NORMATIVE",
        "source_kind": "OCR_TEXT",
        "accepted_source_line_ids": ["ocrline_grounded"],
    }

    incorrect = identities.article_id(**common, chapter_number="2")
    corrected = identities.article_id(**common, chapter_number="4")

    assert incorrect != corrected
    assert identities.chunk_id(
        article_id=incorrect, chunk_index=0, chunk_text=common["article_text"]
    ) != identities.chunk_id(
        article_id=corrected, chunk_index=0, chunk_text=common["article_text"]
    )
    assert identities.chapter_id(
        standard_checksum=common["standard_source_checksum"],
        chapter_number="2",
        chapter_title="Incorrect structure",
    ) != identities.chapter_id(
        standard_checksum=common["standard_source_checksum"],
        chapter_number="4",
        chapter_title="Correct structure",
    )
    assert common["article_number"] == "4.0.1"
    assert common["accepted_source_line_ids"] == ["ocrline_grounded"]


def test_unchanged_grounded_identity_inputs_remain_stable() -> None:
    identities = StableIdentityService()
    values = {
        "standard_source_checksum": "b" * 64,
        "standard_code": "STD-2026",
        "chapter_number": "3.10",
        "article_number": "3.10.4",
        "article_text": "Unchanged grounded content.",
        "source_page_start": 8,
        "source_page_end": 9,
        "region_type": "NORMATIVE_BODY",
        "article_type": "NORMATIVE",
        "source_kind": "OCR_TEXT",
        "accepted_source_line_ids": ["ocrline_a", "ocrline_b"],
    }

    assert identities.article_id(**values) == identities.article_id(**values)
