"""Content identities for OCR configuration and raw executions."""

import hashlib
import json

from app.schemas.ocr import (
    OCRPageResult,
    OCRProviderIdentity,
    OCRRenderConfig,
    OCRRenderMetadata,
    OCRRun,
)


OCR_ADAPTER_VERSION = "v0.4-b.3b.1b-adapter"
OCR_QUALITY_GATE_VERSION = "v0.4-b.3b-q1-r1-quality"
OCR_CORPUS_SEMANTICS_VERSION = "v0.4-b.3b-q1-r1-ocr-corpus"


def _digest(prefix: str, payload: object) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"{prefix}_{hashlib.sha256(serialized.encode('utf-8')).hexdigest()}"


def semantic_ocr_run_id(
    *,
    official_source_checksum: str,
    provider: OCRProviderIdentity,
    render_config: OCRRenderConfig,
    adapter_version: str = OCR_ADAPTER_VERSION,
    quality_gate_version: str = OCR_QUALITY_GATE_VERSION,
    corpus_semantics_version: str = OCR_CORPUS_SEMANTICS_VERSION,
    render_metadata: list[OCRRenderMetadata] | None = None,
) -> str:
    return _digest(
        "ocr",
        {
            "official_source_checksum": official_source_checksum.lower(),
            "provider": provider.model_dump(mode="json"),
            "render_config": render_config.model_dump(mode="json"),
            "adapter_version": adapter_version,
            "quality_gate_version": quality_gate_version,
            "corpus_semantics_version": corpus_semantics_version,
            "rendered_pages": [
                item.model_dump(mode="json")
                for item in sorted(
                    render_metadata or [], key=lambda value: value.page_number
                )
            ],
        },
    )


def execution_id(pages: list[OCRPageResult]) -> str:
    payload = []
    for page in sorted(pages, key=lambda item: item.page_number):
        payload.append(
            {
                "page_number": page.page_number,
                "image_sha256": page.render.image_sha256,
                "error": page.error,
                "lines": [
                    {
                        "line_id": line.line_id,
                        "raw_text": line.raw_text,
                        "polygon": line.polygon,
                        "confidence": line.confidence,
                        "reading_order_index": line.reading_order_index,
                        "characters": [
                            character.model_dump(mode="json")
                            for character in line.characters
                        ],
                    }
                    for line in page.lines
                ],
            }
        )
    return _digest("execution", payload)


def stable_line_id(
    *, page_number: int, reading_order_index: int, raw_text: str, polygon: list[list[float]]
) -> str:
    return _digest(
        "ocrline",
        [page_number, reading_order_index, raw_text, polygon],
    )


def validate_ocr_run_identity(run: OCRRun) -> None:
    expected_run_id = semantic_ocr_run_id(
        official_source_checksum=run.official_source_checksum,
        provider=run.provider,
        render_config=run.render_config,
        adapter_version=run.adapter_version,
        quality_gate_version=run.quality_gate_version,
        corpus_semantics_version=run.corpus_semantics_version,
        render_metadata=[page.render for page in run.pages],
    )
    if run.ocr_run_id != expected_run_id:
        raise ValueError("OCR semantic run identity does not match its configuration.")
    if run.execution_id != execution_id(run.pages):
        raise ValueError("OCR execution identity does not match its raw page results.")
