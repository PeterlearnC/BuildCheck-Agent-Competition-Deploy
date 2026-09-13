"""Strict loading and compatibility checks for OCR qualification authority."""

import hashlib
import json
from pathlib import Path

from app.schemas.ocr_qualification import (
    OCRQualificationManifest,
    OCRQualificationStatus,
)
from app.services.ocr.run_identity import (
    OCR_CORPUS_SEMANTICS_VERSION,
    OCR_QUALITY_GATE_VERSION,
)


class OCRQualificationManifestError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(payload: object) -> str:
    data = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(data)


def load_qualification_manifest(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_source_checksum: str,
    expected_parse_result_sha256: str,
) -> OCRQualificationManifest:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise OCRQualificationManifestError(
            "OCR qualification manifest is missing or unreadable."
        ) from exc
    if sha256_bytes(data) != expected_file_sha256:
        raise OCRQualificationManifestError(
            "OCR qualification manifest byte identity mismatch."
        )
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_strict_object)
        manifest = OCRQualificationManifest.model_validate(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise OCRQualificationManifestError(
            "OCR qualification manifest is invalid."
        ) from exc
    checks = (
        manifest.official_source_checksum == expected_source_checksum,
        manifest.qualified_parse_result_artifact_sha256
        == expected_parse_result_sha256,
        manifest.quality_gate_version == OCR_QUALITY_GATE_VERSION,
        manifest.ocr_corpus_semantics_version == OCR_CORPUS_SEMANTICS_VERSION,
        manifest.qualification_status == OCRQualificationStatus.QUALIFIED,
    )
    if not all(checks):
        raise OCRQualificationManifestError(
            "OCR corpus is stale or its qualification authority is incompatible."
        )
    return manifest


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate OCR qualification key: {key}")
        result[key] = value
    return result
