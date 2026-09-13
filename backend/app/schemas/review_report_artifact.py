"""Deterministic in-memory artifact metadata for a frozen ReviewReportModel."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


REPORT_ARTIFACT_IDENTITY_VERSION = "v0.1-d.7-p3a-report-artifact"
JSON_REPORT_RENDERER_VERSION = "v0.1-d.7-p3a-json"
HTML_REPORT_RENDERER_VERSION = "v0.1-d.7-p3a-html"
JSON_REPORT_MEDIA_TYPE = "application/json; charset=utf-8"
HTML_REPORT_MEDIA_TYPE = "text/html; charset=utf-8"


class ReviewReportArtifactFormat(str, Enum):
    JSON = "JSON"
    HTML = "HTML"


def deterministic_report_artifact_id(
    *,
    report_model_id: str,
    artifact_format: ReviewReportArtifactFormat,
    renderer_version: str,
    content_sha256: str,
    media_type: str,
    identity_version: str = REPORT_ARTIFACT_IDENTITY_VERSION,
) -> str:
    """Return presentation identity without changing report semantic identity."""

    payload = {
        "identity_version": identity_version,
        "report_model_id": report_model_id,
        "format": artifact_format.value,
        "renderer_version": renderer_version,
        "content_sha256": content_sha256,
        "media_type": media_type,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "reportartifact_" + hashlib.sha256(canonical).hexdigest()


def _format_contract(
    artifact_format: ReviewReportArtifactFormat,
) -> tuple[str, str, str]:
    if artifact_format == ReviewReportArtifactFormat.JSON:
        return JSON_REPORT_RENDERER_VERSION, JSON_REPORT_MEDIA_TYPE, ".json"
    return HTML_REPORT_RENDERER_VERSION, HTML_REPORT_MEDIA_TYPE, ".html"


class ReviewReportArtifact(BaseModel):
    """Immutable in-memory bytes plus semantic-neutral artifact metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity_version: Literal[REPORT_ARTIFACT_IDENTITY_VERSION] = (
        REPORT_ARTIFACT_IDENTITY_VERSION
    )
    artifact_id: str = Field(pattern=r"^reportartifact_[0-9a-f]{64}$")
    report_model_id: str = Field(pattern=r"^reviewreport_[0-9a-f]{64}$")
    format: ReviewReportArtifactFormat
    renderer_version: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    filename: str = Field(pattern=r"^reviewreport_[0-9a-f]{64}\.(json|html)$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_length: int = Field(ge=0)
    content: bytes = Field(strict=True)

    @model_validator(mode="after")
    def validate_artifact(self):
        renderer_version, media_type, extension = _format_contract(self.format)
        if self.renderer_version != renderer_version:
            raise ValueError("Renderer version does not match artifact format.")
        if self.media_type != media_type:
            raise ValueError("Media type does not match artifact format.")
        if self.filename != f"{self.report_model_id}{extension}":
            raise ValueError("Artifact filename does not match report authority.")
        if self.byte_length != len(self.content):
            raise ValueError("Artifact byte length does not match content.")
        actual_sha256 = hashlib.sha256(self.content).hexdigest()
        if self.content_sha256 != actual_sha256:
            raise ValueError("Artifact content hash does not match content.")
        expected_id = deterministic_report_artifact_id(
            report_model_id=self.report_model_id,
            artifact_format=self.format,
            renderer_version=self.renderer_version,
            content_sha256=self.content_sha256,
            media_type=self.media_type,
            identity_version=self.identity_version,
        )
        if self.artifact_id != expected_id:
            raise ValueError("Artifact identity does not match byte authority.")
        return self
