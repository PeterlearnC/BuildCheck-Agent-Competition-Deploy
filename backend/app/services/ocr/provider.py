"""Core-side OCR provider abstraction and isolated worker adapter."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable

from app.schemas.ocr import (
    OCRCharacterResult,
    OCRLineResult,
    OCRPageResult,
    OCRProviderIdentity,
)
from app.services.ocr.page_renderer import RenderedOCRPage
from app.services.ocr.raster_support import OCRRasterSupportAnalyzer
from app.services.ocr.run_identity import stable_line_id


class OCRProviderError(RuntimeError):
    pass


class OCRProvider(ABC):
    @property
    @abstractmethod
    def identity(self) -> OCRProviderIdentity:
        raise NotImplementedError

    @abstractmethod
    def recognize_page(self, page: RenderedOCRPage) -> OCRPageResult:
        raise NotImplementedError


@dataclass(frozen=True)
class RapidOCRWorkerConfig:
    python_executable: Path
    worker_script: Path
    detector_model_path: Path
    recognizer_model_path: Path
    classifier_model_path: Path
    identity: OCRProviderIdentity
    timeout_seconds: float = 120.0
    intra_op_threads: int = 6
    inter_op_threads: int = 1
    detector_long_side_limit: int = 2048


class RapidOCRSubprocessProvider(OCRProvider):
    """Invoke RapidOCR without importing OCR dependencies in the core process."""

    def __init__(
        self,
        config: RapidOCRWorkerConfig,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        raster_support: OCRRasterSupportAnalyzer | None = None,
    ) -> None:
        self.config = config
        self._runner = runner
        self._raster_support = raster_support or OCRRasterSupportAnalyzer()

    @property
    def identity(self) -> OCRProviderIdentity:
        return self.config.identity

    def recognize_page(self, page: RenderedOCRPage) -> OCRPageResult:
        self._verify_model_artifacts()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temporary:
            image_path = Path(temporary.name)
            temporary.write(page.png_bytes)
        try:
            request = {
                "image_path": str(image_path),
                "page_number": page.metadata.page_number,
                "expected_identity": self.identity.model_dump(mode="json"),
                "detector_model_path": str(self.config.detector_model_path),
                "recognizer_model_path": str(self.config.recognizer_model_path),
                "classifier_model_path": str(self.config.classifier_model_path),
                "intra_op_threads": self.config.intra_op_threads,
                "inter_op_threads": self.config.inter_op_threads,
                "detector_long_side_limit": self.config.detector_long_side_limit,
            }
            completed = self._runner(
                [str(self.config.python_executable), str(self.config.worker_script)],
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="strict",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                timeout=self.config.timeout_seconds,
                check=False,
                shell=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise OCRProviderError("The isolated OCR worker could not be executed.") from exc
        finally:
            image_path.unlink(missing_ok=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-1000:]
            raise OCRProviderError(f"The isolated OCR worker failed closed: {detail}")
        try:
            payload = json.loads(completed.stdout)
        except (TypeError, ValueError) as exc:
            raise OCRProviderError("The isolated OCR worker returned invalid JSON.") from exc
        if payload.get("provider_identity") != self.identity.model_dump(mode="json"):
            raise OCRProviderError("The isolated OCR worker identity did not match the request.")
        lines = []
        for index, item in enumerate(payload.get("lines", [])):
            polygon = item["polygon"]
            lines.append(
                OCRLineResult(
                    line_id=stable_line_id(
                        page_number=page.metadata.page_number,
                        reading_order_index=index,
                        raw_text=item["text"],
                        polygon=polygon,
                    ),
                    page_number=page.metadata.page_number,
                    raw_text=item["text"],
                    polygon=polygon,
                    confidence=item["confidence"],
                    reading_order_index=index,
                    characters=[OCRCharacterResult.model_validate(value) for value in item.get("characters", [])],
                )
            )
        try:
            lines = self._raster_support.annotate(page.png_bytes, lines)
        except (RuntimeError, ValueError) as exc:
            raise OCRProviderError(
                "The rendered OCR page could not produce deterministic raster support."
            ) from exc
        return OCRPageResult(
            page_number=page.metadata.page_number,
            lines=lines,
            render=page.metadata,
            provider=self.identity,
            elapsed_seconds=payload.get("elapsed_seconds"),
            warnings=payload.get("warnings", []),
            error=payload.get("error"),
        )

    def _verify_model_artifacts(self) -> None:
        expected = (
            (self.config.detector_model_path, self.identity.detector),
            (self.config.recognizer_model_path, self.identity.recognizer),
            (self.config.classifier_model_path, self.identity.classifier),
        )
        for path, artifact in expected:
            if artifact is None or not path.is_file():
                raise OCRProviderError("A pinned OCR model artifact is missing.")
            if artifact.model_id != path.name:
                raise OCRProviderError("A pinned OCR model identity does not match its artifact.")
            digest_state = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest_state.update(chunk)
            digest = digest_state.hexdigest()
            if digest.lower() != artifact.sha256.lower():
                raise OCRProviderError("A pinned OCR model artifact hash does not match.")
