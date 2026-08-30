"""Immutable, run-scoped persistence for raw OCR and derived assessments."""

import hashlib
import json
from pathlib import Path

from app.schemas.ocr import (
    OCRAcceptedPage,
    OCRRun,
    OCRRunQualityAssessment,
)
from app.services.standards.standard_repository import StandardRepository
from app.services.ocr.run_identity import validate_ocr_run_identity


class OCRRepositoryError(RuntimeError):
    pass


class OCRRepository:
    def __init__(self, standards: StandardRepository) -> None:
        self.standards = standards

    def execution_dir(self, standard_id: str, run: OCRRun) -> Path:
        if (
            not standard_id
            or standard_id in {".", ".."}
            or Path(standard_id).name != standard_id
        ):
            raise OCRRepositoryError("Invalid standard identity for OCR storage.")
        directory = (
            self.standards.document_dir(standard_id)
            / "ocr"
            / "runs"
            / self.storage_key(run)
        )
        self._validate_existing_identity(directory, run)
        return directory

    @staticmethod
    def storage_key(run: OCRRun) -> str:
        payload = f"{run.ocr_run_id}\0{run.execution_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:32]

    def save_raw_run(self, standard_id: str, run: OCRRun) -> Path:
        validate_ocr_run_identity(run)
        directory = self.execution_dir(standard_id, run)
        self._write_immutable(directory / "run.json", run.model_dump(mode="json", exclude={"pages"}))
        self._write_immutable(
            directory / "raw_pages.json",
            [page.model_dump(mode="json") for page in run.pages],
        )
        return directory

    def save_quality(
        self, standard_id: str, run: OCRRun, assessment: OCRRunQualityAssessment
    ) -> None:
        self._write_immutable(
            self.execution_dir(standard_id, run) / "quality_assessment.json",
            assessment.model_dump(mode="json"),
        )

    def save_accepted_pages(
        self, standard_id: str, run: OCRRun, pages: list[OCRAcceptedPage]
    ) -> None:
        self._write_immutable(
            self.execution_dir(standard_id, run) / "accepted_parser_pages.json",
            [page.model_dump(mode="json") for page in pages],
        )

    @staticmethod
    def _validate_existing_identity(directory: Path, run: OCRRun) -> None:
        if not directory.exists():
            return
        run_path = directory / "run.json"
        if not run_path.is_file():
            raise OCRRepositoryError(
                "Existing OCR storage key has no verifiable run identity."
            )
        try:
            stored = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise OCRRepositoryError(
                "Existing OCR storage key has unreadable run identity."
            ) from exc
        if (
            stored.get("ocr_run_id") != run.ocr_run_id
            or stored.get("execution_id") != run.execution_id
        ):
            raise OCRRepositoryError(
                "OCR storage-key collision does not match full run identities."
            )

    @staticmethod
    def _write_immutable(path: Path, payload: object) -> None:
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        if path.exists():
            try:
                existing = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise OCRRepositoryError("Stored OCR artifact is unreadable.") from exc
            if existing != serialized:
                raise OCRRepositoryError("Immutable OCR artifact already exists with different content.")
            return
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise OCRRepositoryError("Failed to persist immutable OCR artifact.") from exc
