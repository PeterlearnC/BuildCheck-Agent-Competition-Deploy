"""Local-file repository abstraction for uploaded and parsed standards."""

import json
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas.standards import (
    StandardArticle,
    StandardChapter,
    StandardDocument,
    StandardPage,
)


class StandardRepositoryError(Exception):
    pass


class StandardRepository:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.standards_dir / "documents"

    def document_dir(self, standard_id: str) -> Path:
        return self.root / standard_id

    def source_path(self, standard_id: str) -> Path:
        return self.document_dir(standard_id) / "source.pdf"

    def save_source(self, standard_id: str, content: bytes) -> Path:
        path = self.source_path(standard_id)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(content)
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise StandardRepositoryError("Failed to save the standard PDF.") from exc
        return path

    def save_document(self, document: StandardDocument) -> None:
        self._write_model(self.document_dir(document.standard_id) / "metadata.json", document)

    def load_document(self, standard_id: str) -> StandardDocument | None:
        path = self.document_dir(standard_id) / "metadata.json"
        return self._read_model(path, StandardDocument)

    def list_documents(self) -> list[StandardDocument]:
        if not self.root.is_dir():
            return []
        result: list[StandardDocument] = []
        for path in sorted(self.root.glob("*/metadata.json")):
            document = self._read_model(path, StandardDocument)
            if document is not None:
                result.append(document)
        return sorted(result, key=lambda item: item.created_at, reverse=True)

    def save_pages(self, standard_id: str, pages: list[StandardPage]) -> None:
        self._write_list(self.document_dir(standard_id) / "pages.json", pages)

    def load_pages(self, standard_id: str) -> list[StandardPage]:
        return self._read_list(
            self.document_dir(standard_id) / "pages.json", StandardPage
        )

    def save_chapters(
        self, standard_id: str, chapters: list[StandardChapter]
    ) -> None:
        self._write_list(self.document_dir(standard_id) / "chapters.json", chapters)

    def load_chapters(self, standard_id: str) -> list[StandardChapter]:
        return self._read_list(
            self.document_dir(standard_id) / "chapters.json", StandardChapter
        )

    def save_articles(
        self, standard_id: str, articles: list[StandardArticle]
    ) -> None:
        self._write_list(self.document_dir(standard_id) / "articles.json", articles)

    def load_articles(self, standard_id: str) -> list[StandardArticle]:
        return self._read_list(
            self.document_dir(standard_id) / "articles.json", StandardArticle
        )

    def articles_have_fields(self, standard_id: str, required: set[str]) -> bool:
        path = self.document_dir(standard_id) / "articles.json"
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        return bool(payload) and all(
            isinstance(item, dict) and required.issubset(item) for item in payload
        )

    @staticmethod
    def _read_model(path: Path, model_type):
        if not path.is_file():
            return None
        try:
            return model_type.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StandardRepositoryError(
                f"Stored standards data is unreadable: {path.name}."
            ) from exc

    @classmethod
    def _read_list(cls, path: Path, model_type) -> list:
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [model_type.model_validate(item) for item in payload]
        except (OSError, ValueError, TypeError) as exc:
            raise StandardRepositoryError(
                f"Stored standards data is unreadable: {path.name}."
            ) from exc

    @classmethod
    def _write_model(cls, path: Path, model) -> None:
        cls._write_json(path, model.model_dump(mode="json"))

    @classmethod
    def _write_list(cls, path: Path, items: list) -> None:
        cls._write_json(path, [item.model_dump(mode="json") for item in items])

    @staticmethod
    def _write_json(path: Path, payload) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError as exc:
            raise StandardRepositoryError(
                f"Failed to save standards data: {path.name}."
            ) from exc
