"""Minimal offline registry for explicit standards filtering."""

import json
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas.standards import StandardRegistryEntry, StandardRegistryStatus
from app.services.standards.standard_repository import StandardRepositoryError


class StandardRegistryService:
    def __init__(
        self,
        settings: Settings | None = None,
        entries: list[StandardRegistryEntry] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._entries = entries

    @property
    def path(self) -> Path:
        return self.settings.standards_dir / "registry.json"

    def list_entries(self) -> list[StandardRegistryEntry]:
        entries = self._entries if self._entries is not None else self._load()
        return sorted(
            entries,
            key=lambda item: (
                item.standard_code.casefold(),
                item.effective_date.isoformat() if item.effective_date else "",
                item.standard_name.casefold(),
            ),
        )

    def filter(
        self,
        *,
        status: StandardRegistryStatus | None = None,
        discipline: str | None = None,
        jurisdiction: str | None = None,
    ) -> list[StandardRegistryEntry]:
        entries = self.list_entries()
        if status is not None:
            entries = [entry for entry in entries if entry.status == status]
        if discipline is not None:
            entries = [
                entry
                for entry in entries
                if (entry.discipline or "").casefold() == discipline.casefold()
            ]
        if jurisdiction is not None:
            entries = [
                entry
                for entry in entries
                if (entry.jurisdiction or "").casefold() == jurisdiction.casefold()
            ]
        return entries

    def active_codes(
        self,
        *,
        discipline: str | None = None,
        jurisdiction: str | None = None,
    ) -> set[str]:
        return {
            entry.standard_code.casefold()
            for entry in self.filter(
                status=StandardRegistryStatus.ACTIVE,
                discipline=discipline,
                jurisdiction=jurisdiction,
            )
        }

    def _load(self) -> list[StandardRegistryEntry]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("Registry root must be a list.")
            return [StandardRegistryEntry.model_validate(item) for item in payload]
        except (OSError, ValueError, TypeError) as exc:
            raise StandardRepositoryError("Stored standards registry is unreadable.") from exc
