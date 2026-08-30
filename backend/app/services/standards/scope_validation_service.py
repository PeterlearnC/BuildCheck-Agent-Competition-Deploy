"""Validate an explicit standards scope without selecting applicable standards."""

from dataclasses import dataclass

from app.schemas.standards import StandardRegistryStatus
from app.schemas.standards_scope import StandardScope
from app.services.standards.standard_registry_service import StandardRegistryService
from app.services.standards.standard_repository import StandardRepository


class ScopeValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ScopeValidationResult:
    scope: StandardScope
    warnings: tuple[str, ...] = ()


class ScopeValidationService:
    def __init__(
        self,
        repository: StandardRepository | None = None,
        registry: StandardRegistryService | None = None,
    ) -> None:
        self.repository = repository or StandardRepository()
        self.registry = registry or StandardRegistryService(self.repository.settings)

    def validate_scope(self, scope: StandardScope) -> ScopeValidationResult:
        if not scope.standard_ids:
            raise ScopeValidationError("Compliance standard scope must not be empty.")

        documents = {
            document.standard_id: document for document in self.repository.list_documents()
        }
        missing = sorted(set(scope.standard_ids).difference(documents))
        if missing:
            raise ScopeValidationError("Unknown standard_id values: " + ", ".join(missing))

        registry_by_code = {
            entry.standard_code.casefold(): entry for entry in self.registry.list_entries()
        }
        warnings: list[str] = []
        superseded: list[str] = []
        for standard_id in scope.standard_ids:
            document = documents[standard_id]
            entry = registry_by_code.get((document.standard_code or "").casefold())
            if entry is not None and entry.status == StandardRegistryStatus.SUPERSEDED:
                superseded.append(standard_id)
            elif entry is None or entry.status == StandardRegistryStatus.UNKNOWN:
                warnings.append(f"Registry status is unknown for standard_id: {standard_id}")
        if superseded:
            raise ScopeValidationError(
                "Superseded standards are not allowed in compliance scope: "
                + ", ".join(sorted(superseded))
            )
        return ScopeValidationResult(scope=scope, warnings=tuple(warnings))
