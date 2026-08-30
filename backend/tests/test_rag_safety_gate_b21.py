import pytest

from app.core.config import Settings
from app.schemas.standards import StandardRegistryEntry
from app.schemas.standards_retrieval import StandardSearchRequest
from app.schemas.standards_scope import StandardScope
from app.services.retrieval.retrieval_manifest_service import RetrievalManifestService
from app.services.retrieval.standards_search_service import StandardsSearchService
from app.services.retrieval.vector_store import LocalMemoryVectorStore
from app.services.standards.scope_validation_service import (
    ScopeValidationError,
    ScopeValidationService,
)
from app.services.standards.standard_registry_service import StandardRegistryService
from app.services.standards.standard_repository import StandardRepository
from tests.retrieval_helpers import make_record


def _manifest(model_id: str):
    return RetrievalManifestService().build(
        [make_record("std", "1.1.1", "正文")],
        embedding_provider="local",
        embedding_model_id=model_id,
        embedding_dimension=1024,
    )


def _repository(tmp_path):
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    record = make_record(
        "std-a", "1.1.1", "正文", standard_code="JGJ 999-2024", standard_name="测试标准"
    )
    repository.save_document(record.document)
    repository.save_articles("std-a", [record.article])
    return repository


def test_same_dimension_different_model_ids_have_different_manifest_fingerprints() -> None:
    assert _manifest("Qwen/Qwen3-Embedding-0.6B").fingerprint != _manifest("BAAI/bge-m3").fingerprint


def test_model_id_change_invalidates_vector_cache() -> None:
    service = RetrievalManifestService()
    assert not service.compatible(
        _manifest("Qwen/Qwen3-Embedding-0.6B"),
        _manifest("BAAI/bge-m3"),
    )


def test_blank_embedding_model_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="embedding_model_id"):
        _manifest(" ")


def test_empty_compliance_scope_is_rejected(tmp_path) -> None:
    with pytest.raises(ScopeValidationError, match="must not be empty"):
        ScopeValidationService(_repository(tmp_path)).validate_scope(StandardScope(standard_ids=[]))


def test_unknown_standard_in_scope_is_rejected(tmp_path) -> None:
    with pytest.raises(ScopeValidationError, match="Unknown standard_id"):
        ScopeValidationService(_repository(tmp_path)).validate_scope(
            StandardScope(standard_ids=["missing"])
        )


def test_superseded_standard_in_scope_is_rejected(tmp_path) -> None:
    repository = _repository(tmp_path)
    registry = StandardRegistryService(
        entries=[
            StandardRegistryEntry(
                standard_code="JGJ 999-2024",
                standard_name="测试标准",
                status="SUPERSEDED",
            )
        ]
    )
    with pytest.raises(ScopeValidationError, match="Superseded"):
        ScopeValidationService(repository, registry).validate_scope(
            StandardScope(standard_ids=["std-a"])
        )


def test_unscoped_search_remains_compatible_and_warns(tmp_path) -> None:
    response = StandardsSearchService(_repository(tmp_path)).search(
        StandardSearchRequest(query="正文")
    )
    assert response.hits
    assert response.scope_warning == "Search without explicit standard scope may include multiple standards"


def test_explicit_search_scope_has_no_warning(tmp_path) -> None:
    response = StandardsSearchService(_repository(tmp_path)).search(
        StandardSearchRequest(query="正文", standard_ids=["std-a"])
    )
    assert response.hits
    assert response.scope_warning is None


def test_local_memory_vector_store_persist_contract_exists() -> None:
    with pytest.raises(NotImplementedError, match="does not support persistence"):
        LocalMemoryVectorStore().persist()


def test_local_memory_vector_store_load_contract_exists() -> None:
    with pytest.raises(NotImplementedError, match="does not support persistence"):
        LocalMemoryVectorStore().load()
