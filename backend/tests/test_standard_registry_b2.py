import json

from app.core.config import Settings
from app.schemas.standards import StandardRegistryEntry, StandardRegistryStatus
from app.services.standards.standard_registry_service import StandardRegistryService


def _entries():
    return [
        StandardRegistryEntry(
            standard_code="JGJ 1-2024", standard_name="现行标准", discipline="结构", status="ACTIVE", jurisdiction="CN"
        ),
        StandardRegistryEntry(
            standard_code="JGJ 1-2010", standard_name="旧标准", discipline="结构", status="SUPERSEDED", jurisdiction="CN"
        ),
    ]


def test_active_filter_returns_only_active_entries() -> None:
    result = StandardRegistryService(entries=_entries()).filter(status=StandardRegistryStatus.ACTIVE)
    assert [entry.standard_code for entry in result] == ["JGJ 1-2024"]


def test_superseded_filter_returns_only_superseded_entries() -> None:
    result = StandardRegistryService(entries=_entries()).filter(status=StandardRegistryStatus.SUPERSEDED)
    assert [entry.standard_code for entry in result] == ["JGJ 1-2010"]


def test_empty_registry_returns_empty_list(tmp_path) -> None:
    service = StandardRegistryService(Settings(standards_dir=tmp_path / "standards"))
    assert service.list_entries() == []


def test_registry_filters_discipline_and_jurisdiction() -> None:
    service = StandardRegistryService(entries=_entries())
    assert len(service.filter(discipline="结构", jurisdiction="cn")) == 2
    assert service.filter(discipline="给排水") == []


def test_registry_loads_offline_json(tmp_path) -> None:
    settings = Settings(standards_dir=tmp_path / "standards")
    settings.standards_dir.mkdir(parents=True)
    (settings.standards_dir / "registry.json").write_text(
        json.dumps([_entries()[0].model_dump(mode="json")], ensure_ascii=False), encoding="utf-8"
    )
    assert StandardRegistryService(settings).active_codes() == {"jgj 1-2024"}
