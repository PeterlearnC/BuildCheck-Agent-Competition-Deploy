"""Structured, traceable output models for document analysis."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _first_value(item: dict[str, Any], names: tuple[str, ...]) -> Any:
    return next((item[name] for name in names if item.get(name) is not None), None)


def _page_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        return int(value) if value > 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else None
    return None


class SourceValue(BaseModel):
    value: str | int | None = None
    source_page: int | None = None
    source_text: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, value):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        if payload.get("value") is None:
            payload["value"] = _first_value(payload, ("name", "title", "text"))
        if not isinstance(payload.get("value"), (str, int)) or isinstance(
            payload.get("value"), bool
        ):
            payload["value"] = None
        if payload.get("source_page") is None:
            payload["source_page"] = _first_value(
                payload, ("page", "page_number")
            )
        payload["source_page"] = _page_number(payload.get("source_page"))
        if payload.get("source_text") is None:
            payload["source_text"] = _first_value(payload, ("quote", "original_text"))
        if payload.get("source_text") is not None and not isinstance(
            payload["source_text"], str
        ):
            payload["source_text"] = None
        return payload


class ProjectInfo(BaseModel):
    project_name: SourceValue | None = None
    project_location: SourceValue | None = None
    construction_unit: SourceValue | None = None
    design_unit: SourceValue | None = None
    supervision_unit: SourceValue | None = None
    general_contractor: SourceValue | None = None
    subcontractor: SourceValue | None = None
    construction_company: SourceValue | None = None

    @field_validator("*", mode="before")
    @classmethod
    def wrap_plain_values(cls, value):
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return {"value": value}
        return value if isinstance(value, dict) or value is None else None


class ScheduleInfo(BaseModel):
    plan_start_date: SourceValue | None = None
    plan_completion_date: SourceValue | None = None
    duration_days: SourceValue | None = None

    @field_validator("*", mode="before")
    @classmethod
    def wrap_plain_values(cls, value):
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return {"value": value}
        return value if isinstance(value, dict) or value is None else None


class ChapterInfo(BaseModel):
    title: str
    start_page: int
    end_page: int | None = None
    level: int | None = None


class StandardReference(BaseModel):
    name: str | None = None
    code: str
    source_page: int | None = None
    source_text: str | None = None


class MaterialInfo(BaseModel):
    name: str
    specification: str | None = None
    usage: str | None = None
    source_page: int | None = None
    source_text: str | None = None


class MethodInfo(BaseModel):
    name: str
    source_page: int | None = None
    source_text: str | None = None


class SectionPresence(BaseModel):
    quality: bool = False
    safety: bool = False
    environment: bool = False
    emergency: bool = False


class DocumentAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    document_type: SourceValue | None = None
    engineering_category: SourceValue | None = None
    specialty: SourceValue | None = None
    project: ProjectInfo = Field(default_factory=ProjectInfo)
    schedule: ScheduleInfo = Field(default_factory=ScheduleInfo)
    chapters: list[ChapterInfo] = Field(default_factory=list)
    compilation_basis: list[SourceValue | str] = Field(default_factory=list)
    standards: list[StandardReference] = Field(default_factory=list)
    main_work_items: list[SourceValue | str] = Field(default_factory=list)
    main_materials: list[MaterialInfo] = Field(default_factory=list)
    main_methods: list[MethodInfo] = Field(default_factory=list)
    section_presence: SectionPresence = Field(default_factory=SectionPresence)

    @model_validator(mode="before")
    @classmethod
    def apply_safe_container_defaults(cls, value):
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        for key in ("project", "schedule", "section_presence"):
            if not isinstance(payload.get(key), dict):
                payload[key] = {}
        for key in (
            "chapters", "compilation_basis", "standards", "main_work_items",
            "main_materials", "main_methods",
        ):
            if payload.get(key) is None:
                payload[key] = []
        if isinstance(payload.get("document_type"), str):
            payload["document_type"] = {"value": payload["document_type"]}
        elif payload.get("document_type") is not None and not isinstance(
            payload["document_type"], dict
        ):
            payload["document_type"] = None
        for key in ("engineering_category", "specialty"):
            if isinstance(payload.get(key), str):
                payload[key] = {"value": payload[key]}
            elif payload.get(key) is not None and not isinstance(payload[key], dict):
                payload[key] = None
        for key in ("quality", "safety", "environment", "emergency"):
            if key in payload["section_presence"] and not isinstance(
                payload["section_presence"][key], (bool, str, int)
            ):
                payload["section_presence"][key] = False
        payload["chapters"] = cls._normalize_chapters(payload["chapters"])
        payload["standards"] = cls._normalize_named_items(
            payload["standards"], ("code", "standard_code", "code_number"), "code"
        )
        payload["main_materials"] = cls._normalize_named_items(
            payload["main_materials"], ("name", "material", "material_name"), "name"
        )
        payload["main_methods"] = cls._normalize_named_items(
            payload["main_methods"],
            ("name", "method", "method_name", "title", "method_title", "value"),
            "name",
            accept_strings=True,
        )
        payload["compilation_basis"] = cls._normalize_source_lists(
            payload["compilation_basis"]
        )
        payload["main_work_items"] = cls._normalize_source_lists(
            payload["main_work_items"]
        )
        return payload

    @staticmethod
    def _normalize_chapters(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        chapters: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            title = _first_value(item, ("title", "name", "chapter", "heading"))
            start_page = _page_number(
                _first_value(item, ("start_page", "page", "page_number", "source_page"))
            )
            if not isinstance(title, str) or not title.strip() or start_page is None:
                continue
            end_page = _page_number(_first_value(item, ("end_page", "last_page")))
            if end_page is not None and end_page < start_page:
                end_page = None
            level = _page_number(item.get("level"))
            chapters.append(
                {
                    "title": title.strip(),
                    "start_page": start_page,
                    "end_page": end_page,
                    "level": level,
                }
            )
        return chapters

    @staticmethod
    def _normalize_named_items(
        value: Any,
        aliases: tuple[str, ...],
        canonical_name: str,
        accept_strings: bool = False,
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        result: list[dict[str, Any]] = []
        for item in value:
            if accept_strings and isinstance(item, str) and item.strip():
                result.append({canonical_name: item.strip()})
                continue
            if not isinstance(item, dict):
                continue
            name = _first_value(item, aliases)
            if not isinstance(name, str) or not name.strip():
                continue
            normalized = dict(item)
            normalized[canonical_name] = name.strip()
            normalized["source_page"] = _page_number(
                _first_value(normalized, ("source_page", "page", "page_number"))
            )
            if normalized.get("source_text") is not None and not isinstance(
                normalized["source_text"], str
            ):
                normalized["source_text"] = None
            if canonical_name == "name":
                if "specification" not in normalized:
                    normalized["specification"] = _first_value(
                        normalized, ("spec", "model")
                    )
                if "usage" not in normalized:
                    normalized["usage"] = _first_value(
                        normalized, ("use", "purpose")
                    )
                for optional_name in ("specification", "usage"):
                    if normalized.get(optional_name) is not None and not isinstance(
                        normalized[optional_name], str
                    ):
                        normalized[optional_name] = None
            elif normalized.get("name") is not None and not isinstance(
                normalized["name"], str
            ):
                normalized["name"] = None
            result.append(normalized)
        return result

    @staticmethod
    def _normalize_source_lists(value: Any) -> list[Any]:
        if not isinstance(value, list):
            return []
        result: list[Any] = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                normalized = dict(item)
                if normalized.get("value") is None:
                    normalized["value"] = _first_value(
                        normalized, ("name", "title", "item", "text")
                    )
                if isinstance(normalized.get("value"), (str, int)) and not isinstance(
                    normalized["value"], bool
                ):
                    result.append(normalized)
        return result


class DocumentAnalysisResponse(BaseModel):
    success: bool = True
    document_id: str
    cached: bool = False
    analysis: DocumentAnalysis
