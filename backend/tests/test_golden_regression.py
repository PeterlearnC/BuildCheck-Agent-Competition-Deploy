import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from app.schemas.analysis import DocumentAnalysis, ProjectInfo, ScheduleInfo
from app.services.document_analysis_service import DocumentAnalysisService
from app.services.metadata_extraction_service import MetadataExtractionService
from app.services.pdf_service import PDFExtractionResult, PDFPageText
from app.services.source_locator_service import SourceLocatorService
from app.services.text_preprocessing_service import TextPreprocessingService
from app.services.chapter_detection_service import ChapterDetectionService


TESTS_DIR = Path(__file__).parent
GOLDEN_DIR = TESTS_DIR / "golden"
FIXTURE_DIR = TESTS_DIR / "fixtures" / "golden"


class _NoNetworkLLM:
    def analyze_document(self, _pages):  # pragma: no cover - must never be called
        raise AssertionError("Golden regression must not call an external LLM")


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("—", "-").replace("–", "-").replace("－", "-")
    text = re.sub(r"\s+", "", text).casefold()
    return text.strip("《》“”‘’\"'，,。；;：:、")


def _code_identity(value: Any) -> str:
    return _normalized(value).upper()


def _source_value(item: Any) -> Any:
    return item.value if hasattr(item, "value") else item


def _assert_scalar(actual: Any, rule: dict[str, Any], path: str) -> None:
    if rule.get("is_null") is True:
        assert actual is None, f"{path}: expected null, got {actual!r}"
        return
    if rule.get("not_null") is True:
        assert actual is not None, f"{path}: expected non-null"
    if actual is None:
        assert not ({"equals", "contains", "source_page"} & rule.keys()), (
            f"{path}: value is null"
        )
        return

    value = _source_value(actual)
    if "equals" in rule:
        expected = rule["equals"]
        if isinstance(expected, (str, int)) and isinstance(value, (str, int)):
            assert _normalized(value) == _normalized(expected), (
                f"{path}: expected {expected!r}, got {value!r}"
            )
        else:
            assert value == expected, f"{path}: expected {expected!r}, got {value!r}"
    for expected_part in rule.get("contains", []):
        assert _normalized(expected_part) in _normalized(value), (
            f"{path}: {expected_part!r} not found in {value!r}"
        )
    if "source_page" in rule:
        assert getattr(actual, "source_page", None) == rule["source_page"], (
            f"{path}: expected source_page={rule['source_page']}, "
            f"got {getattr(actual, 'source_page', None)}"
        )


def _assert_list(
    actual: list[Any], rule: dict[str, Any], value_getter, path: str
) -> None:
    values = [value_getter(item) for item in actual]
    identities = {_normalized(value) for value in values}
    for expected in rule.get("must_include", []):
        assert _normalized(expected) in identities, (
            f"{path}: missing {expected!r}; actual={values!r}"
        )
    for forbidden in rule.get("must_not_include", []):
        assert _normalized(forbidden) not in identities, (
            f"{path}: forbidden item {forbidden!r} present"
        )


def assert_golden_analysis(analysis: DocumentAnalysis, golden: dict[str, Any]) -> None:
    for field in ("document_type", "engineering_category", "specialty"):
        if field in golden:
            _assert_scalar(getattr(analysis, field), golden[field], field)

    for field, rule in golden.get("project", {}).items():
        _assert_scalar(getattr(analysis.project, field), rule, f"project.{field}")
    for field, rule in golden.get("schedule", {}).items():
        _assert_scalar(getattr(analysis.schedule, field), rule, f"schedule.{field}")

    _assert_list(
        analysis.main_work_items,
        golden.get("main_work_items", {}),
        lambda item: _source_value(item),
        "main_work_items",
    )
    _assert_list(
        analysis.main_methods,
        golden.get("main_methods", {}),
        lambda item: item.name,
        "main_methods",
    )

    actual_codes = {_code_identity(item.code) for item in analysis.standards}
    for expected in golden.get("standards", {}).get("must_include_codes", []):
        assert _code_identity(expected) in actual_codes, (
            f"standards: missing code {expected!r}; actual={sorted(actual_codes)!r}"
        )

    if "section_presence" in golden:
        assert analysis.section_presence.model_dump() == golden["section_presence"]


def _run_deterministic_fixture(fixture: dict[str, Any]) -> DocumentAnalysis:
    raw_pages = [PDFPageText(**item) for item in fixture["pages"]]
    text = "\n".join(page.text for page in raw_pages)
    page_count = max(page.page_number for page in raw_pages)
    extracted = PDFExtractionResult(
        page_count=page_count, char_count=len(text), text=text, pages=raw_pages
    )

    preprocessing = TextPreprocessingService()
    detector = ChapterDetectionService()
    metadata = MetadataExtractionService()
    service = DocumentAnalysisService(_NoNetworkLLM())
    pages = preprocessing.preprocess(raw_pages)
    chapters = detector.detect(pages)
    local_project, local_schedule = metadata.extract(pages)

    analysis = DocumentAnalysis.model_validate(fixture.get("seed_analysis", {}))
    analysis.document_type = metadata.extract_document_type(
        raw_pages, local_project.project_name
    )
    analysis.chapters = service._merge_chapters(
        chapters, analysis.chapters, extracted.page_count
    )
    analysis.project = service._merge_metadata(local_project, analysis.project)
    analysis.project = metadata.validate_organizations(pages, analysis.project)
    analysis.schedule = service._merge_metadata(local_schedule, analysis.schedule)

    local_category = metadata.extract_engineering_category(pages, local_project)
    if local_category is not None:
        analysis.engineering_category = local_category
    else:
        analysis.engineering_category = metadata.validate_engineering_category(
            pages, analysis.project, analysis.engineering_category
        )

    analysis.standards = service._merge_standards(
        service._extract_standards(extracted), analysis.standards
    )
    analysis.compilation_basis = service._merge_compilation_basis(
        service._extract_compilation_basis(pages, chapters),
        analysis.compilation_basis,
    )
    llm_presence = analysis.section_presence
    located = SourceLocatorService().locate(analysis, raw_pages)
    deterministic_presence = detector.section_presence(located.chapters, raw_pages)
    located.section_presence = service._merge_section_presence(
        llm_presence, deterministic_presence
    )
    return located


GOLDEN_FILES = sorted(GOLDEN_DIR.glob("*.json"))


@pytest.mark.parametrize("golden_path", GOLDEN_FILES, ids=lambda path: path.stem)
def test_deterministic_golden_regression(golden_path: Path) -> None:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    fixture = json.loads(
        (FIXTURE_DIR / golden["fixture"]).read_text(encoding="utf-8")
    )
    assert_golden_analysis(_run_deterministic_fixture(fixture), golden)
