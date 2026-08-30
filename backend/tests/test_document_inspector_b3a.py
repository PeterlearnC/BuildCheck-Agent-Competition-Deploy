import pytest
from datetime import datetime, timezone

from app.core.config import Settings
from app.schemas.standards import (
    DocumentCapabilityStatus,
    StandardDocument,
    StandardPage,
    StandardParseStatus,
)
from app.services.standards.document_inspector_service import DocumentInspectorService
from app.services.standards.standard_document_service import (
    StandardDocumentService,
    StandardParseFailure,
)
from app.services.standards.standard_repository import StandardRepository


def _inspect(pages):
    return DocumentInspectorService().inspect(pages)


def test_image_watermark_document_requires_ocr() -> None:
    pages = [
        StandardPage(page_number=i, text=f"浏览专用\n{i}", image_count=1)
        for i in range(1, 18)
    ]
    result = _inspect(pages)
    assert result.status == DocumentCapabilityStatus.OCR_REQUIRED
    assert result.effective_text_char_count == 0
    assert result.image_page_ratio == 1.0


def test_empty_page_list_is_empty_document() -> None:
    assert _inspect([]).status == DocumentCapabilityStatus.EMPTY_DOCUMENT


def test_blank_pages_are_empty_document() -> None:
    pages = [StandardPage(page_number=1, text=" \n ")]
    assert _inspect(pages).status == DocumentCapabilityStatus.EMPTY_DOCUMENT


def test_image_only_pages_require_ocr() -> None:
    pages = [StandardPage(page_number=1, text="", image_count=1)]
    assert _inspect(pages).status == DocumentCapabilityStatus.OCR_REQUIRED


def test_parse_boundary_persists_ocr_required(tmp_path) -> None:
    repository = StandardRepository(Settings(standards_dir=tmp_path / "standards"))
    now = datetime.now(timezone.utc)
    document = StandardDocument(
        standard_id="11111111-1111-1111-1111-111111111111",
        source_filename="scan.pdf",
        source_checksum="a" * 64,
        page_count=17,
        created_at=now,
        updated_at=now,
    )
    pages = [
        StandardPage(page_number=i, text=f"浏览专用\n{i}", image_count=1)
        for i in range(1, 18)
    ]
    repository.save_document(document)
    repository.save_pages(document.standard_id, pages)
    with pytest.raises(StandardParseFailure, match="OCR_REQUIRED"):
        StandardDocumentService(repository=repository).parse(document.standard_id)
    stored = repository.load_document(document.standard_id)
    assert stored.parse_status == StandardParseStatus.OCR_REQUIRED
    assert stored.parse_error == "OCR_REQUIRED"


def test_sparse_non_image_text_is_insufficient() -> None:
    result = _inspect([StandardPage(page_number=1, text="封面")])
    assert result.status == DocumentCapabilityStatus.TEXT_INSUFFICIENT


def test_structured_text_is_ready() -> None:
    result = _inspect([StandardPage(page_number=1, text="1 总则\n1.0.1 构件应可靠连接。")])
    assert result.status == DocumentCapabilityStatus.TEXT_READY


def test_repeated_watermark_is_not_effective_text() -> None:
    pages = [
        StandardPage(page_number=i, text=f"浏览专用\n{i}", image_count=1)
        for i in range(1, 5)
    ]
    result = _inspect(pages)
    assert result.repeated_text_ratio > 0.5
    assert result.pages_with_meaningful_text == 0


def test_isolated_page_numbers_are_excluded() -> None:
    result = _inspect([StandardPage(page_number=1, text="17")])
    assert result.effective_text_char_count == 0


def test_mixed_sparse_image_document_is_insufficient() -> None:
    pages = [
        StandardPage(page_number=1, text="1 总则\n1.0.1 有效正文。"),
        StandardPage(page_number=2, text="2", image_count=1),
        StandardPage(page_number=3, text="3", image_count=1),
        StandardPage(page_number=4, text="4"),
    ]
    assert _inspect(pages).status == DocumentCapabilityStatus.TEXT_INSUFFICIENT


def test_text_rich_pages_with_images_can_be_ready() -> None:
    pages = [
        StandardPage(
            page_number=i,
            text=f"{i}.0.1 " + "构件应进行可靠连接和安全检查。" * 8,
            image_count=1,
        )
        for i in range(1, 4)
    ]
    assert _inspect(pages).status == DocumentCapabilityStatus.TEXT_READY


def test_capability_counts_are_deterministic() -> None:
    pages = [StandardPage(page_number=1, text="1.0.1 构件应可靠连接。")]
    assert _inspect(pages) == _inspect(pages)


@pytest.mark.parametrize("image_count", [-1, -10])
def test_negative_image_count_is_rejected(image_count: int) -> None:
    with pytest.raises(ValueError):
        StandardPage(page_number=1, text="正文", image_count=image_count)
