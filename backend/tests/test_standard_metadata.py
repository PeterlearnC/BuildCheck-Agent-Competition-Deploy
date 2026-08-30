import pytest

from app.schemas.standards import MetadataConfidence, StandardPage
from app.services.standards.standard_metadata_service import StandardMetadataService


def _extract(*texts: str):
    return StandardMetadataService().extract(
        [StandardPage(page_number=index + 1, text=text) for index, text in enumerate(texts)]
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("标准编号 JGJ 130-2011\n建筑施工安全技术规范", "JGJ 130-2011"),
        ("标准编号 GB/T 50326-2017\n建设工程管理规范", "GB/T 50326-2017"),
        ("编号 JGJ/T 194-2009\n施工安全技术规程", "JGJ/T 194-2009"),
        ("编号 T/CECS 1000-2022\n工程应用技术规程", "T/CECS 1000-2022"),
    ],
)
def test_standard_code_common_formats(text: str, expected: str) -> None:
    assert _extract(text).standard_code == expected


@pytest.mark.parametrize("noise", ["2024-05-17", "6.3.4", "1200-1500", "5.0kN", "2008年"])
def test_standard_code_rejects_non_codes(noise: str) -> None:
    assert _extract(f"发布日期 {noise}\n普通说明文字").standard_code is None


def test_standard_name_joins_cover_title_lines() -> None:
    result = _extract(
        "中华人民共和国行业标准\n建筑施工扣件式\n钢管脚手架安全技术规范\n标准编号 JGJ 130-2011"
    )
    assert result.standard_name == "建筑施工扣件式钢管脚手架安全技术规范"
    assert result.confidence == MetadataConfidence.HIGH


def test_generic_banner_is_not_standard_name() -> None:
    result = _extract("中华人民共和国行业标准\n住房和城乡建设部发布")
    assert result.standard_name is None


def test_edition_uses_code_year_without_explicit_conflict() -> None:
    assert _extract("JGJ 130-2011\n建筑施工技术规范").edition == "2011"


def test_conflicting_explicit_edition_is_conservative() -> None:
    result = _extract("JGJ 130-2011\n建筑施工技术规范\n2020年版")
    assert result.edition is None
    assert result.confidence == MetadataConfidence.LOW
