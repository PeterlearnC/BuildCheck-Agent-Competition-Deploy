import pytest

from app.schemas.standards import StandardIdentityStatus, StandardPage
from app.services.standards.standard_identity_service import StandardIdentityService


def _identify(text: str):
    return StandardIdentityService().identify([StandardPage(page_number=1, text=text)])


@pytest.mark.parametrize(
    ("source", "canonical", "display"),
    [
        ("GB 55023-2022施工脚手架通用规范", "GB55023-2022", "GB 55023-2022"),
        ("GB55023-2022施工脚手架通用规范", "GB55023-2022", "GB 55023-2022"),
        ("GB55023—2022施工脚手架通用规范", "GB55023-2022", "GB 55023-2022"),
        ("GB55023–2022施工脚手架通用规范", "GB55023-2022", "GB 55023-2022"),
        ("GB55023－2022施工脚手架通用规范", "GB55023-2022", "GB 55023-2022"),
        ("GB55023一2022施工脚手架通用规范", "GB55023-2022", "GB 55023-2022"),
        ("GB/T 50326-2017建设工程管理规范", "GB/T50326-2017", "GB/T 50326-2017"),
        ("JGJ 130-2011建筑施工安全技术规范", "JGJ130-2011", "JGJ 130-2011"),
        ("JGJ/T194-2009施工安全技术规程", "JGJ/T194-2009", "JGJ/T 194-2009"),
        ("CJJ 1-2008城镇道路工程标准", "CJJ1-2008", "CJJ 1-2008"),
        ("CJJ/T 135-2009工程技术规程", "CJJ/T135-2009", "CJJ/T 135-2009"),
    ],
)
def test_code_variants_are_canonicalized(source, canonical, display) -> None:
    result = _identify(source)
    assert result.canonical_code == canonical
    assert result.display_code == display


@pytest.mark.parametrize(
    "noise",
    ["2022-05-01", "4.4.6", "5.2kN", "1.2×1.5", "第17页", "发布日期2022-05-01"],
)
def test_non_code_values_are_rejected(noise: str) -> None:
    assert _identify(noise).canonical_code is None


def test_inline_code_is_removed_from_name() -> None:
    result = _identify("GB55023一2022施工脚手架通用规范")
    assert result.standard_name == "施工脚手架通用规范"
    assert result.status == StandardIdentityStatus.CONFIRMED


def test_banners_are_not_names() -> None:
    result = _identify("中华人民共和国国家标准\n住房和城乡建设部\nGB 55023-2022")
    assert result.standard_name is None
    assert result.status == StandardIdentityStatus.IDENTITY_UNCERTAIN


def test_canonicalize_rejects_partial_candidate() -> None:
    assert StandardIdentityService().canonicalize("55023-2022") is None
