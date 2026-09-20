"""P3 competition adapter/UI qualification for frozen controlled Case D."""

from __future__ import annotations

import ast
import hashlib
from html.parser import HTMLParser
import inspect
import json
from pathlib import Path
import shutil
import subprocess

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from app.main import app
from app.schemas.competition_internal_consistency import (
    CompetitionConsistencyCandidateView,
    CompetitionConsistencySourceView,
    CompetitionConsistencyValueGroupView,
    CompetitionInternalConsistencyResponse,
)
import app.services.competition_internal_consistency_service as service_module
from app.services.review.internal_consistency_service import (
    InternalConsistencyReviewService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_D_ROOT = PROJECT_ROOT / "backend/tests/fixtures/internal_consistency/case_d"
CASE_D_ASSET = CASE_D_ROOT / "a38da41d-3bb1-5cbb-8402-8d76545dfca9.pdf"
CASE_D_METADATA = CASE_D_ROOT / "controlled-case-d.json"
STATIC_ROOT = PROJECT_ROOT / "backend/app/static/competition"
SERVICE_PATH = (
    PROJECT_ROOT / "backend/app/services/competition_internal_consistency_service.py"
)
CASE_D_ENDPOINT = "/api/v1/competition/demo/cases/CASE-D/run"
EXPECTED_ASSET_SHA256 = (
    "732c04c8d0ad282292ba0529af04bc0777c83e67d6a2c99cb606b7883d6764e9"
)
EXPECTED_CANDIDATE_ID = (
    "consistencycandidate_7887f2e54729711565854dd5eba57bfd1a2c34dffe23225e6e523e3d330b42b7"
)
EXPECTED_REVIEW_ID = (
    "consistencyreview_1b2d6e11c9e943972e6ae6a314ca23d656ad7276171970795f19bd5c028ac390"
)
CLIENT = TestClient(app)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | {
            nested
            for item in value.values()
            for nested in _all_keys(item)
        }
    if isinstance(value, list):
        return {nested for item in value for nested in _all_keys(item)}
    return set()


def _install_temp_case_d(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    asset_bytes: bytes | None,
    metadata_update: dict[str, object] | None = None,
) -> None:
    root = tmp_path / "case_d"
    root.mkdir()
    metadata = json.loads(CASE_D_METADATA.read_text(encoding="utf-8"))
    metadata.update(metadata_update or {})
    metadata_path = root / "controlled-case-d.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    asset_path = root / CASE_D_ASSET.name
    if asset_bytes is not None:
        asset_path.write_bytes(asset_bytes)
    monkeypatch.setattr(service_module, "CASE_D_ROOT", root.resolve())
    monkeypatch.setattr(service_module, "CASE_D_METADATA_PATH", metadata_path)
    monkeypatch.setattr(service_module, "CASE_D_ASSET_PATH", asset_path)


@pytest.fixture(scope="module")
def case_d_response() -> dict:
    response = CLIENT.post(CASE_D_ENDPOINT)
    assert response.status_code == 200
    return response.json()


class _IdentityParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        del tag
        for key, value in attrs:
            if key == "id" and value is not None:
                self.ids.append(value)


def test_response_models_are_frozen_extra_forbid_and_non_normative() -> None:
    models = (
        CompetitionConsistencySourceView,
        CompetitionConsistencyValueGroupView,
        CompetitionConsistencyCandidateView,
        CompetitionInternalConsistencyResponse,
    )
    forbidden = {
        "compliance",
        "compliant",
        "non_compliant",
        "finding",
        "severity",
        "risk",
        "unsafe",
        "safe",
        "violation",
        "defect",
        "overall_result",
        "whole_plan_verdict",
        "standard",
        "article",
        "normative_requirement",
        "score",
        "ranking",
    }
    for model in models:
        assert model.model_config["frozen"] is True
        assert model.model_config["extra"] == "forbid"
        assert forbidden.isdisjoint(model.model_fields)

    payload = CLIENT.post(CASE_D_ENDPOINT).json()
    with pytest.raises(ValidationError):
        CompetitionInternalConsistencyResponse.model_validate(
            {**payload, "severity": "HIGH"}
        )


def test_exact_literal_case_d_route_precedes_dynamic_route() -> None:
    paths = [getattr(route, "path", "") for route in app.routes]
    literal = "/api/v1/competition/demo/cases/CASE-D/run"
    dynamic = "/api/v1/competition/demo/cases/{case_id}/run"
    assert paths.count(literal) == 1
    assert paths.index(literal) < paths.index(dynamic)
    operation = CLIENT.get("/openapi.json").json()["paths"][literal]["post"]
    assert "requestBody" not in operation


def test_case_d_post_returns_controlled_synthetic_response(
    case_d_response: dict,
) -> None:
    assert case_d_response["case_id"] == "CASE-D"
    assert case_d_response["classification"] == "CONTROLLED_SYNTHETIC"
    assert case_d_response["display_label"] == "受控合成演示案例"
    assert case_d_response["purpose"] == "INTERNAL_CONSISTENCY_DEMONSTRATION"
    assert case_d_response["document_sha256"] == EXPECTED_ASSET_SHA256


@pytest.mark.parametrize(
    "payload",
    [
        {"pdf_path": "C:/caller.pdf"},
        {"facts": [{"value": 30, "unit": "mm"}]},
        {"source_text": "caller authority", "numeric_value": 20},
    ],
)
def test_case_d_route_rejects_every_request_body(payload: dict) -> None:
    response = CLIENT.post(CASE_D_ENDPOINT, json=payload)
    assert response.status_code == 422
    assert "不接受请求体" in response.json()["detail"]


@pytest.mark.parametrize("query", [{"pdf_path": "x"}, {"facts": "x"}])
def test_case_d_route_rejects_query_authority(query: dict[str, str]) -> None:
    response = CLIENT.post(CASE_D_ENDPOINT, params=query)
    assert response.status_code == 422


def test_unknown_case_still_uses_existing_dynamic_route() -> None:
    response = CLIENT.post("/api/v1/competition/demo/cases/CASE-UNKNOWN/run")
    assert response.status_code == 404
    assert response.json()["detail"] == "未找到该固定局部审查目标。"


def test_adapter_invokes_only_frozen_public_review_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = InternalConsistencyReviewService.review_document

    def counted(self: InternalConsistencyReviewService, document_id: str):
        calls.append(document_id)
        return original(self, document_id)

    monkeypatch.setattr(InternalConsistencyReviewService, "review_document", counted)
    response = CLIENT.post(CASE_D_ENDPOINT)
    assert response.status_code == 200
    assert calls == ["a38da41d-3bb1-5cbb-8402-8d76545dfca9"]

    source = inspect.getsource(
        service_module.CompetitionInternalConsistencyDemoService
    )
    assert "._group_facts(" not in source
    assert "._fact_from_clause(" not in source
    assert "._review_verified_pages(" not in source


def test_qualified_asset_and_metadata_are_validated_server_side() -> None:
    metadata = json.loads(CASE_D_METADATA.read_text(encoding="utf-8"))
    assert metadata["case_id"] == "CASE-D"
    assert metadata["document_id"] == "a38da41d-3bb1-5cbb-8402-8d76545dfca9"
    assert metadata["classification"] == "CONTROLLED_SYNTHETIC"
    assert metadata["purpose"] == "INTERNAL_CONSISTENCY_DEMONSTRATION"
    assert metadata["asset_sha256"] == _sha256(CASE_D_ASSET)
    assert metadata["asset_sha256"] == EXPECTED_ASSET_SHA256


def test_tampered_case_d_asset_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_temp_case_d(
        monkeypatch,
        tmp_path,
        asset_bytes=CASE_D_ASSET.read_bytes() + b"tampered",
    )
    response = CLIENT.post(CASE_D_ENDPOINT)
    assert response.status_code == 409
    assert response.json() == {
        "detail": "系统已安全停止，本次未生成一致性检查结果。请检查受控演示资产。"
    }


def test_missing_case_d_asset_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_temp_case_d(monkeypatch, tmp_path, asset_bytes=None)
    response = CLIENT.post(CASE_D_ENDPOINT)
    assert response.status_code == 409


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "CASE-X"),
        ("document_id", "00000000-0000-0000-0000-000000000000"),
        ("classification", "REAL_PROJECT"),
        ("purpose", "OTHER"),
        ("asset_sha256", "0" * 64),
    ],
)
def test_invalid_controlled_metadata_fails_closed(
    field: str,
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_temp_case_d(
        monkeypatch,
        tmp_path,
        asset_bytes=CASE_D_ASSET.read_bytes(),
        metadata_update={field: value},
    )
    assert CLIENT.post(CASE_D_ENDPOINT).status_code == 409


def test_live_result_contains_one_candidate_and_exact_source_authority(
    case_d_response: dict,
) -> None:
    assert case_d_response["candidate_count"] == 1
    candidate = case_d_response["candidates"][0]
    assert candidate["status"] == "INTERNAL_CONSISTENCY_CANDIDATE"
    assert candidate["relation"] == "DIFFERENT_VALUE"
    assert candidate["reason"] == "DIFFERENT_EXPLICIT_NUMERIC_VALUES"
    assert candidate["review_status"] == "NEEDS_HUMAN_REVIEW"
    assert candidate["object_display_name"] == "支护结构"
    assert candidate["parameter_display_name"] == "水平位移报警值"
    assert candidate["unit"] == "mm"
    assert [group["value"] for group in candidate["value_groups"]] == ["20", "30"]
    sources = [
        source
        for group in candidate["value_groups"]
        for source in group["sources"]
    ]
    assert {source["physical_page"] for source in sources} == {1, 2}
    assert {source["source_text"] for source in sources} == {
        "支护结构水平位移报警值为20 mm。",
        "支护结构水平位移报警值为30 mm。",
    }
    assert all(source["fact_id"].startswith("consistencyfact_") for source in sources)
    assert all(len(source["source_text_sha256"]) == 64 for source in sources)
    assert all(len(source["page_text_sha256"]) == 64 for source in sources)


def test_candidate_and_review_identity_are_deterministic(
    case_d_response: dict,
) -> None:
    repeated = CLIENT.post(CASE_D_ENDPOINT)
    assert repeated.status_code == 200
    payload = repeated.json()
    assert payload["review_id"] == case_d_response["review_id"] == EXPECTED_REVIEW_ID
    assert (
        payload["candidates"][0]["candidate_id"]
        == case_d_response["candidates"][0]["candidate_id"]
        == EXPECTED_CANDIDATE_ID
    )
    assert payload == case_d_response


def test_response_has_no_authority_escalation(case_d_response: dict) -> None:
    forbidden = {
        "compliance",
        "compliant",
        "non_compliant",
        "finding",
        "finding_id",
        "severity",
        "risk",
        "unsafe",
        "safe",
        "violation",
        "defect",
        "standard",
        "article",
        "normative_requirement",
        "whole_plan_verdict",
        "overall_result",
        "score",
    }
    assert forbidden.isdisjoint({key.casefold() for key in _all_keys(case_d_response)})
    serialized = json.dumps(case_d_response, ensure_ascii=False)
    assert "REAL_PROJECT" not in serialized
    assert "NON_COMPLIANT" not in serialized


@pytest.mark.parametrize(
    ("case_id", "item_kind", "decision", "terminal_status"),
    [
        ("CASE-A", "FINDING", "COMPLIANT", None),
        ("CASE-B", "FINDING", "NON_COMPLIANT", None),
        ("CASE-C", "REVIEW_GAP", None, "NO_STANDARD_SCOPE"),
    ],
)
def test_existing_case_semantics_remain_unchanged(
    case_id: str,
    item_kind: str,
    decision: str | None,
    terminal_status: str | None,
) -> None:
    response = CLIENT.post(f"/api/v1/competition/demo/cases/{case_id}/run")
    assert response.status_code == 200
    payload = response.json()
    assert payload["case_id"] == case_id
    assert payload["item_kind"] == item_kind
    if decision is not None:
        assert payload["decision"] == decision
    if terminal_status is not None:
        assert payload["terminal_status"] == terminal_status


def test_existing_metadata_remains_exactly_three_abc_cases() -> None:
    response = CLIENT.get("/api/v1/competition/demo")
    assert response.status_code == 200
    assert [item["case_id"] for item in response.json()["cases"]] == [
        "CASE-A",
        "CASE-B",
        "CASE-C",
    ]


def test_case_d_ui_is_additive_parseable_and_has_unique_ids() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    parser = _IdentityParser()
    parser.feed(html)
    parser.close()
    assert len(parser.ids) == len(set(parser.ids))
    assert "跨章节方案内部一致性检查" in html
    assert "受控合成演示案例" in html
    assert "运行一致性检查" in html
    assert "不构成规范不合规、风险或工程缺陷结论" in html
    assert 'id="case-grid"' in html
    assert 'id="result-section"' in html


def test_case_d_javascript_uses_safe_server_evidence_rendering() -> None:
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    endpoint = "/api/v1/competition/demo/cases/CASE-D/run"
    assert source.count(endpoint) == 1
    assert "CONSISTENCY_CASE_D_API" in source
    assert 'method: "POST"' in source
    assert "group.value" in source
    assert "source.physical_page" in source
    assert "source.source_text" in source
    assert "待人工复核" in source
    assert "createElement" in source
    assert "textContent" in source
    assert "replaceChildren" in source
    assert "innerHTML" not in source
    assert "支护结构水平位移报警值为30 mm。" not in source
    assert "支护结构水平位移报警值为20 mm。" not in source


def test_case_d_ui_preserves_server_returned_purpose_without_rewriting() -> None:
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    purpose_render = 'element("p", "consistency-result-code", result.purpose)'
    assert source.count("result.purpose") == 1
    assert purpose_render in source
    assert "INTERNAL_CONSISTENCY_DEMONSTRATION" not in source
    assert all(
        forbidden not in purpose_render.casefold()
        for forbidden in ("compliance", "finding", "severity", "risk", "defect")
    )


def test_production_adapter_has_no_prebuilt_result_or_forbidden_runtime() -> None:
    source = SERVICE_PATH.read_text(encoding="utf-8")
    assert EXPECTED_CANDIDATE_ID not in source
    assert EXPECTED_REVIEW_ID not in source
    assert "支护结构水平位移报警值为30 mm。" not in source
    assert "支护结构水平位移报警值为20 mm。" not in source
    assert "expected_review" not in source
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_prefixes = (
        "app.services.ocr",
        "app.services.review.compliance_comparison_service",
        "app.services.review.review_finding_service",
        "requests",
        "httpx",
        "openai",
    )
    assert not any(
        imported.startswith(prefix)
        for imported in imports
        for prefix in forbidden_prefixes
    )
    assert "AppData" not in source
    assert "BuildCheck-Agent-Competition-D9-P3" not in source


def test_case_d_asset_is_tracked_and_packaged_by_existing_rules() -> None:
    relative_asset = CASE_D_ASSET.relative_to(PROJECT_ROOT).as_posix()
    tracked = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "ls-files", "--error-unmatch", relative_asset],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tracked.returncode == 0

    deployment_root = PROJECT_ROOT.parent / "BuildCheck-Agent-Competition-Deploy"
    dockerfile = (deployment_root / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (deployment_root / ".dockerignore").read_text(encoding="utf-8")
    assert "COPY . ." in dockerfile
    assert "backend/tests" not in dockerignore
    assert "internal_consistency" not in dockerignore
    assert "case_d" not in dockerignore
