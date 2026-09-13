"""Focused qualification-style tests for the F2 same-origin demo slice."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
import app.services.competition_demo_service as competition_demo_service_module
from app.services.ocr.run_identity import (
    OCR_CORPUS_SEMANTICS_VERSION,
    OCR_QUALITY_GATE_VERSION,
)
from app.services.competition_demo_service import CompetitionDemoService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = PROJECT_ROOT / "backend" / "app" / "static" / "competition"
client = TestClient(app)

EXPECTED_CASES = {
    "CASE-A": {
        "item_kind": "FINDING",
        "finding_id": "finding_993e8847b86de084c9d1942a9771bf3dab07200260499f7d38d00292060e818b",
        "decision": "COMPLIANT",
        "label": "局部符合",
        "reason": "NUMERIC_LIMIT_SATISFIED",
        "article": "4.4.15",
    },
    "CASE-B": {
        "item_kind": "FINDING",
        "finding_id": "finding_530869523a03ddacf658e167fcff74f5b32b68f324533711a0b38538298f0c73",
        "decision": "NON_COMPLIANT",
        "label": "局部不符合",
        "reason": "NUMERIC_LIMIT_VIOLATED",
        "article": "4.4.16",
    },
    "CASE-C": {
        "item_kind": "REVIEW_GAP",
        "document_id": "04039d98-4131-422a-b29a-256bade04a6a",
        "document_sha256": "41f4ea2e7d999f298309f4ecb25491cc79125d422cb84484963de36325166b95",
        "terminal_class": "CANDIDATE_TERMINAL",
        "terminal_status": "NO_STANDARD_SCOPE",
        "physical_page": 8,
        "page_char_start": 107,
        "page_char_end": 142,
        "source_text": "（3）、套管预埋必须做到同水平标高的套管标高偏差必控制在5mm 之内，",
        "source_text_sha256": "7450ef276b244603d4c2bc4930bba8b94d9381b20d4eb39c33bec7862ad40faf",
    },
}

FORBIDDEN_RESPONSE_KEYS = {
    "qualification_assertion",
    "runtime_result",
    "runtime_result_override",
    "prebuilt_finding",
    "expected_decision",
    "expected_reason",
    "expected_reason_code",
    "expected_finding_id",
    "expected_comparison_id",
    "expected_review_gap_id",
    "expected_workspace_id",
    "expected_scope",
}


@pytest.fixture(scope="module", autouse=True)
def qualified_ocr_authority_for_presentation_tests(tmp_path_factory):
    """Keep presentation tests isolated from the committed qualification artifact."""
    source = PROJECT_ROOT / "competition/corpus/gb55023-ocr-qualification-manifest.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw.update(
        {
            "execution_artifact_sha256": "1" * 64,
            "raw_ocr_result_artifact_sha256": "2" * 64,
            "quality_assessment_artifact_sha256": "3" * 64,
            "accepted_boundary_artifact_sha256": "4" * 64,
            "quality_gate_version": OCR_QUALITY_GATE_VERSION,
            "ocr_corpus_semantics_version": OCR_CORPUS_SEMANTICS_VERSION,
            "qualification_status": "QUALIFIED",
            "qualification_reason": "Synthetic complete chain for presentation tests only.",
        }
    )
    path = tmp_path_factory.mktemp("ocr-ui-qualification") / "qualified.json"
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    original = competition_demo_service_module.load_corpus_package

    def load_qualified(manifest, project_root):
        return original(
            manifest,
            project_root,
            qualification_manifest_path=path,
            qualification_manifest_sha256=digest,
        )

    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        competition_demo_service_module,
        "load_corpus_package",
        load_qualified,
    )
    yield
    patcher.undo()


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def metadata_payload() -> dict:
    response = client.get("/api/v1/competition/demo")
    assert response.status_code == 200
    return response.json()


@pytest.fixture(scope="module")
def live_results() -> dict[str, dict]:
    results = {}
    for case_id in EXPECTED_CASES:
        response = client.post(f"/api/v1/competition/demo/cases/{case_id}/run")
        assert response.status_code == 200, response.text
        results[case_id] = response.json()
    return results


def test_competition_homepage_loads() -> None:
    response = client.get("/competition/")
    assert response.status_code == 200
    assert "筑审智核" in response.text
    assert "QUALIFIED SAMPLE MODE" in response.text
    assert "局部范围声明" in response.text


@pytest.mark.parametrize("asset", ["app.js", "styles.css"])
def test_competition_static_assets_load(asset: str) -> None:
    response = client.get(f"/competition/{asset}")
    assert response.status_code == 200
    assert response.content


@pytest.mark.parametrize("path", ["/health", "/docs", "/openapi.json"])
def test_existing_application_routes_remain_available(path: str) -> None:
    assert client.get(path).status_code == 200


def test_openapi_success_response_discriminates_finding_and_review_gap() -> None:
    schema = client.get("/openapi.json").json()
    response_schema = schema["paths"][
        "/api/v1/competition/demo/cases/{case_id}/run"
    ]["post"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["discriminator"] == {
        "propertyName": "item_kind",
        "mapping": {
            "FINDING": "#/components/schemas/CompetitionFindingCaseRunResult",
            "REVIEW_GAP": "#/components/schemas/CompetitionReviewGapCaseRunResult",
        },
    }
    assert len(response_schema["oneOf"]) == 2
    gap_properties = schema["components"]["schemas"][
        "CompetitionReviewGapCaseRunResult"
    ]["properties"]
    assert {
        "finding_id",
        "comparison_id",
        "decision",
        "decision_scope",
        "standard_evidence",
        "requirement_id",
        "article_number",
        "standard_id",
    }.isdisjoint(gap_properties)


def test_metadata_reports_verified_runtime_ready(metadata_payload: dict) -> None:
    assert metadata_payload["mode"] == "QUALIFIED_SAMPLE_MODE"
    assert metadata_payload["status"] == "READY"
    assert len(metadata_payload["plans"]) == 2
    assert len(metadata_payload["cases"]) == 3
    assert all(check["ready"] for check in metadata_payload["readiness_checks"])
    assert all(plan["verified"] for plan in metadata_payload["plans"])


def test_metadata_exposes_only_neutral_case_information(metadata_payload: dict) -> None:
    cases = {item["case_id"]: item for item in metadata_payload["cases"]}
    assert set(cases) == set(EXPECTED_CASES)
    for case_id, item in cases.items():
        common = {
            "case_id",
            "item_kind",
            "label",
            "description",
            "document_id",
            "page_number",
        }
        if item["item_kind"] == "FINDING":
            assert set(item) == common | {"standard_code", "article_number"}
            assert EXPECTED_CASES[case_id]["decision"] not in str(item)
            assert EXPECTED_CASES[case_id]["finding_id"] not in str(item)
        else:
            assert set(item) == common
            assert case_id == "CASE-C"


def test_metadata_contains_no_qualification_assertion_leakage(metadata_payload: dict) -> None:
    keys = {key.casefold() for key in _keys(metadata_payload)}
    assert keys.isdisjoint(FORBIDDEN_RESPONSE_KEYS)
    assert not any(key.startswith("expected_") for key in keys)


def test_metadata_not_ready_is_read_only(tmp_path: Path) -> None:
    settings = Settings(
        upload_dir=tmp_path / "absent-uploads",
        standards_dir=tmp_path / "absent-standards",
    )
    result = CompetitionDemoService(settings=settings).metadata()
    assert result.status == "NOT_READY"
    assert not all(check.ready for check in result.readiness_checks)
    assert not (tmp_path / "absent-uploads").exists()
    assert not (tmp_path / "absent-standards").exists()


def test_unknown_case_is_rejected() -> None:
    response = client.post("/api/v1/competition/demo/cases/CASE-UNKNOWN/run")
    assert response.status_code == 404
    assert "固定局部审查目标" in response.json()["detail"]


@pytest.mark.parametrize(
    "case_path",
    [
        "..%2FCASE-A",
        "%2e%2e%2fCASE-A",
        "CASE-A%2F..%2FCASE-B",
    ],
)
def test_path_traversal_case_identifier_is_rejected(case_path: str) -> None:
    assert client.post(f"/api/v1/competition/demo/cases/{case_path}/run").status_code in {404, 405}


@pytest.mark.parametrize(
    "payload",
    [
        {"document_id": "override"},
        {"plan_sha256": "0" * 64},
        {"standard_sha256": "0" * 64},
        {"evidence_id": "override"},
        {"plan_fact": {"char_start": 0}},
        {"expected_decision": "COMPLIANT"},
        {"review_finding": {"decision": "COMPLIANT"}},
        {"workspace_id": "findingsworkspace_" + "0" * 64},
        {"review_gap_id": "reviewgap_" + "0" * 64},
        {"status": "NO_STANDARD_SCOPE"},
        {"decision": "REVIEW_GAP"},
        {"manifest_path": "../manifest.json"},
        {"corpus_path": "../corpus.json"},
        {"filesystem_path": "C:/authority.json"},
        {"overall_status": "PASS"},
    ],
)
def test_run_endpoint_rejects_all_caller_authority_payloads(payload: dict) -> None:
    response = client.post("/api/v1/competition/demo/cases/CASE-A/run", json=payload)
    assert response.status_code == 422
    assert "不接受请求体" in response.json()["detail"]


@pytest.mark.parametrize("case_id", ["CASE-A", "CASE-B"])
def test_live_finding_cases_match_qualified_c3_authority(
    case_id: str, live_results: dict[str, dict]
) -> None:
    result = live_results[case_id]
    expected = EXPECTED_CASES[case_id]
    assert result["status"] == "SUCCESS"
    assert result["item_kind"] == "FINDING"
    assert result["finding_id"] == expected["finding_id"]
    assert result["decision"] == expected["decision"]
    assert result["decision_label"] == expected["label"]
    assert result["reason_code"] == expected["reason"]
    assert result["decision_scope"] == "REVIEW_UNIT_REQUIREMENT"
    assert result["standard_evidence"]["article_number"] == expected["article"]
    assert result["plan_evidence"]["exact_text"]
    assert result["standard_evidence"]["exact_requirement_text"]
    assert result["summary"]


def test_case_c_is_genuine_d6_review_gap_without_finding_authority(
    live_results: dict[str, dict]
) -> None:
    result = live_results["CASE-C"]
    expected = EXPECTED_CASES["CASE-C"]
    assert result["status"] == "SUCCESS"
    assert result["item_kind"] == "REVIEW_GAP"
    assert result["terminal_class"] == expected["terminal_class"]
    assert result["terminal_status"] == expected["terminal_status"]
    assert result["review_gap_id"].startswith("reviewgap_")
    assert result["technical_provenance"]["workspace_id"].startswith(
        "findingsworkspace_"
    )
    assert result["finding_absent"] is True
    assert result["comparison_absent"] is True
    assert result["decision_absent"] is True
    assert result["standard_authority_absent"] is True
    assert result["article_authority_absent"] is True
    assert result["requirement_authority_absent"] is True
    assert {
        "finding_id",
        "comparison_id",
        "decision",
        "decision_scope",
        "standard_evidence",
        "requirement_id",
        "article_number",
        "standard_id",
    }.isdisjoint(result)
    evidence = result["plan_evidence"]
    assert evidence["document_id"] == expected["document_id"]
    assert evidence["pdf_sha256"] == expected["document_sha256"]
    assert evidence["physical_page"] == expected["physical_page"]
    assert evidence["page_char_start"] == expected["page_char_start"]
    assert evidence["page_char_end"] == expected["page_char_end"]
    assert evidence["exact_text"] == expected["source_text"]
    assert evidence["text_sha256"] == expected["source_text_sha256"]


@pytest.mark.parametrize("case_id", ["CASE-A", "CASE-B", "CASE-C"])
def test_run_projection_contains_no_expected_result_leakage(
    case_id: str, live_results: dict[str, dict]
) -> None:
    keys = {key.casefold() for key in _keys(live_results[case_id])}
    assert keys.isdisjoint(FORBIDDEN_RESPONSE_KEYS)
    assert not any(key.startswith("expected_") for key in keys)


def test_case_b_presents_plan_control_conflict_not_field_measurement(live_results: dict[str, dict]) -> None:
    result = live_results["CASE-B"]
    assert result["local_explanation"] == "当前方案控制值与该项规范限值发生局部冲突。"
    serialized = str(result)
    assert "实测3mm" not in serialized
    assert "现场间隙为3mm" not in serialized


def test_case_c_is_review_gap_not_compliance_uncertainty_finding(
    live_results: dict[str, dict]
) -> None:
    result = live_results["CASE-C"]
    assert result["status"] == "SUCCESS"
    assert result["item_kind"] == "REVIEW_GAP"
    assert "规范范围权威" in result["local_explanation"]
    for false_verdict in ("方案合规", "方案不合规", "符合 GB", "违反 GB"):
        assert false_verdict not in result["local_explanation"]
    assert "SAFE_FAILURE" not in str(result)


def test_presentation_dto_keeps_dual_citations_and_provenance_separate(live_results: dict[str, dict]) -> None:
    for result in (live_results["CASE-A"], live_results["CASE-B"]):
        assert result["plan_evidence"]["document_id"] == result["technical_provenance"]["document_id"]
        assert result["plan_evidence"]["pdf_sha256"] == result["technical_provenance"]["document_sha256"]
        assert result["standard_evidence"]["evidence_id"] == result["technical_provenance"]["evidence_id"]
        assert result["standard_evidence"]["requirement_id"] == result["technical_provenance"]["requirement_id"]
        assert result["finding_id"] == result["technical_provenance"]["finding_id"]
    gap = live_results["CASE-C"]
    assert gap["plan_evidence"]["document_id"] == gap["technical_provenance"]["document_id"]
    assert gap["plan_evidence"]["pdf_sha256"] == gap["technical_provenance"]["document_sha256"]
    assert gap["review_gap_id"] == gap["technical_provenance"]["review_gap_id"]


def test_frontend_contains_no_prebuilt_case_authority() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    combined = f"{html}\n{javascript}"
    assert "expected_" not in combined.casefold()
    assert "qualification_assertion" not in combined.casefold()
    assert "runtime_result_override" not in combined.casefold()
    assert "finding_993e8847" not in combined
    assert "finding_53086952" not in combined
    assert "finding_9c02ed35" not in combined
    assert "reviewgap_09b31a" not in combined
    assert "findingsworkspace_90fb" not in combined


def test_frontend_javascript_uses_safe_dom_and_same_origin_fetch() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "textContent" in javascript
    assert "createElement" in javascript
    assert "innerHTML" not in javascript
    assert "eval(" not in javascript
    assert "http://" not in javascript
    assert "https://" not in javascript
    assert 'const API_ROOT = "/api/v1/competition/demo"' in javascript


def test_frontend_has_distinct_review_gap_rendering_without_fake_finding_card() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    assert 'if (result.item_kind === "FINDING")' in javascript
    assert 'if (result.item_kind === "REVIEW_GAP")' in javascript
    assert "function renderReviewGapResult" in javascript
    assert "明确审查缺口" in javascript
    assert "NO_STANDARD_SCOPE" in javascript
    assert ".review-gap-header" in styles
    assert "Finding / Review Gap" in html
    gap_renderer = javascript.split("function renderReviewGapResult", 1)[1].split(
        "function renderResult", 1
    )[0]
    assert "DECISION_LABELS" not in gap_renderer
    assert "result.standard_evidence" not in gap_renderer
    assert "result.finding_id" not in gap_renderer
    assert "result.comparison_id" not in gap_renderer


def test_competition_ui_contains_no_c4_workflow_controls() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in STATIC_ROOT.iterdir())
    for term in ("reviewer_note", "CALLER_ASSERTED_UNVERIFIED", "接受 Finding", "拒绝 Finding", "最终裁决"):
        assert term not in combined


def test_affirmative_whole_plan_wording_is_absent() -> None:
    production_paths = [
        PROJECT_ROOT / "backend" / "app" / "schemas" / "competition_demo.py",
        PROJECT_ROOT / "backend" / "app" / "services" / "competition_demo_service.py",
        PROJECT_ROOT / "backend" / "app" / "api" / "routes" / "competition.py",
        *STATIC_ROOT.iterdir(),
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in production_paths)
    for forbidden in ("审查通过", "方案合格", "方案不合格", "项目合规", "全部符合", "全部不符合", "整体不合格", "overall pass", "overall fail"):
        assert forbidden not in combined
    assert "不代表整份方案总体合规状态" in combined


def test_packaged_demo_authority_files_remain_byte_identical() -> None:
    expected = {
        "competition/demo-manifest.json": "30ff01416e651cf72404e2b2808f496e5bd6323fe0b4415fa13ef1c1d678da84",
        "competition/corpus/gb55023-qualified-parse-result.json": "4a240614213ff5652dae5f7a5a07063b9848df1e54ccd42830320745947f6fd7",
    }
    assert {path: _sha256(PROJECT_ROOT / path) for path in expected} == expected
