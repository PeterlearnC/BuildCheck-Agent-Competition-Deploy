"""Focused presentation-boundary tests for the D.7-P3-C report viewer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = PROJECT_ROOT / "backend" / "app" / "static" / "competition"
CLIENT = TestClient(app)

FROZEN_UPSTREAM_SHA256 = {
    "backend/app/schemas/review_report_artifact.py": "c5ffb6b20b2e75df7161127f128783832ced2bd03c51d14df6108b78f1599101",
    "backend/app/services/review/review_report_renderer.py": "a2a10a12d388be67f474dfef46f4a7022b1ae457f4589f659a8489b128090579",
    "backend/app/api/routes/review_report.py": "cf90bf47218f4c0c339691f89bfbad8177bf0b331389b9214bcad26edd740436",
    "backend/app/schemas/review_report_transport.py": "03d3ef40f9c0242ec00ec9ecf2b0a6370b543861a001da0a91bd61910dc2bbbe",
    "backend/app/api/routes/review.py": "a7dbbef2d9424b9bb70439e8eaf3fb1c817fcf98754a7127ee39b914bf294d0e",
}


def _asset(name: str) -> str:
    return (STATIC_ROOT / name).read_text(encoding="utf-8")


def test_report_viewer_is_served_by_existing_competition_mount() -> None:
    response = CLIENT.get("/competition/")
    assert response.status_code == 200
    assert 'id="report-viewer"' in response.text
    assert 'id="report-artifact-frame"' in response.text


@pytest.mark.parametrize("name", ["index.html", "app.js", "styles.css"])
def test_only_repository_native_static_assets_are_used(name: str) -> None:
    response = CLIENT.get(f"/competition/{name}")
    assert response.status_code == 200


def test_viewer_consumes_only_frozen_report_transport_endpoint() -> None:
    javascript = _asset("app.js")
    assert "/api/v1/documents/" in javascript
    assert "/review/report`" in javascript
    assert 'method: "POST"' in javascript
    assert 'delivery: "INLINE"' in javascript
    assert "ReviewReportService" not in javascript
    assert "HumanReviewService" not in javascript
    assert "FindingsWorkspaceService" not in javascript


def test_report_artifact_is_read_and_reused_as_unmodified_bytes() -> None:
    javascript = _asset("app.js")
    assert "await response.arrayBuffer()" in javascript
    assert "new Blob([artifactBytes]" in javascript
    assert "download.href = reportViewerState.objectUrl" in javascript
    assert "frame.src = reportViewerState.objectUrl" in javascript
    assert "response.clone().json" not in javascript
    assert "response.clone().text" not in javascript


def test_viewer_uses_artifact_owned_transport_metadata() -> None:
    javascript = _asset("app.js")
    for header in (
        "Content-Disposition",
        "Content-Type",
        "X-Report-Model-ID",
        "X-Report-Artifact-ID",
        "X-Content-SHA256",
    ):
        assert header in javascript
    assert "download.download = filename" in javascript


def test_preview_is_sandboxed_and_has_no_script_authority() -> None:
    html = _asset("index.html")
    assert '<iframe\n          id="report-artifact-frame"' in html
    assert "          sandbox\n" in html
    assert "srcdoc" not in html


def test_new_viewer_code_contains_no_semantic_decision_logic() -> None:
    javascript = _asset("app.js")
    viewer = javascript[javascript.index("const reportViewerState") :]
    forbidden = (
        "NON_COMPLIANT",
        "COMPLIANT",
        "INSUFFICIENT_INFORMATION",
        "final_decision",
        "effective_compliance",
        "overall_compliance",
        "risk_score",
        "severity",
        "PlanFact",
    )
    assert all(token not in viewer for token in forbidden)


def test_viewer_creates_no_regenerated_report_payload() -> None:
    javascript = _asset("app.js")
    viewer = javascript[javascript.index("const reportViewerState") :]
    assert "JSON.stringify({ commands, format, delivery" in viewer
    assert "JSON.stringify(artifact" not in viewer
    assert "innerHTML" not in viewer
    assert "document.write" not in viewer


def test_frozen_upstream_authority_files_remain_byte_identical() -> None:
    for relative_path, expected_sha256 in FROZEN_UPSTREAM_SHA256.items():
        assert hashlib.sha256((PROJECT_ROOT / relative_path).read_bytes()).hexdigest() == expected_sha256


def test_false_authority_matrix_is_zero_by_construction() -> None:
    javascript = _asset("app.js")
    viewer = javascript[javascript.index("const reportViewerState") :]
    assert "fetch(REPORT_API_PATH(documentId)" in viewer
    assert "indexedDB" not in viewer
    assert "localStorage" not in viewer
    assert "WebSocket" not in viewer
    assert "XMLHttpRequest" not in viewer
    assert "fetch(" in viewer
    assert viewer.count("fetch(") == 1
