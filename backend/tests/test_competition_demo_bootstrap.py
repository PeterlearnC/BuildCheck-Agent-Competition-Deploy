"""Competition demo bootstrap preserves and reproduces qualified C.3 authority."""

from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from app.core.config import Settings
from app.services.standards.standard_repository import StandardRepository
from scripts.prepare_competition_demo import (
    CompetitionBootstrapError,
    DemoManifest,
    PROTECTED_CORE_AUTHORITY_PATHS,
    PROTECTED_F1_AUTHORITY_PATHS,
    QUALIFIED_CORE_BASELINE,
    QUALIFIED_F1_BASELINE,
    _load_json,
    _assert_live_finding,
    _finding_service,
    _verify_qualified_repository,
    build_comparison_request,
    install_qualified_corpus,
    load_corpus_package,
    load_manifest,
    prepare_runtime,
    provision_plans,
    run_preflight,
    verify_core_baseline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "competition" / "demo-manifest.json"
SCRIPT_PATH = PROJECT_ROOT / "backend" / "scripts" / "prepare_competition_demo.py"
DEFAULT_ASSET_DIR = PROJECT_ROOT / "backend" / "data" / "uploads"
ASSET_DIR = Path(os.getenv("BUILDCHECK_DEMO_ASSET_DIR", DEFAULT_ASSET_DIR))


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write(repository: Path, relative_path: str, content: str) -> None:
    destination = repository / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


@pytest.fixture
def qualified_guard_repository(tmp_path):
    repository = tmp_path / "qualified-repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "guard-test@example.invalid")
    _git(repository, "config", "user.name", "Guard Test")

    for path in sorted(PROTECTED_CORE_AUTHORITY_PATHS):
        _write(repository, path, f"qualified-core:{path}\n")
    _write(repository, "backend/app/main.py", "qualified composition root\n")
    _git(repository, "add", "--", ".")
    _git(repository, "commit", "--quiet", "-m", "qualified core")
    core_head = _git(repository, "rev-parse", "HEAD").lower()

    for path in sorted(PROTECTED_F1_AUTHORITY_PATHS):
        _write(repository, path, f"qualified-f1:{path}\n")
    _git(repository, "add", "--", ".")
    _git(repository, "commit", "--quiet", "-m", "qualified F1")
    f1_head = _git(repository, "rev-parse", "HEAD").lower()
    return {
        "repository": repository,
        "core_head": core_head,
        "f1_head": f1_head,
    }


def _verify_guard_fixture(fixture: dict) -> str:
    return _verify_qualified_repository(
        fixture["repository"],
        expected_core_head=fixture["core_head"],
        qualified_f1_head=fixture["f1_head"],
    )


def _raw_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _validated(raw: dict) -> DemoManifest:
    return DemoManifest.model_validate(raw)


def _duplicate_key(
    source: str,
    *,
    anchor: str,
    key: str,
    duplicate_value: str,
) -> str:
    before, selected = source.split(anchor, 1)
    marker = f'"{key}": '
    start = selected.index(marker)
    line_start = selected.rfind("\n", 0, start) + 1
    line_end = selected.index("\n", start)
    original_line = selected[line_start:line_end]
    indentation = original_line[: len(original_line) - len(original_line.lstrip())]
    duplicate_line = f'{indentation}"{key}": {duplicate_value}'
    if original_line.rstrip().endswith(","):
        insertion = "\n" + duplicate_line + ","
    else:
        insertion = ",\n" + duplicate_line
    selected = selected[:line_end] + insertion + selected[line_end:]
    return before + anchor + selected


def _require_assets(manifest: DemoManifest) -> None:
    missing = [
        plan.document_id
        for plan in manifest.plans
        if not (ASSET_DIR / f"{plan.document_id}.pdf").is_file()
    ]
    if missing:
        pytest.skip(
            "Operator-provided qualified competition plan assets are unavailable: "
            + ", ".join(missing)
        )


@pytest.fixture(scope="session")
def manifest() -> DemoManifest:
    return load_manifest(MANIFEST_PATH)


@pytest.fixture(scope="session")
def clean_runtime(tmp_path_factory, manifest):
    _require_assets(manifest)
    root = tmp_path_factory.mktemp("competition-runtime")
    uploads = root / "uploads"
    standards = root / "standards"
    assert not (standards / "documents").exists()
    result = prepare_runtime(
        project_root=PROJECT_ROOT,
        manifest_path=MANIFEST_PATH,
        asset_dir=ASSET_DIR,
        upload_dir=uploads,
        standards_dir=standards,
        preflight=True,
    )
    return {
        "root": root,
        "uploads": uploads,
        "standards": standards,
        "result": result,
    }


def test_protected_authority_boundary_matches_qualified_c1_c4_semantics() -> None:
    assert PROTECTED_CORE_AUTHORITY_PATHS == {
        "backend/app/api/routes/review.py",
        "backend/app/schemas/compliance_comparison.py",
        "backend/app/schemas/compliance_review.py",
        "backend/app/schemas/finding_review.py",
        "backend/app/schemas/normative_requirement.py",
        "backend/app/schemas/review_finding.py",
        "backend/app/schemas/review_unit.py",
        "backend/app/services/review/__init__.py",
        "backend/app/services/review/compliance_comparison_service.py",
        "backend/app/services/review/compliance_review_service.py",
        "backend/app/services/review/finding_review_service.py",
        "backend/app/services/review/plan_fact_service.py",
        "backend/app/services/review/requirement_service.py",
        "backend/app/services/review/review_finding_service.py",
        "backend/app/services/review/review_unit_service.py",
    }
    assert PROTECTED_F1_AUTHORITY_PATHS == {
        "competition/demo-manifest.json",
        "competition/corpus/gb55023-qualified-parse-result.json",
    }
    assert "backend/app/main.py" not in PROTECTED_CORE_AUTHORITY_PATHS


def test_current_qualified_f1_baseline_accepts_downstream_f2_wip() -> None:
    assert QUALIFIED_CORE_BASELINE == "7a0250a52f84fdd332775dacb2d5683877625f1a"
    assert QUALIFIED_F1_BASELINE == "b35638577cae511ed40660bf9748c15873180726"
    current_head = _git(PROJECT_ROOT, "rev-parse", "HEAD").lower()
    assert verify_core_baseline(PROJECT_ROOT, QUALIFIED_CORE_BASELINE) == current_head
    _git(
        PROJECT_ROOT,
        "merge-base",
        "--is-ancestor",
        QUALIFIED_F1_BASELINE,
        current_head,
    )


def test_qualified_f1_baseline_itself_is_accepted(qualified_guard_repository) -> None:
    assert _verify_guard_fixture(qualified_guard_repository) == qualified_guard_repository["f1_head"]


@pytest.mark.parametrize(
    "path",
    [
        "backend/app/static/future/index.html",
        "README.md",
        "docs/competition-demo.md",
    ],
)
def test_legitimate_downstream_additions_are_accepted(
    qualified_guard_repository, path
) -> None:
    repository = qualified_guard_repository["repository"]
    _write(repository, path, "legitimate downstream product addition\n")
    _git(repository, "add", "--", path)
    _git(repository, "commit", "--quiet", "-m", "downstream addition")
    assert _verify_guard_fixture(qualified_guard_repository) == _git(
        repository, "rev-parse", "HEAD"
    ).lower()


def test_downstream_main_composition_commit_is_accepted(qualified_guard_repository) -> None:
    repository = qualified_guard_repository["repository"]
    _write(repository, "backend/app/main.py", "qualified composition root\nnew router mount\n")
    _git(repository, "add", "--", "backend/app/main.py")
    _git(repository, "commit", "--quiet", "-m", "extend composition root")
    assert _verify_guard_fixture(qualified_guard_repository) == _git(
        repository, "rev-parse", "HEAD"
    ).lower()


@pytest.mark.parametrize(
    "phase,path",
    [
        ("C.1", "backend/app/services/review/review_unit_service.py"),
        ("C.2", "backend/app/services/review/compliance_comparison_service.py"),
        ("C.3", "backend/app/services/review/review_finding_service.py"),
        ("C.4", "backend/app/services/review/finding_review_service.py"),
    ],
)
def test_committed_c1_c4_semantic_mutations_are_rejected(
    qualified_guard_repository, phase, path
) -> None:
    repository = qualified_guard_repository["repository"]
    _write(repository, path, f"tampered {phase} authority\n")
    _git(repository, "add", "--", path)
    _git(repository, "commit", "--quiet", "-m", f"tamper {phase}")
    with pytest.raises(CompetitionBootstrapError, match="Committed qualified core"):
        _verify_guard_fixture(qualified_guard_repository)


@pytest.mark.parametrize(
    "path",
    [
        "competition/demo-manifest.json",
        "competition/corpus/gb55023-qualified-parse-result.json",
    ],
)
def test_committed_f1_authority_mutations_are_rejected(
    qualified_guard_repository, path
) -> None:
    repository = qualified_guard_repository["repository"]
    _write(repository, path, "tampered F1 authority\n")
    _git(repository, "add", "--", path)
    _git(repository, "commit", "--quiet", "-m", "tamper F1 authority")
    with pytest.raises(CompetitionBootstrapError, match="Committed qualified F1"):
        _verify_guard_fixture(qualified_guard_repository)


def test_unstaged_protected_authority_tampering_is_rejected(
    qualified_guard_repository,
) -> None:
    repository = qualified_guard_repository["repository"]
    path = "backend/app/schemas/compliance_review.py"
    _write(repository, path, "unstaged protected tampering\n")
    with pytest.raises(CompetitionBootstrapError, match="Unstaged protected authority"):
        _verify_guard_fixture(qualified_guard_repository)


def test_staged_only_protected_authority_tampering_is_rejected(
    qualified_guard_repository,
) -> None:
    repository = qualified_guard_repository["repository"]
    path = "backend/app/schemas/compliance_comparison.py"
    _write(repository, path, "staged protected tampering\n")
    _git(repository, "add", "--", path)
    assert _git(repository, "diff", "--name-only") == ""
    assert _git(repository, "diff", "--cached", "--name-only") == path
    with pytest.raises(CompetitionBootstrapError, match="Staged protected authority"):
        _verify_guard_fixture(qualified_guard_repository)


def test_wrong_qualified_f1_ancestry_is_rejected(qualified_guard_repository) -> None:
    repository = qualified_guard_repository["repository"]
    _git(repository, "reset", "--hard", qualified_guard_repository["core_head"])
    _write(repository, "README.md", "branch that bypasses qualified F1\n")
    _git(repository, "add", "--", "README.md")
    _git(repository, "commit", "--quiet", "-m", "wrong F1 ancestry")
    with pytest.raises(CompetitionBootstrapError, match="qualified F1 baseline"):
        _verify_guard_fixture(qualified_guard_repository)


def test_manifest_is_strict_versioned_selector_authority(manifest) -> None:
    assert manifest.schema_version == "competition-demo-manifest-v1"
    assert len(manifest.plans) == 2
    assert [case.case_id for case in manifest.cases] == ["CASE-A", "CASE-B", "CASE-C"]
    assert all(
        case.qualification_assertion.expected_scope == "REVIEW_UNIT_REQUIREMENT"
        for case in manifest.cases
    )
    raw = _raw_manifest()
    assert "finding" not in raw
    assert "runtime_result" not in raw


@pytest.mark.parametrize(
    ("attack", "anchor", "key", "duplicate_value"),
    [
        (
            "top-level-identical",
            "{\n",
            "schema_version",
            '"competition-demo-manifest-v1"',
        ),
        ("top-level-conflicting", "{\n", "schema_version", '"evil-version"'),
        ("nested-standard-source", '"standard": {', "source_sha256", '"0"'),
        ("nested-plan-sha", '"plans": [', "pdf_sha256", '"0"'),
        (
            "nested-case-document",
            '"cases": [',
            "document_id",
            '"00000000-0000-0000-0000-000000000000"',
        ),
        (
            "nested-expected-decision",
            '"qualification_assertion": {',
            "expected_decision",
            '"NON_COMPLIANT"',
        ),
        (
            "nested-evidence-selector",
            '"request": {',
            "evidence_id",
            '"chunk_0000000000000000000000000000000000000000000000000000000000000000"',
        ),
    ],
)
def test_duplicate_manifest_authority_keys_are_rejected(
    tmp_path, attack, anchor, key, duplicate_value
) -> None:
    attacked = _duplicate_key(
        MANIFEST_PATH.read_text(encoding="utf-8"),
        anchor=anchor,
        key=key,
        duplicate_value=duplicate_value,
    )
    path = tmp_path / f"{attack}.json"
    path.write_text(attacked, encoding="utf-8")
    with pytest.raises(
        CompetitionBootstrapError,
        match=rf"DUPLICATE_JSON_KEY.*{key}",
    ):
        load_manifest(path)


def test_duplicate_nested_corpus_authority_key_is_rejected(tmp_path) -> None:
    source = PROJECT_ROOT / "competition/corpus/gb55023-qualified-parse-result.json"
    attacked = _duplicate_key(
        source.read_text(encoding="utf-8"),
        anchor='"authority": {',
        key="source_checksum",
        duplicate_value='"0"',
    )
    path = tmp_path / "duplicate-corpus-authority.json"
    path.write_text(attacked, encoding="utf-8")
    with pytest.raises(
        CompetitionBootstrapError,
        match=r"DUPLICATE_JSON_KEY.*source_checksum",
    ):
        _load_json(path)


def test_original_authority_json_remains_strictly_loadable(manifest) -> None:
    assert load_manifest(MANIFEST_PATH) == manifest
    package = load_corpus_package(manifest, PROJECT_ROOT)
    assert package.authority.parse_result_canonical_sha256 == (
        manifest.standard.qualified_parse_result_sha256
    )


def test_qualified_corpus_package_matches_frozen_authority(manifest) -> None:
    package = load_corpus_package(manifest, PROJECT_ROOT)
    assert package.parse_result.document.standard_id == manifest.standard.standard_id
    assert package.parse_result.document.source_checksum == manifest.standard.source_sha256
    assert len(package.parse_result.articles) == 64
    by_number = {article.article_number: article for article in package.parse_result.articles}
    assert by_number["4.4.15"].article_id.endswith("74dfb6")
    assert by_number["4.4.16"].article_id.endswith("8209d")


def test_clean_runtime_bootstrap_installs_one_qualified_standard(clean_runtime) -> None:
    result = clean_runtime["result"]
    assert result["status"] == "PASS"
    assert result["corpus_provision"] == "INSTALLED"
    assert result["parsed_standard_count"] == 1
    assert result["qualified_article_count"] == 64
    assert result["ocr_calls"] == 0
    assert result["external_llm_calls"] == 0
    assert result["prebuilt_findings_used"] == 0


def test_three_live_c3_cases_match_qualified_authority(clean_runtime) -> None:
    cases = clean_runtime["result"]["cases"]
    assert [(case["case_id"], case["decision"]) for case in cases] == [
        ("CASE-A", "COMPLIANT"),
        ("CASE-B", "NON_COMPLIANT"),
        ("CASE-C", "INSUFFICIENT_INFORMATION"),
    ]
    assert all(case["status"] == "PASS" for case in cases)
    assert all(
        case["manifest_expected_result_used_as_runtime_finding"] is False
        for case in cases
    )


def test_repeat_bootstrap_is_idempotent(manifest, clean_runtime) -> None:
    result = prepare_runtime(
        project_root=PROJECT_ROOT,
        manifest_path=MANIFEST_PATH,
        asset_dir=ASSET_DIR,
        upload_dir=clean_runtime["uploads"],
        standards_dir=clean_runtime["standards"],
        preflight=False,
    )
    assert result["corpus_provision"] == "ALREADY_VERIFIED"
    assert set(result["plan_provision"].values()) == {"ALREADY_VERIFIED"}
    assert result["parsed_standard_count"] == 1


@pytest.mark.parametrize("mode", ["missing", "named-wrong", "modified"])
def test_wrong_or_missing_plan_assets_fail_closed(tmp_path, manifest, mode) -> None:
    assets = tmp_path / "assets"
    uploads = tmp_path / "uploads"
    assets.mkdir()
    plan = manifest.plans[0]
    if mode == "named-wrong":
        (assets / f"{plan.document_id}.pdf").write_bytes(b"not the qualified PDF")
    elif mode == "modified":
        _require_assets(manifest)
        source = ASSET_DIR / f"{plan.document_id}.pdf"
        (assets / f"{plan.document_id}.pdf").write_bytes(
            source.read_bytes() + b"modified"
        )
    with pytest.raises(CompetitionBootstrapError):
        provision_plans(manifest, assets, uploads)
    assert not (uploads / f"{plan.document_id}.pdf").exists()


def test_wrong_runtime_plan_is_never_overwritten(tmp_path, manifest) -> None:
    assets = tmp_path / "assets"
    uploads = tmp_path / "uploads"
    assets.mkdir()
    uploads.mkdir()
    plan = manifest.plans[0]
    destination = uploads / f"{plan.document_id}.pdf"
    destination.write_bytes(b"conflicting runtime bytes")
    with pytest.raises(CompetitionBootstrapError):
        provision_plans(manifest, assets, uploads)
    assert destination.read_bytes() == b"conflicting runtime bytes"


def test_modified_corpus_package_fails_sha_before_install(tmp_path, manifest) -> None:
    destination = tmp_path / manifest.standard.corpus_package_path
    destination.parent.mkdir(parents=True)
    source = PROJECT_ROOT / manifest.standard.corpus_package_path
    destination.write_bytes(source.read_bytes() + b"tampered")
    with pytest.raises(CompetitionBootstrapError, match="package SHA mismatch"):
        load_corpus_package(manifest, tmp_path)


def test_wrong_source_sha_metadata_fails_corpus_authority(manifest) -> None:
    raw = _raw_manifest()
    raw["standard"]["source_sha256"] = "0" * 64
    tampered = _validated(raw)
    with pytest.raises(CompetitionBootstrapError, match="authority mismatch"):
        load_corpus_package(tampered, PROJECT_ROOT)


def test_conflicting_installed_corpus_is_not_replaced(tmp_path, manifest) -> None:
    package = load_corpus_package(manifest, PROJECT_ROOT)
    settings = Settings(standards_dir=tmp_path / "standards")
    assert install_qualified_corpus(package, settings) == "INSTALLED"
    metadata = (
        settings.standards_dir
        / "documents"
        / manifest.standard.standard_id
        / "metadata.json"
    )
    raw = json.loads(metadata.read_text(encoding="utf-8"))
    raw["standard_code"] = "WRONG"
    metadata.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(CompetitionBootstrapError, match="conflicting"):
        install_qualified_corpus(package, settings)
    assert json.loads(metadata.read_text(encoding="utf-8"))["standard_code"] == "WRONG"


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown-schema",
        "missing-sha",
        "malformed-sha",
        "duplicate-case",
        "unknown-standard",
        "case-document-mismatch",
    ],
)
def test_manifest_identity_attacks_are_rejected(mutation) -> None:
    raw = _raw_manifest()
    if mutation == "unknown-schema":
        raw["schema_version"] = "unknown"
    elif mutation == "missing-sha":
        del raw["plans"][0]["pdf_sha256"]
    elif mutation == "malformed-sha":
        raw["standard"]["source_sha256"] = "not-a-sha"
    elif mutation == "duplicate-case":
        raw["cases"][1]["case_id"] = raw["cases"][0]["case_id"]
    elif mutation == "unknown-standard":
        raw["cases"][0]["standard_id"] = "unknown"
        raw["cases"][0]["request"]["standard_ids"] = ["unknown"]
    else:
        raw["cases"][0]["document_id"] = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(ValueError):
        DemoManifest.model_validate(raw)


@pytest.mark.parametrize(
    "field",
    [
        "whole_plan_decision",
        "document_status",
        "compliance_score",
        "finding",
        "runtime_result",
        "runtime_result_override",
    ],
)
def test_manifest_runtime_and_widened_authority_fields_are_rejected(tmp_path, field) -> None:
    raw = _raw_manifest()
    raw[field] = {"decision": "COMPLIANT"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CompetitionBootstrapError, match="forbidden"):
        load_manifest(path)


def test_wrong_evidence_requirement_and_plan_fact_selectors_fail_closed(
    clean_runtime, manifest
) -> None:
    settings = Settings(
        upload_dir=clean_runtime["uploads"],
        standards_dir=clean_runtime["standards"],
    )
    for mutation in ("evidence", "requirement", "plan-fact"):
        raw = _raw_manifest()
        case = raw["cases"][0]
        if mutation == "evidence":
            fake = "chunk_" + "0" * 64
            case["request"]["evidence_id"] = fake
            case["qualification_assertion"]["expected_evidence_id"] = fake
        elif mutation == "requirement":
            case["request"]["requirement"]["source_text"] = (
                "错" + case["request"]["requirement"]["source_text"][1:]
            )
        else:
            case["request"]["plan_facts"][0]["source_text"] = "999mm"
        attacked = _validated(raw)
        with pytest.raises(CompetitionBootstrapError, match="failed closed"):
            run_preflight(attacked, settings)


def test_expected_decision_is_assertion_not_runtime_authority(clean_runtime, manifest) -> None:
    settings = Settings(
        upload_dir=clean_runtime["uploads"],
        standards_dir=clean_runtime["standards"],
    )
    case = manifest.cases[0]
    service = _finding_service(settings)
    live = service.create(case.document_id, build_comparison_request(case)).finding
    assert live.decision == "COMPLIANT"
    wrong_assertion = case.qualification_assertion.model_copy(
        update={"expected_decision": "NON_COMPLIANT"}
    )
    attacked_case = case.model_copy(update={"qualification_assertion": wrong_assertion})
    with pytest.raises(CompetitionBootstrapError, match="decision"):
        _assert_live_finding(attacked_case, live)
    assert live.decision == "COMPLIANT"


def test_production_bootstrap_imports_no_ocr_or_llm_service() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name.startswith("app.services.ocr") for name in imported)
    assert "app.services.llm_service" not in imported


def test_no_c4_route_report_or_aggregation_authority_added(manifest) -> None:
    script = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "FindingReviewService" not in script
    assert "APIRouter" not in script
    assert "ReportBuilder" not in script
    assert "DocumentCompliance" not in script
    assert "OverallCompliance" not in script
    assert all(
        case.qualification_assertion.expected_scope == "REVIEW_UNIT_REQUIREMENT"
        for case in manifest.cases
    )
