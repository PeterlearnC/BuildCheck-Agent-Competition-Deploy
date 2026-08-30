"""Provision and preflight the three qualified competition demo cases.

This module packages existing authority only.  It does not parse standards,
run OCR/LLMs, decide compliance, or manufacture ReviewFinding responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import Settings  # noqa: E402
from app.schemas.compliance_comparison import (  # noqa: E402
    ComparisonDecision,
    ComparisonDecisionScope,
    ComparisonReasonCode,
    ComplianceComparisonRequest,
    SourceSpanSelector,
)
from app.schemas.standards import (  # noqa: E402
    StandardArticle,
    StandardChapter,
    StandardDocument,
    StandardPage,
)
from app.services.retrieval.standards_search_service import (  # noqa: E402
    StandardsSearchService,
)
from app.services.review.compliance_comparison_service import (  # noqa: E402
    ComplianceComparisonService,
)
from app.services.review.compliance_review_service import (  # noqa: E402
    ComplianceReviewService,
)
from app.services.review.plan_fact_service import PlanFactService  # noqa: E402
from app.services.review.requirement_service import RequirementService  # noqa: E402
from app.services.review.review_finding_service import ReviewFindingService  # noqa: E402
from app.services.review.review_unit_service import ReviewUnitService  # noqa: E402
from app.services.standards.scope_validation_service import (  # noqa: E402
    ScopeValidationService,
)
from app.services.standards.standard_registry_service import (  # noqa: E402
    StandardRegistryService,
)
from app.services.standards.standard_repository import StandardRepository  # noqa: E402


MANIFEST_SCHEMA_VERSION = "competition-demo-manifest-v1"
CORPUS_SCHEMA_VERSION = "qualified-parsed-corpus-v1"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
DOCUMENT_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
FORBIDDEN_AUTHORITY_KEYS = {
    "finding",
    "review_finding",
    "runtime_result",
    "runtime_result_override",
    "whole_plan_decision",
    "document_decision",
    "document_status",
    "overall_result",
    "overall_compliance",
    "pass_fail",
    "compliance_score",
    "coverage_percentage",
}
QUALIFIED_CORE_BASELINE = "7a0250a52f84fdd332775dacb2d5683877625f1a"
QUALIFIED_F1_BASELINE = "b35638577cae511ed40660bf9748c15873180726"

# This set is derived from the production semantic files introduced or modified
# by the qualified C.1, C.2, C.3, and C.4 commits.  Composition entry points
# such as app/main.py and qualification tests are intentionally excluded.
PROTECTED_CORE_AUTHORITY_PATHS = frozenset(
    {
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
)
PROTECTED_F1_AUTHORITY_PATHS = frozenset(
    {
    "competition/demo-manifest.json",
    "competition/corpus/gb55023-qualified-parse-result.json",
    }
)


class CompetitionBootstrapError(RuntimeError):
    """Fail-closed competition bootstrap error."""


class _DuplicateJsonKeyError(ValueError):
    """Internal signal raised when an authority object repeats a key."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(key)


class ExactSelector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    source_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def span_matches_text(self):
        if self.char_end != self.char_start + len(self.source_text):
            raise ValueError("Selector span does not match source_text.")
        return self


class PlanAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    description: str = Field(min_length=1)
    pdf_sha256: str = Field(pattern=SHA256_PATTERN)


class StandardAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    standard_id: str = Field(min_length=1)
    standard_code: str = Field(min_length=1)
    canonical_standard_code: str = Field(min_length=1)
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    qualified_pipeline_file_sha256: str = Field(pattern=SHA256_PATTERN)
    qualified_parse_result_sha256: str = Field(pattern=SHA256_PATTERN)
    qualified_article_count: int = Field(gt=0)
    corpus_package_path: str = Field(min_length=1)
    corpus_package_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def package_path_is_repository_relative(self):
        path = Path(self.corpus_package_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Corpus package path must remain repository-relative.")
        return self


class DemoComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    page_number: int = Field(gt=0)
    source_text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    retrieval_query: str = Field(min_length=1)
    standard_ids: list[str] = Field(min_length=1)
    top_k: int = Field(ge=1, le=20)
    evidence_id: str = Field(min_length=1)
    requirement: ExactSelector
    plan_facts: list[ExactSelector]


class QualificationAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_decision: ComparisonDecision
    expected_reason_code: ComparisonReasonCode
    expected_scope: Literal[ComparisonDecisionScope.REVIEW_UNIT_REQUIREMENT]
    expected_finding_id: str = Field(pattern=r"^finding_[0-9a-f]{64}$")
    expected_comparison_id: str = Field(pattern=r"^comparison_[0-9a-f]{64}$")
    expected_review_unit_id: str = Field(pattern=r"^reviewunit_[0-9a-f]{64}$")
    expected_evidence_id: str = Field(pattern=r"^chunk_[0-9a-f]{64}$")
    expected_requirement_id: str = Field(pattern=r"^requirement_[0-9a-f]{64}$")
    expected_plan_fact_ids: list[str]
    expected_article_id: str = Field(pattern=r"^article_[0-9a-f]{64}$")


class DemoCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^CASE-[A-Z0-9-]+$")
    document_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    standard_id: str = Field(min_length=1)
    article_number: str = Field(min_length=1)
    request: DemoComparisonRequest
    qualification_assertion: QualificationAssertion

    @model_validator(mode="after")
    def request_scope_is_exact(self):
        if self.request.standard_ids != [self.standard_id]:
            raise ValueError("Case request must use only its declared standard_id.")
        if self.request.evidence_id != self.qualification_assertion.expected_evidence_id:
            raise ValueError("Case evidence selector differs from qualification authority.")
        return self


class DemoManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[MANIFEST_SCHEMA_VERSION]
    qualified_core_baseline: str = Field(pattern=r"^[0-9a-f]{40}$")
    standard: StandardAuthority
    plans: list[PlanAuthority] = Field(min_length=1)
    cases: list[DemoCase] = Field(min_length=1)

    @model_validator(mode="after")
    def identities_are_unique_and_referenced(self):
        plan_ids = [item.document_id for item in self.plans]
        case_ids = [item.case_id for item in self.cases]
        if len(plan_ids) != len(set(plan_ids)):
            raise ValueError("Manifest contains duplicate document_id values.")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Manifest contains duplicate case_id values.")
        if len(self.cases) != 3:
            raise ValueError("Competition manifest must contain exactly three qualified cases.")
        known_plans = set(plan_ids)
        for case in self.cases:
            if case.document_id not in known_plans:
                raise ValueError("Case references an unknown plan document.")
            if case.standard_id != self.standard.standard_id:
                raise ValueError("Case references an unknown standard authority.")
        return self


class CorpusPackageAuthority(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pipeline_file_sha256: str = Field(pattern=SHA256_PATTERN)
    parse_result_canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    source_checksum: str = Field(pattern=SHA256_PATTERN)
    standard_id: str = Field(min_length=1)
    standard_code: str = Field(min_length=1)
    article_count: int = Field(gt=0)
    extraction: Literal["EXACT_PARSE_RESULT_FROM_QUALIFIED_PIPELINE_NO_REPARSE"]


class QualifiedParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document: StandardDocument
    pages: list[StandardPage]
    chapters: list[StandardChapter]
    articles: list[StandardArticle]


class QualifiedCorpusPackage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[CORPUS_SCHEMA_VERSION]
    authority: CorpusPackageAuthority
    parse_result: QualifiedParseResult


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CompetitionBootstrapError(f"Unable to read authority file: {path}") from exc
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(serialized)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except _DuplicateJsonKeyError as exc:
        raise CompetitionBootstrapError(
            f"DUPLICATE_JSON_KEY: duplicate key {exc.key!r} in authority file: {path}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompetitionBootstrapError(f"Invalid JSON authority: {path}") from exc


def _scan_forbidden_authority_keys(payload: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = key.casefold()
            if normalized in FORBIDDEN_AUTHORITY_KEYS:
                location = ".".join((*path, key))
                raise CompetitionBootstrapError(
                    f"Manifest contains forbidden runtime/document authority field: {location}"
                )
            _scan_forbidden_authority_keys(value, (*path, key))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _scan_forbidden_authority_keys(value, (*path, str(index)))


def load_manifest(path: Path) -> DemoManifest:
    raw = _load_json(path)
    _scan_forbidden_authority_keys(raw)
    try:
        return DemoManifest.model_validate(raw)
    except ValueError as exc:
        raise CompetitionBootstrapError("Competition demo manifest validation failed.") from exc


def _verify_qualified_repository(
    project_root: Path,
    *,
    expected_core_head: str,
    qualified_f1_head: str,
    protected_core_paths: frozenset[str] = PROTECTED_CORE_AUTHORITY_PATHS,
    protected_f1_paths: frozenset[str] = PROTECTED_F1_AUTHORITY_PATHS,
) -> str:
    """Verify qualified ancestry and protected authority, not product shape."""

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise CompetitionBootstrapError(
                "Unable to verify the frozen Git baseline."
            ) from exc
        return result.stdout.strip()

    def require_ancestor(ancestor: str, descendant: str, label: str) -> None:
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", ancestor, descendant],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise CompetitionBootstrapError(
                f"Runtime HEAD does not descend from qualified {label} {ancestor}."
            ) from exc

    def changed_paths(*diff_arguments: str, paths: frozenset[str]) -> set[str]:
        if not paths:
            return set()
        return {
            item.replace("\\", "/")
            for item in git(
                "diff",
                "--name-only",
                *diff_arguments,
                "--",
                *sorted(paths),
            ).splitlines()
            if item
        }

    actual = git("rev-parse", "HEAD").lower()
    require_ancestor(expected_core_head, actual, "core")
    require_ancestor(qualified_f1_head, actual, "F1 baseline")

    committed_core = changed_paths(
        f"{expected_core_head}..{actual}", paths=protected_core_paths
    )
    if committed_core:
        raise CompetitionBootstrapError(
            "Committed qualified core authority changed: "
            + ", ".join(sorted(committed_core))
        )

    committed_f1 = changed_paths(
        f"{qualified_f1_head}..{actual}", paths=protected_f1_paths
    )
    if committed_f1:
        raise CompetitionBootstrapError(
            "Committed qualified F1 authority changed: "
            + ", ".join(sorted(committed_f1))
        )

    protected_paths = protected_core_paths | protected_f1_paths
    unstaged = changed_paths(paths=protected_paths)
    if unstaged:
        raise CompetitionBootstrapError(
            "Unstaged protected authority changed: " + ", ".join(sorted(unstaged))
        )

    staged = changed_paths("--cached", paths=protected_paths)
    if staged:
        raise CompetitionBootstrapError(
            "Staged protected authority changed: " + ", ".join(sorted(staged))
        )
    return actual


def verify_core_baseline(project_root: Path, expected_head: str) -> str:
    if expected_head != QUALIFIED_CORE_BASELINE:
        raise CompetitionBootstrapError(
            "Manifest qualified core baseline differs from the qualified release authority."
        )
    return _verify_qualified_repository(
        project_root,
        expected_core_head=expected_head,
        qualified_f1_head=QUALIFIED_F1_BASELINE,
    )


def load_corpus_package(
    manifest: DemoManifest, project_root: Path
) -> QualifiedCorpusPackage:
    package_path = (project_root / manifest.standard.corpus_package_path).resolve()
    try:
        package_path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise CompetitionBootstrapError("Corpus package escaped the repository root.") from exc
    if not package_path.is_file():
        raise CompetitionBootstrapError("Qualified competition corpus package is missing.")
    if sha256_file(package_path) != manifest.standard.corpus_package_sha256:
        raise CompetitionBootstrapError("Qualified competition corpus package SHA mismatch.")

    raw = _load_json(package_path)
    parse_result = raw.get("parse_result") if isinstance(raw, dict) else None
    if canonical_sha256(parse_result) != manifest.standard.qualified_parse_result_sha256:
        raise CompetitionBootstrapError("Qualified parsed-corpus identity mismatch.")
    try:
        package = QualifiedCorpusPackage.model_validate(raw)
    except ValueError as exc:
        raise CompetitionBootstrapError("Qualified corpus package schema is invalid.") from exc

    authority = package.authority
    standard = manifest.standard
    document = package.parse_result.document
    checks = {
        "pipeline SHA": authority.pipeline_file_sha256
        == standard.qualified_pipeline_file_sha256,
        "parse-result SHA": authority.parse_result_canonical_sha256
        == standard.qualified_parse_result_sha256,
        "source SHA": authority.source_checksum == standard.source_sha256,
        "standard ID": authority.standard_id == standard.standard_id,
        "standard code": authority.standard_code == standard.standard_code,
        "article count": authority.article_count == standard.qualified_article_count,
        "document standard ID": document.standard_id == standard.standard_id,
        "document standard code": document.standard_code == standard.standard_code,
        "document canonical code": document.canonical_standard_code
        == standard.canonical_standard_code,
        "document source SHA": document.source_checksum == standard.source_sha256,
        "document article count": len(package.parse_result.articles)
        == standard.qualified_article_count,
        "official source binding": bool(
            document.official_source_binding
            and document.official_source_binding.confirmed
            and document.official_source_binding.source_checksum == standard.source_sha256
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise CompetitionBootstrapError(
            "Qualified corpus authority mismatch: " + ", ".join(failed)
        )
    if any(article.standard_id != standard.standard_id for article in package.parse_result.articles):
        raise CompetitionBootstrapError("Qualified corpus contains a foreign article owner.")
    return package


def _repository_payload(repository: StandardRepository, standard_id: str) -> dict[str, Any] | None:
    document = repository.load_document(standard_id)
    if document is None:
        return None
    return {
        "document": document.model_dump(mode="json"),
        "pages": [item.model_dump(mode="json") for item in repository.load_pages(standard_id)],
        "chapters": [
            item.model_dump(mode="json") for item in repository.load_chapters(standard_id)
        ],
        "articles": [
            item.model_dump(mode="json") for item in repository.load_articles(standard_id)
        ],
    }


def install_qualified_corpus(
    package: QualifiedCorpusPackage, settings: Settings
) -> Literal["INSTALLED", "ALREADY_VERIFIED"]:
    repository = StandardRepository(settings)
    expected = package.parse_result.model_dump(mode="json")
    standard_id = package.parse_result.document.standard_id
    existing = _repository_payload(repository, standard_id)
    if existing is not None:
        if canonical_sha256(existing) != canonical_sha256(expected):
            raise CompetitionBootstrapError(
                "Runtime repository contains conflicting GB55023 authority."
            )
        return "ALREADY_VERIFIED"

    root = repository.root
    target = repository.document_dir(standard_id)
    temporary = root / f".{standard_id}.competition-demo-{uuid4().hex}"
    try:
        root.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
        files = {
            "metadata.json": expected["document"],
            "pages.json": expected["pages"],
            "chapters.json": expected["chapters"],
            "articles.json": expected["articles"],
        }
        for filename, payload in files.items():
            (temporary / filename).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        os.replace(temporary, target)
    except OSError as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise CompetitionBootstrapError("Unable to install qualified corpus atomically.") from exc

    installed = _repository_payload(repository, standard_id)
    if installed is None or canonical_sha256(installed) != canonical_sha256(expected):
        raise CompetitionBootstrapError("Installed qualified corpus failed reread verification.")
    return "INSTALLED"


def _find_verified_asset(asset_dir: Path, plan: PlanAuthority) -> Path:
    named = asset_dir / f"{plan.document_id}.pdf"
    if named.exists():
        if not named.is_file() or sha256_file(named) != plan.pdf_sha256:
            raise CompetitionBootstrapError(
                f"Plan {plan.document_id} has the expected filename but wrong bytes."
            )
        return named
    if not asset_dir.is_dir():
        raise CompetitionBootstrapError(f"Plan asset directory is missing: {asset_dir}")
    matches = [
        candidate
        for candidate in sorted(asset_dir.glob("*.pdf"))
        if candidate.is_file() and sha256_file(candidate) == plan.pdf_sha256
    ]
    if len(matches) != 1:
        raise CompetitionBootstrapError(
            f"Plan {plan.document_id} requires exactly one SHA-matching local PDF asset."
        )
    return matches[0]


def provision_plans(
    manifest: DemoManifest, asset_dir: Path, upload_dir: Path
) -> dict[str, str]:
    results: dict[str, str] = {}
    upload_dir.mkdir(parents=True, exist_ok=True)
    for plan in manifest.plans:
        destination = upload_dir / f"{plan.document_id}.pdf"
        if destination.exists():
            if not destination.is_file() or sha256_file(destination) != plan.pdf_sha256:
                raise CompetitionBootstrapError(
                    f"Runtime plan {plan.document_id} conflicts with qualified authority."
                )
            results[plan.document_id] = "ALREADY_VERIFIED"
            continue
        source = _find_verified_asset(asset_dir, plan)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != plan.pdf_sha256:
                raise CompetitionBootstrapError(
                    f"Copied plan {plan.document_id} failed SHA verification."
                )
            os.replace(temporary, destination)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise CompetitionBootstrapError(
                f"Unable to provision plan {plan.document_id}."
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        results[plan.document_id] = "INSTALLED"
    return results


def build_comparison_request(case: DemoCase) -> ComplianceComparisonRequest:
    request = case.request
    return ComplianceComparisonRequest(
        page_number=request.page_number,
        source_text=request.source_text,
        char_start=request.char_start,
        retrieval_query=request.retrieval_query,
        standard_ids=request.standard_ids,
        top_k=request.top_k,
        evidence_id=request.evidence_id,
        requirement=SourceSpanSelector(**request.requirement.model_dump()),
        plan_facts=[
            SourceSpanSelector(**selector.model_dump()) for selector in request.plan_facts
        ],
    )


def _finding_service(settings: Settings) -> ReviewFindingService:
    repository = StandardRepository(settings)
    registry = StandardRegistryService(settings)
    review_units = ReviewUnitService(settings=settings)
    search = StandardsSearchService(repository=repository, registry=registry)
    scope = ScopeValidationService(repository=repository, registry=registry)
    reviews = ComplianceReviewService(
        search_service=search,
        scope_validator=scope,
        review_unit_service=review_units,
    )
    comparisons = ComplianceComparisonService(
        RequirementService(), PlanFactService(review_units)
    )
    return ReviewFindingService(
        settings=settings,
        review_unit_service=review_units,
        compliance_review_service=reviews,
        comparison_service=comparisons,
    )


def _assert_live_finding(case: DemoCase, finding) -> None:
    expected = case.qualification_assertion
    actual = {
        "decision": finding.decision,
        "reason": finding.reason_code,
        "scope": finding.decision_scope,
        "finding_id": finding.finding_id,
        "comparison_id": finding.comparison_id,
        "review_unit_id": finding.review_unit_id,
        "evidence_id": finding.evidence_id,
        "requirement_id": finding.requirement_id,
        "plan_fact_ids": finding.plan_fact_ids,
        "article_id": finding.standard_citation.article_id,
        "article_number": finding.standard_citation.article_number,
    }
    required = {
        "decision": expected.expected_decision,
        "reason": expected.expected_reason_code,
        "scope": ComparisonDecisionScope.REVIEW_UNIT_REQUIREMENT,
        "finding_id": expected.expected_finding_id,
        "comparison_id": expected.expected_comparison_id,
        "review_unit_id": expected.expected_review_unit_id,
        "evidence_id": expected.expected_evidence_id,
        "requirement_id": expected.expected_requirement_id,
        "plan_fact_ids": expected.expected_plan_fact_ids,
        "article_id": expected.expected_article_id,
        "article_number": case.article_number,
    }
    mismatches = [key for key in required if actual[key] != required[key]]
    if mismatches:
        raise CompetitionBootstrapError(
            f"{case.case_id} live C.3 authority mismatch: " + ", ".join(mismatches)
        )


def run_preflight(manifest: DemoManifest, settings: Settings) -> list[dict[str, Any]]:
    service = _finding_service(settings)
    plan_by_id = {plan.document_id: plan for plan in manifest.plans}
    reports: list[dict[str, Any]] = []
    for case in manifest.cases:
        try:
            response = service.create(case.document_id, build_comparison_request(case))
        except Exception as exc:
            raise CompetitionBootstrapError(
                f"{case.case_id} live C.3 reconstruction failed closed."
            ) from exc
        finding = response.finding
        _assert_live_finding(case, finding)
        plan = plan_by_id[case.document_id]
        if finding.document_sha256 != plan.pdf_sha256:
            raise CompetitionBootstrapError(
                f"{case.case_id} Finding PDF SHA differs from plan authority."
            )
        if finding.standard_citation.source_checksum != manifest.standard.source_sha256:
            raise CompetitionBootstrapError(
                f"{case.case_id} standard source SHA differs from qualified authority."
            )
        reports.append(
            {
                "case_id": case.case_id,
                "status": "PASS",
                "document_id": finding.document_id,
                "document_sha256": finding.document_sha256,
                "standard_id": finding.standard_citation.standard_id,
                "standard_source_sha256": finding.standard_citation.source_checksum,
                "evidence_id": finding.evidence_id,
                "finding_id": finding.finding_id,
                "decision": finding.decision.value,
                "reason_code": finding.reason_code.value,
                "decision_scope": finding.decision_scope.value,
                "plan_citation": finding.plan_citation.model_dump(mode="json"),
                "standard_citation": finding.standard_citation.model_dump(mode="json"),
                "summary": finding.summary,
                "manifest_expected_result_used_as_runtime_finding": False,
            }
        )
    return reports


def prepare_runtime(
    *,
    project_root: Path,
    manifest_path: Path,
    asset_dir: Path,
    upload_dir: Path,
    standards_dir: Path,
    preflight: bool,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    verified_head = verify_core_baseline(project_root, manifest.qualified_core_baseline)
    package = load_corpus_package(manifest, project_root)
    settings = Settings(upload_dir=upload_dir, standards_dir=standards_dir)
    plan_status = provision_plans(manifest, asset_dir, upload_dir)
    corpus_status = install_qualified_corpus(package, settings)
    repository = StandardRepository(settings)
    documents = repository.list_documents()
    if not any(item.standard_id == manifest.standard.standard_id for item in documents):
        raise CompetitionBootstrapError("Qualified GB55023 is unavailable after bootstrap.")
    result: dict[str, Any] = {
        "status": "PASS",
        "manifest_schema_version": manifest.schema_version,
        "verified_core_baseline": verified_head,
        "plan_provision": plan_status,
        "corpus_provision": corpus_status,
        "parsed_standard_count": len(documents),
        "qualified_standard_id": manifest.standard.standard_id,
        "qualified_article_count": len(
            repository.load_articles(manifest.standard.standard_id)
        ),
        "ocr_calls": 0,
        "external_llm_calls": 0,
        "prebuilt_findings_used": 0,
    }
    if preflight:
        result["cases"] = run_preflight(manifest, settings)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision and preflight the verified BuildCheck competition demo."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "competition" / "demo-manifest.json",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=BACKEND_DIR / "data" / "uploads",
        help="Local directory containing operator-provided real plan PDFs.",
    )
    parser.add_argument(
        "--upload-dir", type=Path, default=BACKEND_DIR / "data" / "uploads"
    )
    parser.add_argument(
        "--standards-dir", type=Path, default=BACKEND_DIR / "data" / "standards"
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run all three live production C.3 reconstructions after provisioning.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = prepare_runtime(
            project_root=PROJECT_ROOT,
            manifest_path=arguments.manifest.resolve(),
            asset_dir=arguments.assets_dir.resolve(),
            upload_dir=arguments.upload_dir.resolve(),
            standards_dir=arguments.standards_dir.resolve(),
            preflight=arguments.preflight,
        )
    except CompetitionBootstrapError as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
