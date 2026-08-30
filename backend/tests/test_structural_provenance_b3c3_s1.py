from dataclasses import replace

import pytest

from app.schemas.ocr import SourceKind
from app.schemas.standards import DocumentRegionType
from app.schemas.standards import StandardChapter, StandardDocument, StandardPage
from app.schemas.structural import (
    StructuralEvidenceKind,
    StructuralEvidenceProducer,
    StructuralEvidenceRole,
    StructuralSourceReference,
    StructuralTransitionProvenance,
    StructuralValidationBasis,
    ValidatedStructuralEvidence,
)
from app.services.standards.structural_evidence_resolver import (
    STRUCTURAL_RESOLVER_VERSION,
    StructuralEvidenceCandidate,
    StructuralEvidenceResolverService,
)
from app.services.standards.article_merge_service import ArticleMergeService
from app.services.standards.standard_chapter_service import ChapterPosition
from app.services.standards.standard_parser_service import StandardParserService


CHECKSUM = "a" * 64


def _pdf_reference(order: int = 0) -> StructuralSourceReference:
    return StructuralSourceReference(
        page_number=2,
        source_order_index=order,
        parser_line_index=order,
    )


def _ocr_reference(order: int = 0) -> StructuralSourceReference:
    return StructuralSourceReference(
        page_number=2,
        source_order_index=order,
        raw_line_id=f"ocrline_{order}",
        polygon=[[10, 10], [90, 10], [90, 30], [10, 30]],
    )


def _candidate(
    *,
    source_kind: SourceKind = SourceKind.PDF_TEXT,
    producer: StructuralEvidenceProducer = StructuralEvidenceProducer.PDF_TEXT,
    kind: StructuralEvidenceKind = StructuralEvidenceKind.CHAPTER,
    number: str = "3",
    title: str = "Safety controls",
    region: DocumentRegionType = DocumentRegionType.NORMATIVE_BODY,
    fragments: tuple[str, ...] = ("3 Safety controls",),
    references: tuple[StructuralSourceReference, ...] | None = None,
    previous: str | None = None,
    following: str | None = "3.1.1",
    inside: bool = False,
    transition: StructuralTransitionProvenance | None = None,
    bases: tuple[StructuralValidationBasis, ...] = (
        StructuralValidationBasis.EXPLICIT_HEADING_SYNTAX,
    ),
) -> StructuralEvidenceCandidate:
    if references is None:
        references = (
            _ocr_reference() if source_kind == SourceKind.OCR_TEXT else _pdf_reference(),
        )
    return StructuralEvidenceCandidate(
        source_checksum=CHECKSUM,
        source_kind=source_kind,
        producer=producer,
        structure_kind=kind,
        structure_number=number,
        structure_title=title,
        region_role=region,
        source_fragments=fragments,
        source_references=references,
        previous_article_number=previous,
        next_article_number=following,
        inside_article_interval=inside,
        transition=transition,
        validation_basis=bases,
    )


def test_resolver_version_is_frozen_for_s1() -> None:
    assert STRUCTURAL_RESOLVER_VERSION == "v0.4-b.3c.3-s1-r1-structural"


def test_front_matter_copy_is_rejected_but_body_copy_validates() -> None:
    resolver = StructuralEvidenceResolverService()
    heading = _candidate(
        kind=StructuralEvidenceKind.SECTION,
        number="3.1",
        title="General controls",
        fragments=("3.1 General controls",),
        following="3.1.1",
    )

    assert resolver.resolve(
        replace(heading, region_role=DocumentRegionType.FRONT_MATTER)
    ) is None
    evidence = resolver.resolve(heading)
    assert evidence is not None
    assert evidence.region_role == DocumentRegionType.NORMATIVE_BODY


@pytest.mark.parametrize(
    ("candidate", "expected_bases"),
    [
        (
            _candidate(following="3.1.1"),
            {
                StructuralValidationBasis.REGION_ROLE,
                StructuralValidationBasis.NUMBER_GRAMMAR,
                StructuralValidationBasis.SOURCE_ORDER,
                StructuralValidationBasis.ARTICLE_PREFIX,
            },
        ),
        (
            _candidate(
                kind=StructuralEvidenceKind.SECTION,
                number="3.10",
                title="Electrical controls",
                fragments=("3.10 Electrical controls",),
                previous="3.9.4",
                following="3.10.1",
            ),
            {
                StructuralValidationBasis.ARTICLE_PREFIX,
                StructuralValidationBasis.PREFIX_TRANSITION,
            },
        ),
        (
            _candidate(
                number="4",
                title="Environmental controls",
                fragments=("4 Environmental controls",),
                previous="3.15.3",
                following="4.0.1",
            ),
            {
                StructuralValidationBasis.ARTICLE_PREFIX,
                StructuralValidationBasis.PREFIX_TRANSITION,
            },
        ),
    ],
)
def test_true_numeric_structures_validate_with_auditable_bases(
    candidate: StructuralEvidenceCandidate,
    expected_bases: set[StructuralValidationBasis],
) -> None:
    evidence = StructuralEvidenceResolverService().resolve(candidate)
    assert evidence is not None
    assert expected_bases.issubset(set(evidence.validation_basis))


def test_structural_evidence_is_never_normative_article_text() -> None:
    evidence = StructuralEvidenceResolverService().resolve(_candidate())
    assert evidence is not None
    assert evidence.evidence_role == StructuralEvidenceRole.STRUCTURAL_ONLY
    assert evidence.accepted_as_normative is False
    assert "parser_boundary_index" not in ValidatedStructuralEvidence.model_fields


def test_ocr_and_pdf_references_share_one_semantic_resolver() -> None:
    resolver = StructuralEvidenceResolverService()
    pdf = resolver.resolve(_candidate())
    ocr = resolver.resolve(
        _candidate(
            source_kind=SourceKind.OCR_TEXT,
            producer=StructuralEvidenceProducer.OCR_TRANSITION,
            references=(_ocr_reference(),),
            transition=StructuralTransitionProvenance(
                issue_codes=["STRUCTURAL_TRANSITION"],
                next_article_number="3.1.1",
            ),
            bases=(StructuralValidationBasis.OCR_TRANSITION,),
        )
    )

    assert pdf is not None and ocr is not None
    assert pdf.source_references[0].raw_line_id is None
    assert pdf.source_references[0].polygon is None
    assert pdf.source_references[0].parser_line_index == 0
    assert ocr.source_references[0].raw_line_id == "ocrline_0"
    assert ocr.source_references[0].polygon is not None
    assert ocr.source_references[0].parser_line_index is None


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(
            source_kind=SourceKind.OCR_TEXT,
            producer=StructuralEvidenceProducer.OCR_CONTEXT,
            references=(_pdf_reference(),),
        ),
        _candidate(
            references=(
                StructuralSourceReference(page_number=2, source_order_index=0),
            ),
        ),
    ],
)
def test_source_specific_reference_requirements_fail_closed(
    candidate: StructuralEvidenceCandidate,
) -> None:
    assert StructuralEvidenceResolverService().resolve(candidate) is None


@pytest.mark.parametrize(
    ("kind", "number", "title", "text", "following"),
    [
        (StructuralEvidenceKind.APPENDIX, "A", "Test methods", "A Test methods", "A.1.1"),
        (
            StructuralEvidenceKind.APPENDIX_SECTION,
            "A.1",
            "Loading tests",
            "A.1 Loading tests",
            "A.1.1",
        ),
    ],
)
def test_appendix_structures_require_appendix_region(
    kind: StructuralEvidenceKind,
    number: str,
    title: str,
    text: str,
    following: str,
) -> None:
    candidate = _candidate(
        kind=kind,
        number=number,
        title=title,
        fragments=(text,),
        region=DocumentRegionType.APPENDIX,
        following=following,
    )
    resolver = StructuralEvidenceResolverService()
    assert resolver.resolve(candidate) is not None
    assert resolver.resolve(
        replace(candidate, region_role=DocumentRegionType.NORMATIVE_BODY)
    ) is None


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(
            number="1",
            title="Install supports without terminal punctuation",
            fragments=("1 Install supports without terminal punctuation",),
            previous="3.10.3",
            following="3.10.4",
            inside=True,
        ),
        _candidate(
            number="2",
            title="Wrapped normative requirement continues",
            fragments=("2 Wrapped normative requirement continues",),
            previous="3.10.3",
            following="3.10.4",
        ),
        _candidate(
            number="3",
            title="A long prose fragment that remains ordinary article content",
            fragments=(
                "3 A long prose fragment that remains ordinary article content",
            ),
            previous="3.12.5",
            following="3.13.1",
        ),
    ],
)
def test_number_plus_prose_is_not_promoted_without_a_real_prefix_transition(
    candidate: StructuralEvidenceCandidate,
) -> None:
    assert StructuralEvidenceResolverService().resolve(candidate) is None


def test_candidate_incompatible_with_next_article_is_rejected() -> None:
    candidate = _candidate(following="4.0.1")
    assert StructuralEvidenceResolverService().resolve(candidate) is None


def test_candidate_compatible_with_previous_article_is_not_a_new_transition() -> None:
    candidate = _candidate(previous="3.9.4", following="3.10.1")
    assert StructuralEvidenceResolverService().resolve(candidate) is None


@pytest.mark.parametrize(
    "candidate",
    [
        _candidate(title="", fragments=("3",)),
        _candidate(number="4", fragments=("3 Safety controls",)),
        _candidate(region=DocumentRegionType.EXPLANATION),
        _candidate(
            references=(_pdf_reference(2), _pdf_reference(1)),
            fragments=("3", "Safety controls"),
        ),
        _candidate(
            references=(_pdf_reference(0), _pdf_reference(2)),
            fragments=("3", "Safety controls"),
        ),
    ],
)
def test_missing_ambiguous_or_ungrounded_source_fails_closed(
    candidate: StructuralEvidenceCandidate,
) -> None:
    assert StructuralEvidenceResolverService().resolve(candidate) is None


def test_ambiguous_ocr_transition_without_corroboration_fails_closed() -> None:
    candidate = _candidate(
        source_kind=SourceKind.OCR_TEXT,
        producer=StructuralEvidenceProducer.OCR_TRANSITION,
        references=(_ocr_reference(),),
        following=None,
        transition=StructuralTransitionProvenance(
            issue_codes=["STRUCTURAL_TRANSITION"],
            ambiguous=True,
        ),
        bases=(StructuralValidationBasis.OCR_TRANSITION,),
    )
    assert StructuralEvidenceResolverService().resolve(candidate) is None


def test_multifragment_heading_requires_contiguous_explicit_references() -> None:
    candidate = _candidate(
        number="4",
        title="Safety controls",
        fragments=("4", "Safety controls"),
        references=(_pdf_reference(4), _pdf_reference(5)),
        previous="3.2.1",
        following="4.0.1",
    )
    evidence = StructuralEvidenceResolverService().resolve(candidate)
    assert evidence is not None
    assert evidence.source_text == "4\nSafety controls"
    assert [item.source_order_index for item in evidence.source_references] == [4, 5]


@pytest.mark.parametrize(
    ("structure", "article", "expected"),
    [
        ("3", "3.10.3", True),
        ("3.10", "3.10.3", True),
        ("2", "4.0.1", False),
        ("3.10", "3.11.1", False),
        ("3.10.3", "3.10.3", False),
    ],
)
def test_article_prefix_compatibility(
    structure: str, article: str, expected: bool
) -> None:
    assert StructuralEvidenceResolverService.is_compatible(structure, article) is expected


def test_evidence_identity_is_deterministic_and_semantically_sensitive() -> None:
    resolver = StructuralEvidenceResolverService()
    reference = _pdf_reference()
    arguments = {
        "source_checksum": CHECKSUM,
        "structure_kind": StructuralEvidenceKind.CHAPTER,
        "structure_number": "3",
        "structure_title": "Safety controls",
        "source_references": (reference,),
    }
    first = resolver.evidence_id(**arguments)
    second = resolver.evidence_id(**arguments)

    assert first == second
    assert first == resolver.evidence_id(
        **{**arguments, "structure_number": "03", "structure_title": "Safety   controls"}
    )
    assert first != resolver.evidence_id(
        **{**arguments, "structure_title": "Different controls"}
    )
    assert first != resolver.evidence_id(
        **{**arguments, "source_references": (_pdf_reference(1),)}
    )
    assert first != resolver.evidence_id(
        **arguments, resolver_version="v0.4-b.3c.3-s1-r2-structural"
    )


def test_resolve_many_orders_validated_evidence_by_source() -> None:
    resolver = StructuralEvidenceResolverService()
    later = _candidate(
        number="4",
        title="Later controls",
        fragments=("4 Later controls",),
        references=(_pdf_reference(8),),
        previous="3.1.1",
        following="4.0.1",
    )
    earlier = _candidate(references=(_pdf_reference(2),))

    evidence = resolver.resolve_many([later, earlier])
    assert [item.structure_number for item in evidence] == ["3", "4"]


def _document() -> StandardDocument:
    return StandardDocument(
        standard_id="structural-wiring",
        source_filename="synthetic.pdf",
        source_checksum=CHECKSUM,
        page_count=3,
    )


def test_pdf_pipeline_separates_contents_and_body_structural_copies() -> None:
    pages = [
        StandardPage(
            page_number=1,
            text="Contents\n1.1 Generic controls ...... 2",
        ),
        StandardPage(
            page_number=2,
            text=(
                "第1章 Root controls\n"
                "1.0.1 Introductory body.\n"
                "1.1 Generic controls\n"
                "1.1.1 Grounded normative requirement."
            ),
        ),
    ]

    result = StandardParserService().parse(_document(), pages)

    assert [item.chapter_number for item in result.chapters] == ["1", "1.1"]
    assert all(item.structural_evidence is not None for item in result.chapters)
    assert all(item.source_page_start == 2 for item in result.chapters)
    normative_text = "\n".join(article.source_text for article in result.articles)
    assert "Generic controls" not in normative_text
    assert "Root controls" not in normative_text


def test_numbered_continuation_remains_owned_until_next_article() -> None:
    pages = [
        StandardPage(
            page_number=1,
            text=(
                "第1章 Root controls\n"
                "1.0.1 Root article.\n"
                "第3章 Operational controls\n"
                "3.10 Electrical controls\n"
                "3.10.3 Requirements include:\n"
                "1 first numbered requirement\n"
                "wrapped continuation without punctuation\n"
                "2 second numbered requirement"
            ),
        ),
        StandardPage(
            page_number=2,
            text="next-page continuation\n3.10.4 Next grounded article.",
        ),
    ]

    result = StandardParserService().parse(_document(), pages)
    article = next(item for item in result.articles if item.article_number == "3.10.3")

    assert article.source_page_end == 2
    assert "1 first numbered requirement" in article.content
    assert "wrapped continuation" in article.content
    assert "2 second numbered requirement" in article.content
    assert "next-page continuation" in article.content
    assert article.chapter_number == "3.10"


def test_missing_compatible_structure_preserves_article_without_ownership() -> None:
    pages = [
        StandardPage(
            page_number=1,
            text="第1章 Root controls\n1.0.1 Root article.\n4.0.1 Grounded orphan.",
        )
    ]

    result = StandardParserService().parse(_document(), pages)
    orphan = next(item for item in result.articles if item.article_number == "4.0.1")

    assert orphan.chapter_id is None
    assert orphan.chapter_number is None
    assert orphan.content == "Grounded orphan."


def test_validated_chapter_and_section_transitions_flush_normally() -> None:
    pages = [
        StandardPage(
            page_number=1,
            text=(
                "第1章 Root controls\n"
                "1.0.1 Root article.\n"
                "第3章 Operational controls\n"
                "3.9 Prior section\n"
                "3.9.1 Prior article.\n"
                "3.10 New section\n"
                "3.10.1 New article.\n"
                "第4章 Later controls\n"
                "4.0.1 Later article."
            ),
        )
    ]

    result = StandardParserService().parse(_document(), pages)
    articles = {item.article_number: item for item in result.articles}

    assert articles["3.9.1"].chapter_number == "3.9"
    assert articles["3.10.1"].chapter_number == "3.10"
    assert articles["4.0.1"].chapter_number == "4"
    assert articles["3.9.1"].content == "Prior article."


def test_unvalidated_structure_position_cannot_flush_active_article() -> None:
    pages = [
        StandardPage(
            page_number=1,
            text="1.0.1 Grounded body.\n2 candidate prose\ncontinued body",
        )
    ]
    legacy = StandardChapter(
        chapter_id="legacy",
        standard_id="structural-wiring",
        chapter_number="2",
        chapter_title="candidate prose",
        level=1,
    )

    articles = ArticleMergeService().merge(
        "structural-wiring",
        pages,
        [ChapterPosition(1, 1, legacy)],
        standard_source_checksum=CHECKSUM,
    )

    assert len(articles) == 1
    assert "candidate prose" in articles[0].content
    assert "continued body" in articles[0].content
