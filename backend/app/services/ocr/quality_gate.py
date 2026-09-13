"""Conservative deterministic quality checks for raw OCR geometry and structure."""

import math
import re
from statistics import median

from app.schemas.ocr import (
    OCRIssueCode,
    OCRPageQualityAssessment,
    OCRQualityIssue,
    OCRQualityState,
    OCRRun,
    OCRRunQualityAssessment,
    OfficialSourceBinding,
)
from app.services.ocr.article_structure import OCRArticleStructureService
from app.services.ocr.run_identity import OCR_QUALITY_GATE_VERSION, validate_ocr_run_identity


class OCRQualityGate:
    # Provider confidence is a probability-like [0, 1] signal, not a semantic
    # decision.  Values below REVIEW_THRESHOLD are rejected outright; values
    # below ACCEPT_THRESHOLD require review.  Neither class may enter the
    # normative parser channel.  The two boundaries intentionally distinguish
    # unusable recognition from auditable review-required recognition.
    CONFIDENCE_REJECT_THRESHOLD = 0.50
    CONFIDENCE_ACCEPT_THRESHOLD = 0.80
    ARTICLE = re.compile(r"^\d+\.\d+\.\d+(?:\.\d+)?(?:-\d+)?(?:\s|$)")
    ARTICLE_LIKE = re.compile(r"^\d+(?:\.\d+){2,4}(?:\s|$)")
    FUSED_ARTICLE_LIKE = re.compile(r"^(?P<major>\d+)\.(?P<tail>\d{2,})(?:\s|$)")
    LIST_ITEM = re.compile(r"^(?P<number>\d{1,2})(?:\s|[、.)）])")
    STANDALONE_NUMBER = re.compile(r"^\d{1,2}$")
    # The height boundary matches the existing conservative geometry-outlier
    # boundary but cannot reject alone. The aspect bound limits the rule to
    # compact shapes, and the 25% ratio requires a drastic page-relative ink
    # deficit. The combined boundary separated the diagnosed phantom from all
    # accepted lines in the two qualification corpora; dark negative controls
    # exercise the same geometry without being rejected.
    RASTER_COMPACT_MIN_HEIGHT_RATIO = 1.35
    RASTER_COMPACT_MAX_ASPECT_RATIO = 3.0
    RASTER_MAX_RELATIVE_SUPPORT_RATIO = 0.25

    def __init__(
        self, article_structure: OCRArticleStructureService | None = None
    ) -> None:
        self.article_structure = article_structure or OCRArticleStructureService()

    def assess(
        self, run: OCRRun, binding: OfficialSourceBinding | None = None
    ) -> OCRRunQualityAssessment:
        validate_ocr_run_identity(run)
        pages = [self._assess_page(page) for page in run.pages]
        self._apply_article_safety(run, pages)
        if binding is not None and (
            not binding.confirmed
            or binding.source_checksum.lower() != run.official_source_checksum.lower()
        ):
            for page in pages:
                page.issues.append(
                    OCRQualityIssue(
                        code=OCRIssueCode.IDENTITY_CONFLICT,
                        reason="Curated official-source binding does not match the OCR source.",
                    )
                )
                page.state = OCRQualityState.OCR_REJECTED
                page.accepted_line_ids = []
        if not pages or all(page.state == OCRQualityState.OCR_REJECTED for page in pages):
            state = OCRQualityState.OCR_REJECTED
        elif any(page.state != OCRQualityState.OCR_ACCEPTED for page in pages):
            state = OCRQualityState.OCR_LOW_CONFIDENCE
        else:
            state = OCRQualityState.OCR_ACCEPTED
        return OCRRunQualityAssessment(
            ocr_run_id=run.ocr_run_id,
            execution_id=run.execution_id,
            state=state,
            pages=pages,
            quality_gate_version=OCR_QUALITY_GATE_VERSION,
        )

    def _apply_article_safety(
        self, run: OCRRun, pages: list[OCRPageQualityAssessment]
    ) -> None:
        intervals = self.article_structure.intervals(run)
        transitions = self.article_structure.transitions(run)
        assessment_by_page = {page.page_number: page for page in pages}
        raw_line_by_id = {
            line.line_id: line for page in run.pages for line in page.lines
        }
        for transition in transitions:
            trigger_ids = list(transition.structural_heading_line_ids) or [
                transition.line_ids[0]
            ]
            if transition.next_article_heading_id:
                trigger_ids.append(transition.next_article_heading_id)
            assessment_by_page[transition.page_number].issues.append(
                OCRQualityIssue(
                    code=OCRIssueCode.STRUCTURAL_TRANSITION,
                    reason=(
                        "Validated or fail-closed structural transition text is "
                        "not normative article body."
                    ),
                    triggering_line_ids=list(dict.fromkeys(trigger_ids)),
                    affected_article_heading_id=transition.previous_article_heading_id,
                    affected_line_ids=list(transition.line_ids),
                    affected_polygons=[
                        raw_line_by_id[line_id].polygon
                        for line_id in transition.line_ids
                    ],
                    structural_numbers=list(transition.structural_numbers),
                    previous_article_heading_id=(
                        transition.previous_article_heading_id
                    ),
                    next_article_heading_id=transition.next_article_heading_id,
                )
            )
        for transition in self.article_structure.section_start_gaps(transitions):
            assessment_by_page[transition.page_number].issues.append(
                OCRQualityIssue(
                    code=OCRIssueCode.SECTION_ARTICLE_START_GAP,
                    reason=(
                        "The first recognized article in a validated section does "
                        "not start at one; missing headings are not reconstructed."
                    ),
                    triggering_line_ids=list(
                        dict.fromkeys(
                            [
                                *transition.structural_heading_line_ids,
                                *(
                                    [transition.next_article_heading_id]
                                    if transition.next_article_heading_id
                                    else []
                                ),
                            ]
                        )
                    ),
                    affected_article_heading_id=transition.next_article_heading_id,
                    affected_line_ids=list(transition.line_ids),
                    affected_polygons=[
                        raw_line_by_id[line_id].polygon
                        for line_id in transition.line_ids
                    ],
                    structural_numbers=list(transition.structural_numbers),
                    previous_article_heading_id=(
                        transition.previous_article_heading_id
                    ),
                    next_article_heading_id=transition.next_article_heading_id,
                )
            )
        quarantine_codes = {
            OCRIssueCode.ARTICLE_NUMBER_AMBIGUITY,
            OCRIssueCode.STRUCTURAL_SEQUENCE_GAP,
            OCRIssueCode.READING_ORDER_ANOMALY,
            OCRIssueCode.CONFIDENCE_REJECTED,
            OCRIssueCode.CONFIDENCE_REVIEW_REQUIRED,
        }
        for page in pages:
            for issue in page.issues:
                if issue.code not in quarantine_codes:
                    continue
                original_ids = list(issue.affected_line_ids)
                interval = next(
                    (
                        candidate
                        for candidate in intervals
                        if (
                            issue.affected_article_heading_id
                            == candidate.heading_line_id
                            or any(
                                line_id in candidate.line_ids
                                for line_id in original_ids
                            )
                        )
                    ),
                    None,
                )
                if interval is None:
                    continue
                issue.triggering_line_ids = (
                    issue.triggering_line_ids or original_ids
                )
                issue.affected_article_heading_id = interval.heading_line_id
                issue.affected_line_ids = list(interval.line_ids)

        for current, following in self.article_structure.sequence_gaps(intervals):
            assessment_by_page[current.heading_page_number].issues.append(
                OCRQualityIssue(
                    code=OCRIssueCode.ARTICLE_SEQUENCE_GAP,
                    reason=(
                        "Recognized article headings are non-consecutive; the "
                        "intervening structural ownership is uncertain and missing "
                        "headings are not reconstructed."
                    ),
                    triggering_line_ids=[
                        current.heading_line_id,
                        following.heading_line_id,
                    ],
                    affected_article_heading_id=current.heading_line_id,
                    affected_line_ids=list(current.line_ids),
                )
            )

        affected_ids = {
            line_id
            for page in pages
            for issue in page.issues
            for line_id in issue.affected_line_ids
        }
        for raw_page, page_assessment in zip(run.pages, pages, strict=True):
            raw_ids = [line.line_id for line in raw_page.lines]
            page_assessment.accepted_line_ids = [
                line_id for line_id in raw_ids if line_id not in affected_ids
            ]
            if not page_assessment.accepted_line_ids:
                page_assessment.state = OCRQualityState.OCR_REJECTED
            elif page_assessment.issues or set(raw_ids) & affected_ids:
                page_assessment.state = OCRQualityState.OCR_LOW_CONFIDENCE
            else:
                page_assessment.state = OCRQualityState.OCR_ACCEPTED

    def _assess_page(self, page) -> OCRPageQualityAssessment:
        issues: list[OCRQualityIssue] = []
        if page.error or not page.lines:
            issues.append(
                OCRQualityIssue(
                    code=OCRIssueCode.EMPTY_OR_FAILED_OCR,
                    reason=page.error or "OCR returned no recognized lines.",
                )
            )
            return OCRPageQualityAssessment(
                page_number=page.page_number,
                state=OCRQualityState.OCR_REJECTED,
                issues=issues,
                accepted_line_ids=[],
            )

        meaningful_chars = sum(len(re.sub(r"\s+", "", line.raw_text)) for line in page.lines)
        if meaningful_chars < 12:
            issues.append(
                OCRQualityIssue(
                    code=OCRIssueCode.LOW_TEXT_COVERAGE,
                    reason="Recognized text is too sparse for controlled parser entry.",
                    affected_line_ids=[line.line_id for line in page.lines],
                )
            )

        line_ids = [line.line_id for line in page.lines]
        if len(set(line_ids)) != len(line_ids):
            issues.append(
                OCRQualityIssue(
                    code=OCRIssueCode.SOURCE_MAPPING_FAILURE,
                    reason="Raw OCR line identities are not unique within the page.",
                    affected_line_ids=line_ids,
                )
            )

        heights = [self._height(line.polygon) for line in page.lines if self._height(line.polygon) > 0]
        typical_height = median(heights) if heights else 1.0
        article_prefixes = {
            ".".join(match.group(0).strip().split(".")[:2])
            for line in page.lines
            if (match := self.ARTICLE.match(line.raw_text.strip()))
        }
        for line in page.lines:
            if line.confidence < self.CONFIDENCE_REJECT_THRESHOLD:
                issues.append(
                    OCRQualityIssue(
                        code=OCRIssueCode.CONFIDENCE_REJECTED,
                        reason=(
                            "OCR line confidence is below the deterministic "
                            "rejection boundary "
                            f"({line.confidence:.6f} < "
                            f"{self.CONFIDENCE_REJECT_THRESHOLD:.6f})."
                        ),
                        affected_line_ids=[line.line_id],
                        affected_polygons=[line.polygon],
                    )
                )
            elif line.confidence < self.CONFIDENCE_ACCEPT_THRESHOLD:
                issues.append(
                    OCRQualityIssue(
                        code=OCRIssueCode.CONFIDENCE_REVIEW_REQUIRED,
                        reason=(
                            "OCR line confidence requires human review before "
                            "normative parser entry "
                            f"({line.confidence:.6f} < "
                            f"{self.CONFIDENCE_ACCEPT_THRESHOLD:.6f})."
                        ),
                        affected_line_ids=[line.line_id],
                        affected_polygons=[line.polygon],
                    )
                )
            height = self._height(line.polygon)
            width = self._width(line.polygon)
            angle = abs(self._top_edge_angle(line.polygon))
            rotated_outlier = angle >= 8.0 and height >= typical_height * 1.35
            large_compact_outlier = (
                height >= typical_height * 2.35 and width <= height * 3.0
            )
            if rotated_outlier or large_compact_outlier:
                issues.append(
                    OCRQualityIssue(
                        code=OCRIssueCode.SUSPICIOUS_OVERLAY_GEOMETRY,
                        reason=(
                            "Line geometry is inconsistent with surrounding reading bands "
                            f"(angle={angle:.1f}, height_ratio={height / typical_height:.2f})."
                        ),
                        affected_line_ids=[line.line_id],
                        affected_polygons=[line.polygon],
                    )
                )

            visual_support = line.visual_support
            if (
                visual_support is not None
                and visual_support.page_reference_support > 0
                and visual_support.height_ratio
                >= self.RASTER_COMPACT_MIN_HEIGHT_RATIO
                and visual_support.aspect_ratio
                <= self.RASTER_COMPACT_MAX_ASPECT_RATIO
                and visual_support.relative_support_ratio
                <= self.RASTER_MAX_RELATIVE_SUPPORT_RATIO
            ):
                issues.append(
                    OCRQualityIssue(
                        code=OCRIssueCode.INSUFFICIENT_RASTER_TEXT_SUPPORT,
                        reason=(
                            "Compact height-outlying OCR geometry lacks dark raster "
                            "support relative to normal page text "
                            f"(support={visual_support.dark_support:.4f}, "
                            f"reference={visual_support.page_reference_support:.4f}, "
                            f"ratio={visual_support.relative_support_ratio:.4f}, "
                            f"height_ratio={visual_support.height_ratio:.2f}, "
                            f"aspect_ratio={visual_support.aspect_ratio:.2f}, "
                            f"rule={visual_support.rule_version})."
                        ),
                        triggering_line_ids=[line.line_id],
                        affected_line_ids=[line.line_id],
                        affected_polygons=[line.polygon],
                        visual_support=visual_support,
                    )
                )

            value = line.raw_text.strip()
            fused = self.FUSED_ARTICLE_LIKE.match(value)
            fused_matches_context = bool(
                fused
                and f"{fused.group('major')}.{fused.group('tail')[0]}" in article_prefixes
            )
            if (
                (self.ARTICLE_LIKE.match(value) and not self.ARTICLE.match(value))
                or fused_matches_context
            ):
                issues.append(
                    OCRQualityIssue(
                        code=OCRIssueCode.ARTICLE_NUMBER_AMBIGUITY,
                        reason="Article-like prefix is not a structurally valid article number.",
                        affected_line_ids=[line.line_id],
                        affected_polygons=[line.polygon],
                    )
                )

        issues.extend(self._detached_number_issues(page.lines, typical_height))
        issues.extend(self._sequence_gap_issues(page))
        affected = {
            line_id
            for issue in issues
            for line_id in issue.affected_line_ids
        }
        accepted = [line.line_id for line in page.lines if line.line_id not in affected]
        if not accepted:
            state = OCRQualityState.OCR_REJECTED
        elif issues:
            state = OCRQualityState.OCR_LOW_CONFIDENCE
        else:
            state = OCRQualityState.OCR_ACCEPTED
        return OCRPageQualityAssessment(
            page_number=page.page_number,
            state=state,
            issues=issues,
            accepted_line_ids=accepted,
        )

    def _detached_number_issues(self, lines, typical_height: float) -> list[OCRQualityIssue]:
        issues = []
        for number_line in lines:
            if not self.STANDALONE_NUMBER.fullmatch(number_line.raw_text.strip()):
                continue
            number_top, number_bottom = self._vertical_bounds(number_line.polygon)
            for body_line in lines:
                if body_line.line_id == number_line.line_id:
                    continue
                body_top, body_bottom = self._vertical_bounds(body_line.polygon)
                overlap = min(number_bottom, body_bottom) - max(number_top, body_top)
                if overlap <= min(number_bottom - number_top, body_bottom - body_top) * 0.35:
                    continue
                if self._left(number_line.polygon) > (
                    self._left(body_line.polygon) + typical_height * 1.5
                ):
                    continue
                if abs(self._top_edge_angle(body_line.polygon)) >= 8.0:
                    continue
                issues.append(
                    OCRQualityIssue(
                        code=OCRIssueCode.READING_ORDER_ANOMALY,
                        reason="A detached list number overlaps a body line and cannot be ordered safely.",
                        affected_line_ids=[number_line.line_id, body_line.line_id],
                        affected_polygons=[number_line.polygon, body_line.polygon],
                    )
                )
                break
        return issues

    def _sequence_gap_issues(self, page) -> list[OCRQualityIssue]:
        groups = []
        candidates = []
        current_article = None
        for line in sorted(page.lines, key=lambda item: (self._top(item.polygon), self._left(item.polygon))):
            value = line.raw_text.strip()
            if self.ARTICLE.match(value):
                if candidates:
                    groups.append((current_article, candidates))
                candidates = []
                current_article = line
                continue
            match = self.LIST_ITEM.match(value)
            if match:
                candidates.append((int(match.group("number")), line))
            elif (
                self.STANDALONE_NUMBER.fullmatch(value)
                and self._top(line.polygon) < page.render.pixel_height * 0.88
            ):
                candidates.append((int(value), line))
        if candidates:
            groups.append((current_article, candidates))
        issues = []
        for article_line, group in groups:
            if len(group) < 4:
                continue
            for (previous, previous_line), (current, current_line) in zip(group, group[1:]):
                if current > previous + 1 and previous >= 1:
                    top = self._top(previous_line.polygon)
                    bottom = self._top(current_line.polygon)
                    between = [
                        line.line_id
                        for line in page.lines
                        if top <= self._top(line.polygon) <= bottom
                    ]
                    related = [article_line.line_id] if article_line is not None else []
                    issues.append(
                        OCRQualityIssue(
                            code=OCRIssueCode.STRUCTURAL_SEQUENCE_GAP,
                            reason=(
                                "A numeric list sequence has a gap; the missing identity is not reconstructed."
                            ),
                            triggering_line_ids=[
                                previous_line.line_id,
                                current_line.line_id,
                            ],
                            affected_article_heading_id=(
                                article_line.line_id
                                if article_line is not None
                                else None
                            ),
                            affected_line_ids=list(dict.fromkeys(
                                [*related, previous_line.line_id, *between, current_line.line_id]
                            )),
                        )
                    )
        return issues

    @staticmethod
    def _top_edge_angle(polygon: list[list[float]]) -> float:
        if len(polygon) < 2:
            return 0.0
        dx = polygon[1][0] - polygon[0][0]
        dy = polygon[1][1] - polygon[0][1]
        return math.degrees(math.atan2(dy, dx)) if dx or dy else 0.0

    @staticmethod
    def _vertical_bounds(polygon):
        values = [point[1] for point in polygon]
        return min(values), max(values)

    @classmethod
    def _height(cls, polygon):
        top, bottom = cls._vertical_bounds(polygon)
        return bottom - top

    @staticmethod
    def _width(polygon):
        values = [point[0] for point in polygon]
        return max(values) - min(values)

    @staticmethod
    def _top(polygon):
        return min(point[1] for point in polygon)

    @staticmethod
    def _left(polygon):
        return min(point[0] for point in polygon)
