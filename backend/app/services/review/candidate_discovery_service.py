"""Conservative deterministic discovery of source-grounded review candidates."""

import hashlib
import re
from collections.abc import Sequence

from app.schemas.review_candidate import (
    REVIEW_CANDIDATE_DISCOVERY_VERSION,
    CandidateSourceSpan,
    ReviewCandidate,
    ReviewCandidateClass,
    ReviewCandidateDiscoveryMethod,
    ReviewCandidateStatus,
    deterministic_candidate_id,
)
from app.services.pdf_service import PDFPageText


MAX_CANDIDATE_CHARS = 280
GENERAL_TOPIC = "一般审查目标"

_NUMBER_WITH_UNIT = re.compile(
    r"(?<![\d.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*"
    r"(?:MPa|kPa|kN|mm|cm|N|m|%|°|℃|毫米|厘米|千牛|兆帕|千帕|米)(?![A-Za-z])",
    re.IGNORECASE,
)
_NUMERIC_CONTROL_OPERATOR = re.compile(
    r"不(?:得)?(?:大于|小于|超过|高于|低于)|至少|至多|最大不得超过|"
    r"最小不得低于|控制在|控制值|上限|下限|应达到|应满足"
)
_OBLIGATION = re.compile(r"不得|严禁|必须|应当|需要|宜|应")
_INSPECTION = re.compile(r"检查|验收|检测|监测|复核")
_SAFETY = re.compile(r"安全|防护|临边|坠落|消防|警戒")
_PROCEDURE = re.compile(r"安装|拆除|搭设|施工|浇筑|吊装|开挖|回填|顺序|工序")
_MATERIAL = re.compile(r"材料|钢管|混凝土|钢筋|水泥|砂浆|型钢|扣件|木材|强度等级")
_STRUCTURAL = re.compile(r"立杆|水平杆|剪刀撑|连墙件|模板支撑|脚手架|支撑架|梁|柱|节点")
_CONSTRUCTION_CONTEXT = re.compile(
    r"施工|作业|安装|拆除|搭设|浇筑|开挖|回填|模板|脚手架|支撑|钢管|"
    r"混凝土|钢筋|构件|材料|设备|人员|现场|工序|管理"
)
_DISCOVERY_SIGNAL = re.compile(
    "|".join(
        pattern.pattern
        for pattern in (
            _NUMBER_WITH_UNIT,
            _NUMERIC_CONTROL_OPERATOR,
            _OBLIGATION,
            _INSPECTION,
            _SAFETY,
            _PROCEDURE,
            _MATERIAL,
            _STRUCTURAL,
        )
    ),
    re.IGNORECASE,
)
_PRIMARY_SEGMENT = re.compile(
    "[^。！？!?；;，," + chr(13) + chr(10) + "]+[。！？!?；;，,]?"
)

_TOPIC_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("可调托撑", re.compile(r"可调托撑")),
    ("立杆", re.compile(r"立杆")),
    ("水平杆", re.compile(r"水平杆")),
    ("剪刀撑", re.compile(r"剪刀撑")),
    ("连墙件", re.compile(r"连墙件")),
    ("模板支撑", re.compile(r"模板支撑|支撑架")),
    ("钢管", re.compile(r"钢管")),
    ("脚手架", re.compile(r"脚手架")),
    ("材料", re.compile(r"材料|混凝土|钢筋|水泥|砂浆|型钢|扣件|木材")),
    ("施工荷载", re.compile(r"施工荷载|荷载")),
    ("安全防护", re.compile(r"安全|防护|临边|坠落|消防|警戒")),
    ("检查验收", re.compile(r"检查|验收|检测|监测|复核")),
)


class CandidateDiscoveryInputError(ValueError):
    pass


class CandidateDiscoveryService:
    """Discover bounded exact spans; never retrieve evidence or decide compliance.

    Segmentation uses Chinese/ASCII sentence, clause, comma, and newline
    boundaries. A segment longer than 280 characters is reduced to a stable
    280-character window around each explicit discovery signal. Confidence is
    discovery confidence only: 0.45 base, +0.35 numeric-control evidence,
    +0.10 obligation, +0.05 construction context, +0.10 classified domain;
    it is capped at 0.95 (0.55 for unresolved candidates).
    """

    def discover(
        self,
        *,
        document_id: str,
        document_sha256: str,
        pages: Sequence[PDFPageText],
    ) -> list[ReviewCandidate]:
        self._validate_input(document_id, document_sha256, pages)
        candidates: dict[tuple[int, int, int, ReviewCandidateClass], ReviewCandidate] = {}
        for page in sorted(pages, key=lambda item: item.page_number):
            for start, end in self._bounded_segments(page.text):
                source_text = page.text[start:end]
                if not self._is_review_worthy(source_text):
                    continue
                candidate_class = self._classify(source_text)
                status = (
                    ReviewCandidateStatus.UNRESOLVED
                    if candidate_class == ReviewCandidateClass.UNRESOLVED
                    else ReviewCandidateStatus.DISCOVERED
                )
                source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
                span = CandidateSourceSpan(
                    page_number=page.page_number,
                    char_start=start,
                    char_end=end,
                    source_text=source_text,
                    source_text_sha256=source_hash,
                )
                candidate = ReviewCandidate(
                    candidate_id=deterministic_candidate_id(
                        document_sha256=document_sha256,
                        source_spans=[span],
                        candidate_class=candidate_class,
                    ),
                    document_id=document_id,
                    document_sha256=document_sha256,
                    source_spans=(span,),
                    source_text=source_text,
                    source_text_sha256=source_hash,
                    topic=self._topic(source_text),
                    candidate_class=candidate_class,
                    discovery_method=ReviewCandidateDiscoveryMethod.DETERMINISTIC_SIGNAL_RULES,
                    discovery_version=REVIEW_CANDIDATE_DISCOVERY_VERSION,
                    confidence=self._confidence(source_text, candidate_class),
                    status=status,
                )
                key = (page.page_number, start, end, candidate_class)
                candidates.setdefault(key, candidate)
        return sorted(
            candidates.values(),
            key=lambda item: (
                item.source_spans[0].page_number,
                item.source_spans[0].char_start,
                item.source_spans[0].char_end,
                item.candidate_class.value,
                item.candidate_id,
            ),
        )

    @staticmethod
    def _validate_input(
        document_id: str,
        document_sha256: str,
        pages: Sequence[PDFPageText],
    ) -> None:
        if not document_id.strip() or document_id != document_id.strip():
            raise CandidateDiscoveryInputError(
                "document_id must be non-blank without edge whitespace."
            )
        if not re.fullmatch(r"[0-9a-f]{64}", document_sha256):
            raise CandidateDiscoveryInputError(
                "document_sha256 must be a lowercase SHA-256 identity."
            )
        if isinstance(pages, (str, bytes)) or not isinstance(pages, Sequence):
            raise CandidateDiscoveryInputError("pages must be parsed PDFPageText records.")
        if any(not isinstance(page, PDFPageText) for page in pages):
            raise CandidateDiscoveryInputError("pages must be parsed PDFPageText records.")
        page_numbers = [page.page_number for page in pages]
        if any(number <= 0 for number in page_numbers) or len(page_numbers) != len(
            set(page_numbers)
        ):
            raise CandidateDiscoveryInputError(
                "Parsed physical page numbers must be positive and unique."
            )

    @classmethod
    def _bounded_segments(cls, text: str) -> list[tuple[int, int]]:
        spans: set[tuple[int, int]] = set()
        for match in _PRIMARY_SEGMENT.finditer(text):
            start, end = cls._trim_span(text, match.start(), match.end())
            if start >= end:
                continue
            if end - start <= MAX_CANDIDATE_CHARS:
                spans.add((start, end))
                continue
            segment = text[start:end]
            for signal in _DISCOVERY_SIGNAL.finditer(segment):
                center = start + (signal.start() + signal.end()) // 2
                window_start = max(start, center - MAX_CANDIDATE_CHARS // 2)
                window_end = min(end, window_start + MAX_CANDIDATE_CHARS)
                window_start = max(start, window_end - MAX_CANDIDATE_CHARS)
                window_start, window_end = cls._trim_span(
                    text, window_start, window_end
                )
                if window_start < window_end:
                    spans.add((window_start, window_end))
        return sorted(spans)

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    @staticmethod
    def _is_review_worthy(text: str) -> bool:
        numeric_control = bool(_NUMBER_WITH_UNIT.search(text)) and bool(
            _NUMERIC_CONTROL_OPERATOR.search(text)
        )
        inspection = bool(_INSPECTION.search(text)) and bool(
            _CONSTRUCTION_CONTEXT.search(text)
        )
        obligation = bool(_OBLIGATION.search(text)) and len("".join(text.split())) >= 4
        domain_control = bool((_SAFETY.search(text) or _PROCEDURE.search(text))) and bool(
            _CONSTRUCTION_CONTEXT.search(text)
        )
        return numeric_control or inspection or obligation or domain_control

    @staticmethod
    def _classify(text: str) -> ReviewCandidateClass:
        if _NUMBER_WITH_UNIT.search(text) and _NUMERIC_CONTROL_OPERATOR.search(text):
            return ReviewCandidateClass.NUMERIC_CONTROL
        if _INSPECTION.search(text) and _CONSTRUCTION_CONTEXT.search(text):
            return ReviewCandidateClass.INSPECTION_REQUIREMENT
        if _MATERIAL.search(text) and _OBLIGATION.search(text):
            return ReviewCandidateClass.MATERIAL_REQUIREMENT
        if _SAFETY.search(text) and (_OBLIGATION.search(text) or _PROCEDURE.search(text)):
            return ReviewCandidateClass.SAFETY_REQUIREMENT
        if _STRUCTURAL.search(text) and _OBLIGATION.search(text):
            return ReviewCandidateClass.STRUCTURAL_CONFIGURATION
        if _PROCEDURE.search(text) and _OBLIGATION.search(text):
            return ReviewCandidateClass.PROCEDURAL_REQUIREMENT
        if _OBLIGATION.search(text) and _CONSTRUCTION_CONTEXT.search(text):
            return ReviewCandidateClass.CONSTRUCTION_REQUIREMENT
        return ReviewCandidateClass.UNRESOLVED

    @staticmethod
    def _topic(text: str) -> str:
        matches = [label for label, pattern in _TOPIC_RULES if pattern.search(text)]
        return matches[0] if len(matches) == 1 else GENERAL_TOPIC

    @staticmethod
    def _confidence(text: str, candidate_class: ReviewCandidateClass) -> float:
        score = 0.45
        if _NUMBER_WITH_UNIT.search(text) and _NUMERIC_CONTROL_OPERATOR.search(text):
            score += 0.35
        if _OBLIGATION.search(text):
            score += 0.10
        if _CONSTRUCTION_CONTEXT.search(text):
            score += 0.05
        if candidate_class != ReviewCandidateClass.UNRESOLVED:
            score += 0.10
        limit = 0.55 if candidate_class == ReviewCandidateClass.UNRESOLVED else 0.95
        return round(min(score, limit), 2)
