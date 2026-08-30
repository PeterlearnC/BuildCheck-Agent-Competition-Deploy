"""Conservative, layout-aware capability gate for standards PDFs."""

from collections import Counter
import math
import re
import unicodedata

from app.schemas.standards import (
    DocumentCapabilityResult,
    DocumentCapabilityStatus,
    StandardPage,
)


class DocumentInspectorService:
    PAGE_NUMBER = re.compile(r"^(?:第\s*)?\d{1,4}(?:\s*页)?$")
    STRUCTURAL_TEXT = re.compile(
        r"(?:第\s*[一二三四五六七八九十百\d]+\s*章|"
        r"(?<!\d)(?:\d+\.){2,3}\d+(?!\d)|"
        r"(?:规范|标准|规程|附录|条文说明))"
    )
    NORMATIVE_SENTENCE = re.compile(r"应当|应|不得|不应|严禁|宜|可")
    IMAGE_HEAVY_THRESHOLD = 0.60

    def inspect(self, pages: list[StandardPage]) -> DocumentCapabilityResult:
        page_count = len(pages)
        raw_count = sum(len(page.text or "") for page in pages)
        if page_count == 0:
            return self._result(
                page_count, raw_count, 0, 0, 0, 0, 0.0,
                DocumentCapabilityStatus.EMPTY_DOCUMENT,
                "document has no extracted text pages",
            )

        coverages = [self._image_coverage(page) for page in pages]
        image_pages = sum(page.image_count > 0 for page in pages)
        heavy_pages = sum(value >= self.IMAGE_HEAVY_THRESHOLD for value in coverages)
        if not any((page.text or "").strip() for page in pages):
            if image_pages:
                return self._result(
                    page_count, raw_count, 0, 0, image_pages, heavy_pages, 0.0,
                    DocumentCapabilityStatus.OCR_REQUIRED,
                    "image-only document has no extracted text layer",
                    mean_image_coverage=sum(coverages) / page_count,
                )
            return self._result(
                page_count, raw_count, 0, 0, 0, 0, 0.0,
                DocumentCapabilityStatus.EMPTY_DOCUMENT,
                "document pages contain neither text nor images",
            )

        normalized_lines = [self._page_lines(page.text) for page in pages]
        repeated = self._repeated_edge_identities(normalized_lines, page_count)
        effective_by_page: list[int] = []
        repeated_chars = 0
        nonspace_chars = 0
        for lines in normalized_lines:
            effective = 0
            last_index = len(lines) - 1
            for index, line in enumerate(lines):
                compact = self._identity(line)
                nonspace_chars += len(compact)
                edge_slot = self._edge_slot(index, last_index)
                if edge_slot and (edge_slot, compact) in repeated:
                    repeated_chars += len(compact)
                    continue
                if self.PAGE_NUMBER.fullmatch(compact):
                    continue
                meaningful_chars = sum(character.isalnum() for character in compact)
                if meaningful_chars < 4:
                    continue
                effective += meaningful_chars
            effective_by_page.append(effective)

        effective_count = sum(effective_by_page)
        meaningful_pages = sum(
            count >= 24
            or (count >= 8 and bool(self.STRUCTURAL_TEXT.search(pages[index].text or "")))
            for index, count in enumerate(effective_by_page)
        )
        meaningful_ratio = meaningful_pages / page_count
        image_ratio = image_pages / page_count
        heavy_ratio = heavy_pages / page_count
        repeated_ratio = repeated_chars / max(nonspace_chars, 1)
        mean_coverage = sum(coverages) / page_count

        status, reason = self._classify(
            page_count=page_count,
            effective_count=effective_count,
            meaningful_pages=meaningful_pages,
            meaningful_ratio=meaningful_ratio,
            image_heavy_ratio=heavy_ratio,
            repeated_ratio=repeated_ratio,
            pages=pages,
        )
        return self._result(
            page_count,
            raw_count,
            effective_count,
            meaningful_pages,
            image_pages,
            heavy_pages,
            repeated_ratio,
            status,
            reason,
            mean_image_coverage=mean_coverage,
        )

    def _classify(
        self,
        *,
        page_count: int,
        effective_count: int,
        meaningful_pages: int,
        meaningful_ratio: float,
        image_heavy_ratio: float,
        repeated_ratio: float,
        pages: list[StandardPage],
    ) -> tuple[DocumentCapabilityStatus, str]:
        has_structure = any(self.STRUCTURAL_TEXT.search(page.text or "") for page in pages)
        if image_heavy_ratio > 0.5 and (
            meaningful_ratio < 0.5
            or effective_count < max(120, page_count * 35)
            or repeated_ratio >= 0.55
        ):
            return (
                DocumentCapabilityStatus.OCR_REQUIRED,
                "image-heavy pages contain too little usable structured text",
            )
        if has_structure and meaningful_pages >= 1 and image_heavy_ratio < 0.5:
            return DocumentCapabilityStatus.TEXT_READY, "usable structured text layer detected"
        if image_heavy_ratio == 0 and effective_count >= 8:
            return DocumentCapabilityStatus.TEXT_READY, "usable text layer detected"
        if meaningful_pages == 0 or effective_count < 10:
            return (
                DocumentCapabilityStatus.TEXT_INSUFFICIENT,
                "extracted text is present but lacks meaningful standard content",
            )
        if image_heavy_ratio >= 0.2 and meaningful_ratio < 0.75:
            return (
                DocumentCapabilityStatus.TEXT_INSUFFICIENT,
                "mixed document has insufficient text-ready page coverage",
            )
        if has_structure or effective_count >= max(80, page_count * 30):
            return DocumentCapabilityStatus.TEXT_READY, "usable standard text layer detected"
        return (
            DocumentCapabilityStatus.TEXT_INSUFFICIENT,
            "text layer is too sparse for reliable structure parsing",
        )

    def _repeated_edge_identities(
        self, pages: list[list[str]], page_count: int
    ) -> set[tuple[str, str]]:
        if page_count < 3:
            return set()
        counts: Counter[tuple[str, str]] = Counter()
        for lines in pages:
            last_index = len(lines) - 1
            per_page: set[tuple[str, str]] = set()
            for index, line in enumerate(lines):
                slot = self._edge_slot(index, last_index)
                identity = self._identity(line)
                if (
                    slot
                    and 2 <= len(identity) <= 80
                    and not self.PAGE_NUMBER.fullmatch(identity)
                    and not self.NORMATIVE_SENTENCE.search(line)
                ):
                    per_page.add((slot, identity))
            counts.update(per_page)
        threshold = max(3, math.ceil(page_count * 0.6))
        return {value for value, count in counts.items() if count >= threshold}

    @staticmethod
    def _edge_slot(index: int, last_index: int) -> str | None:
        if index <= 1:
            return f"top-{index}"
        if index >= max(last_index - 1, 0):
            return f"bottom-{last_index - index}"
        return None

    @staticmethod
    def _image_coverage(page: StandardPage) -> float:
        if page.image_coverage_ratio is not None:
            return page.image_coverage_ratio
        return 1.0 if page.image_count > 0 else 0.0

    @staticmethod
    def _page_lines(text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text or "")
        return [line.strip() for line in normalized.splitlines() if line.strip()]

    @staticmethod
    def _identity(text: str) -> str:
        return re.sub(r"\s+", "", text).casefold()

    @staticmethod
    def _result(
        page_count: int,
        raw_count: int,
        effective_count: int,
        meaningful_pages: int,
        image_pages: int,
        heavy_pages: int,
        repeated_ratio: float,
        status: DocumentCapabilityStatus,
        reason: str,
        *,
        mean_image_coverage: float = 0.0,
    ) -> DocumentCapabilityResult:
        denominator = max(page_count, 1)
        return DocumentCapabilityResult(
            page_count=page_count,
            raw_text_char_count=raw_count,
            effective_text_char_count=effective_count,
            pages_with_meaningful_text=meaningful_pages,
            pages_with_images=image_pages,
            meaningful_text_page_ratio=meaningful_pages / denominator,
            image_page_ratio=image_pages / denominator,
            image_heavy_page_ratio=heavy_pages / denominator,
            mean_image_coverage_ratio=mean_image_coverage,
            repeated_text_ratio=repeated_ratio,
            status=status,
            reason=reason,
        )
