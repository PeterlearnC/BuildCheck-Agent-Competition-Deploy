"""Conservative line-preserving preprocessing for standards pages."""

from collections import Counter
import math
import re
import unicodedata

from app.schemas.standards import StandardPage


class StandardTextPreprocessingService:
    PAGE_NUMBER = re.compile(r"^\s*(?:[-—–]\s*)?\d{1,4}(?:\s*[-—–])?\s*$")

    def preprocess(self, pages: list[StandardPage]) -> list[StandardPage]:
        normalized = [
            StandardPage(
                page_number=page.page_number,
                text=self._normalize(page.text),
                image_count=page.image_count,
                image_coverage_ratio=page.image_coverage_ratio,
                source_kind=page.source_kind,
                ocr_run_id=page.ocr_run_id,
                ocr_execution_id=page.ocr_execution_id,
                ocr_quality_state=page.ocr_quality_state,
                ocr_provider=page.ocr_provider,
                ocr_render=page.ocr_render,
                source_mappings=self._remap_source_mappings(
                    page, self._normalize(page.text).splitlines()
                ),
            )
            for page in pages
        ]
        repeated = self._repeated_edge_lines(normalized)
        result: list[StandardPage] = []
        for page in normalized:
            lines = page.text.splitlines()
            nonempty_indexes = [index for index, line in enumerate(lines) if line.strip()]
            edge_indexes = set(nonempty_indexes[:2] + nonempty_indexes[-2:])
            kept: list[str] = []
            for index, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    kept.append("")
                    continue
                identity = self._identity(stripped)
                if index in edge_indexes and identity in repeated:
                    kept.append("")
                    continue
                if index in edge_indexes and self.PAGE_NUMBER.fullmatch(stripped):
                    kept.append("")
                    continue
                kept.append(stripped)
            result.append(
                StandardPage(
                    page_number=page.page_number,
                    text="\n".join(kept),
                    image_count=page.image_count,
                    image_coverage_ratio=page.image_coverage_ratio,
                    source_kind=page.source_kind,
                    ocr_run_id=page.ocr_run_id,
                    ocr_execution_id=page.ocr_execution_id,
                    ocr_quality_state=page.ocr_quality_state,
                    ocr_provider=page.ocr_provider,
                    ocr_render=page.ocr_render,
                    source_mappings=self._remap_source_mappings(page, kept),
                )
            )
        return result

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text or "")
        normalized = normalized.replace("\u00a0", " ").replace("\u3000", " ")
        normalized = normalized.replace("：", ":").replace("；", ";")
        return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in normalized.splitlines())

    def _repeated_edge_lines(self, pages: list[StandardPage]) -> set[str]:
        if len(pages) < 3:
            return set()
        counts: Counter[str] = Counter()
        for page in pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            identities = {
                self._identity(line)
                for line in lines[:2] + lines[-2:]
                if 2 <= len(line) <= 80 and not self.PAGE_NUMBER.fullmatch(line)
            }
            counts.update(identities)
        threshold = max(3, math.ceil(len(pages) * 0.6))
        return {identity for identity, count in counts.items() if count >= threshold}

    @staticmethod
    def _identity(line: str) -> str:
        return re.sub(r"\s+", "", line).casefold()

    @staticmethod
    def _remap_source_mappings(page: StandardPage, lines: list[str]):
        if not page.source_mappings:
            return []
        offsets: dict[int, tuple[int, int]] = {}
        position = 0
        for index, line in enumerate(lines):
            offsets[index] = (position, position + len(line))
            position += len(line) + 1
        result = []
        for mapping in page.source_mappings:
            line = lines[mapping.parser_line_index] if mapping.parser_line_index < len(lines) else ""
            if not line:
                continue
            start, end = offsets[mapping.parser_line_index]
            result.append(
                mapping.model_copy(
                    update={"normalized_start": start, "normalized_end": end}
                )
            )
        return result
