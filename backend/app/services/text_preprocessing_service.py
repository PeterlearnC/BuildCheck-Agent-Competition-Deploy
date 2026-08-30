"""Conservative cleanup that preserves page-level traceability."""

import re
from collections import Counter

from app.services.pdf_service import PDFPageText


class TextPreprocessingService:
    """Remove layout noise without rewriting meaningful PDF text."""

    def preprocess(self, pages: list[PDFPageText]) -> list[PDFPageText]:
        page_lines = [self._clean_lines(page.text) for page in pages]
        repeated_edges = self._find_repeated_headers_and_footers(page_lines)
        result: list[PDFPageText] = []
        for page, lines in zip(pages, page_lines, strict=True):
            kept = [line for line in lines if self._normalise_edge(line) not in repeated_edges]
            text = "\n".join(kept).strip()
            text = re.sub(r"\n{3,}", "\n\n", text)
            result.append(PDFPageText(page_number=page.page_number, text=text))
        return result

    @staticmethod
    def _clean_lines(text: str) -> list[str]:
        lines: list[str] = []
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            cleaned = re.sub(r"[ \t\u3000]+", " ", line).strip()
            if cleaned or (lines and lines[-1]):
                lines.append(cleaned)
        while lines and not lines[-1]:
            lines.pop()
        return lines

    def _find_repeated_headers_and_footers(self, pages: list[list[str]]) -> set[str]:
        if len(pages) < 3:
            return set()
        candidates: Counter[str] = Counter()
        for lines in pages:
            nonempty = [line for line in lines if line]
            # Only the outermost lines are safe basic header/footer candidates.
            # Looking at the second line can accidentally remove repeated chapter titles.
            for line in nonempty[:1] + nonempty[-1:]:
                normalised = self._normalise_edge(line)
                if normalised and len(normalised) <= 80:
                    candidates[normalised] += 1
        threshold = max(3, (len(pages) + 1) // 2)
        return {line for line, count in candidates.items() if count >= threshold}

    @staticmethod
    def _normalise_edge(line: str) -> str:
        # Page numbers vary, so replace digit runs only for header/footer comparison.
        return re.sub(r"\d+", "#", line.strip())
