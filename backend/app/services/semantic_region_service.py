"""Shared structural regions for long documents and embedded sub-documents."""

import re

from app.services.pdf_service import PDFPageText


class SemanticRegionService:
    CALCULATION_HEADING = re.compile(
        r"^(?:附图及计算书|计算书|设计计算|计算分析|验算|相关计算书)$"
    )
    DOCUMENT_TITLE = re.compile(
        r"[^，,。；;：:\n]{2,80}(?:专项施工方案|施工组织设计|施工方案)$"
    )
    STRUCTURE_TERMS = (
        re.compile(r"(?:^|\n)\s*(?:\d+(?:\.\d+)*[、.．]?\s*)?编制(?:说明及)?依据"),
        re.compile(r"(?:^|\n)\s*(?:\d+(?:\.\d+)*[、.．]?\s*)?工程概况"),
        re.compile(r"(?:^|\n)\s*(?:\d+(?:\.\d+)*[、.．]?\s*)?施工(?:工艺|方法|工艺技术)"),
    )
    HEADING_PREFIX = re.compile(
        r"^\s*(?:第[一二三四五六七八九十百\d]+[章节篇]|"
        r"[一二三四五六七八九十]+、|\d+[、.．]|\d+\s+|"
        r"\d+(?:\.\d+)+[、.．]?)\s*"
    )

    def classify(self, pages: list[PDFPageText]) -> dict[int, str]:
        embedded_starts = self._embedded_starts(pages)
        result: dict[int, str] = {}
        region = "normal"
        for index, page in enumerate(pages):
            if index in embedded_starts:
                region = "embedded"
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            starts_calculation = any(
                self.is_calculation_heading(line) for line in lines
            )
            if starts_calculation and index not in embedded_starts:
                region = "calculation"
            result[page.page_number] = region
        return result

    def _embedded_starts(self, pages: list[PDFPageText]) -> set[int]:
        starts: set[int] = set()
        calculation_seen = False
        for index, page in enumerate(pages):
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            if any(self.is_calculation_heading(line) for line in lines):
                calculation_seen = True
                continue
            if not calculation_seen or not any(self.DOCUMENT_TITLE.search(line) for line in lines):
                continue
            window = "\n".join(item.text for item in pages[index : index + 12])
            if all(pattern.search(window) for pattern in self.STRUCTURE_TERMS):
                starts.add(index)
        return starts

    @classmethod
    def is_calculation_heading(cls, line: str) -> bool:
        cleaned = cls.HEADING_PREFIX.sub("", line).strip(" ：:、，,。")
        return bool(cls.CALCULATION_HEADING.fullmatch(cleaned))

    @staticmethod
    def is_formula_or_unit_expression(title: str) -> bool:
        compact = re.sub(r"\s+", "", title)
        if re.search(r"(?:kN(?:\.m)?|N/mm|MPa|kPa|mm|cm|m²|m³)", compact, re.I):
            return True
        if (
            re.fullmatch(r"[A-Za-zα-ωΑ-Ω\d_{}()[\].,+\-*/=<>≤≥×÷^%：:]+", compact)
            and re.search(r"\d|[=<>≤≥×÷+*/^]", compact)
        ):
            return True
        if re.search(r"[=×÷]", compact) and re.search(r"\d", compact):
            return True
        return False
