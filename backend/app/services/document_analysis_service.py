"""Orchestration, deterministic extraction, and JSON cache for V0.2."""

import json
import re
from pathlib import Path

from app.core.config import Settings, get_settings
from app.schemas.analysis import (
    DocumentAnalysis,
    ProjectInfo,
    ScheduleInfo,
    SectionPresence,
    SourceValue,
    StandardReference,
)
from app.services.chapter_detection_service import ChapterDetectionService
from app.services.llm_service import LLMService
from app.services.metadata_extraction_service import MetadataExtractionService
from app.services.pdf_service import PDFExtractionResult, PDFService
from app.services.source_locator_service import SourceLocatorService
from app.services.semantic_region_service import SemanticRegionService
from app.services.text_preprocessing_service import TextPreprocessingService


class AnalysisCacheError(Exception):
    pass


class DocumentAnalysisService:
    STANDARD_CODE = (
        r"(?:(?:GB|JGJ|CJJ|JTG|TB|JT)(?:/T)?|DL/T|DBJ|DB\d*(?:/T)?|CECS)"
        r"\s*[A-Z]*\s*\d[\d.\-—–/]*\d|"
        r"T\s*/\s*[A-Z0-9]+(?:\s*[-./]\s*[A-Z0-9]+)+|"
        r"[A-Z]{2,}[A-Z0-9]*\s*-\s*[A-Z]{2,}[A-Z0-9]*"
        r"(?:\s*-\s*[A-Z0-9]+)+"
    )
    STANDARD_PATTERNS = (
        re.compile(
            rf"《(?P<name>[^》\n]{{2,80}})》\s*[（(]?\s*"
            rf"(?P<code>{STANDARD_CODE})\s*[）)]?",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?m)^\s*[※*·]?\s*(?P<name>[^\n（）()]{{2,80}}?)\s*"
            rf"[（(]\s*(?P<code>{STANDARD_CODE})\s*[）)]\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?m)^\s*[※*·]?\s*(?P<code>{STANDARD_CODE})\s+"
            rf"(?P<name>[^\n（）()]{{2,80}}?)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?P<code>{STANDARD_CODE})",
            re.IGNORECASE,
        ),
    )

    def __init__(
        self,
        llm_service: LLMService,
        settings: Settings | None = None,
        pdf_service: PDFService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_service = llm_service
        self.pdf_service = pdf_service or PDFService()
        self.preprocessing = TextPreprocessingService()
        self.chapter_detection = ChapterDetectionService()
        self.metadata_extraction = MetadataExtractionService()
        self.source_locator = SourceLocatorService()

    def find_pdf(self, document_id: str) -> Path | None:
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", document_id):
            return None
        path = self.settings.upload_dir / f"{document_id}.pdf"
        return path if path.is_file() else None

    def load_cache(self, document_id: str) -> DocumentAnalysis | None:
        path = self.settings.analysis_dir / f"{document_id}.json"
        if not path.is_file():
            return None
        try:
            return DocumentAnalysis.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AnalysisCacheError("Saved analysis cache is unreadable.") from exc

    def analyze(self, pdf_path: Path) -> DocumentAnalysis:
        extracted = self.pdf_service.extract_text(pdf_path)
        pages = self.preprocessing.preprocess(extracted.pages)
        chapters = self.chapter_detection.detect(pages)
        local_project, local_schedule = self.metadata_extraction.extract(pages)
        local_document_type = self.metadata_extraction.extract_document_type(
            extracted.pages, local_project.project_name
        )
        local_engineering_category = self.metadata_extraction.extract_engineering_category(
            pages, local_project
        )
        deterministic_standards = self._extract_standards(extracted)
        deterministic_basis = self._extract_compilation_basis(pages, chapters)
        analysis = self.llm_service.analyze_document(pages)
        if local_document_type is not None:
            analysis.document_type = local_document_type
        analysis.chapters = self._merge_chapters(chapters, analysis.chapters, extracted.page_count)
        analysis.project = self._merge_metadata(local_project, analysis.project)
        analysis.project = self.metadata_extraction.validate_organizations(
            pages, analysis.project
        )
        analysis.schedule = self._merge_metadata(local_schedule, analysis.schedule)
        if local_engineering_category is not None:
            analysis.engineering_category = local_engineering_category
        else:
            analysis.engineering_category = self.metadata_extraction.validate_engineering_category(
                pages, analysis.project, analysis.engineering_category
            )
        analysis.standards = self._merge_standards(deterministic_standards, analysis.standards)
        analysis.compilation_basis = self._merge_compilation_basis(
            deterministic_basis, analysis.compilation_basis
        )

        llm_presence = analysis.section_presence
        final_analysis = self.source_locator.locate(analysis, extracted.pages)
        deterministic_presence = self.chapter_detection.section_presence(
            final_analysis.chapters, extracted.pages
        )
        # This is the single final merge point. Nothing after this line rebuilds
        # DocumentAnalysis before the route writes the cache and response.
        final_analysis.section_presence = self._merge_section_presence(
            llm_presence, deterministic_presence
        )
        return final_analysis

    def save_cache(self, document_id: str, analysis: DocumentAnalysis) -> None:
        try:
            self.settings.analysis_dir.mkdir(parents=True, exist_ok=True)
            target = self.settings.analysis_dir / f"{document_id}.json"
            temporary = self.settings.analysis_dir / f".{document_id}.tmp"
            temporary.write_text(
                json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(target)
        except OSError as exc:
            raise AnalysisCacheError("Failed to save document analysis cache.") from exc

    def _extract_standards(self, extracted: PDFExtractionResult) -> list[StandardReference]:
        candidates: dict[str, list[tuple[int, StandardReference]]] = {}
        order: list[str] = []
        regions = SemanticRegionService().classify(extracted.pages)
        for page in extracted.pages:
            page_bonus = 0
            if re.search(r"编制(?:说明及)?依据|依据文件|主要规范标准", page.text):
                page_bonus += 300
            if regions.get(page.page_number) == "calculation":
                page_bonus -= 500
            page_items = self._standards_from_rows(page)
            matches = sorted(
                (
                    match
                    for pattern in self.STANDARD_PATTERNS
                    for match in pattern.finditer(page.text)
                ),
                key=lambda match: match.start(),
            )
            page_items.extend(
                (
                    self._clean_standard_name(match.groupdict().get("name")),
                    re.sub(r"\s+", "", match.group("code")).replace("—", "-").replace("–", "-"),
                    match.group(0).strip()[:300] or None,
                    0,
                )
                for match in matches
            )
            for name, code, source_text, row_bonus in page_items:
                if not self._is_standard_code(code):
                    continue
                key = code.upper()
                if key not in candidates:
                    candidates[key] = []
                    order.append(key)
                candidates[key].append(
                    (
                        page_bonus + row_bonus + (100 if name else 0),
                        StandardReference(
                        name=name,
                        code=code,
                        source_page=page.page_number,
                        source_text=source_text,
                        ),
                    )
                )
        return [
            max(candidates[key], key=lambda item: (item[0], -item[1].source_page))[1]
            for key in order
        ]

    def _standards_from_rows(
        self, page
    ) -> list[tuple[str | None, str, str, int]]:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        result: list[tuple[str | None, str, str, int]] = []
        code_pattern = re.compile(self.STANDARD_CODE, re.IGNORECASE)
        for index, line in enumerate(lines):
            matches = list(code_pattern.finditer(line))
            if not matches:
                continue
            for match in matches:
                code = re.sub(r"\s+", "", match.group(0)).replace("—", "-").replace("–", "-")
                same_row = f"{line[:match.start()]} {line[match.end():]}"
                name = self._clean_standard_name(same_row)
                if not self._is_standard_name(name):
                    name = None
                source_lines = [line]
                if name is None:
                    neighbours = []
                    if index > 0:
                        neighbours.append((index - 1, lines[index - 1]))
                    if index + 1 < len(lines):
                        neighbours.append((index + 1, lines[index + 1]))
                    for neighbour_index, neighbour in neighbours:
                        candidate = self._clean_standard_name(neighbour)
                        if (
                            self._is_standard_name(candidate)
                            and not code_pattern.search(neighbour)
                        ):
                            name = candidate
                            source_lines = (
                                [neighbour, line]
                                if neighbour_index < index
                                else [line, neighbour]
                            )
                            break
                result.append((name, code, "\n".join(source_lines)[:300], 200))
        return result

    @staticmethod
    def _is_standard_name(value: str | None) -> bool:
        return bool(
            value
            and re.search(r"标准|规范|规程|导则|指南|技术条件|技术要求", value)
        )

    @staticmethod
    def _is_standard_code(value: str) -> bool:
        compact = re.sub(r"\s+", "", value).upper().replace("—", "-").replace("–", "-")
        if not 5 <= len(compact) <= 48:
            return False
        if not re.search(r"[A-Z]", compact) or not re.search(r"\d", compact):
            return False
        if re.fullmatch(r"[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}", compact):
            return False
        if re.fullmatch(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", compact):
            return False
        known_family = re.match(
            r"^(?:GB|JGJ|CJJ|JTG|TB|JT|DL/T|DBJ|DB\d*|CECS|T/)", compact
        )
        mixed_family = bool(
            re.fullmatch(
                r"[A-Z]{2,}[A-Z0-9]*-[A-Z]{2,}[A-Z0-9]*(?:-[A-Z0-9]+)+",
                compact,
            )
        )
        return bool(known_family or mixed_family)

    @staticmethod
    def _extract_compilation_basis(pages, chapters) -> list[SourceValue]:
        basis_title = re.compile(r"编制依据|编制说明及依据|依据文件|主要规范标准")
        regions = SemanticRegionService().classify(pages)
        relevant_pages: set[int] = set()
        for chapter_index, chapter in enumerate(chapters):
            if not basis_title.search(chapter.title):
                continue
            end_page = chapter.end_page or chapter.start_page
            for following in chapters[chapter_index + 1 :]:
                if (following.level or 1) <= (chapter.level or 1):
                    if following.start_page > chapter.start_page:
                        end_page = min(end_page, following.start_page - 1)
                    break
            relevant_pages.update(
                page_number
                for page_number in range(chapter.start_page, end_page + 1)
                if regions.get(page_number) != "calculation"
            )
        for index, page in enumerate(pages):
            if regions.get(page.page_number) == "calculation" or not re.search(
                r"(?:^|\n)\s*(?:第[一二三四五六七八九十百\d]+[章节篇]\s*|"
                r"\d+(?:\.\d+)*[、.．]?\s*)?"
                r"(?:编制依据|编制说明及依据|依据文件|主要规范标准)\s*$",
                page.text,
                re.MULTILINE,
            ):
                continue
            relevant_pages.add(page.page_number)
            for following in pages[index + 1 : index + 3]:
                if regions.get(following.page_number) == "calculation" or re.search(
                    r"(?:^|\n)\s*(?:第[一二三四五六七八九十百\d]+[章节篇]\s*|"
                    r"\d+(?:\.\d+)*[、.．]?\s*)?(?:工程概况|施工(?:工艺|方法)|"
                    r"质量|安全|验收|应急)",
                    following.text,
                ):
                    break
                relevant_pages.add(following.page_number)

        keyword = re.compile(r"合同|勘察报告|施工图纸|设计图纸|设计文件|规范|标准")
        result: list[SourceValue] = []
        seen: set[str] = set()
        for page in pages:
            if page.page_number not in relevant_pages:
                continue
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            for index, line in enumerate(lines):
                if not keyword.search(line) or line.strip(" ：:、") == "编制依据":
                    continue
                source_lines = [line]
                if index > 0 and re.search(
                    r"提供的?$|编制的?$|出具的?$|(?:有限公司|研究院|设计院)$",
                    lines[index - 1],
                ):
                    source_lines.insert(0, lines[index - 1])
                value = "".join(source_lines)
                value = re.sub(
                    r"^\s*(?:[（(]?\d+[）)]?[、.．]|\d+(?:\.\d+)+[、.．]?)\s*",
                    "",
                    value,
                ).strip()
                identity = re.sub(r"[\s、，,。；;：:（）()]+", "", value)
                if not value or identity in seen:
                    continue
                result.append(
                    SourceValue(
                        value=value,
                        source_page=page.page_number,
                        source_text="\n".join(source_lines)[:300],
                    )
                )
                seen.add(identity)
        return result

    @staticmethod
    def _clean_standard_name(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = re.sub(
            r"^\s*(?:(?:[（(]\s*\d+\s*[）)]|[①-⑳]|"
            r"\d+(?:\.\d+)*[、.．)]?)\s*[、.．)]*|[※*·•])\s*",
            "",
            value,
        )
        cleaned = re.sub(r"^\s*\d+\s*[|｜]\s*", "", cleaned)
        cleaned = re.sub(r"[（(]\s*[）)]", "", cleaned)
        cleaned = cleaned.strip(" |｜《》“”‘’\"'：:，,；;。-—")
        if not cleaned or re.fullmatch(
            r"国家及|行业及|地方及|国家|行业|地方|标准|规范|规范名称|标准名称|编号|序号",
            cleaned,
        ):
            return None
        return cleaned

    @staticmethod
    def _merge_metadata(
        local: ProjectInfo | ScheduleInfo, llm: ProjectInfo | ScheduleInfo
    ) -> ProjectInfo | ScheduleInfo:
        payload = llm.model_dump()
        for field, value in local.model_dump().items():
            if value is not None:
                payload[field] = value
        return type(local).model_validate(payload)

    @staticmethod
    def _merge_chapters(deterministic, llm_values, page_count):
        # Local headings retain their original PDF page positions and are authoritative.
        if deterministic:
            return list(deterministic)

        result = []
        known = set()
        for item in llm_values:
            if item.title in known or not 1 <= item.start_page <= page_count:
                continue
            end_page = item.end_page
            if end_page is not None and not item.start_page <= end_page <= page_count:
                end_page = None
            result.append(item.model_copy(update={"end_page": end_page}))
            known.add(item.title)
        result.sort(key=lambda item: (item.start_page, item.title))
        for index, item in enumerate(result):
            if item.end_page is not None:
                continue
            current_level = item.level or 1
            next_start = page_count + 1
            for following in result[index + 1 :]:
                if (following.level or 1) <= current_level:
                    next_start = following.start_page
                    break
            inferred_end = max(item.start_page, min(next_start, page_count))
            result[index] = item.model_copy(update={"end_page": inferred_end})
        return result

    @staticmethod
    def _merge_standards(
        deterministic: list[StandardReference], llm_values: list[StandardReference]
    ) -> list[StandardReference]:
        result = list(deterministic)
        known = {item.code.upper().replace(" ", "") for item in result}
        for item in llm_values:
            key = item.code.upper().replace(" ", "")
            if key not in known:
                result.append(item)
                known.add(key)
        return result

    @staticmethod
    def _merge_compilation_basis(
        deterministic: list[SourceValue], llm_values: list[SourceValue | str]
    ) -> list[SourceValue | str]:
        result: list[SourceValue | str] = list(deterministic)
        if deterministic:
            return result

        def identity(item: SourceValue | str) -> str:
            value = item if isinstance(item, str) else str(item.value or "")
            return re.sub(r"[\s、，,。；;：:（）()]+", "", value)

        known = {identity(item) for item in result}
        for item in llm_values:
            key = identity(item)
            if key and key not in known:
                result.append(item)
                known.add(key)
        return result

    @staticmethod
    def _merge_section_presence(
        llm: SectionPresence, deterministic: SectionPresence
    ) -> SectionPresence:
        return SectionPresence(
            quality=llm.quality or deterministic.quality,
            safety=llm.safety or deterministic.safety,
            environment=llm.environment or deterministic.environment,
            emergency=llm.emergency or deterministic.emergency,
        )
