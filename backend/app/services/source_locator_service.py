"""Score evidence in original PDF pages instead of trusting LLM page claims."""

import re
from typing import Any

from app.schemas.analysis import DocumentAnalysis
from app.services.deterministic_postprocessing_service import DeterministicPostProcessingService
from app.services.pdf_service import PDFPageText
from app.services.semantic_region_service import SemanticRegionService


class SourceLocatorService:
    PROJECT_LABELS = {
        "project_name": ("工程名称", "项目名称"),
        "project_location": ("工程地址", "项目地址", "建设地点", "工程地点", "项目地点", "项目所在地"),
        "construction_unit": ("建设单位", "业主单位"),
        "design_unit": ("设计单位", "勘察、设计单位", "勘察设计单位"),
        "supervision_unit": ("监理单位",),
        "general_contractor": ("施工总包单位", "施工总承包单位", "总承包单位"),
        "subcontractor": ("分包单位", "专业分包单位"),
        "construction_company": ("施工单位", "编制单位", "编制主体"),
    }
    SCHEDULE_LABELS = {
        "plan_start_date": ("计划开工日期", "计划开工时间"),
        "plan_completion_date": ("计划竣工日期", "计划完工日期", "计划完成日期"),
        "duration_days": ("计划总工期", "总工期", "计划工期"),
    }
    DEFAULT_WEIGHTS = {
        "source_text": 90,
        "value": 50,
        "name": 50,
        "code": 45,
        "specification": 35,
        "usage": 35,
    }

    def locate(self, analysis: DocumentAnalysis, pages: list[PDFPageText]) -> DocumentAnalysis:
        contents_pages = self._find_contents_pages(pages)
        regions = SemanticRegionService().classify(pages)
        payload = DeterministicPostProcessingService().process(analysis, pages).model_dump()
        work_context_pages = self._work_item_context_pages(payload, pages)
        method_context_pages = self._method_context_pages(payload, regions)
        standard_context_pages = self._text_context_pages(
            pages, regions, r"编制(?:说明及)?依据|依据文件|主要规范标准", 350
        )
        material_context_pages = self._text_context_pages(
            pages, regions, r"材料计划|主要材料|材料清单|材料表", 60
        )

        for field in ("document_type", "engineering_category", "specialty"):
            if isinstance(payload.get(field), dict):
                context_patterns = ()
                if field == "engineering_category":
                    context_patterns = ("项目安装工程",)
                elif field == "specialty":
                    context_patterns = ("给排水工程",)
                preferred_pages = None
                if field == "engineering_category":
                    project_name = payload.get("project", {}).get("project_name")
                    if isinstance(project_name, dict) and project_name.get("source_page"):
                        preferred_pages = {project_name["source_page"]: 400}
                        context_patterns = (*context_patterns, "工程名称", "项目名称")
                elif field == "document_type":
                    preferred_pages = {
                        page.page_number: 500
                        for page in pages
                        if page.page_number <= 3
                    }
                self._locate_item(
                    payload[field],
                    pages,
                    ("source_text", "value"),
                    contents_pages,
                    preferred_pages=preferred_pages,
                    context_patterns=context_patterns,
                    context_pattern_bonus=100,
                )

        for field, item in payload["project"].items():
            if isinstance(item, dict):
                preferred_pages = None
                if field == "project_location" and item.get("source_page"):
                    preferred_pages = {item["source_page"]: 400}
                elif field == "construction_company" and item.get("source_page") in {
                    page.page_number for page in pages if page.page_number <= 3
                }:
                    preferred_pages = {item["source_page"]: 300}
                self._locate_item(
                    item,
                    pages,
                    ("source_text", "value"),
                    contents_pages,
                    preferred_labels=self.PROJECT_LABELS.get(field, ()),
                    preferred_pages=preferred_pages,
                )
        for field, item in payload["schedule"].items():
            if isinstance(item, dict):
                self._locate_item(
                    item,
                    pages,
                    ("source_text", "value"),
                    contents_pages,
                    preferred_labels=self.SCHEDULE_LABELS.get(field, ()),
                )

        for item in payload["standards"]:
            self._locate_item(
                item,
                pages,
                ("source_text", "name", "code"),
                contents_pages,
                preferred_pages=standard_context_pages,
            )

        located_work_items: list[dict[str, Any]] = []
        for item in payload["main_work_items"]:
            normalized = {"value": item} if isinstance(item, str) else item
            if isinstance(normalized, dict):
                keywords = self._work_item_keywords(str(normalized.get("value") or ""))
                if not self._has_verified_numbered_work_source(normalized, pages):
                    self._locate_item(
                        normalized,
                        pages,
                        ("source_text", "value"),
                        contents_pages,
                        preferred_pages=work_context_pages,
                        weights={"source_text": 120, "value": 50},
                        extra_candidates=keywords,
                    )
                located_work_items.append(normalized)
        payload["main_work_items"] = located_work_items

        located_basis: list[Any] = []
        for item in payload["compilation_basis"]:
            normalized = {"value": item} if isinstance(item, str) else item
            if isinstance(normalized, dict):
                self._locate_item(
                    normalized,
                    pages,
                    ("source_text", "value"),
                    contents_pages,
                    preferred_pages=standard_context_pages,
                )
                if regions.get(normalized.get("source_page")) != "calculation":
                    located_basis.append(normalized)
        payload["compilation_basis"] = located_basis

        located_materials: list[dict[str, Any]] = []
        for item in payload["main_materials"]:
            self._locate_item(
                item,
                pages,
                ("source_text", "name", "specification", "usage"),
                contents_pages,
                weights={"source_text": 50, "name": 50, "specification": 35, "usage": 35},
                preferred_pages=material_context_pages,
                context_patterns=(
                    "室内", "立管", "横管", "支管", "采用", "使用部位",
                    "连接方式", "热熔", "粘接", "承插",
                ),
            )
            if regions.get(item.get("source_page")) != "calculation":
                located_materials.append(item)
        payload["main_materials"] = located_materials
        for item in payload["main_methods"]:
            keywords = self._method_keywords(str(item.get("name") or ""))
            if not self._has_verified_method_source(item, pages):
                self._locate_item(
                    item,
                    pages,
                    ("source_text", "name"),
                    contents_pages,
                    preferred_pages=method_context_pages,
                    weights={"source_text": 100, "name": 30},
                    extra_candidates=keywords,
                    context_patterns=(
                        "施工步骤", "操作程序", "冲洗、消毒", "生活给水管道及设备消毒",
                        "冲洗步骤", "消毒方法",
                    ),
                    penalty_patterns=("系统试验包括", "试验包括", "概述"),
                )
            title_matches: list[tuple[int, int, str]] = []
            for page in pages:
                if (
                    page.page_number in contents_pages
                    or regions.get(page.page_number) == "calculation"
                ):
                    continue
                matched = self._method_title_match(
                    str(item.get("name") or ""), page.text, keywords
                )
                if matched:
                    score, excerpt = matched
                    title_matches.append((score, page.page_number, excerpt))
            if title_matches:
                # A matching numbered body title is stronger evidence than a
                # later continuation paragraph or an isolated flow-chart term.
                _, page_number, excerpt = max(
                    title_matches, key=lambda value: (value[0], -value[1])
                )
                item["source_page"] = page_number
                item["source_text"] = excerpt
        located = DocumentAnalysis.model_validate(payload)
        return DeterministicPostProcessingService().final_cleanup(located, pages)

    @staticmethod
    def _has_verified_method_source(
        item: dict[str, Any], pages: list[PDFPageText]
    ) -> bool:
        page_number = item.get("source_page")
        source_text = str(item.get("source_text") or "").strip()
        first_line = source_text.splitlines()[0] if source_text else ""
        if not re.match(r"^\s*\d+(?:\.\d+)*[、.．]", first_line):
            return False
        return any(
            page.page_number == page_number and first_line in page.text
            for page in pages
        )

    @staticmethod
    def _has_verified_numbered_work_source(
        item: dict[str, Any], pages: list[PDFPageText]
    ) -> bool:
        page_number = item.get("source_page")
        source_text = str(item.get("source_text") or "").strip()
        first_line = source_text.splitlines()[0] if source_text else ""
        if not re.match(r"^\s*[（(]?\d+[）)]?、", first_line):
            return False
        return any(
            page.page_number == page_number and first_line in page.text for page in pages
        )

    def _locate_item(
        self,
        item: dict[str, Any],
        pages: list[PDFPageText],
        candidate_fields: tuple[str, ...],
        contents_pages: set[int],
        *,
        preferred_labels: tuple[str, ...] = (),
        preferred_pages: dict[int, int] | None = None,
        weights: dict[str, int] | None = None,
        extra_candidates: list[tuple[str, int]] | None = None,
        context_patterns: tuple[str, ...] = (),
        context_pattern_bonus: int = 25,
        penalty_patterns: tuple[str, ...] = (),
    ) -> None:
        field_weights = self.DEFAULT_WEIGHTS | (weights or {})
        candidates = [
            (field, str(item[field]).strip(), field_weights.get(field, 20))
            for field in candidate_fields
            if item.get(field) is not None and str(item[field]).strip()
        ]
        candidates.extend(
            (f"keyword_{index}", candidate, weight)
            for index, (candidate, weight) in enumerate(extra_candidates or [])
            if candidate
        )
        item["source_page"] = None
        item["source_text"] = None
        page_results: list[tuple[int, int, str]] = []

        for page in pages:
            page_score = preferred_pages.get(page.page_number, 0) if preferred_pages else 0
            evidence_count = 0
            anchors: list[tuple[int, str]] = []
            evidence_spans: list[tuple[int, int]] = []
            for field, candidate, candidate_weight in candidates:
                pattern = self._flexible_pattern(candidate)
                if pattern is None:
                    continue
                occurrences = []
                for match in pattern.finditer(page.text):
                    snippet = self._source_snippet(page.text, match.start(), match.end())
                    quality = self._match_score(snippet, candidate, page.page_number in contents_pages)
                    occurrences.append((quality, snippet, match.start(), match.end()))
                if occurrences:
                    quality, snippet, start, end = max(
                        occurrences, key=lambda value: value[0]
                    )
                    page_score += candidate_weight + quality
                    evidence_count += 1
                    anchors.append((candidate_weight + quality, snippet))
                    evidence_spans.append((start, end))

            if evidence_count and context_patterns:
                context_hits = sum(
                    1 for pattern in context_patterns if re.search(re.escape(pattern), page.text)
                )
                page_score += context_hits * context_pattern_bonus
            if evidence_count and penalty_patterns:
                page_score -= sum(
                    250 for pattern in penalty_patterns if re.search(re.escape(pattern), page.text)
                )

            if len(evidence_spans) > 1:
                anchors.append(
                    (
                        200,
                        self._source_span(
                            page.text,
                            min(span[0] for span in evidence_spans),
                            max(span[1] for span in evidence_spans),
                        ),
                    )
                )

            label_anchor = self._label_anchor(page.text, preferred_labels)
            if label_anchor and evidence_count:
                page_score += 250
                anchors.append((300, label_anchor))
            if evidence_count:
                page_score += evidence_count * 10
                if page.page_number in contents_pages:
                    page_score -= 100
                anchor = max(anchors, key=lambda value: value[0])[1]
                page_results.append((page_score, page.page_number, anchor))

        if page_results:
            _, page_number, snippet = max(page_results, key=lambda value: (value[0], -value[1]))
            item["source_page"] = page_number
            item["source_text"] = snippet

    @staticmethod
    def _label_anchor(text: str, labels: tuple[str, ...]) -> str | None:
        for label in labels:
            match = re.search(
                rf"(?:^|\n)\s*{re.escape(label)}\s*(?:[：:|｜]\s*|(?=\s)|$)",
                text,
                re.MULTILINE,
            )
            if match:
                line_end = text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(text)
                anchor = text[match.start() : line_end].strip()
                remainder = re.sub(
                    rf"^\s*{re.escape(label)}\s*[：:|｜]?\s*", "", anchor
                )
                if not remainder and line_end < len(text):
                    next_end = text.find("\n", line_end + 1)
                    if next_end == -1:
                        next_end = len(text)
                    anchor = text[match.start() : next_end].strip()
                return anchor
        return None

    @staticmethod
    def _flexible_pattern(value: str) -> re.Pattern[str] | None:
        compact = re.sub(r"\s+", "", value)
        if not compact:
            return None
        return re.compile(r"\s*".join(re.escape(character) for character in compact), re.IGNORECASE)

    @staticmethod
    def _source_snippet(text: str, start: int, end: int) -> str:
        return SourceLocatorService._source_span(text, start, end)

    @staticmethod
    def _source_span(text: str, start: int, end: int) -> str:
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        if line_end == -1:
            line_end = len(text)
        snippet = text[line_start:line_end].strip()
        if len(snippet) <= 300:
            return snippet
        window_start = max(0, start - line_start - 100)
        return snippet[window_start : window_start + 300].strip()

    @staticmethod
    def _find_contents_pages(pages: list[PDFPageText]) -> set[int]:
        contents_pages: set[int] = set()
        in_contents = False
        for page in pages:
            lines = [line.strip() for line in page.text.splitlines() if line.strip()]
            if any(re.fullmatch(r"(?:目录|目\s*录|contents)", line, re.IGNORECASE) for line in lines):
                in_contents = True
            has_toc_lines = any(
                re.search(r"(?:\.{2,}|…{2,}|·{2,})\s*\d+\s*$", line) for line in lines
            )
            if in_contents and has_toc_lines:
                contents_pages.add(page.page_number)
            elif in_contents and contents_pages:
                in_contents = False
        return contents_pages

    @staticmethod
    def _work_item_context_pages(payload: dict[str, Any], pages: list[PDFPageText]) -> dict[int, int]:
        preferred: dict[int, int] = {}

        def prefer(page_number: int, score: int) -> None:
            preferred[page_number] = max(preferred.get(page_number, 0), score)

        for chapter in payload.get("chapters", []):
            if re.search(r"项目主要施工内容|主要施工内容|施工范围|工程内容|工程概况", chapter["title"]):
                end_page = chapter.get("end_page") or chapter["start_page"]
                chapter_score = (
                    180
                    if re.search(r"项目主要施工内容|主要施工内容|施工范围|工程内容", chapter["title"])
                    else 100
                )
                for page_number in range(chapter["start_page"], end_page + 1):
                    prefer(page_number, chapter_score)
        for index, page in enumerate(pages):
            if re.search(r"项目主要施工内容|主要施工内容|施工范围|工程内容", page.text):
                prefer(page.page_number, 250)
                if (
                    index + 1 < len(pages)
                    and pages[index + 1].page_number == page.page_number + 1
                ):
                    prefer(pages[index + 1].page_number, 180)
        return preferred

    @staticmethod
    def _method_context_pages(
        payload: dict[str, Any], regions: dict[int, str]
    ) -> dict[int, int]:
        preferred: dict[int, int] = {}
        for chapter in payload.get("chapters", []):
            title = chapter["title"]
            if (
                DeterministicPostProcessingService.EXCLUDED_METHOD_CONTEXT.search(title)
                or not DeterministicPostProcessingService.METHOD_PARENT.search(title)
            ):
                continue
            score = 180
            end_page = chapter.get("end_page") or chapter["start_page"]
            for page_number in range(chapter["start_page"], end_page + 1):
                if regions.get(page_number) == "calculation":
                    continue
                preferred[page_number] = max(preferred.get(page_number, 0), score)
        return preferred

    @staticmethod
    def _text_context_pages(
        pages: list[PDFPageText],
        regions: dict[int, str],
        pattern: str,
        score: int,
    ) -> dict[int, int]:
        preferred: dict[int, int] = {}
        remaining = 0
        for page in pages:
            if regions.get(page.page_number) == "calculation":
                remaining = 0
                continue
            if re.search(pattern, page.text):
                remaining = 2
            if remaining:
                preferred[page.page_number] = score
                remaining -= 1
        return preferred

    @staticmethod
    def _method_keywords(name: str) -> list[tuple[str, int]]:
        """Derive search evidence only; returned source text always comes from the PDF."""
        if not name:
            return []
        keywords: list[tuple[str, int]] = []

        def add(value: str, weight: int) -> None:
            cleaned = value.strip()
            if len(cleaned) >= 2 and all(existing != cleaned for existing, _ in keywords):
                keywords.append((cleaned, weight))

        core = SourceLocatorService._method_core(name)
        if len(core) >= 4:
            add(core, 70)

        for pattern, weight in (
            (r"PP-?R", 55),
            (r"UPVC", 55),
            (r"热熔连接", 55),
            (r"塑料排水管", 45),
            (r"铸铁排水管", 50),
            (r"柔性", 35),
            (r"管道支架", 50),
            (r"管道试压", 55),
            (r"管道冲洗", 50),
            (r"消毒", 40),
        ):
            if match := re.search(pattern, name, re.IGNORECASE):
                add(match.group(0), weight)

        if "支架" in name:
            add("支吊架", 40)
            add("支架安装", 40)
        if "试压" in name:
            add("试验压力", 45)
            add("压力试验", 45)
        if "冲洗" in name:
            add("冲洗", 35)
        if "安装" in name:
            add("安装", 10)
        return keywords

    @staticmethod
    def _method_core(name: str) -> str:
        return re.sub(
            r"(?:施工工艺流程|施工工艺|施工方法|安装方法|试验方法|方法|工艺流程)$",
            "",
            name,
        ).strip(" ：:、，,")

    @classmethod
    def _method_title_match(
        cls, method_name: str, page_text: str, keywords: list[tuple[str, int]]
    ) -> tuple[int, str] | None:
        heading_pattern = re.compile(
            r"(?m)^\s*(?P<prefix>第[一二三四五六七八九十百]+[章节篇]|"
            r"[一二三四五六七八九十百]+、|\d+、|\d+(?:\.\d+)+[、.．]?)\s*"
            r"(?P<title>[^\n]{2,80})$"
        )
        headings = list(heading_pattern.finditer(page_text))
        if not headings:
            return None
        method_core = re.sub(r"\s+", "", cls._method_core(method_name))
        scored: list[tuple[int, int]] = []
        for index, heading in enumerate(headings):
            title = heading.group("title").strip(" ：:、，,。")
            compact_title = re.sub(r"\s+", "", title)
            score = 0
            if method_core and compact_title == method_core:
                score += 140
            elif (
                method_core
                and min(len(method_core), len(compact_title)) >= 4
                and (method_core in compact_title or compact_title in method_core)
            ):
                score += 90
            for keyword, weight in keywords:
                if re.sub(r"\s+", "", keyword).lower() in compact_title.lower():
                    score += min(weight, 60)
            if score >= 40:
                scored.append((score, index))
        if not scored:
            return None
        selected_score, selected_index = max(scored, key=lambda value: value[0])
        start = headings[selected_index].start()
        selected_level = cls._heading_prefix_level(
            headings[selected_index].group("prefix")
        )
        end = len(page_text)
        for following in headings[selected_index + 1 :]:
            if cls._heading_prefix_level(following.group("prefix")) <= selected_level:
                end = following.start()
                break
        return selected_score, page_text[start:end].strip()[:300]

    @staticmethod
    def _heading_prefix_level(prefix: str) -> int:
        number = re.match(r"\d+(?:\.\d+)+", prefix)
        return number.group(0).count(".") + 1 if number else 1

    @staticmethod
    def _work_item_keywords(value: str) -> list[tuple[str, int]]:
        if not value:
            return []
        keywords: list[tuple[str, int]] = []
        core = re.sub(r"(?:系统)?(?:施工|工程|安装)$", "", value).strip()
        if len(core) >= 2 and core != value:
            keywords.append((core, 50))
        # Preserve a meaningful system phrase when only the activity suffix is semantic.
        system_core = re.sub(r"(?:施工|工程|安装)$", "", value).strip()
        if (
            len(system_core) >= 2
            and system_core != value
            and all(existing != system_core for existing, _ in keywords)
        ):
            keywords.append((system_core, 60))
        return keywords

    @staticmethod
    def _match_score(snippet: str, candidate: str, is_contents_page: bool) -> int:
        score = 0
        if re.search(r"(?:\.{2,}|…{2,}|·{2,})\s*\d+\s*$", snippet):
            score -= 100
        compact_candidate = re.sub(r"\s+", "", candidate).strip("。；;，,：:")
        heading = re.sub(
            r"^\s*(?:第[一二三四五六七八九十百]+[章节篇]|"
            r"[一二三四五六七八九十百]+、|\d+、|\d+(?:\.\d+)+[、.．]?)\s*",
            "",
            snippet,
        )
        compact_heading = re.sub(r"\s+", "", heading).strip("。；;，,：:")
        if compact_heading == compact_candidate:
            score += 60
        elif compact_candidate in compact_heading and len(compact_heading) <= len(compact_candidate) + 20:
            score += 20
        if is_contents_page:
            score -= 50
        return score
