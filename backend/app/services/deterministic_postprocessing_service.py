"""Stable V0.2 completion from explicit numbered PDF text."""

import re
from typing import Any

from app.schemas.analysis import (
    DocumentAnalysis,
    MethodInfo,
    SourceValue,
)
from app.services.pdf_service import PDFPageText
from app.services.semantic_region_service import SemanticRegionService


class DeterministicPostProcessingService:
    NUMBERED_HEADING = re.compile(
        r"(?m)^\s*(?P<number>\d+(?:\.\d+)*)"
        r"(?:[、.．][ \t]*|[ \t]+)(?P<title>[^\n]{2,80})$"
    )
    METHOD_PARENT = re.compile(
        r"施工方法|系统安装及施工|施工工艺|工艺流程与施工方法|安装施工|"
        r"搭设工艺|支撑.*(?:搭设|拆除)|拆除工艺|拆除施工|"
        r"专项施工方案|施工技术|施工方案|管道试验"
    )
    WORK_PARENT_TITLE = re.compile(
        r"^(?:施工方法|系统安装及施工|施工工艺(?:技术)?|安装施工|专项施工方案|"
        r"施工技术|施工方案|管道试验)$"
    )
    WORK_SKELETON_AUXILIARY = re.compile(
        r"总体.*工艺流程|施工工艺流程$|施工流程$|技术参数|主要施工设备|"
        r"操作要求|注意事项|检验$|检查与维护|维护措施$"
    )
    METHOD_SEMANTICS = re.compile(
        r"安装|施工|敷设|配线|布线|调试|测试|接地|连接|制作|试压|试验|"
        r"冲洗|消毒|灌水|通球|通水|支架|吊装|铺设|组装|校准|测量|放样|"
        r"开挖|喷播|回填|浇筑|注浆|支护|张拉|焊接|监测|定位|设置|搭设|拆除|卸荷"
    )
    EXPLICIT_TEST_METHOD = re.compile(r"试压|灌水|通球试验|通水试验|冲洗|消毒")
    EXCLUDED_METHOD_CONTEXT = re.compile(
        r"编制(?:说明及)?依据|依据文件|施工图纸|设计图纸|合同(?:文件)?|规范标准|"
        r"工程概况|项目简介|工程简介|系统简介|材料计划|材料选型|材料设备|"
        r"设备计划|机具表|施工组织|组织架构|人员配备|劳动力(?:计划|配置)|"
        r"工程管理|项目管理|岗位职责|常见质量通病|质量保证|质量管理|质量控制|"
        r"成品保护|安全防护|安全措施|安全管理|文明施工|职业健康安全|环境保护|"
        r"应急|验收|管理制度|售后服务|培训|维护保养|计算书|设计计算|计算分析|验算"
    )
    STATEMENT_HEADING = re.compile(
        r"^(?:安装|施工|调试|测试|敷设)?(?:前|后)应|^(?:应|必须|不得|严禁|要|需|按)|"
        r"应当|应在|应按|应保证|应提前|应严格|应符合|应采用|应将|应由|应先|"
        r"应及时|应进行|应设置|应安装|必须|不得|严禁|要安装|需在|方可"
    )
    AUXILIARY_METHOD_TITLE = re.compile(
        r"认证测试(?:标准|模型|参数)|测试工具|施工前(?:的)?准备|准备工作|"
        r"检查项目|领导小组|危险源"
    )
    METHOD_NOUN_ENDING = re.compile(
        r"(?:安装|施工|施工要求|施工工艺|施工工艺流程|敷设|配线|布线|调试|测试|"
        r"接地|连接|制作|试压|试验|吊装|铺设|组装|校准|冲洗|消毒|灌水|通球试验|"
        r"通水试验)$"
    )
    ORDINARY_ENUMERATION = re.compile(r"^\s*(?:[（(]\d+[）)]、|\d+、)")
    WORK_CONTEXT = re.compile(r"项目主要施工内容|主要施工内容|施工范围")
    WORK_ITEM = re.compile(
        r"^\s*[（(]?(?P<number>\d+)[）)]?、\s*"
        r"(?P<title>[^：:\n]{2,30})\s*[：:]\s*(?P<body>.*)$"
    )
    WORK_SEMANTICS = re.compile(r"给水|排水|雨水|污水|废水|卫生洁具|消防|电气|暖通|通风")
    WORK_ACTION = re.compile(
        r"开挖|施工|安装|浇筑|支护|注浆|喷播|铺设|回填|吊装|拆除|砌筑|"
        r"敷设|调试|连接|制作|加固|搭设|张拉|焊接|监测|定位|设置"
    )
    WORK_ENTITY = re.compile(r"沟|槽|墙|孔|边坡|基坑|路面|管道|系统|桥梁|涵洞")
    MATERIAL_FORM = re.compile(r"(?:碎石|砂|水泥|布|板材|箱板)$")
    QUANTITY_TABLE = re.compile(r"工程量表|工程数量表|主要工程数量|项目\s*单位\s*数量")
    CALCULATION_TITLE = re.compile(r"计算书|设计计算|计算分析|验算")
    CONSTRUCTION_TYPE_ONLY = re.compile(
        r"(?:构件|组件|装置|架|支撑体系|承重体系|构造形式|结构形式|结构型式|方案类型)$"
    )
    AREA_ONLY = re.compile(
        r"^(?:地下|地上)?[一二三四五六七八九十百\d]+层.*(?:用房|大厅|露台|区域|楼面|屋面)$"
    )

    def __init__(self) -> None:
        self.semantic_regions = SemanticRegionService()

    def process(
        self, analysis: DocumentAnalysis, pages: list[PDFPageText]
    ) -> DocumentAnalysis:
        payload = analysis.model_dump()
        chapters = payload.get("chapters", [])
        deterministic_methods = self._extract_methods(chapters, pages)
        payload["main_methods"] = self._merge_methods(
            deterministic_methods, payload.get("main_methods", []), chapters
        )
        deterministic_work = self._extract_work_items(
            chapters,
            pages,
            allow_chapter_fallback=len(payload.get("main_work_items", [])) < 2,
        )
        payload["main_work_items"] = self._merge_work_items(
            deterministic_work, payload.get("main_work_items", [])
        )
        return DocumentAnalysis.model_validate(payload)

    def final_cleanup(
        self, analysis: DocumentAnalysis, pages: list[PDFPageText]
    ) -> DocumentAnalysis:
        payload = analysis.model_dump()
        chapters = payload.get("chapters", [])
        regions = self.semantic_regions.classify(pages)
        filtered_methods = [
            MethodInfo.model_validate(item)
            for item in payload.get("main_methods", [])
            if self._is_allowed_final_method(item, chapters)
            and regions.get(item.get("source_page")) != "calculation"
        ]
        payload["main_methods"] = [
            item.model_dump()
            for item in self._deduplicate_methods(filtered_methods, chapters)
        ]

        deterministic_work = self._extract_work_items(
            chapters,
            pages,
            allow_chapter_fallback=len(payload.get("main_work_items", [])) < 2,
        )
        if deterministic_work:
            payload["main_work_items"] = self._merge_work_items(
                deterministic_work,
                payload.get("main_work_items", []),
                restrict_to_deterministic=(
                    len(deterministic_work) >= 2
                    or len(payload.get("main_work_items", [])) <= 1
                ),
            )
        payload["main_work_items"] = self._cleanup_work_items(payload, pages)
        return DocumentAnalysis.model_validate(payload)

    def _extract_methods(
        self, chapters: list[dict[str, Any]], pages: list[PDFPageText]
    ) -> list[MethodInfo]:
        relevant_pages: set[int] = set()
        explicit_process_starts: dict[int, int] = {}
        regions = self.semantic_regions.classify(pages)
        for chapter in chapters:
            title = str(chapter.get("title") or "")
            if not (self.METHOD_PARENT.search(title) or self.EXPLICIT_TEST_METHOD.search(title)):
                continue
            start = chapter["start_page"]
            end = chapter.get("end_page") or start
            relevant_pages.update(
                page_number
                for page_number in range(start, end + 1)
                if regions.get(page_number) != "calculation"
            )
            marker_pages = [
                (page.page_number, match.end())
                for page in pages
                if start <= page.page_number <= end
                and (match := re.search(
                    r"工艺流程与施工方法|施工方法及操作要求", page.text
                ))
            ]
            if marker_pages:
                first_page, first_offset = min(marker_pages)
                for page_number in range(first_page, end + 1):
                    if regions.get(page_number) != "calculation":
                        explicit_process_starts[page_number] = (
                            first_offset if page_number == first_page else 0
                        )

        methods: list[MethodInfo] = []
        for page in pages:
            if page.page_number not in relevant_pages:
                continue
            headings = list(self.NUMBERED_HEADING.finditer(page.text))
            for index, heading in enumerate(headings):
                title = heading.group("title").strip(" ：:、，,。")
                depth = heading.group("number").count(".") + 1
                if (
                    (
                        not self.METHOD_SEMANTICS.search(title)
                        and (
                            page.page_number not in explicit_process_starts
                            or heading.start()
                            < explicit_process_starts[page.page_number]
                        )
                    )
                    or self.WORK_PARENT_TITLE.fullmatch(title)
                    or self.EXCLUDED_METHOD_CONTEXT.search(title)
                    or not self._is_method_heading(title, depth)
                ):
                    continue
                chapter_context = self._method_chapter_context(title, chapters)
                if chapter_context is False:
                    continue
                end = len(page.text)
                for following in headings[index + 1 :]:
                    following_depth = following.group("number").count(".") + 1
                    if following_depth <= depth:
                        end = following.start()
                        break
                source_text = page.text[heading.start() : end].strip()[:300]
                methods.append(
                    MethodInfo(
                        name=title,
                        source_page=page.page_number,
                        source_text=source_text,
                    )
                )
        return self._deduplicate_methods(methods, chapters)

    def _extract_work_items(
        self,
        chapters: list[dict[str, Any]],
        pages: list[PDFPageText],
        allow_chapter_fallback: bool = True,
    ) -> list[SourceValue]:
        context_pages: set[int] = set()
        regions = self.semantic_regions.classify(pages)
        for chapter in chapters:
            if re.search(r"工程概况|主要施工内容|施工范围", str(chapter.get("title") or "")):
                start = chapter["start_page"]
                end = chapter.get("end_page") or start
                context_pages.update(
                    page_number
                    for page_number in range(start, end + 1)
                    if regions.get(page_number) != "calculation"
                )
        for index, page in enumerate(pages):
            if self.WORK_CONTEXT.search(page.text):
                if regions.get(page.page_number) == "calculation":
                    continue
                context_pages.add(page.page_number)
                if (
                    index + 1 < len(pages)
                    and pages[index + 1].page_number == page.page_number + 1
                ):
                    context_pages.add(pages[index + 1].page_number)

        contextual_lines = [
            (page.page_number, line.strip())
            for page in pages
            if page.page_number in context_pages
            for line in page.text.splitlines()
            if line.strip()
        ]
        work_items: list[SourceValue] = []
        for index, (page_number, line) in enumerate(contextual_lines):
            match = self.WORK_ITEM.match(line)
            if not match:
                continue
            title = match.group("title").strip()
            if not self.WORK_SEMANTICS.search(title):
                continue

            source_lines = [line]
            for _, continuation in contextual_lines[index + 1 :]:
                if self.WORK_ITEM.match(continuation) or self.NUMBERED_HEADING.match(
                    continuation
                ):
                    break
                source_lines.append(continuation)
                if len("\n".join(source_lines)) >= 300:
                    break
            work_items.append(
                SourceValue(
                    value=title,
                    source_page=page_number,
                    source_text="\n".join(source_lines)[:300],
                )
            )
        if allow_chapter_fallback and len(work_items) < 2:
            work_items.extend(self._work_items_from_construction_chapters(chapters, pages))
        return self._deduplicate_work_items(work_items)

    def _work_items_from_construction_chapters(
        self, chapters: list[dict[str, Any]], pages: list[PDFPageText]
    ) -> list[SourceValue]:
        """Build a conservative work skeleton from real construction headings."""
        full_text = "\n".join(page.text for page in pages)
        regions = self.semantic_regions.classify(pages)
        result: list[SourceValue] = []
        for index, chapter in enumerate(chapters):
            title = str(chapter.get("title") or "").strip()
            if (
                not title
                or self.WORK_PARENT_TITLE.fullmatch(title)
                or self.WORK_SKELETON_AUXILIARY.search(title)
                or self.EXCLUDED_METHOD_CONTEXT.search(title)
                or self.CALCULATION_TITLE.search(title)
                or self.AREA_ONLY.fullmatch(title)
                or not (
                    self.WORK_ACTION.search(title)
                    or self.WORK_SEMANTICS.search(title)
                    or re.search(r"施工方案$", title)
                )
                or not self._is_direct_construction_child(index, chapters)
                or regions.get(chapter.get("start_page")) == "calculation"
            ):
                continue
            value = re.sub(r"(?:专项)?施工方案$|方案$", "", title).strip()
            value = re.sub(r"施工$", "", value).strip()
            if re.search(r"(?:桩|放坡)$", value) and "支护" in full_text:
                value = f"{value}支护"
            if value == "基坑开挖" and re.search(r"土方开挖|基坑土方", full_text):
                value = "基坑土方开挖"
            if not value:
                continue
            page_number = chapter.get("start_page")
            source_text = self._chapter_source_text(title, page_number, pages)
            result.append(
                SourceValue(
                    value=value,
                    source_page=page_number if source_text else None,
                    source_text=source_text,
                )
            )
        return result

    def _is_direct_construction_child(
        self, index: int, chapters: list[dict[str, Any]]
    ) -> bool:
        chapter = chapters[index]
        level = chapter.get("level") or 1
        for parent in reversed(chapters[:index]):
            parent_level = parent.get("level") or 1
            if parent_level >= level:
                continue
            parent_title = str(parent.get("title") or "")
            return bool(
                level == parent_level + 1
                and self.METHOD_PARENT.search(parent_title)
                and not self.EXCLUDED_METHOD_CONTEXT.search(parent_title)
            )
        return False

    @staticmethod
    def _chapter_source_text(
        title: str, page_number: int | None, pages: list[PDFPageText]
    ) -> str | None:
        for page in pages:
            if page.page_number != page_number:
                continue
            lines = page.text.splitlines()
            for index, line in enumerate(lines):
                if title not in line:
                    continue
                return "\n".join(lines[index : index + 3]).strip()[:300] or None
        return None

    def _merge_methods(
        self,
        deterministic: list[MethodInfo],
        llm_items: list[dict[str, Any]],
        chapters: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged = [item.model_dump() for item in deterministic]
        merged.extend(llm_items)
        return [item.model_dump() for item in self._deduplicate_methods(
            [MethodInfo.model_validate(item) for item in merged], chapters
        )]

    def _merge_work_items(
        self,
        deterministic: list[SourceValue],
        llm_items: list[Any],
        restrict_to_deterministic: bool = True,
    ) -> list[Any]:
        result: list[Any] = [item.model_dump() for item in deterministic]
        known = {
            self._canonical_work_name(str(item.value or "")) for item in deterministic
        }
        for item in llm_items:
            value = item if isinstance(item, str) else item.get("value")
            identity = self._canonical_work_name(str(value or ""))
            if identity and not any(
                self._same_work_item(identity, known_item) for known_item in known
            ):
                if (
                    deterministic
                    and restrict_to_deterministic
                    and not self._has_numbered_work_evidence(item)
                ):
                    continue
                result.append(item)
                known.add(identity)
        return result

    def _is_allowed_final_method(
        self, item: dict[str, Any], chapters: list[dict[str, Any]]
    ) -> bool:
        name = str(item.get("name") or "").strip()
        source_text = str(item.get("source_text") or "")
        if not name or self.ORDINARY_ENUMERATION.match(name):
            return False
        source_heading = self.NUMBERED_HEADING.match(source_text)
        depth = (
            source_heading.group("number").count(".") + 1
            if source_heading is not None
            else None
        )
        if not self._is_method_heading(name, depth):
            return False
        source_heading_line = source_text.splitlines()[0] if source_text else ""
        if self.EXCLUDED_METHOD_CONTEXT.search(name) or self.EXCLUDED_METHOD_CONTEXT.search(
            source_heading_line
        ):
            return False
        if (
            self.CONSTRUCTION_TYPE_ONLY.search(name)
            and not self.METHOD_SEMANTICS.search(name)
            and not self._has_explicit_method_heading_evidence(item, chapters)
        ):
            return False
        if not chapters:
            return True

        chapter_match = self._method_chapter_context(name, chapters)
        if chapter_match is not None:
            return chapter_match

        source_page = item.get("source_page")
        allowed_ranges = [
            chapter
            for chapter in chapters
            if not self.EXCLUDED_METHOD_CONTEXT.search(str(chapter.get("title") or ""))
            and (
                self.METHOD_PARENT.search(str(chapter.get("title") or ""))
                or self.EXPLICIT_TEST_METHOD.search(str(chapter.get("title") or ""))
            )
        ]
        return any(
            source_page is not None
            and chapter["start_page"]
            <= source_page
            <= (chapter.get("end_page") or chapter["start_page"])
            for chapter in allowed_ranges
        )

    def _has_explicit_method_heading_evidence(
        self, item: dict[str, Any], chapters: list[dict[str, Any]]
    ) -> bool:
        source_text = str(item.get("source_text") or "").strip()
        first_line = source_text.splitlines()[0] if source_text else ""
        heading = self.NUMBERED_HEADING.match(first_line)
        if heading is None:
            return False
        if self._normalized_title(heading.group("title")) != self._normalized_title(
            str(item.get("name") or "")
        ):
            return False
        return self._method_chapter_context(str(item.get("name") or ""), chapters) is True

    def _method_chapter_context(
        self, method_name: str, chapters: list[dict[str, Any]]
    ) -> bool | None:
        method_key = self._normalized_title(method_name)
        for index, chapter in enumerate(chapters):
            if self._normalized_title(str(chapter.get("title") or "")) != method_key:
                continue
            level = chapter.get("level") or 1
            context_titles = [str(chapter.get("title") or "")]
            parent_level = level
            for parent in reversed(chapters[:index]):
                candidate_level = parent.get("level") or 1
                if candidate_level < parent_level:
                    context_titles.append(str(parent.get("title") or ""))
                    parent_level = candidate_level
            context = " ".join(context_titles)
            if self.EXCLUDED_METHOD_CONTEXT.search(context):
                return False
            return bool(
                self.METHOD_PARENT.search(context)
                or self.EXPLICIT_TEST_METHOD.search(context)
            )
        return None

    def _has_numbered_work_evidence(self, item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        source_text = str(item.get("source_text") or "").strip()
        first_line = source_text.splitlines()[0] if source_text else ""
        match = self.WORK_ITEM.match(first_line)
        return bool(match and self.WORK_SEMANTICS.search(match.group("title")))

    def _cleanup_work_items(
        self, payload: dict[str, Any], pages: list[PDFPageText]
    ) -> list[dict[str, Any]]:
        regions = self.semantic_regions.classify(pages)
        references = [
            self._canonical_work_name(str(item.get("name") or ""))
            for item in payload.get("main_methods", [])
        ]
        references.extend(
            self._canonical_work_name(str(chapter.get("title") or ""))
            for chapter in payload.get("chapters", [])
            if self.METHOD_SEMANTICS.search(str(chapter.get("title") or ""))
            and not self.EXCLUDED_METHOD_CONTEXT.search(
                str(chapter.get("title") or "")
            )
        )
        material_names = [
            self._canonical_work_name(str(item.get("name") or ""))
            for item in payload.get("main_materials", [])
        ]
        page_text = {page.page_number: page.text for page in pages}

        result: list[dict[str, Any]] = []
        for raw_item in payload.get("main_work_items", []):
            item = {"value": raw_item} if isinstance(raw_item, str) else raw_item
            if not isinstance(item, dict):
                continue
            value = str(item.get("value") or "").strip()
            identity = self._canonical_work_name(value)
            if not identity:
                continue
            if regions.get(item.get("source_page")) == "calculation":
                continue
            if self.CALCULATION_TITLE.search(value) or self.AREA_ONLY.fullmatch(value):
                continue
            if self.WORK_ACTION.search(value) or self._has_numbered_work_evidence(item):
                result.append(item)
                continue
            if any(self._work_item_related(identity, reference) for reference in references):
                result.append(item)
                continue
            if any(
                identity == material
                or (len(identity) >= 2 and identity in material)
                or (len(material) >= 2 and material in identity)
                for material in material_names
                if material
            ):
                continue
            if self.WORK_ENTITY.search(value):
                result.append(item)
                continue
            source_page_text = page_text.get(item.get("source_page"), "")
            if self.MATERIAL_FORM.search(value) or self.QUANTITY_TABLE.search(
                source_page_text
            ):
                continue
            result.append(item)
        return result

    @staticmethod
    def _work_item_related(left: str, right: str) -> bool:
        if not left or not right:
            return False
        if left in right or right in left:
            return True
        shorter, longer = sorted((left, right), key=len)
        return any(shorter[index : index + 3] in longer for index in range(len(shorter) - 2))

    @classmethod
    def _is_method_heading(cls, title: str, depth: int | None) -> bool:
        cleaned = title.strip(" ：:、，,。；;…")
        if (
            not cleaned
            or cls.STATEMENT_HEADING.search(cleaned)
            or cls.AUXILIARY_METHOD_TITLE.search(cleaned)
            or cls.CALCULATION_TITLE.search(cleaned)
            or SemanticRegionService.is_formula_or_unit_expression(cleaned)
        ):
            return False
        if len(cleaned) > 24 and re.search(r"[，,；;。]", cleaned):
            return False
        if re.match(r"^(?:层|米|毫米|个|项|在|当|待|为|对|本工程)", cleaned):
            return False
        if re.search(r"[：:]", cleaned):
            return False
        if (
            depth == 1
            and len(cleaned) > 12
            and not re.search(
                r"安装|拆除|施工|设置|卸荷|检查|要求|流程|工艺|试验|调试|敷设$",
                cleaned,
            )
        ):
            return False
        if title.rstrip().endswith(("。", "，", ",", "；", ";", "：", ":", "……", "…")):
            return False
        if depth is not None and depth >= 4:
            return len(cleaned) <= 24 and bool(cls.METHOD_NOUN_ENDING.search(cleaned))
        return len(cleaned) <= 40

    @classmethod
    def _deduplicate_methods(
        cls,
        methods: list[MethodInfo],
        chapters: list[dict[str, Any]] | None = None,
    ) -> list[MethodInfo]:
        result: list[MethodInfo] = []
        positions: dict[str, int] = {}
        for method in methods:
            identity = cls.canonical_method_name(method.name)
            if not identity:
                continue
            if identity not in positions:
                positions[identity] = len(result)
                result.append(method)
                continue
            position = positions[identity]
            if cls._provenance_score(method, chapters or []) > cls._provenance_score(
                result[position], chapters or []
            ):
                result[position] = method
        return result

    @classmethod
    def _deduplicate_work_items(cls, items: list[SourceValue]) -> list[SourceValue]:
        result: list[SourceValue] = []
        seen: set[str] = set()
        for item in items:
            identity = cls._canonical_work_name(str(item.value or ""))
            if identity and identity not in seen:
                result.append(item)
                seen.add(identity)
        return result

    @staticmethod
    def canonical_method_name(name: str) -> str:
        normalized = name.upper().replace("－", "-").replace("—", "-")
        normalized = re.sub(r"[（(][^）)]*[）)]", "", normalized)
        normalized = re.sub(
            r"(?:施工方法|安装方法|试验方法|方法)$", "", normalized
        )
        normalized = re.sub(
            r"[\s\-‐‑–—、，,。；;：:（）()\[\]【】]+", "", normalized
        )
        normalized = re.sub(
            r"的(?=安装|施工|调试|测试|敷设|配线|布线|接地|连接|制作|试验|吊装|"
            r"铺设|组装|校准)",
            "",
            normalized,
        )
        if "PPR" in normalized and (
            "管道施工工艺" in normalized or "热熔连接" in normalized
        ):
            return "PPR管道施工工艺"
        if normalized.endswith("施工"):
            base = normalized[:-2]
            if len(base) >= 4 and not base.endswith(("和", "及", "与")):
                return base
        return normalized

    @staticmethod
    def _canonical_work_name(name: str) -> str:
        normalized = re.split(r"[：:]", name.strip(), maxsplit=1)[0]
        normalized = re.sub(r"(?:施工|工程|安装)$", "", normalized)
        return re.sub(r"[\s、，,。；;：:（）()]+", "", normalized)

    @staticmethod
    def _same_work_item(left: str, right: str) -> bool:
        if left == right:
            return True
        if "给水部分" in (left, right) and (
            (left == "给水部分" and "生活给水系统" in right)
            or (right == "给水部分" and "生活给水系统" in left)
        ):
            return True
        shorter, longer = sorted((left, right), key=len)
        return len(shorter) >= 4 and longer.startswith(shorter)

    @staticmethod
    def _normalized_title(value: str) -> str:
        value = re.sub(r"^\s*\d+(?:\.\d+)+[、.．]?\s*", "", value)
        return re.sub(r"[\s、，,。；;：:（）()]+", "", value)

    @classmethod
    def _provenance_score(
        cls, method: MethodInfo, chapters: list[dict[str, Any]]
    ) -> int:
        source_text = method.source_text or ""
        source_heading = cls.NUMBERED_HEADING.match(source_text)
        heading_matches_name = bool(
            source_heading
            and cls.canonical_method_name(source_heading.group("title").strip(" ：:、，,。"))
            == cls.canonical_method_name(method.name)
        )
        exact_chapter = any(
            cls._normalized_title(str(chapter.get("title") or ""))
            == cls._normalized_title(method.name)
            and method.source_page == chapter.get("start_page")
            for chapter in chapters
        )
        return (
            (10000 if exact_chapter else 0)
            + (5000 if heading_matches_name else 0)
            + (2000 if source_heading else 0)
            + (1000 if method.source_page is not None else 0)
            + len(source_text)
        )
