"""Deterministic extraction of explicitly labelled project and schedule facts."""

import re
from datetime import datetime

from app.schemas.analysis import ProjectInfo, ScheduleInfo, SourceValue
from app.services.pdf_service import PDFPageText


class MetadataExtractionService:
    FIELD_BOUNDARY_LABELS = (
        "工程名称", "项目名称", "建设地点", "工程地点", "工程地址", "项目地址",
        "项目地点", "项目所在地", "工程区域位置", "建设单位", "业主单位",
        "施工单位", "总包单位", "施工总包单位", "施工总承包单位", "监理单位",
        "监控单位", "检测单位", "设计单位", "勘察单位", "勘察、设计单位",
        "勘察设计单位", "分包单位", "专业分包单位", "建筑面积", "建筑高度",
        "结构形式", "建设规模", "工程规模", "施工范围", "项目经理",
        "总监理工程师", "编制日期", "编制时间", "工程工期", "合同工期", "工期",
    )
    PROJECT_LABELS = {
        "project_name": ("工程名称", "项目名称"),
        "project_location": (
            "工程地址", "项目地址", "建设地点", "工程地点", "项目地点", "项目所在地"
        ),
        "construction_unit": ("建设单位", "业主单位"),
        "design_unit": ("设计单位", "勘察、设计单位", "勘察设计单位"),
        "supervision_unit": ("监理单位",),
        "general_contractor": (
            "施工总包单位", "总包单位", "总承包单位", "施工总承包单位"
        ),
        "subcontractor": ("分包单位", "专业分包单位"),
        "construction_company": ("施工单位", "编制单位", "编制主体"),
    }
    ORGANIZATION_FIELDS = {
        "construction_unit", "design_unit", "supervision_unit",
        "general_contractor", "subcontractor", "construction_company",
    }
    DATE_LABELS = {
        "plan_start_date": ("计划开工日期", "计划开工时间"),
        "plan_completion_date": ("计划竣工日期", "计划完工日期", "计划完成日期"),
    }
    DURATION_LABELS = ("计划总工期", "总工期", "计划工期", "工期")
    DATE_VALUE = r"(\d{4})\s*[年./-]\s*(\d{1,2})\s*[月./-]\s*(\d{1,2})\s*日?"
    DATE_PATTERN = re.compile(DATE_VALUE)
    DURATION_PATTERN = re.compile(r"(\d+)\s*(?:个?日历天|天|日)")
    NUMBERED_LINE = re.compile(
        r"^\s*(?:第[一二三四五六七八九十百]+[章节篇]|[一二三四五六七八九十百]+、|"
        r"\d+、|\d+(?:\.\d+)+[、.．]?)"
    )

    def extract(self, pages: list[PDFPageText]) -> tuple[ProjectInfo, ScheduleInfo]:
        project = ProjectInfo()
        project_scores: dict[str, int] = {}
        schedule_candidates: list[tuple[int, ScheduleInfo]] = []
        all_labels = tuple(
            dict.fromkeys(
                label
                for labels in (*self.PROJECT_LABELS.values(), *self.DATE_LABELS.values())
                for label in labels
            )
        )
        all_labels = tuple(
            dict.fromkeys((*all_labels, *self.DURATION_LABELS, *self.FIELD_BOUNDARY_LABELS))
        )

        for page in pages:
            lines = [self._clean_line(line) for line in page.text.splitlines()]
            for index, line in enumerate(lines):
                if not line:
                    continue
                for field, labels in self.PROJECT_LABELS.items():
                    if field != "project_location" and getattr(project, field) is not None:
                        continue
                    matched_label = self._matched_label(line, labels)
                    extracted = self._label_value(line, labels, all_labels)
                    if extracted is None:
                        continue
                    if self._has_multiple_labels(line, all_labels):
                        continuation = []
                    elif field == "project_name":
                        continuation = self._project_name_continuation(
                            extracted, lines, index + 1, all_labels
                        )
                    else:
                        continuation = self._continuation_lines(
                            lines, index + 1, all_labels
                        )
                    value = "".join([extracted, *continuation]).strip()
                    value = self._clean_project_field_boundary(value)
                    if field == "project_location":
                        value = re.sub(r"^(?:位于|坐落于?)\s*", "", value)
                    candidate_score = self._project_label_score(field, matched_label)
                    if candidate_score <= project_scores.get(field, -1):
                        continue
                    source_lines = [line, *continuation]
                    nearby_text = "\n".join(lines[max(0, index - 2) : index + 4])
                    if field in self.ORGANIZATION_FIELDS and self._is_contact_person(
                        value, "\n".join(source_lines), nearby_text
                    ):
                        continue
                    if value:
                        setattr(
                            project,
                            field,
                            self._source(value, page.page_number, "\n".join(source_lines)),
                        )
                        project_scores[field] = candidate_score
            page_schedule = ScheduleInfo()
            self._extract_schedule_from_page(page, page_schedule)
            if (
                re.search(r"(?:专项)?施工进度计划(?:表|安排)?", page.text)
                and page_schedule.plan_start_date is None
                and page_schedule.plan_completion_date is None
            ):
                # A task-duration column is not an explicit overall duration.
                page_schedule.duration_days = None
            if any(value is not None for value in page_schedule.model_dump().values()):
                schedule_candidates.append(
                    (self._schedule_scope_score(page.text, page_schedule), page_schedule)
                )
        schedule = self._select_schedule_candidates(schedule_candidates)
        table_candidate = self._extract_schedule_table(pages)
        if table_candidate is not None:
            schedule = self._select_schedule_candidates(
                [*schedule_candidates, (400, table_candidate)]
            )
        if project.project_name is None:
            project.project_name = self._extract_narrative_project_name(pages)
        if project.project_name is None:
            project.project_name = self._extract_title_page_project_name(pages)
        if project.project_location is None:
            project.project_location = self._extract_narrative_location(pages)
        if project.construction_company is None:
            project.construction_company = self._extract_cover_company(pages)
        return project, schedule

    def extract_engineering_category(
        self, pages: list[PDFPageText], project: ProjectInfo
    ) -> SourceValue | None:
        """Return a category only from strong project-definition evidence."""
        project_name = project.project_name
        if project_name is not None:
            evidence = "\n".join(
                value
                for value in (
                    str(project_name.value or ""),
                    project_name.source_text or "",
                )
                if value
            )
            if "安装工程" in evidence:
                return SourceValue(
                    value="安装工程",
                    source_page=project_name.source_page,
                    source_text=project_name.source_text,
                )

        for page_number, text in self._project_definition_evidence(pages, project):
            overview = re.search(r"(?:^|\n)\s*(?:[（(]?[一二三四五六七八九十\d]+[）)]?[、.．]?\s*)?建筑概况", text)
            if overview is None:
                continue
            section = text[overview.start() : overview.start() + 1200]
            evidence_classes = sum(
                bool(pattern.search(section))
                for pattern in (
                    re.compile(r"建筑面积"),
                    re.compile(r"结构形式|框架(?:剪力墙|核心筒)?结构"),
                    re.compile(r"(?:地下|地上)\s*\d+\s*层|楼层"),
                    re.compile(r"主楼|裙房|住宅楼|办公楼|建筑高度"),
                )
            )
            if evidence_classes < 2:
                continue
            return SourceValue(
                value="房屋建筑",
                source_page=page_number,
                source_text=section[:300].strip(),
            )

        return None

    def validate_engineering_category(
        self,
        pages: list[PDFPageText],
        project: ProjectInfo,
        candidate: SourceValue | None,
    ) -> SourceValue | None:
        if candidate is None or not isinstance(candidate.value, str):
            return None
        value = candidate.value.strip()
        if not value:
            return None
        for page_number, text in self._project_definition_evidence(pages, project):
            if value in text:
                return SourceValue(
                    value=value,
                    source_page=page_number,
                    source_text=self._evidence_snippet(text, value),
                )
        return None

    def validate_organizations(
        self, pages: list[PDFPageText], project: ProjectInfo
    ) -> ProjectInfo:
        payload = project.model_dump()
        for field in self.ORGANIZATION_FIELDS:
            item = payload.get(field)
            if not isinstance(item, dict) or not item.get("value"):
                continue
            value = str(item["value"])
            compact_value = re.sub(r"\s+", "", value)
            matching_pages = [
                page.text
                for page in pages
                if page.page_number == item.get("source_page")
                or (
                    compact_value
                    and compact_value
                    in re.sub(r"[\s|｜]+", "", page.text)
                )
            ]
            nearby_text = "\n".join(matching_pages)
            if self._is_contact_person(
                value, str(item.get("source_text") or ""), nearby_text
            ):
                payload[field] = None
        return ProjectInfo.model_validate(payload)

    def _extract_schedule_from_page(self, page: PDFPageText, schedule: ScheduleInfo) -> None:
        text = page.text
        range_pattern = re.compile(
            rf"(?:目标工期|计划工期)\s*(?:为|[：:])?\s*"
            rf"(?P<start_year>\d{{4}})\s*[年./-]\s*"
            rf"(?P<start_month>\d{{1,2}})\s*[月./-]\s*"
            rf"(?P<start_day>\d{{1,2}})\s*日?\s*(?:-|—|–|－|~|～|至|到)\s*"
            rf"(?P<end_year>\d{{4}})\s*[年./-]\s*"
            rf"(?P<end_month>\d{{1,2}})\s*[月./-]\s*"
            rf"(?P<end_day>\d{{1,2}})\s*日?"
        )
        range_match = range_pattern.search(text)
        if range_match:
            trailing = text[range_match.end() : range_match.end() + 120]
            duration_match = re.search(
                r"(?:施工周期|工期)\s*(?:为|[：:])?\s*"
                r"(?P<duration>\d+)\s*(?:个?日历天|天|日)",
                trailing,
            )
            source_end = (
                range_match.end() + duration_match.end()
                if duration_match
                else range_match.end()
            )
            start_value = self._normalize_date_parts(
                range_match.group("start_year", "start_month", "start_day")
            )
            end_value = self._normalize_date_parts(
                range_match.group("end_year", "end_month", "end_day")
            )
            source_text = self._match_source_block(
                text, range_match.start(), source_end
            )
            if start_value and end_value:
                schedule.plan_start_date = self._source(
                    start_value, page.page_number, source_text
                )
                schedule.plan_completion_date = self._source(
                    end_value, page.page_number, source_text
                )
                if duration_match:
                    schedule.duration_days = self._source(
                        int(duration_match.group("duration")), page.page_number, source_text
                    )

        for field, labels in self.DATE_LABELS.items():
            if getattr(schedule, field) is not None:
                continue
            label_pattern = "|".join(re.escape(label) for label in labels)
            pattern = re.compile(
                rf"(?:{label_pattern})(?:\s*[（(][^）)\n]{{0,30}}[）)])?"
                rf"\s*(?:为|[：:])?\s*{self.DATE_VALUE}"
            )
            match = pattern.search(text)
            if match:
                normalized = self._normalize_date_parts(match.groups()[-3:])
                if normalized:
                    setattr(
                        schedule,
                        field,
                        self._source(
                            normalized,
                            page.page_number,
                            self._match_source_text(text, match.start(), match.end()),
                        ),
                    )

        narrative_patterns = {
            "plan_start_date": re.compile(
                rf"(?:本工程|本项目)?\s*(?:计划|预计)\s*(?:从)?\s*{self.DATE_VALUE}"
                rf"\s*(?:开工|开始施工|开始)"
            ),
            "plan_completion_date": re.compile(
                rf"(?:计划|预计)?\s*{self.DATE_VALUE}\s*(?:完工|竣工|完成)"
            ),
        }
        for field, pattern in narrative_patterns.items():
            if getattr(schedule, field) is not None:
                continue
            match = pattern.search(text)
            if match:
                normalized = self._normalize_date_parts(match.groups()[-3:])
                if normalized:
                    setattr(
                        schedule,
                        field,
                        self._source(
                            normalized,
                            page.page_number,
                            self._match_source_text(text, match.start(), match.end()),
                        ),
                    )

        if schedule.duration_days is None:
            labels = "|".join(re.escape(label) for label in self.DURATION_LABELS)
            match = re.search(
                rf"(?:{labels})\s*(?:为|[：:])?\s*(\d+)\s*(?:个?日历天|天|日)", text
            )
            if match:
                schedule.duration_days = self._source(
                    int(match.group(1)),
                    page.page_number,
                    self._match_source_text(text, match.start(), match.end()),
                )
            else:
                match = re.search(r"共需\s*(\d+)\s*(?:个?日历天|天|日)", text)
                if match:
                    schedule.duration_days = self._source(
                        int(match.group(1)),
                        page.page_number,
                        self._match_source_text(text, match.start(), match.end()),
                    )

    def _extract_schedule_table(
        self, pages: list[PDFPageText]
    ) -> ScheduleInfo | None:
        """Aggregate dates only inside an explicitly scoped construction schedule."""
        scoped_pages: list[PDFPageText] = []
        remaining = 0
        for page in pages:
            if re.search(r"(?:专项)?施工进度计划(?:表|安排)?", page.text):
                remaining = 3
            if remaining:
                scoped_pages.append(page)
                remaining -= 1

        rows: list[tuple[str, str, str, int]] = []
        summary_rows: list[tuple[str, str, str, int]] = []
        for page in scoped_pages:
            for raw_line in page.text.splitlines():
                line = self._clean_line(raw_line)
                if not line or re.search(r"编制日期|专家论证|论证日期|成立日期|材料进场", line):
                    continue
                matches = list(self.DATE_PATTERN.finditer(line))
                if len(matches) < 2:
                    continue
                start = self._normalize_date_parts(matches[-2].groups())
                end = self._normalize_date_parts(matches[-1].groups())
                if start is None or end is None or start > end:
                    continue
                row = (start, end, line[:300], page.page_number)
                rows.append(row)
                if re.search(r"总计|合计|总体|全部工程|总工期|整个工程", line):
                    summary_rows.append(row)
        selected = summary_rows or rows
        if not selected:
            return None

        start_row = min(selected, key=lambda item: item[0])
        end_row = max(selected, key=lambda item: item[1])
        schedule = ScheduleInfo()
        schedule.plan_start_date = self._source(
            start_row[0], start_row[3], start_row[2]
        )
        schedule.plan_completion_date = self._source(
            end_row[1], end_row[3], end_row[2]
        )
        if summary_rows:
            summary = summary_rows[0]
            date_tail = self.DATE_PATTERN.sub("", summary[2])
            duration = self.DURATION_PATTERN.search(date_tail)
            if duration:
                schedule.duration_days = self._source(
                    int(duration.group(1)), summary[3], summary[2]
                )
        return schedule

    @staticmethod
    def _schedule_scope_score(text: str, schedule: ScheduleInfo) -> int:
        score = 0
        if re.search(r"目标工期|计划工期|施工周期", text):
            score += 300
        if re.search(r"施工进度计划|专项进度计划", text):
            score += 300
        if re.search(r"专项施工|本方案|分项工程|施工工期", text):
            score += 160
        if re.search(r"项目总工期|工程总工期|合同总工期", text):
            score -= 250
        values = schedule.model_dump()
        score += sum(value is not None for value in values.values()) * 20
        if schedule.plan_start_date is not None and schedule.plan_completion_date is not None:
            score += 100
            if schedule.duration_days is not None:
                score += 100
        return score

    @staticmethod
    def _select_schedule_candidates(
        candidates: list[tuple[int, ScheduleInfo]],
    ) -> ScheduleInfo:
        complete_groups = [
            (score, schedule)
            for score, schedule in candidates
            if schedule.plan_start_date is not None
            and schedule.plan_completion_date is not None
        ]
        if complete_groups:
            _, selected = max(complete_groups, key=lambda item: item[0])
            result = selected.model_copy(deep=True)
            if result.duration_days is not None:
                return result
        else:
            result = ScheduleInfo()
        for field in ("plan_start_date", "plan_completion_date", "duration_days"):
            if getattr(result, field) is not None:
                continue
            available = [
                (score, getattr(schedule, field))
                for score, schedule in candidates
                if getattr(schedule, field) is not None
            ]
            if available:
                setattr(result, field, max(available, key=lambda item: item[0])[1])
        return result

    def extract_document_type(
        self, pages: list[PDFPageText], project_name: SourceValue | None = None
    ) -> SourceValue | None:
        """Prefer a repeated, explicit PDF title over a generic LLM type."""
        suffix = r"(?:专项施工方案|施工组织设计|施工方案)"
        candidates: dict[str, list[tuple[int, str]]] = {}
        project_value = (
            re.sub(r"\s+", "", str(project_name.value or ""))
            if project_name is not None
            else ""
        )
        project_base = re.sub(
            r"(?:建设)?项目$|工程$", "", project_value.rstrip("。．.")
        )
        for page in pages:
            lines = [self._clean_line(line) for line in page.text.splitlines() if line.strip()]
            for index, line in enumerate(lines):
                values = [(line, line)]
                if index + 1 < len(lines) and not re.match(
                    r"^(?:工程名称|项目名称|编制单位|建设单位)\s*[：:]", line
                ):
                    next_line = lines[index + 1]
                    overlap = next(
                        (
                            size
                            for size in range(min(len(line), len(next_line)), 0, -1)
                            if line.endswith(next_line[:size])
                        ),
                        0,
                    )
                    values.append(
                        (f"{line}{next_line[overlap:]}", f"{line}\n{next_line}")
                    )
                for value, source_text in values:
                    if re.search(r"(?:\.{2,}|…{2,}|·{2,})\s*\d+\s*$", source_text):
                        continue
                    match = re.search(rf"([^：:，,。；;]{{0,60}}?{suffix})", value)
                    if not match:
                        continue
                    title = re.sub(
                        r"^\s*(?:第[一二三四五六七八九十百\d]+[章节篇]|"
                        r"[一二三四五六七八九十百]+、|"
                        r"\d+(?:\.\d+)*(?:[、.．]\s*|\s+))",
                        "",
                        match.group(1),
                    ).strip(" 《》：:，,。；;")
                    title = re.sub(
                        r"^(?:[ivxlcdm]{1,6}|第?\d+页)\s*[《》]?\s*",
                        "",
                        title,
                        flags=re.IGNORECASE,
                    ).strip(" 《》：:，,。；;")
                    compact_title = re.sub(r"\s+", "", title)
                    descriptor_match = re.match(
                        rf"^(.{{0,60}}?(?:工程总承包(?:[（(][^）)]{{1,20}}[）)])?项目|"
                        rf"建设项目|项目))(?=.+{suffix}$)",
                        compact_title,
                    )
                    if descriptor_match and descriptor_match.group(1) in project_value:
                        compact_title = compact_title[descriptor_match.end() :]
                        title = compact_title.lstrip("-—：:")
                    if project_value and compact_title.startswith(project_value):
                        compact_title = compact_title[len(project_value) :].lstrip("-—：:")
                        title = compact_title
                    elif project_base and compact_title.startswith(project_base):
                        compact_title = compact_title[len(project_base) :]
                        compact_title = re.sub(r"^(?:建设)?项目", "", compact_title)
                        title = compact_title.lstrip("-—：:")
                    elif project_base:
                        for start in range(len(project_base)):
                            suffix_value = project_base[start:]
                            if len(suffix_value) >= 4 and compact_title.startswith(
                                suffix_value
                            ):
                                compact_title = compact_title[len(suffix_value) :]
                                title = compact_title.lstrip("-—：:")
                                break
                    compact_title = re.sub(
                        r"^(?:(?:工程)?施工[^，,。；;]{0,40}?标段|"
                        r"(?:施工)?[A-Z0-9№.（）()+-]{1,30}标段)",
                        "",
                        re.sub(r"\s+", "", title),
                    )
                    if compact_title:
                        title = compact_title
                    if title in {"专项施工方案", "施工方案"}:
                        continue
                    if re.match(r"^(?:某|本|该)(?:工程|项目)", title):
                        continue
                    if re.search(
                        r"^(?:编制(?:说明及)?依据|工程概况|施工(?:方法|工艺|技术)|"
                        r"主要施工方案|施工部署|施工计划)(?:及|与|、|$)",
                        title,
                    ):
                        continue
                    candidates.setdefault(title, []).append((page.page_number, source_text))
        if not candidates:
            return None
        title, evidence = max(
            candidates.items(),
            key=lambda item: (
                any(page <= 3 for page, _ in item[1]),
                len(item[0]),
                len(item[1]),
            ),
        )
        page_number, source_text = min(
            evidence, key=lambda item: (0 if item[0] <= 3 else 1, item[0])
        )
        return self._source(title, page_number, source_text)

    @staticmethod
    def _is_contact_person(value: str, source_text: str, nearby_text: str) -> bool:
        context = f"{source_text}\n{nearby_text}"
        if not re.search(r"联系人|抢险人员|联系方式|通讯录|手机号|手机号码|联系电话", context):
            return False
        if not re.search(r"1[3-9]\d{9}", context):
            return False
        cleaned = re.sub(r"1[3-9]\d{9}", "", value)
        cleaned = re.sub(r"联系人|手机号|手机号码|联系电话|电话", "", cleaned)
        cleaned = re.sub(r"[\s|｜：:，,；;\d\-]+", "", cleaned)
        return bool(re.fullmatch(r"[\u4e00-\u9fff]{2,4}", cleaned))

    def _extract_cover_company(self, pages: list[PDFPageText]) -> SourceValue | None:
        pattern = re.compile(
            r"[\u4e00-\u9fffA-Za-z0-9（）()]{4,60}"
            r"(?:集团有限公司|有限责任公司|有限公司|集团公司)"
        )
        for page in pages:
            if page.page_number > 3:
                continue
            lines = [self._clean_line(line) for line in page.text.splitlines() if line.strip()]
            document_title_index = next(
                (
                    index
                    for index, line in enumerate(lines)
                    if re.search(r"施工组织设计|专项施工方案|施工方案", line)
                ),
                None,
            )
            if document_title_index is None:
                continue
            for line in lines[document_title_index + 1 :]:
                match = pattern.search(self._clean_line(line))
                if match:
                    return self._source(
                        match.group(0), page.page_number, line.strip()
                    )
        return None

    def _extract_narrative_project_name(
        self, pages: list[PDFPageText]
    ) -> SourceValue | None:
        overview_quoted_pattern = re.compile(
            r"[“\"‘'](?P<quoted>[^”\"’'\n]{2,60})[”\"’']\s*"
            r"(?P<suffix>[^，,。；;\n]{0,40}(?:工程|项目))"
        )
        quoted_pattern = re.compile(
            r"投资的\s*[“\"‘'](?P<quoted>[^”\"’'\n]{2,60})[”\"’']"
            r"\s*(?P<suffix>[^，,。；;\n]{0,40})"
        )
        located_pattern = re.compile(
            r"(?P<name>[^，,。；;\n]{4,80}?(?:工程|项目))\s*[，,]?\s*"
            r"(?:坐落于?|位于)"
        )
        for page in pages:
            patterns = [quoted_pattern, located_pattern]
            if re.search(r"(?:^|\n)\s*(?:\d+(?:\.\d+)*[、.．]?\s*)?工程概况", page.text):
                patterns.insert(0, overview_quoted_pattern)
            for pattern in patterns:
                match = pattern.search(page.text)
                if not match:
                    continue
                if "quoted" in match.groupdict():
                    value = f"{match.group('quoted')}{match.group('suffix') or ''}"
                else:
                    value = match.group("name")
                    value = re.sub(r"^.*?(?:投资的|建设的)", "", value)
                    value = re.sub(
                        r"^\s*(?:\d+(?:\.\d+)*[、.．]?\s*)?工程概况\s*",
                        "",
                        value,
                    )
                value = value.strip(" “\"”：《》")
                if value not in {"本工程", "本项目", "该工程", "该项目"} and re.search(
                    r"(?:工程|项目)$", value
                ):
                    return self._source(
                        value,
                        page.page_number,
                        self._match_source_text(page.text, match.start(), match.end()),
                    )
        return None

    def _extract_title_page_project_name(
        self, pages: list[PDFPageText]
    ) -> SourceValue | None:
        candidates: list[tuple[int, int, str, str]] = []
        for page in (page for page in pages if page.page_number <= 3):
            lines = [self._clean_line(line) for line in page.text.splitlines() if line.strip()]
            for index, line in enumerate(lines):
                combinations = [(line, line)]
                if index + 1 < len(lines) and not re.search(
                    r"(?:有限公司|公司|集团|编制单位|建设单位)$", line
                ):
                    combinations.append((f"{line}{lines[index + 1]}", f"{line}\n{lines[index + 1]}"))
                for value, source_text in combinations:
                    cleaned = value.strip(" “\"”：《》")
                    cleaned = re.sub(r"(工程|项目)(?:施工组织设计|施工方案|施工)$", r"\1", cleaned)
                    if not re.search(r"(?:工程|项目)$", cleaned):
                        continue
                    if re.search(r"施工组织设计|施工方案|专项方案|编制单位|建设单位", cleaned):
                        continue
                    if not 4 <= len(cleaned) <= 80:
                        continue
                    score = (40 if cleaned.endswith("工程") else 20) + min(len(cleaned), 40)
                    candidates.append((score, page.page_number, cleaned, source_text))
        if not candidates:
            return None
        _, page_number, value, source_text = max(candidates, key=lambda item: item[0])
        return self._source(value, page_number, source_text)

    def _extract_narrative_location(
        self, pages: list[PDFPageText]
    ) -> SourceValue | None:
        for page in pages:
            lines = [self._clean_line(line) for line in page.text.splitlines()]
            for index, line in enumerate(lines):
                match = re.search(
                    r"(?:本工程|本项目|该工程|该项目)\s*(?:坐落于?|位于)\s*"
                    r"(?P<location>.+)$",
                    line,
                )
                if match is None and re.search(
                    r"(?:^|\n)\s*(?:\d+(?:\.\d+)*[、.．]?\s*)?工程概况",
                    page.text,
                ):
                    match = re.search(r"^\s*(?:坐落于?|位于)\s*(?P<location>.+)$", line)
                if match is None:
                    match = re.search(
                        r"(?:本工程|本项目|该工程|该项目)?\s*(?:路线|线路)"
                        r"(?P<location>西起.+?(?:东至|北至|南至).+)$",
                        line,
                    )
                if not match:
                    continue
                context = line[max(0, match.start() - 20) : match.end()]
                if re.search(
                    r"地质构造|孔位|桩位|构件位置|安装位置|模板位置|受力位置|"
                    r"测点位置|施工位置|梁体位置|设备位置",
                    context,
                ):
                    continue
                source_lines = [line[match.start() :].strip()]
                value_lines = [match.group("location").strip(" ：:")]
                if not line.rstrip().endswith(("。", "；", ";")):
                    for continuation in lines[index + 1 : index + 4]:
                        if not continuation or self.NUMBERED_LINE.match(continuation):
                            break
                        source_lines.append(continuation)
                        value_lines.append(continuation)
                        if continuation.endswith(("。", "；", ";")):
                            break
                value = "".join(value_lines).strip(" ：:")
                return self._source(
                    value.rstrip("。；;"),
                    page.page_number,
                    "\n".join(source_lines),
                )
        return None

    def _project_definition_evidence(
        self, pages: list[PDFPageText], project: ProjectInfo
    ) -> list[tuple[int, str]]:
        evidence: list[tuple[int, str]] = []
        if project.project_name is not None and project.project_name.source_page is not None:
            evidence.append(
                (
                    project.project_name.source_page,
                    "\n".join(
                        part
                        for part in (
                            str(project.project_name.value or ""),
                            project.project_name.source_text or "",
                        )
                        if part
                    ),
                )
            )
        evidence.extend(
            (page.page_number, page.text) for page in pages if page.page_number <= 3
        )
        for page in pages:
            match = re.search(
                r"(?:^|\n)\s*(?:(?:第[一二三四五六七八九十百\d]+章|"
                r"[一二三四五六七八九十百]+、|"
                r"\d+(?:\.\d+)*[、.．]?)\s*)?工程概况",
                page.text,
            )
            if match:
                evidence.append((page.page_number, page.text[match.start() : match.start() + 1200]))
        return evidence

    @staticmethod
    def _evidence_snippet(text: str, value: str) -> str:
        start = max(0, text.find(value) - 80)
        end = min(len(text), text.find(value) + len(value) + 120)
        return text[start:end].strip()[:300]

    def _continuation_lines(
        self, lines: list[str], start: int, all_labels: tuple[str, ...]
    ) -> list[str]:
        result: list[str] = []
        for line in lines[start:]:
            if (
                not line
                or self.NUMBERED_LINE.match(line)
                or self._starts_with_label(line, all_labels)
                or self._is_section_heading(line)
            ):
                break
            if len(result) >= 3:
                break
            result.append(line)
            if line.endswith(("。", "；", ";")):
                break
        return result

    def _project_name_continuation(
        self,
        extracted: str,
        lines: list[str],
        start: int,
        all_labels: tuple[str, ...],
    ) -> list[str]:
        if re.search(r"(?:工程|项目)[。.]?$", extracted):
            return []
        if start >= len(lines):
            return []
        line = lines[start]
        if (
            not line
            or self._starts_with_label(line, all_labels)
            or self._is_section_heading(line)
            or re.search(r"专项施工方案|施工组织设计|施工方案", line)
            or not re.search(r"(?:工程|项目)[。.]?$", line)
        ):
            return []
        return [line]

    @staticmethod
    def _is_section_heading(line: str) -> bool:
        return bool(
            re.match(
                r"^\s*(?:第[一二三四五六七八九十百\d]+[章节篇]|"
                r"[（(][一二三四五六七八九十\d]+[）)]|"
                r"[一二三四五六七八九十\d]+[、.．])\s*\S+",
                line,
            )
        )

    @staticmethod
    def _starts_with_label(line: str, labels: tuple[str, ...]) -> bool:
        return any(
            re.match(rf"^{re.escape(label)}\s*(?:[：:|｜]|\s)", line)
            for label in labels
        )

    @staticmethod
    def _has_multiple_labels(line: str, labels: tuple[str, ...]) -> bool:
        return sum(bool(re.search(re.escape(label), line)) for label in labels) > 1

    @staticmethod
    def _matched_label(line: str, labels: tuple[str, ...]) -> str | None:
        matches = [
            (match.start(), label)
            for label in labels
            if (match := re.search(re.escape(label), line))
        ]
        return min(matches, default=(0, None))[1]

    @staticmethod
    def _project_label_score(field: str, label: str | None) -> int:
        if field != "project_location":
            return 100
        if label in {"工程地址", "项目地址"}:
            return 300
        if label in {"工程地点", "项目地点"}:
            return 250
        return 200

    @classmethod
    def _clean_project_field_boundary(cls, value: str) -> str:
        boundary_pattern = "|".join(
            re.escape(label)
            for label in sorted(cls.FIELD_BOUNDARY_LABELS, key=len, reverse=True)
        )
        cleaned = re.split(boundary_pattern, value, maxsplit=1)[0]
        return cleaned.strip(" |｜：:，,；;")

    @staticmethod
    def _clean_line(line: str) -> str:
        return re.sub(r"[ \t\u3000]+", " ", line).strip()

    @staticmethod
    def _label_value(line: str, labels: tuple[str, ...], all_labels: tuple[str, ...]) -> str | None:
        if any(re.fullmatch(rf"{re.escape(label)}\s*[：:]?", line) for label in labels):
            return ""
        label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
        next_label_pattern = "|".join(
            re.escape(label) for label in sorted(all_labels, key=len, reverse=True)
        )
        match = re.search(
            rf"(?:{label_pattern})\s*(?:[：:|｜]\s*|\s+)(.*?)"
            rf"(?=\s+(?:{next_label_pattern})\s*(?:[：:|｜]|\s|$)|$)",
            line,
        )
        if not match:
            return None
        return match.group(1).strip(" ：:，,；;")

    @classmethod
    def _normalize_date_parts(cls, parts: tuple[str, str, str]) -> str | None:
        try:
            parsed = datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            return None
        return parsed.strftime("%Y-%m-%d")

    @staticmethod
    def _match_source_text(text: str, start: int, end: int) -> str:
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        if line_end == -1:
            line_end = len(text)
        return text[line_start:line_end].strip()[:300]

    @staticmethod
    def _match_source_block(text: str, start: int, end: int) -> str:
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", end)
        if line_end == -1:
            line_end = len(text)
        return text[line_start:line_end].strip()[:300]

    @staticmethod
    def _source(value: str | int, page_number: int, source_text: str) -> SourceValue:
        return SourceValue(value=value, source_page=page_number, source_text=source_text[:300])
