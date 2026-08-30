import json
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

import app.core.config as config_module
from app.core.config import get_settings
from app.main import app
from app.schemas.analysis import (
    ChapterInfo,
    DocumentAnalysis,
    DocumentAnalysisResponse,
    SectionPresence,
    SourceValue,
)
from app.services.chapter_detection_service import ChapterDetectionService
from app.services.document_analysis_service import DocumentAnalysisService
from app.services.deterministic_postprocessing_service import DeterministicPostProcessingService
from app.services.llm_service import LLMConfigurationError, LLMService, get_llm_service
from app.services.metadata_extraction_service import MetadataExtractionService
from app.services.pdf_service import PDFExtractionResult, PDFPageText, PDFService
from app.services.source_locator_service import SourceLocatorService
from app.services.text_preprocessing_service import TextPreprocessingService


client = TestClient(app)


class FakeLLMService:
    calls = 0

    def analyze_document(self, pages: list[PDFPageText]) -> DocumentAnalysis:
        self.calls += 1
        return DocumentAnalysis.model_validate(
            {
                "document_type": {
                    "value": "给排水专项施工方案",
                    "source_page": 1,
                    "source_text": "给排水专项施工方案",
                },
                "engineering_category": "建筑工程",
                "specialty": "给排水工程",
                "project": {
                    "project_name": {
                        "value": "测试工程",
                        "source_page": 1,
                        "source_text": "工程名称：测试工程",
                    }
                },
                "main_work_items": ["室内给水系统"],
                "main_methods": [{"name": "热熔连接", "source_page": 1}],
            }
        )


@pytest.fixture(autouse=True)
def temporary_data_directories(tmp_path: Path):
    settings = get_settings()
    old_upload_dir = settings.upload_dir
    old_analysis_dir = settings.analysis_dir
    settings.upload_dir = tmp_path / "uploads"
    settings.analysis_dir = tmp_path / "analysis"
    fake = FakeLLMService()
    app.dependency_overrides[get_llm_service] = lambda: fake
    yield settings, fake
    app.dependency_overrides.clear()
    settings.upload_dir = old_upload_dir
    settings.analysis_dir = old_analysis_dir


def make_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def upload_pdf() -> str:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("plan.pdf", make_pdf("Construction plan GB 50015-2019"), "application/pdf")},
    )
    assert response.status_code == 200
    return response.json()["document_id"]


def test_missing_document_returns_404() -> None:
    response = client.post("/api/v1/documents/00000000-0000-0000-0000-000000000000/analyze")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_valid_pdf_with_fake_llm_returns_structured_analysis(temporary_data_directories) -> None:
    settings, fake = temporary_data_directories
    document_id = upload_pdf()

    response = client.post(f"/api/v1/documents/{document_id}/analyze")

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["cached"] is False
    assert body["analysis"]["document_type"]["value"] == "给排水专项施工方案"
    assert body["analysis"]["standards"][0]["code"] == "GB50015-2019"
    assert fake.calls == 1
    cache = settings.analysis_dir / f"{document_id}.json"
    assert cache.is_file()
    assert "给排水专项施工方案" in cache.read_text(encoding="utf-8")


def test_cached_result_skips_llm_and_force_reanalyzes(temporary_data_directories) -> None:
    _, fake = temporary_data_directories
    document_id = upload_pdf()
    assert client.post(f"/api/v1/documents/{document_id}/analyze").status_code == 200

    cached = client.post(f"/api/v1/documents/{document_id}/analyze")
    forced = client.post(f"/api/v1/documents/{document_id}/analyze?force=true")

    assert cached.json()["cached"] is True
    assert forced.json()["cached"] is False
    assert fake.calls == 2


def test_parse_plain_mock_json() -> None:
    result = LLMService.parse_json_response(
        json.dumps({"document_type": None, "main_work_items": []}), DocumentAnalysis
    )
    assert result.document_type is None
    assert result.main_work_items == []


def test_null_containers_and_plain_project_values_get_safe_defaults() -> None:
    result = LLMService.parse_json_response(
        '{"document_type":"施工组织设计","project":{"project_name":"测试项目"},'
        '"schedule":null,"standards":null,"section_presence":null}'
    )
    assert result.document_type.value == "施工组织设计"
    assert result.project.project_name.value == "测试项目"
    assert result.schedule.duration_days is None
    assert result.standards == []
    assert result.section_presence.quality is False


def test_parse_json_code_fence_and_surrounding_text() -> None:
    raw = '说明文字\n```json\n{"specialty":"给排水", "project": {}}\n```\n结束'
    result = LLMService.parse_json_response(raw, DocumentAnalysis)
    assert result.specialty.value == "给排水"


def test_invalid_llm_json_has_readable_error() -> None:
    with pytest.raises(Exception, match="valid JSON object"):
        LLMService.parse_json_response("not json", DocumentAnalysis)


def test_text_preprocessing_keeps_page_numbers_and_removes_repeated_edges() -> None:
    pages = [
        PDFPageText(index, f"Company header {index}\n  Chapter   title  \nBody {index}\nPage {index}")
        for index in range(1, 4)
    ]
    cleaned = TextPreprocessingService().preprocess(pages)
    assert [page.page_number for page in cleaned] == [1, 2, 3]
    assert "Chapter title" in cleaned[0].text
    assert "Company header" not in cleaned[0].text
    assert "Page" not in cleaned[0].text


def test_chapter_detection_and_presence_are_non_llm() -> None:
    pages = [
        PDFPageText(1, "第一章 工程概况\n内容"),
        PDFPageText(2, "第二章 质量保证措施\n内容"),
        PDFPageText(4, "第三章 应急预案\n内容"),
    ]
    detector = ChapterDetectionService()
    chapters = detector.detect(pages)
    presence = detector.section_presence(chapters, pages)
    assert [(item.title, item.start_page, item.end_page) for item in chapters] == [
        ("工程概况", 1, 2),
        ("质量保证措施", 2, 4),
        ("应急预案", 4, 4),
    ]
    assert presence.quality is True
    assert presence.emergency is True
    assert presence.safety is False


def test_long_text_is_split_with_traceable_page_markers() -> None:
    chunks = LLMService._build_chunks([PDFPageText(7, "x" * 2500)], 1000)
    assert len(chunks) == 3
    assert all("PDF第7页" in chunk for chunk in chunks)
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_llm_configuration_error_lists_missing_variables() -> None:
    settings = get_settings().model_copy(
        update={"llm_api_key": None, "llm_base_url": None, "llm_model": None}
    )
    service = LLMService(settings=settings)
    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY.*LLM_BASE_URL.*LLM_MODEL"):
        service.call_chat_completion("system", "user")


def test_settings_load_project_env_independently_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_project_root = tmp_path / "project"
    unrelated_working_directory = tmp_path / "elsewhere"
    fake_project_root.mkdir()
    unrelated_working_directory.mkdir()
    env_file = fake_project_root / ".env"
    env_file.write_text(
        "LLM_API_KEY=test-only-key\n"
        "LLM_BASE_URL=https://example.invalid/v1\n"
        "LLM_MODEL=test-model\n",
        encoding="utf-8",
    )

    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_module, "ENV_FILE", env_file)
    monkeypatch.chdir(unrelated_working_directory)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.llm_api_key == "test-only-key"
        assert settings.llm_base_url == "https://example.invalid/v1"
        assert settings.llm_model == "test-model"
    finally:
        get_settings.cache_clear()


def test_llm_chapter_name_and_page_are_normalized() -> None:
    result = LLMService.parse_json_response(
        '{"chapters":[{"name":"工程概况","page":4}]}'
    )
    assert result.chapters == [
        ChapterInfo(title="工程概况", start_page=4, end_page=None)
    ]


def test_llm_chapter_and_page_number_are_normalized() -> None:
    result = LLMService.parse_json_response(
        '{"chapters":[{"chapter":"施工方法","page_number":14}]}'
    )
    assert result.chapters == [
        ChapterInfo(title="施工方法", start_page=14, end_page=None)
    ]


def test_completely_invalid_llm_chapters_are_ignored() -> None:
    result = LLMService.parse_json_response(
        '{"specialty":"给排水工程","chapters":["工程概况","施工方法"]}'
    )
    assert result.specialty.value == "给排水工程"
    assert result.chapters == []


def test_invalid_llm_chapters_do_not_make_analyze_return_502() -> None:
    class InvalidChaptersLLMService:
        def analyze_document(self, pages: list[PDFPageText]) -> DocumentAnalysis:
            return LLMService.parse_json_response(
                '{"document_type":"施工组织设计",'
                '"chapters":["工程概况","施工方法"]}'
            )

    app.dependency_overrides[get_llm_service] = InvalidChaptersLLMService
    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "plan.pdf",
                make_pdf("1.1 Project Overview\nConstruction plan content"),
                "application/pdf",
            )
        },
    )
    document_id = response.json()["document_id"]

    analyzed = client.post(f"/api/v1/documents/{document_id}/analyze")

    assert analyzed.status_code == 200
    assert analyzed.json()["analysis"]["document_type"]["value"] == "施工组织设计"
    assert analyzed.json()["analysis"]["chapters"] == [
        {"title": "Project Overview", "start_page": 1, "end_page": 1, "level": 2}
    ]


def test_local_chapters_take_priority_over_llm_chapters() -> None:
    local = [ChapterInfo(title="工程概况", start_page=2, end_page=5)]
    llm = [
        ChapterInfo(title="工程概况", start_page=4),
        ChapterInfo(title="施工方法", start_page=8),
    ]
    merged = DocumentAnalysisService._merge_chapters(local, llm, page_count=20)
    assert merged == local


def test_one_invalid_chapter_does_not_discard_valid_analysis_fields() -> None:
    result = LLMService.parse_json_response(
        '{"document_type":"施工组织设计","main_methods":[{"method":"管道试压",'
        '"page_number":9},{"unexpected":true}],"chapters":['
        '{"heading":"施工准备","source_page":6},{"missing_page":"无效章节"}]}'
    )
    assert result.document_type.value == "施工组织设计"
    assert result.main_methods[0].name == "管道试压"
    assert result.main_methods[0].source_page == 9
    assert result.chapters == [
        ChapterInfo(title="施工准备", start_page=6, end_page=None)
    ]


def test_malformed_noncritical_items_are_dropped_without_losing_valid_facts() -> None:
    result = LLMService.parse_json_response(
        '{"specialty":["错误类型"],"project":{"project_name":["错误类型"],'
        '"construction_unit":"建设单位A"},"standards":[{"name":"无编号"},'
        '{"standard_code":"GB 50015-2019","source_page":"unknown"}],'
        '"main_materials":[{"usage":"给水"},{"material_name":"PP-R管",'
        '"spec":{"unexpected":true}}],"main_methods":[123,'
        '{"method_name":"热熔连接","page":"14"}]}'
    )
    assert result.specialty is None
    assert result.project.project_name is None
    assert result.project.construction_unit.value == "建设单位A"
    assert [standard.code for standard in result.standards] == ["GB 50015-2019"]
    assert result.standards[0].source_page is None
    assert [material.name for material in result.main_materials] == ["PP-R管"]
    assert result.main_materials[0].specification is None
    assert [method.name for method in result.main_methods] == ["热熔连接"]
    assert result.main_methods[0].source_page == 14


def test_explicit_project_and_schedule_labels_are_extracted_from_page_four() -> None:
    pages = [
        PDFPageText(1, "封面"),
        PDFPageText(
            4,
            "工程名称：测试中心给排水工程\n"
            "项目所在地：上海市浦东新区\n"
            "建设单位：测试建设有限公司\n"
            "设计单位：测试设计院\n"
            "监理单位：测试监理有限公司\n"
            "施工总包单位：测试总承包有限公司\n"
            "计划开工日期：2022年4月15日\n"
            "计划竣工日期：2022年8月12日\n"
            "计划总工期：120日历天",
        ),
    ]
    project, schedule = MetadataExtractionService().extract(pages)

    assert project.project_name.value == "测试中心给排水工程"
    assert project.project_location.value == "上海市浦东新区"
    assert project.construction_unit.value == "测试建设有限公司"
    assert project.design_unit.value == "测试设计院"
    assert project.supervision_unit.value == "测试监理有限公司"
    assert project.general_contractor.value == "测试总承包有限公司"
    assert project.project_name.source_page == 4
    assert project.project_name.source_text == "工程名称：测试中心给排水工程"
    assert schedule.plan_start_date.value == "2022-04-15"
    assert schedule.plan_completion_date.value == "2022-08-12"
    assert schedule.duration_days.value == 120
    assert schedule.duration_days.source_page == 4


def test_chapter_detection_rejects_metadata_parameters_and_list_items() -> None:
    pages = [
        PDFPageText(
            4,
            "建设单位：测试建设有限公司\n"
            "设计单位：测试设计院\n"
            "1.0MPa;\nMpa\n）、采用专用PPR配套支架\n~2mm为宜\n"
            "1.1.2 采用热熔连接施工\n"
            "一、工程概况\n6.2.3 施工方法",
        )
    ]
    chapters = ChapterDetectionService().detect(pages)
    assert [(item.title, item.level) for item in chapters] == [
        ("工程概况", 1),
        ("施工方法", 3),
    ]


def test_contents_candidates_are_matched_to_actual_pdf_body_pages() -> None:
    pages = [
        PDFPageText(2, "目录\n第一章 工程概况........1"),
        PDFPageText(3, "第二章 施工方法........4"),
        PDFPageText(4, "第一章 工程概况\n项目内容"),
        PDFPageText(7, "第二章 施工方法\n施工内容"),
    ]
    chapters = ChapterDetectionService().detect(pages)
    assert chapters == [
        ChapterInfo(title="工程概况", start_page=4, end_page=7, level=1),
        ChapterInfo(title="施工方法", start_page=7, end_page=7, level=1),
    ]


def test_source_locator_replaces_llm_pages_with_actual_pdf_pages() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "project": {
                "project_name": {
                    "value": "测试工程",
                    "source_page": 99,
                    "source_text": "工程名称：测试工程",
                }
            },
            "schedule": {
                "plan_start_date": {
                    "value": "2022-04-15",
                    "source_page": 88,
                    "source_text": "计划开工日期：2022年4月15日",
                }
            },
            "standards": [
                {"name": "建筑给水排水设计标准", "code": "GB50015-2019", "source_page": 77}
            ],
            "main_work_items": ["室内给水系统"],
            "main_materials": [{"name": "PP-R管", "source_page": 66}],
            "main_methods": [
                {"name": "热熔连接", "source_page": 55},
                {"name": "原文不存在的工法", "source_page": 44, "source_text": "伪造片段"},
            ],
        }
    )
    pages = [
        PDFPageText(4, "工程名称：测试工程\n计划开工日期：2022年4月15日"),
        PDFPageText(5, "主要施工内容包括室内给水系统。"),
        PDFPageText(6, "给水管采用PP-R管。"),
        PDFPageText(7, "管道采用热熔连接。"),
        PDFPageText(8, "《建筑给水排水设计标准》\n(GB 50015-2019)"),
    ]
    located = SourceLocatorService().locate(analysis, pages)

    assert located.project.project_name.source_page == 4
    assert located.schedule.plan_start_date.source_page == 4
    assert located.standards[0].source_page == 8
    assert located.main_work_items[0].source_page == 5
    assert located.main_materials[0].source_page == 6
    assert located.main_methods[0].source_page == 7
    assert located.main_methods[1].source_page is None
    assert located.main_methods[1].source_text is None
    assert "工程名称：测试工程" in located.project.project_name.source_text


def test_cross_line_standard_name_and_code_are_combined() -> None:
    extracted = PDFExtractionResult(
        page_count=1,
        char_count=40,
        text="《建筑给水排水设计标准》\n(GB 50015-2019)",
        pages=[PDFPageText(5, "《建筑给水排水设计标准》\n(GB 50015-2019)")],
    )
    standards = DocumentAnalysisService(FakeLLMService())._extract_standards(extracted)
    assert len(standards) == 1
    assert standards[0].name == "建筑给水排水设计标准"
    assert standards[0].code == "GB50015-2019"
    assert standards[0].source_page == 5


def test_schedule_requirement_phrasing_and_spacing_are_extracted() -> None:
    pages = [
        PDFPageText(
            4,
            "1.1.2、工期要求：计划开工日期为2022 年4 月15 日……\n"
            "计划竣工日期（中间交接日期）为2023 年05 月10 日，\n"
            "计划总工期为390 日历天。",
        )
    ]
    _, schedule = MetadataExtractionService().extract(pages)
    assert schedule.plan_start_date.value == "2022-04-15"
    assert schedule.plan_completion_date.value == "2023-05-10"
    assert schedule.duration_days.value == 390
    assert schedule.plan_start_date.source_page == 4
    assert "计划开工日期为2022 年4 月15 日" in schedule.plan_start_date.source_text


def test_project_name_and_location_continuation_lines_are_joined_safely() -> None:
    pages = [
        PDFPageText(
            4,
            "工程名称：长沙市轨道交通工程（EPC+F）\n"
            "项目安装工程。\n"
            "项目所在地：湖南省长沙市\n"
            "岳麓区\n"
            "1.1.2、工期要求：计划总工期为390日历天",
        )
    ]
    project, _ = MetadataExtractionService().extract(pages)
    assert project.project_name.value == "长沙市轨道交通工程（EPC+F）项目安装工程。"
    assert project.project_location.value == "湖南省长沙市岳麓区"
    assert "工期要求" not in project.project_location.value
    assert project.project_name.source_text == (
        "工程名称：长沙市轨道交通工程（EPC+F）\n项目安装工程。"
    )


def test_numbered_body_lists_are_not_chapters() -> None:
    pages = [
        PDFPageText(
            10,
            "1、所有的给排水管材应符合要求。\n"
            "2、安装好的给排水管道应及时保护。\n"
            "3、建立以安全生产责任制为核心的管理制度。\n"
            "7、特殊工种组织专业培训。\n"
            "三、施工部署\n6.2.3 管道支架安装",
        )
    ]
    chapters = ChapterDetectionService().detect(pages)
    assert [(chapter.title, chapter.level) for chapter in chapters] == [
        ("施工部署", 1),
        ("管道支架安装", 3),
    ]


def test_chapter_end_page_uses_next_same_or_higher_level() -> None:
    pages = [
        PDFPageText(14, "一、施工方法"),
        PDFPageText(15, "6.1 管道安装"),
        PDFPageText(16, "6.1.1 支架安装"),
        PDFPageText(17, "6.2 管道试验"),
        PDFPageText(19, "二、质量标准及保护措施"),
    ]
    chapters = ChapterDetectionService().detect(pages)
    by_title = {chapter.title: chapter for chapter in chapters}
    assert by_title["施工方法"].end_page == 19
    assert by_title["管道安装"].end_page == 17
    assert by_title["支架安装"].end_page == 17
    assert by_title["管道试验"].end_page == 19
    assert by_title["质量标准及保护措施"].end_page == 19


def test_source_locator_prefers_body_over_contents_page() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "main_methods": [
                {
                    "name": "管道支架安装",
                    "source_page": 2,
                    "source_text": "管道支架安装",
                }
            ]
        }
    )
    pages = [
        PDFPageText(2, "目录\n6.2 管道支架安装........18"),
        PDFPageText(18, "6.2 管道支架安装\n支架安装应符合施工要求。"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    assert located.main_methods[0].source_page == 18
    assert located.main_methods[0].source_text.startswith("6.2 管道支架安装")
    assert "支架安装应符合施工要求" in located.main_methods[0].source_text


def test_source_locator_prefers_explicit_design_unit_label() -> None:
    company = "中国建筑东北设计研究有限公司"
    analysis = DocumentAnalysis.model_validate(
        {
            "project": {
                "design_unit": {
                    "value": company,
                    "source_page": 20,
                    "source_text": f"{company}设计的施工图纸",
                }
            }
        }
    )
    pages = [
        PDFPageText(4, f"设计单位：{company}"),
        PDFPageText(20, f"本工程依据{company}设计的施工图纸进行施工。"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    assert located.project.design_unit.source_page == 4
    assert located.project.design_unit.source_text == f"设计单位：{company}"


def test_material_location_scores_name_specification_and_usage_together() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "main_materials": [
                {
                    "name": "PP-R给水塑料管",
                    "specification": "S4系列",
                    "usage": "室内给水",
                }
            ]
        }
    )
    pages = [
        PDFPageText(
            10,
            "室内给水采用S4系列PP-R给水塑料管，连接方式为热熔连接。",
        ),
        PDFPageText(12, "PP-R给水塑料管"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    assert located.main_materials[0].source_page == 10
    assert "PP-R给水塑料管" in located.main_materials[0].source_text


def test_work_item_location_prefers_main_work_context() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "工程概况", "start_page": 4, "end_page": 5, "level": 1}
            ],
            "main_work_items": ["排水系统"],
        }
    )
    pages = [
        PDFPageText(4, "项目主要施工内容\n室内给水工程、排水系统及雨水系统。"),
        PDFPageText(5, "施工范围说明。"),
        PDFPageText(23, "排水系统安装完成后进行通球试验。"),
    ]
    assert SourceLocatorService._work_item_context_pages(analysis.model_dump(), pages) == {
        4: 250,
        5: 250,
    }
    located = SourceLocatorService().locate(analysis, pages)
    assert located.main_work_items[0].source_page == 4
    assert "排水系统" in located.main_work_items[0].source_text


@pytest.mark.parametrize(
    ("method_name", "expected_page", "expected_text"),
    [
        ("PP-R管道热熔连接施工方法", 15, "热熔连接"),
        ("UPVC塑料排水管安装方法", 16, "UPVC排水管安装"),
        ("管道试压方法", 20, "试验压力"),
    ],
)
def test_semantic_method_names_are_located_by_keywords_and_chapter_context(
    method_name: str, expected_page: int, expected_text: str
) -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "施工方法", "start_page": 14, "end_page": 19, "level": 1},
                {
                    "title": "质量标准及保护措施",
                    "start_page": 19,
                    "end_page": 22,
                    "level": 1,
                },
                {"title": "管道试验", "start_page": 20, "end_page": 22, "level": 2},
            ],
            "main_methods": [{"name": method_name}],
        }
    )
    pages = [
        PDFPageText(2, "目录\nPP-R给水管安装........15\nUPVC排水管安装........16"),
        PDFPageText(15, "6.1 PP-R给水管安装\n管道连接采用热熔连接施工工艺。"),
        PDFPageText(16, "6.2 UPVC排水管安装\n塑料排水管采用承插粘接安装。"),
        PDFPageText(20, "7.2 管道压力试验\n管道试压时试验压力应符合要求。"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    method = next(method for method in located.main_methods if method.name == method_name)
    assert method.source_page == expected_page
    assert expected_text in method.source_text


def test_upvc_material_complete_context_beats_name_only_plan_table() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "main_materials": [
                {"name": "UPVC内螺旋消音排水管", "usage": "室内排水立管"},
                {"name": "UPVC排水管", "usage": "室内排水横管"},
            ]
        }
    )
    pages = [
        PDFPageText(
            10,
            "安装工程材料选型及连接方式\n"
            "室内排水立管贴临卧室时采用UPVC内螺旋消音排水管，承插粘接。\n"
            "室内排水横管采用UPVC排水管，连接方式为承插粘接。",
        ),
        PDFPageText(12, "主要材料进场计划\nUPVC内螺旋消音排水管\nUPVC排水管"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    assert [material.source_page for material in located.main_materials] == [10, 10]
    assert all("采用" in material.source_text for material in located.main_materials)


def test_semantic_work_item_suffix_locates_colon_scope_text() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "工程概况", "start_page": 4, "end_page": 5, "level": 1}
            ],
            "main_work_items": ["排水系统施工"],
        }
    )
    pages = [
        PDFPageText(
            4,
            "项目主要施工内容\n"
            "（3）、排水系统：完成图纸范围内所有住宅、配套单体雨水、污水、废水及冷凝水排水系统的施工。",
        ),
        PDFPageText(23, "排水系统施工完成后进行检查。"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    assert located.main_work_items[0].value == "排水系统"
    assert located.main_work_items[0].source_page == 4
    assert "排水系统：完成图纸范围" in located.main_work_items[0].source_text


def test_engineering_category_and_specialty_are_traceable_source_values() -> None:
    analysis = DocumentAnalysis.model_validate(
        {"engineering_category": "安装工程", "specialty": "给排水"}
    )
    assert analysis.engineering_category.value == "安装工程"
    assert analysis.specialty.value == "给排水"

    pages = [
        PDFPageText(4, "工程名称：测试项目安装工程\n本方案适用于给排水工程施工。"),
        PDFPageText(10, "安装工程材料选型及连接方式。"),
        PDFPageText(18, "给排水管道安装。"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    assert located.engineering_category.source_page == 4
    assert "项目安装工程" in located.engineering_category.source_text
    assert located.specialty.source_page == 4
    assert "给排水工程" in located.specialty.source_text


def test_cross_line_project_name_deterministically_extracts_installation_category() -> None:
    pages = [
        PDFPageText(
            4,
            "工程名称：长沙市雨花区和景家园（二标段）工程总承包（EPC+F）\n"
            "项目安装工程。\n"
            "项目所在地：长沙市雨花区",
        )
    ]
    service = MetadataExtractionService()
    project, _ = service.extract(pages)
    category = service.extract_engineering_category(pages, project)

    assert category.value == "安装工程"
    assert category.source_page == 4
    assert "项目安装工程" in category.source_text


def test_detailed_flushing_disinfection_text_beats_test_overview() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {
                    "title": "质量标准及保护措施",
                    "start_page": 19,
                    "end_page": 22,
                    "level": 1,
                },
                {"title": "管道试验", "start_page": 20, "end_page": 22, "level": 2},
            ],
            "main_methods": [{"name": "管道试压、冲洗、消毒"}],
        }
    )
    pages = [
        PDFPageText(20, "系统试验包括：管道试压、冲洗、消毒。"),
        PDFPageText(21, "管道冲洗\n冲洗步骤应连续进行。"),
        PDFPageText(22, "管道冲洗、消毒\n生活给水管道及设备消毒应符合要求。"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    assert located.main_methods[0].source_page == 22
    assert "消毒" in located.main_methods[0].source_text


@pytest.mark.parametrize(
    ("method_name", "expected_heading", "forbidden_heading"),
    [
        ("塑料排水管安装", "6.2.3、塑料排水管安装", "6.2.2、支架安装"),
        ("管道支架安装", "6.3、管道支架安装", "6.2.4、柔性铸铁排水管安装"),
    ],
)
def test_method_source_text_starts_at_target_heading_on_shared_page(
    method_name: str, expected_heading: str, forbidden_heading: str
) -> None:
    analysis = DocumentAnalysis.model_validate(
        {"main_methods": [{"name": method_name}]}
    )
    pages = [
        PDFPageText(
            16,
            "6.2.2、支架安装\n支架安装完成后固定管道。\n"
            "6.2.3、塑料排水管安装\nUPVC塑料排水管采用承插粘接。\n"
            "6.2.4、柔性铸铁排水管安装\n柔性接口连接。\n"
            "6.3、管道支架安装\n支吊架间距应符合要求。",
        )
    ]
    located = SourceLocatorService().locate(analysis, pages)
    source_text = located.main_methods[0].source_text
    assert source_text.startswith(expected_heading)
    assert forbidden_heading not in source_text


def test_method_string_and_title_aliases_are_not_filtered() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "main_methods": [
                "管道试压",
                {"title": "管道冲洗、消毒"},
                {"value": "生活给水管道及设备消毒"},
            ]
        }
    )
    assert [method.name for method in analysis.main_methods] == [
        "管道试压",
        "管道冲洗、消毒",
        "生活给水管道及设备消毒",
    ]


def test_source_backed_test_and_flushing_chapters_fill_llm_method_omissions() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "管道试压", "start_page": 20, "end_page": 20, "level": 3},
                {
                    "title": "管道冲洗、消毒",
                    "start_page": 21,
                    "end_page": 21,
                    "level": 3,
                },
            ],
            "main_methods": [],
        }
    )
    pages = [
        PDFPageText(20, "7.4.1、管道试压\n试验压力应符合要求。"),
        PDFPageText(21, "7.4.4、管道冲洗、消毒\n冲洗及消毒应连续进行。"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    assert [method.name for method in located.main_methods] == [
        "管道试压",
        "管道冲洗、消毒",
    ]
    assert [method.source_page for method in located.main_methods] == [20, 21]
    assert located.main_methods[0].source_text.startswith("7.4.1、管道试压")
    assert located.main_methods[1].source_text.startswith("7.4.4、管道冲洗、消毒")


def test_numbered_pressure_test_title_beats_next_page_continuation() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "main_methods": [
                {
                    "name": "管道试压",
                    "source_page": 21,
                    "source_text": "为强度试验合格后进行严密性试验。",
                }
            ]
        }
    )
    pages = [
        PDFPageText(20, "7.4.1、管道试压\n管道试压时试验压力应符合设计要求。"),
        PDFPageText(21, "为强度试验合格后进行严密性试验。"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    method = located.main_methods[0]
    assert method.source_page == 20
    assert method.source_text.startswith("7.4.1、管道试压")


@pytest.mark.parametrize(
    ("method_name", "expected_heading"),
    [
        ("通球试验", "7.4.7、排水管道通球试验"),
        ("通水试验", "7.4.8、管道通水试验"),
    ],
)
def test_numbered_detailed_test_title_beats_isolated_flow_term(
    method_name: str, expected_heading: str
) -> None:
    analysis = DocumentAnalysis.model_validate(
        {"main_methods": [{"name": method_name}]}
    )
    pages = [
        PDFPageText(16, f"施工工艺流程\n安装→检查→{method_name}→验收"),
        PDFPageText(
            23,
            "7.4.7、排水管道通球试验\n通球试验应按系统逐段进行。\n"
            "7.4.8、管道通水试验\n通水试验应检查各接口是否渗漏。",
        ),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    method = located.main_methods[0]
    assert method.source_page == 23
    assert method.source_text.startswith(expected_heading)
    assert "施工工艺流程" not in method.source_text


def test_installation_methods_are_supplemented_only_inside_method_section() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "施工方法", "start_page": 14, "end_page": 19, "level": 1},
                {
                    "title": "柔性铸铁排水管安装",
                    "start_page": 17,
                    "end_page": 18,
                    "level": 3,
                },
                {
                    "title": "管道支架安装",
                    "start_page": 18,
                    "end_page": 19,
                    "level": 2,
                },
                {"title": "安全设施安装", "start_page": 25, "end_page": 25, "level": 2},
            ],
            "main_methods": [],
        }
    )
    pages = [
        PDFPageText(17, "6.2.4、柔性铸铁排水管安装\n管道采用柔性接口连接。"),
        PDFPageText(18, "6.3、管道支架安装\n支吊架安装应牢固。"),
        PDFPageText(25, "8.2、安全设施安装\n安全设施应齐全。"),
    ]
    located = SourceLocatorService().locate(analysis, pages)
    assert [method.name for method in located.main_methods] == [
        "柔性铸铁排水管安装",
        "管道支架安装",
    ]
    assert [method.source_page for method in located.main_methods] == [17, 18]
    assert located.main_methods[0].source_text.startswith("6.2.4、柔性铸铁排水管安装")
    assert located.main_methods[1].source_text.startswith("6.3、管道支架安装")


@pytest.mark.parametrize(
    ("chapter_title", "page_number", "heading"),
    [
        ("管道试压", 20, "7.4.1、管道试压"),
        ("排水管道灌水", 22, "7.4.6、排水管道灌水"),
    ],
)
def test_explicit_test_chapter_completes_missing_llm_method(
    chapter_title: str, page_number: int, heading: str
) -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {
                    "title": chapter_title,
                    "start_page": page_number,
                    "end_page": page_number,
                    "level": 3,
                }
            ],
            "main_methods": [],
        }
    )
    pages = [PDFPageText(page_number, f"{heading}\n详细试验步骤。")]
    located = SourceLocatorService().locate(analysis, pages)
    assert [method.name for method in located.main_methods] == [chapter_title]
    assert located.main_methods[0].source_page == page_number
    assert located.main_methods[0].source_text.startswith(heading)


def test_method_deduplication_ignores_pp_r_spacing_and_keeps_provenance() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "main_methods": [
                {"name": "PP-R管道施工工艺"},
                {
                    "name": "PP-R 管道施工工艺",
                    "source_page": 14,
                    "source_text": "6.1、PP-R 管道施工工艺",
                },
            ]
        }
    )
    completed = DeterministicPostProcessingService().process(analysis, []).main_methods
    assert len(completed) == 1
    assert completed[0].name == "PP-R 管道施工工艺"
    assert completed[0].source_page == 14

    located = SourceLocatorService().locate(
        analysis,
        [PDFPageText(14, "6.1、PP-R 管道施工工艺\n管道采用热熔连接。")],
    )
    assert len(located.main_methods) == 1
    assert located.main_methods[0].source_page == 14


def test_non_method_chapter_is_not_completed_as_main_method() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "安全保证措施", "start_page": 30, "end_page": 32, "level": 1},
                {"title": "安全设施安装", "start_page": 31, "end_page": 31, "level": 2},
            ],
            "main_methods": [],
        }
    )
    located = SourceLocatorService().locate(
        analysis,
        [PDFPageText(31, "9.2、安全设施安装\n安全设施应齐全。")],
    )
    assert located.main_methods == []


def test_deterministic_methods_are_stable_across_different_llm_outputs() -> None:
    chapters = [
        {"title": "施工方法", "start_page": 14, "end_page": 19, "level": 1},
        {"title": "管道试验", "start_page": 20, "end_page": 23, "level": 2},
    ]
    pages = [
        PDFPageText(14, "6.1.1、PP-R 管道施工工艺\n管道采用热熔连接。"),
        PDFPageText(16, "6.2.3、塑料排水管安装\n排水管采用粘接。"),
        PDFPageText(20, "7.4.1、管道试压\n试验压力应符合设计要求。"),
        PDFPageText(22, "7.4.7、排水管道灌水\n灌水试验应按系统进行。"),
    ]
    run_a = DocumentAnalysis.model_validate(
        {
            "chapters": chapters,
            "main_methods": [
                {"name": "PP-R管道施工工艺（热熔连接）"},
                {"name": "管道试压"},
            ],
        }
    )
    run_b = DocumentAnalysis.model_validate(
        {
            "chapters": chapters,
            "main_methods": [{"name": "PP-R 管道施工工艺"}],
        }
    )

    located_a = SourceLocatorService().locate(run_a, pages)
    located_b = SourceLocatorService().locate(run_b, pages)
    canonical = DeterministicPostProcessingService.canonical_method_name
    core_a = {canonical(method.name) for method in located_a.main_methods}
    core_b = {canonical(method.name) for method in located_b.main_methods}

    assert core_a == core_b
    assert canonical("管道试压") in core_b
    assert canonical("排水管道灌水") in core_b
    assert sum(key == canonical("PP-R 管道施工工艺") for key in core_b) == 1
    pressure = next(method for method in located_b.main_methods if method.name == "管道试压")
    assert pressure.source_page == 20
    assert pressure.source_text.startswith("7.4.1、管道试压")


def test_numbered_work_item_replaces_unlocated_llm_item_and_keeps_continuation() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "工程概况", "start_page": 4, "end_page": 5, "level": 1}
            ],
            "main_work_items": [
                {
                    "value": "排水系统施工",
                    "source_page": None,
                    "source_text": None,
                }
            ],
        }
    )
    pages = [
        PDFPageText(
            4,
            "项目主要施工内容\n"
            "（1）、给水部分：完成给水系统施工。\n"
            "（2）、卫生洁具安装：完成洁具安装。\n"
            "（3）、排水系统：完成图纸范围内所有住宅、配套单体雨水、污水",
        ),
        PDFPageText(5, "（含透气立管）、废水及冷凝水排水系统的施工。"),
    ]

    located = SourceLocatorService().locate(analysis, pages)
    drainage = next(item for item in located.main_work_items if item.value == "排水系统")
    assert drainage.source_page == 4
    assert drainage.source_text.startswith("（3）、排水系统：")
    assert "冷凝水排水系统的施工" in drainage.source_text


def test_plain_parenthesized_enumeration_is_not_a_deterministic_method() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "施工方法", "start_page": 14, "end_page": 14, "level": 1}
            ]
        }
    )
    pages = [
        PDFPageText(
            14,
            "6.3、管道支架安装\n支架应安装牢固。\n"
            "（3）、安装在外墙上管道支架必须做在外墙保温层内……",
        )
    ]

    completed = DeterministicPostProcessingService().process(analysis, pages)
    assert [method.name for method in completed.main_methods] == ["管道支架安装"]


def test_product_protection_numbered_list_is_not_a_deterministic_method() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "成品保护", "start_page": 24, "end_page": 24, "level": 1}
            ]
        }
    )
    pages = [
        PDFPageText(
            24,
            "1、安装好的给排水管道及支吊架不得做支撑……\n"
            "2、给排水管道安装完成后，应将所有管口封闭……\n"
            "3、给水水表安装好后，用薄膜包扎好……",
        )
    ]

    completed = DeterministicPostProcessingService().process(analysis, pages)
    assert completed.main_methods == []


def test_quality_defect_chapter_is_not_a_deterministic_method() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {
                    "title": "常见质量通病及防止措施",
                    "start_page": 25,
                    "end_page": 25,
                    "level": 1,
                }
            ]
        }
    )
    pages = [PDFPageText(25, "8.1、给水管支架破坏，管道污染\n应加强成品保护。")]

    completed = DeterministicPostProcessingService().process(analysis, pages)
    assert completed.main_methods == []


def test_pp_r_method_variants_collapse_to_local_numbered_title() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "施工方法", "start_page": 14, "end_page": 14, "level": 1}
            ],
            "main_methods": [
                {"name": "PP-R 管道施工工艺"},
                {"name": "PP-R管道热熔连接施工工艺"},
                {"name": "PP-R管道热熔连接施工方法"},
            ],
        }
    )
    pages = [PDFPageText(14, "6.1.1、PP-R 管道施工工艺\n管道采用热熔连接。")]

    completed = DeterministicPostProcessingService().process(analysis, pages)
    pp_r_methods = [
        method for method in completed.main_methods if "PP-R" in method.name.upper()
    ]
    assert len(pp_r_methods) == 1
    assert pp_r_methods[0].name == "PP-R 管道施工工艺"
    assert pp_r_methods[0].source_page == 14
    assert pp_r_methods[0].source_text.startswith("6.1.1、PP-R 管道施工工艺")


def test_detailed_llm_work_items_merge_into_local_short_titles() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "工程概况", "start_page": 4, "end_page": 4, "level": 1}
            ],
            "main_work_items": [
                "给水部分：完成图纸范围内全部生活给水系统施工",
                "卫生洁具安装：完成全部卫生洁具安装工作",
                "排水系统：完成雨水、污水及废水系统施工",
                "室内生活给水系统施工",
                "预留、预埋工程",
            ],
        }
    )
    pages = [
        PDFPageText(
            4,
            "项目主要施工内容\n"
            "（1）、给水部分：完成图纸范围内全部生活给水系统施工。\n"
            "（2）、卫生洁具安装：完成全部卫生洁具安装工作。\n"
            "（3）、排水系统：完成雨水、污水及废水系统施工。",
        )
    ]

    completed = SourceLocatorService().locate(analysis, pages)
    assert [item.value for item in completed.main_work_items] == [
        "给水部分",
        "卫生洁具安装",
        "排水系统",
    ]
    assert all(item.source_page == 4 for item in completed.main_work_items)
    assert all("：" in (item.source_text or "") for item in completed.main_work_items)


def test_all_thirteen_core_methods_remain_in_deterministic_output() -> None:
    expected = {
        "PP-R 管道施工工艺",
        "排水管安装施工工艺流程",
        "支架安装",
        "塑料排水管安装",
        "柔性铸铁排水管安装",
        "管道支架安装",
        "管道试压",
        "管道冲洗",
        "管道冲洗、消毒",
        "生活给水管道及设备消毒",
        "排水管道灌水",
        "排水管道通球试验",
        "管道通水试验",
    }
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "施工方法", "start_page": 14, "end_page": 19, "level": 1},
                {"title": "管道试验", "start_page": 20, "end_page": 23, "level": 2},
            ]
        }
    )
    pages = [
        PDFPageText(
            14,
            "6.1.1、PP-R 管道施工工艺\n6.2.1、排水管安装施工工艺流程",
        ),
        PDFPageText(16, "6.2.2、支架安装\n6.2.3、塑料排水管安装"),
        PDFPageText(17, "6.2.4、柔性铸铁排水管安装"),
        PDFPageText(18, "6.3、管道支架安装"),
        PDFPageText(20, "7.4.1、管道试压\n7.4.2、管道冲洗"),
        PDFPageText(21, "7.4.4、管道冲洗、消毒"),
        PDFPageText(
            22,
            "7.4.5、生活给水管道及设备消毒\n7.4.7、排水管道灌水",
        ),
        PDFPageText(23, "7.4.7、排水管道通球试验\n7.4.8、管道通水试验"),
    ]

    completed = SourceLocatorService().locate(analysis, pages)
    assert {method.name for method in completed.main_methods} == expected


def test_final_method_cleanup_filters_llm_plain_enumeration_after_merge() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "施工方法", "start_page": 14, "end_page": 19, "level": 1}
            ],
            "main_methods": [
                {"name": "PP-R管道热熔连接施工方法"},
                {"name": "（3）、安装在外墙上管道支架必须做在外墙保温层内……"},
            ],
        }
    )
    pages = [
        PDFPageText(
            14,
            "6.1.1、PP-R 管道施工工艺\n管道采用热熔连接。\n"
            "（3）、安装在外墙上管道支架必须做在外墙保温层内……",
        )
    ]

    located = SourceLocatorService().locate(analysis, pages)
    assert [method.name for method in located.main_methods] == ["PP-R 管道施工工艺"]
    assert located.main_methods[0].source_page == 14


def test_final_method_cleanup_filters_quality_defect_llm_item() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {"title": "施工方法", "start_page": 14, "end_page": 19, "level": 1},
                {
                    "title": "常见质量通病及防止措施",
                    "start_page": 25,
                    "end_page": 26,
                    "level": 1,
                },
                {
                    "title": "给水管支架破坏，管道污染",
                    "start_page": 25,
                    "end_page": 25,
                    "level": 2,
                },
            ],
            "main_methods": [
                {"name": "管道支架安装"},
                {"name": "给水管支架破坏，管道污染"},
            ],
        }
    )
    pages = [
        PDFPageText(18, "6.3、管道支架安装\n支吊架安装应牢固。"),
        PDFPageText(25, "8.1、给水管支架破坏，管道污染\n应加强成品保护。"),
    ]

    located = SourceLocatorService().locate(analysis, pages)
    assert [method.name for method in located.main_methods] == ["管道支架安装"]
    assert located.main_methods[0].source_page == 18


def test_narrative_project_metadata_is_extracted_from_engineering_overview() -> None:
    pages = [
        PDFPageText(
            4,
            "1.1 工程概况\n"
            "长沙市雨花城市建设投资集团有限公司投资的\n"
            "“和景家园（二标段）”智能化弱电系统工程，\n"
            "位于长沙市雨花区劳动东路。",
        )
    ]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.project_name.value == "和景家园（二标段）智能化弱电系统工程"
    assert project.project_name.source_page == 4
    assert "“和景家园（二标段）”智能化弱电系统工程" in project.project_name.source_text
    assert project.project_location.value == "长沙市雨花区劳动东路"
    assert project.project_location.source_page == 4
    assert "位于长沙市雨花区" in project.project_location.source_text
    assert project.construction_unit is None


def test_project_name_is_extracted_from_split_title_page() -> None:
    pages = [
        PDFPageText(
            1,
            "和景家园（二标段）\n智能化系统工程\n施工组织设计\n编制单位：测试单位",
        )
    ]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.project_name.value == "和景家园（二标段）智能化系统工程"
    assert project.project_name.source_page == 1
    assert project.project_name.source_text == "和景家园（二标段）\n智能化系统工程"


@pytest.mark.parametrize(
    ("text", "expected_name"),
    [
        ("智慧园区项目位于长沙市雨花区。", "智慧园区项目"),
        (
            "建设单位投资的‘智慧园区智能化工程’，建设地点明确。",
            "智慧园区智能化工程",
        ),
    ],
)
def test_narrative_project_name_patterns_are_template_independent(
    text: str, expected_name: str
) -> None:
    project, _ = MetadataExtractionService().extract([PDFPageText(4, text)])
    assert project.project_name.value == expected_name
    assert project.project_name.source_page == 4


def test_narrative_schedule_dates_are_normalized() -> None:
    pages = [
        PDFPageText(
            6,
            "本工程计划2022年4月20日开工，\n"
            "2023年5月15日完工，工期为390日历天。",
        )
    ]

    _, schedule = MetadataExtractionService().extract(pages)
    assert schedule.plan_start_date.value == "2022-04-20"
    assert schedule.plan_completion_date.value == "2023-05-15"
    assert schedule.duration_days.value == 390
    assert schedule.plan_start_date.source_page == 6
    assert "2022年4月20日开工" in schedule.plan_start_date.source_text
    assert "2023年5月15日完工" in schedule.plan_completion_date.source_text


def test_engineering_category_rejects_late_incidental_installation_phrase() -> None:
    pages = [
        PDFPageText(1, "和景家园智能化系统工程\n施工组织设计"),
        PDFPageText(4, "1.1 工程概况\n本项目位于长沙市雨花区。"),
        PDFPageText(65, "安装工程施工记录、资料保证措施"),
    ]
    service = MetadataExtractionService()
    project, _ = service.extract(pages)
    candidate = DocumentAnalysis.model_validate(
        {
            "engineering_category": {
                "value": "安装工程",
                "source_page": 65,
                "source_text": "安装工程施工记录、资料保证措施",
            }
        }
    ).engineering_category

    assert service.extract_engineering_category(pages, project) is None
    assert service.validate_engineering_category(pages, project, candidate) is None


def test_standard_names_are_extracted_before_or_after_codes() -> None:
    text = (
        "GB 50314-2015 智能建筑设计标准\n"
        "民用建筑电气设计规范（JGJ 16-2008）\n"
        "※GB 50348-2018 安全防范工程技术标准"
    )
    extracted = PDFExtractionResult(
        page_count=1,
        char_count=len(text),
        text=text,
        pages=[PDFPageText(8, text)],
    )

    standards = DocumentAnalysisService(FakeLLMService())._extract_standards(extracted)
    assert [(item.name, item.code) for item in standards] == [
        ("智能建筑设计标准", "GB50314-2015"),
        ("民用建筑电气设计规范", "JGJ16-2008"),
        ("安全防范工程技术标准", "GB50348-2018"),
    ]
    assert all(item.source_page == 8 for item in standards)
    assert all(item.name in item.source_text for item in standards)


def test_generic_installation_chapter_tree_extracts_cross_specialty_methods() -> None:
    expected = {
        "系统配线施工要求",
        "电话插座安装",
        "电话分线盒安装",
        "线缆桥架施工",
        "管道施工",
        "线路敷设",
        "设备安装",
        "系统安装调试",
        "分配网络安装和施工",
        "系统总调试",
        "电缆敷设",
        "前端设备安装",
        "摄像机安装",
        "报警系统调试",
        "停车场管理系统部件安装",
        "停车场管理系统调试",
        "门禁系统调试",
        "监控系统调试",
    }
    method_titles = list(expected)
    pages = [PDFPageText(20, "系统安装及施工")]
    pages.extend(
        PDFPageText(21 + index, f"4.{index + 1}、{title}\n具体施工操作要求。")
        for index, title in enumerate(method_titles)
    )
    pages.extend(
        [
            PDFPageText(59, "质量保证措施"),
            PDFPageText(60, "5.1、工程管理\n项目组织协调。"),
            PDFPageText(61, "5.2、岗位职责\n明确岗位责任。"),
            PDFPageText(62, "5.3、报警系统测试\n功能测试记录。"),
            PDFPageText(63, "5.4、质量保证\n质量控制措施。"),
            PDFPageText(64, "5.5、安全措施\n现场安全管理。"),
            PDFPageText(65, "5.6、文明施工\n文明施工要求。"),
            PDFPageText(66, "5.7、售后服务\n售后响应安排。"),
            PDFPageText(67, "5.8、培训\n用户培训计划。"),
        ]
    )
    chapters = ChapterDetectionService().detect(pages)
    analysis = DocumentAnalysis.model_validate(
        {"chapters": [chapter.model_dump() for chapter in chapters]}
    )

    completed = DeterministicPostProcessingService().process(analysis, pages)
    assert {method.name for method in completed.main_methods} == expected
    assert all(method.source_page is not None for method in completed.main_methods)
    assert "报警系统测试" not in {method.name for method in completed.main_methods}


def test_title_page_project_name_strips_document_suffix() -> None:
    pages = [
        PDFPageText(1, "和景家园（二标段）智能化弱电系统工程施工\n施工组织设计")
    ]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.project_name.value == "和景家园（二标段）智能化弱电系统工程"
    assert project.project_name.source_page == 1
    assert "工程施工" in project.project_name.source_text


def test_narrative_location_keeps_multiline_site_description() -> None:
    pages = [
        PDFPageText(
            4,
            "1.1 工程概况\n"
            "“和景家园（二标段）”智能化弱电系统工程，\n"
            "位于长沙市雨花区，位于黎托街道川河村，\n"
            "东临沿河规划路，南临规划阳青东路。",
        )
    ]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.project_name.value == "和景家园（二标段）智能化弱电系统工程"
    assert project.project_location.value == (
        "长沙市雨花区，位于黎托街道川河村，东临沿河规划路，南临规划阳青东路"
    )
    assert project.project_location.source_page == 4
    assert "东临沿河规划路" in project.project_location.source_text


def test_method_heading_granularity_keeps_titles_and_drops_statement_details() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {
                    "title": "系统安装及施工",
                    "start_page": 30,
                    "end_page": 40,
                    "level": 1,
                }
            ]
        }
    )
    pages = [
        PDFPageText(30, "5.3.4、摄像机的安装\n摄像机安装应牢固。"),
        PDFPageText(
            31,
            "5.5、监控系统的调试\n"
            "5.5.1.1、电视监控系统调试应在设备安装完成后进行。\n"
            "5.5.3、摄像机的调试\n调整摄像机角度。",
        ),
        PDFPageText(
            32,
            "6.2、报警系统的测试\n进行系统功能测试。\n"
            "6.2.2、报警主机的调试\n设置报警主机参数。",
        ),
        PDFPageText(
            33,
            "7.1.4、控制器的安装\n"
            "7.1.4.1、要安装在防风雨的地方\n"
            "7.1.4.2、控制柜安装固定螺栓并检查紧固情况",
        ),
        PDFPageText(34, "7.2.2、控制器的调试\n完成控制器参数校准。"),
        PDFPageText(35, "8.1.2.2、控制器的安装应保证设备水平。"),
        PDFPageText(36, "8.2、门禁系统的调试\n进行门禁联动调试。"),
    ]

    completed = DeterministicPostProcessingService().process(analysis, pages)
    names = {method.name for method in completed.main_methods}
    assert names == {
        "摄像机的安装",
        "监控系统的调试",
        "摄像机的调试",
        "报警系统的测试",
        "报警主机的调试",
        "控制器的安装",
        "控制器的调试",
        "门禁系统的调试",
    }
    monitor = next(
        method for method in completed.main_methods if method.name == "监控系统的调试"
    )
    assert "5.5.1.1、电视监控系统调试应在" in monitor.source_text
    assert "电视监控系统调试应在" not in names
    assert "要安装在防风雨的地方" not in names


def test_method_aliases_keep_exact_chapter_provenance() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {
                    "title": "系统安装及施工",
                    "start_page": 40,
                    "end_page": 60,
                    "level": 1,
                },
                {
                    "title": "停车场管理系统的调试",
                    "start_page": 53,
                    "end_page": 56,
                    "level": 2,
                },
                {
                    "title": "门禁系统的调试",
                    "start_page": 57,
                    "end_page": 60,
                    "level": 2,
                },
            ],
            "main_methods": [
                {
                    "name": "停车场管理系统调试",
                    "source_page": 49,
                    "source_text": "5.5.5、系统调试\n进行系统联调。",
                },
                {
                    "name": "门禁系统调试",
                    "source_page": 49,
                    "source_text": "5.5.5、系统调试\n进行系统联调。",
                },
            ],
        }
    )
    pages = [
        PDFPageText(49, "5.5.5、系统调试\n进行系统联调。"),
        PDFPageText(53, "7.2、停车场管理系统的调试\n完成停车场系统联调。"),
        PDFPageText(57, "8.2、门禁系统的调试\n完成门禁系统联动调试。"),
    ]

    located = SourceLocatorService().locate(analysis, pages)
    parking = [
        method
        for method in located.main_methods
        if DeterministicPostProcessingService.canonical_method_name(method.name)
        == DeterministicPostProcessingService.canonical_method_name(
            "停车场管理系统的调试"
        )
    ]
    access = [
        method
        for method in located.main_methods
        if DeterministicPostProcessingService.canonical_method_name(method.name)
        == DeterministicPostProcessingService.canonical_method_name("门禁系统的调试")
    ]
    assert [(method.name, method.source_page) for method in parking] == [
        ("停车场管理系统的调试", 53)
    ]
    assert parking[0].source_text.startswith("7.2、停车场管理系统的调试")
    assert [(method.name, method.source_page) for method in access] == [
        ("门禁系统的调试", 57)
    ]
    assert access[0].source_text.startswith("8.2、门禁系统的调试")


def test_final_method_cleanup_removes_requirements_and_test_metadata() -> None:
    invalid_names = [
        "按安装图纸进行安装",
        "施工所需的仪器设备、工具及施工材料应提前准备就绪",
        "系统的防雷接地安装，应严格按设计要求施工",
        "锁具的安装应按产品更新换代说明书要求安装",
        "按系统设计功能对系统功能进行逐项调试",
        "认证测试标准",
        "认证测试模型",
        "认证测试参数",
        "测试工具：FLUKE-4000",
        "施工前的准备工作",
    ]
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {
                    "title": "系统安装及施工",
                    "start_page": 30,
                    "end_page": 40,
                    "level": 1,
                },
                {
                    "title": "报警系统测试",
                    "start_page": 35,
                    "end_page": 36,
                    "level": 2,
                },
            ],
            "main_methods": [{"name": name} for name in invalid_names]
            + [{"name": "报警系统测试"}],
        }
    )
    pages = [
        PDFPageText(
            30,
            "4.1、系统的安装调试\n"
            "4.1.1、认证测试标准\n"
            "4.1.2、认证测试模型\n"
            "4.1.3、认证测试参数\n"
            "4.1.4、测试工具：FLUKE-4000",
        ),
        PDFPageText(
            31,
            "4.1.5.1、按安装图纸进行安装\n"
            "4.1.5.2、施工所需的仪器设备、工具及施工材料应提前准备就绪\n"
            "4.1.5.3、系统的防雷接地安装，应严格按设计要求施工\n"
            "4.1.5.4、锁具的安装应按产品更新换代说明书要求安装\n"
            "4.1.5.5、按系统设计功能对系统功能进行逐项调试",
        ),
        PDFPageText(32, "4.1.6、施工前的准备工作\n核对施工条件。"),
        PDFPageText(35, "4.2、报警系统测试\n逐项完成报警功能测试。"),
    ]

    located = SourceLocatorService().locate(analysis, pages)
    names = {method.name for method in located.main_methods}
    assert names == {"系统的安装调试", "报警系统测试"}
    parent = next(method for method in located.main_methods if method.name == "系统的安装调试")
    assert "认证测试标准" in parent.source_text
    assert all(name not in names for name in invalid_names)


def test_table_project_metadata_stops_at_next_explicit_label() -> None:
    pages = [
        PDFPageText(
            4,
            "工程名称 | 新联路（杉木冲路－春明路）道路工程"
            "工程地址 | 新联路（杉木冲路至春明路段）"
            "工程区域位置 | 长沙市雨花区\n"
            "本工程位于长沙市雨花区。",
        )
    ]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.project_name.value == "新联路（杉木冲路－春明路）道路工程"
    assert project.project_location.value == "新联路（杉木冲路至春明路段）"
    assert project.project_name.source_page == 4
    assert project.project_location.source_page == 4


def test_local_compilation_basis_is_not_overwritten_by_empty_llm_result() -> None:
    pages = [
        PDFPageText(
            18,
            "2、编制依据\n"
            "1、施工合同\n"
            "中航勘察设计研究院有限公司提供的\n"
            "岩土工程详细勘察报告\n"
            "湖南大学设计有限公司提供的\n"
            "设计图纸及其他设计文件\n"
            "工程规范标准",
        )
    ]
    chapters = [
        ChapterInfo(title="编制依据", start_page=18, end_page=18, level=1)
    ]

    local = DocumentAnalysisService._extract_compilation_basis(pages, chapters)
    merged = DocumentAnalysisService._merge_compilation_basis(local, [])
    values = [item.value if not isinstance(item, str) else item for item in merged]
    assert "施工合同" in values
    assert any("岩土工程详细勘察报告" in value for value in values)
    assert any("设计图纸及其他设计文件" in value for value in values)
    assert "工程规范标准" in values
    assert all(not isinstance(item, str) and item.source_page == 18 for item in merged)


def test_engineering_category_prefers_project_name_provenance_over_standard() -> None:
    pages = [
        PDFPageText(
            7,
            "工程名称 | 新联路（杉木冲路－春明路）道路工程\n"
            "工程地址 | 新联路（杉木冲路至春明路段）",
        ),
        PDFPageText(19, "《城镇道路工程施工与质量验收规范》（CJJ1-2008）"),
    ]
    metadata = MetadataExtractionService()
    project, _ = metadata.extract(pages)
    weak_category = DocumentAnalysis.model_validate(
        {
            "engineering_category": {
                "value": "道路工程",
                "source_page": 19,
                "source_text": "《城镇道路工程施工与质量验收规范》",
            }
        }
    ).engineering_category
    category = metadata.validate_engineering_category(
        pages, project, weak_category
    )
    analysis = DocumentAnalysis.model_validate(
        {
            "engineering_category": category.model_dump(),
            "project": project.model_dump(),
        }
    )

    located = SourceLocatorService().locate(analysis, pages)
    assert located.engineering_category.value == "道路工程"
    assert located.engineering_category.source_page == 7
    assert "新联路（杉木冲路－春明路）道路工程" in (
        located.engineering_category.source_text or ""
    )


def test_contact_people_and_mobile_numbers_are_not_organizations() -> None:
    pages = [
        PDFPageText(
            1,
            "道路工程专项施工方案\n中建海嘉建设工程有限公司",
        ),
        PDFPageText(
            70,
            "业主、监理等单位抢险人员联系方式一览表\n"
            "建设单位 | 谢翔 | 18373197756\n"
            "监理单位 | 张杰 | 18684818604",
        ),
    ]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.construction_unit is None
    assert project.supervision_unit is None
    assert project.construction_company.value == "中建海嘉建设工程有限公司"
    assert project.construction_company.source_page == 1

    llm_project = DocumentAnalysis.model_validate(
        {
            "project": {
                "construction_unit": {
                    "value": "谢翔18373197756",
                    "source_page": 70,
                    "source_text": "建设单位 | 谢翔 | 18373197756",
                },
                "supervision_unit": {
                    "value": "张杰18684818604",
                    "source_page": 70,
                    "source_text": "监理单位 | 张杰 | 18684818604",
                },
            }
        }
    ).project
    validated = MetadataExtractionService().validate_organizations(pages, llm_project)
    assert validated.construction_unit is None
    assert validated.supervision_unit is None


def test_scoped_construction_schedule_beats_overall_project_duration() -> None:
    pages = [
        PDFPageText(4, "项目总工期为365天。"),
        PDFPageText(20, "专项施工安排\n施工工期100天。"),
        PDFPageText(
            25,
            "施工进度计划\n"
            "计划从2025年4月25日开始，\n"
            "2025年8月15日完成土方开挖及支护施工，\n"
            "共需112天。",
        ),
    ]

    _, schedule = MetadataExtractionService().extract(pages)
    assert schedule.plan_start_date.value == "2025-04-25"
    assert schedule.plan_completion_date.value == "2025-08-15"
    assert schedule.duration_days.value == 112
    assert schedule.plan_start_date.source_page == 25
    assert schedule.plan_completion_date.source_page == 25
    assert schedule.duration_days.source_page == 25


@pytest.mark.parametrize(
    "heading",
    ["4、施工工艺技术", "4. 施工工艺技术", "第四章 施工工艺技术"],
)
def test_construction_technology_level_one_heading_formats(heading: str) -> None:
    chapters = ChapterDetectionService().detect([PDFPageText(15, heading)])
    assert [(chapter.title, chapter.level) for chapter in chapters] == [
        ("施工工艺技术", 1)
    ]


def test_level_one_construction_technology_closes_previous_plan_section() -> None:
    pages = [
        PDFPageText(10, "3、施工计划"),
        PDFPageText(15, "4、施工工艺技术"),
        PDFPageText(16, "4.1、施工平面布置"),
        PDFPageText(17, "4.2、施工准备"),
        PDFPageText(18, "4.3、边坡施工工艺流程"),
        PDFPageText(46, "5、施工安全保障措施"),
    ]

    chapters = ChapterDetectionService().detect(pages)
    by_title = {chapter.title: chapter for chapter in chapters}
    assert by_title["施工计划"].level == 1
    assert by_title["施工计划"].end_page == 15
    assert by_title["施工工艺技术"].start_page == 15
    assert by_title["施工工艺技术"].level == 1
    assert by_title["施工工艺技术"].end_page == 46
    assert by_title["施工平面布置"].level == 2
    assert by_title["施工准备"].level == 2
    assert by_title["边坡施工工艺流程"].level == 2


def test_construction_technology_titles_form_deterministic_method_skeleton() -> None:
    titles = [
        "测量放样",
        "截水沟施工",
        "土方开挖",
        "路堑挡墙施工",
        "平台沟及盖板沟施工",
        "压密注浆施工",
        "泄水孔安装",
        "锚杆、挂网客土喷播",
    ]
    chapters = [
        {"title": "施工工艺技术", "start_page": 20, "end_page": 35, "level": 1}
    ]
    chapters.extend(
        {
            "title": title,
            "start_page": 21 + index,
            "end_page": 21 + index,
            "level": 3,
        }
        for index, title in enumerate(titles)
    )
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": chapters,
            "main_methods": [
                {"name": "压密注浆"},
                {"name": "分层分段开挖"},
            ],
        }
    )
    pages = [
        PDFPageText(
            21 + index,
            f"4.3.{index + 1}、{title}\n按施工方案完成本工序。",
        )
        for index, title in enumerate(titles)
    ]

    completed = DeterministicPostProcessingService().process(analysis, pages)
    names = {method.name for method in completed.main_methods}
    assert set(titles).issubset(names)
    assert "测量放样" in names
    assert "土方开挖" in names
    grouting = [
        method
        for method in completed.main_methods
        if DeterministicPostProcessingService.canonical_method_name(method.name)
        == DeterministicPostProcessingService.canonical_method_name("压密注浆")
    ]
    assert [(method.name, method.source_page) for method in grouting] == [
        ("压密注浆施工", 26)
    ]


def test_standard_code_is_preserved_even_when_source_may_contain_typo() -> None:
    text = "混凝土结构工程施工质量验收规范（GB50204-015）"
    extracted = PDFExtractionResult(
        page_count=1,
        char_count=len(text),
        text=text,
        pages=[PDFPageText(8, text)],
    )

    standards = DocumentAnalysisService(FakeLLMService())._extract_standards(extracted)
    assert [(item.name, item.code) for item in standards] == [
        ("混凝土结构工程施工质量验收规范", "GB50204-015")
    ]


def test_safety_and_emergency_presence_are_deterministic_from_chapters() -> None:
    chapters = [
        ChapterInfo(title="施工安全保障措施", start_page=40, end_page=50, level=1),
        ChapterInfo(title="危险源关键控制措施", start_page=45, end_page=46, level=2),
        ChapterInfo(title="应急处置措施", start_page=51, end_page=53, level=1),
        ChapterInfo(title="应急组织架构及职责分工", start_page=52, end_page=52, level=2),
    ]

    detected = ChapterDetectionService.section_presence(chapters, [])
    assert detected.safety is True
    assert detected.emergency is True

    merged = DocumentAnalysisService._merge_section_presence(
        SectionPresence(quality=False, safety=False, environment=False, emergency=False),
        detected,
    )
    assert merged.safety is True
    assert merged.emergency is True


@pytest.mark.parametrize("title", ["扬尘治理措施", "施工防噪音措施", "绿色施工"])
def test_environment_presence_supports_common_environment_headings(title: str) -> None:
    detected = ChapterDetectionService.section_presence(
        [ChapterInfo(title=title, start_page=30, end_page=31, level=1)], []
    )
    assert detected.environment is True


def test_explicit_project_address_beats_earlier_narrative_location() -> None:
    pages = [
        PDFPageText(4, "工程概况\n本工程位于湖南省长沙市天心区黑石铺街道。"),
        PDFPageText(
            7,
            "工程名称\n新联路（杉木冲路－春明路）道路工程\n"
            "工程地址\n新联路（杉木冲路至春明路段）\n"
            "工程区域位置\n长沙市天心区",
        ),
    ]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.project_location.value == "新联路（杉木冲路至春明路段）"
    assert project.project_location.source_page == 7

    located = SourceLocatorService().locate(
        DocumentAnalysis.model_validate({"project": project.model_dump()}), pages
    )
    assert located.project.project_location.value == "新联路（杉木冲路至春明路段）"
    assert located.project.project_location.source_page == 7
    assert "工程地址" in (located.project.project_location.source_text or "")
    assert "新联路（杉木冲路至春明路段）" in (
        located.project.project_location.source_text or ""
    )


def test_quantity_table_materials_are_removed_from_final_work_items() -> None:
    expected_work = [
        "土石方开挖",
        "截水沟",
        "平台及平台沟",
        "急流槽",
        "路堑矮墙",
        "矩形盖板沟",
        "坡面压密注浆",
        "三维网喷播植草",
        "泄水孔",
    ]
    material_names = ["路基箱板", "碎石", "彩条布"]
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [
                {
                    "title": "施工工艺技术",
                    "start_page": 20,
                    "end_page": 30,
                    "level": 1,
                }
            ],
            "main_work_items": [
                {
                    "value": name,
                    "source_page": 10,
                    "source_text": f"{name} | 项 | 1",
                }
                for name in [*expected_work, *material_names]
            ],
            "main_materials": [{"name": name} for name in material_names],
            "main_methods": [
                {
                    "name": "土方开挖",
                    "source_page": 20,
                    "source_text": "4.3.1、土方开挖\n分层进行开挖。",
                },
                {
                    "name": "截水沟施工",
                    "source_page": 21,
                    "source_text": "4.3.2、截水沟施工\n完成沟槽施工。",
                },
                {
                    "name": "压密注浆施工",
                    "source_page": 22,
                    "source_text": "4.3.3、压密注浆施工\n按顺序注浆。",
                },
            ],
        }
    )
    pages = [
        PDFPageText(
            10,
            "主要工程数量表\n项目 单位 数量\n"
            + "\n".join(f"{name} | 项 | 1" for name in [*expected_work, *material_names]),
        ),
        PDFPageText(20, "4.3.1、土方开挖\n分层进行开挖。"),
        PDFPageText(21, "4.3.2、截水沟施工\n完成沟槽施工。"),
        PDFPageText(22, "4.3.3、压密注浆施工\n按顺序注浆。"),
    ]

    cleaned = DeterministicPostProcessingService().final_cleanup(analysis, pages)
    assert [item.value for item in cleaned.main_work_items] == expected_work
    assert [material.name for material in cleaned.main_materials] == material_names


class StaticPDFService:
    def __init__(self, pages: list[PDFPageText]) -> None:
        text = "\n".join(page.text for page in pages)
        self.extracted = PDFExtractionResult(
            page_count=len(pages),
            char_count=len(text),
            text=text,
            pages=pages,
        )

    def extract_text(self, _pdf_path: Path) -> PDFExtractionResult:
        return self.extracted


class PresenceLLMService:
    def __init__(self, section_presence: dict[str, bool] | None) -> None:
        self.section_presence = section_presence

    def analyze_document(self, _pages: list[PDFPageText]) -> DocumentAnalysis:
        payload = {}
        if self.section_presence is not None:
            payload["section_presence"] = self.section_presence
        return DocumentAnalysis.model_validate(payload)


def _presence_pipeline_pages() -> list[PDFPageText]:
    return [
        PDFPageText(1, "1.1、工程质量目标\n1.2、验收标准"),
        PDFPageText(2, "2.1、施工安全保障措施\n2.2、危险源关键控制措施"),
        PDFPageText(3, "3.1、扬尘治理措施\n3.2、防噪音措施"),
        PDFPageText(4, "4.1、应急组织架构及职责分工\n4.2、应急响应"),
    ]


@pytest.mark.parametrize(
    "llm_presence",
    [
        {"quality": False, "safety": False, "environment": False, "emergency": False},
        None,
    ],
)
def test_final_response_presence_ors_llm_with_deterministic_evidence(
    llm_presence: dict[str, bool] | None,
) -> None:
    pages = _presence_pipeline_pages()
    analysis = DocumentAnalysisService(
        PresenceLLMService(llm_presence), pdf_service=StaticPDFService(pages)
    ).analyze(Path("presence-test.pdf"))

    response = DocumentAnalysisResponse(
        document_id="00000000-0000-0000-0000-000000000001",
        analysis=analysis,
    )
    dumped = response.model_dump(mode="json")
    restored = DocumentAnalysisResponse.model_validate(dumped)
    expected = {
        "quality": True,
        "safety": True,
        "environment": True,
        "emergency": True,
    }
    assert dumped["analysis"]["section_presence"] == expected
    assert restored.analysis.section_presence.model_dump() == expected


def test_final_response_presence_is_false_without_llm_or_local_evidence() -> None:
    pages = [PDFPageText(1, "1.1、工程概况\n项目基本情况")]
    analysis = DocumentAnalysisService(
        PresenceLLMService(None), pdf_service=StaticPDFService(pages)
    ).analyze(Path("no-presence-evidence.pdf"))

    assert analysis.section_presence == SectionPresence()


def test_presence_is_stable_across_different_llm_results() -> None:
    pages = _presence_pipeline_pages()
    llm_results = [
        {"quality": False, "safety": False, "environment": False, "emergency": False},
        {"quality": True, "safety": False, "environment": True, "emergency": False},
    ]
    final_results = []
    for llm_presence in llm_results:
        analysis = DocumentAnalysisService(
            PresenceLLMService(llm_presence), pdf_service=StaticPDFService(pages)
        ).analyze(Path("stable-presence.pdf"))
        final_results.append(analysis.section_presence.model_dump())

    assert final_results[0] == final_results[1] == {
        "quality": True,
        "safety": True,
        "environment": True,
        "emergency": True,
    }


def test_analyze_endpoint_persists_and_returns_final_deterministic_presence(
    temporary_data_directories, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, _ = temporary_data_directories
    pages = _presence_pipeline_pages()
    text = "\n".join(page.text for page in pages)
    extracted = PDFExtractionResult(
        page_count=len(pages),
        char_count=len(text),
        text=text,
        pages=pages,
    )
    monkeypatch.setattr(PDFService, "extract_text", lambda _self, _path: extracted)

    llm = PresenceLLMService(
        {"quality": False, "safety": False, "environment": False, "emergency": False}
    )
    app.dependency_overrides[get_llm_service] = lambda: llm
    document_id = "00000000-0000-0000-0000-000000000002"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    (settings.upload_dir / f"{document_id}.pdf").write_bytes(b"test-pdf-placeholder")

    forced = client.post(
        f"/api/v1/documents/{document_id}/analyze?force=true"
    )
    assert forced.status_code == 200
    assert forced.json()["cached"] is False
    expected = {
        "quality": True,
        "safety": True,
        "environment": True,
        "emergency": True,
    }
    assert forced.json()["analysis"]["section_presence"] == expected

    cache_payload = json.loads(
        (settings.analysis_dir / f"{document_id}.json").read_text(encoding="utf-8")
    )
    assert cache_payload["section_presence"] == expected

    cached = client.post(f"/api/v1/documents/{document_id}/analyze")
    assert cached.status_code == 200
    assert cached.json()["cached"] is True
    assert cached.json()["analysis"]["section_presence"] == expected


def test_two_column_metadata_fields_stop_at_each_following_label() -> None:
    pages = [
        PDFPageText(
            7,
            "工程名称|某产业园配套项目建设地点|某市新区产业大道\n"
            "建设单位|某城市投资有限公司施工单位|某建设工程局有限公司\n"
            "监理单位|某项目管理有限公司勘察单位|某勘察设计研究院有限公司\n"
            "设计单位|某工程咨询有限公司建设规模|总建筑面积50000平方米",
        )
    ]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.project_name.value == "某产业园配套项目"
    assert project.project_location.value == "某市新区产业大道"
    assert project.construction_unit.value == "某城市投资有限公司"
    assert project.construction_company.value == "某建设工程局有限公司"
    assert project.supervision_unit.value == "某项目管理有限公司"
    assert project.design_unit.value == "某工程咨询有限公司"


def test_target_schedule_range_and_duration_are_one_evidence_group() -> None:
    pages = [
        PDFPageText(65, "材料编号65，进场后分类堆放。"),
        PDFPageText(
            22,
            "根据施工总进度计划安排及支护施工安排，\n"
            "目标工期为2025年10月10日-2025年12月14日，\n"
            "施工周期65天。",
        ),
    ]

    _, schedule = MetadataExtractionService().extract(pages)
    assert schedule.plan_start_date.value == "2025-10-10"
    assert schedule.plan_completion_date.value == "2025-12-14"
    assert schedule.duration_days.value == 65
    assert {
        schedule.plan_start_date.source_page,
        schedule.plan_completion_date.source_page,
        schedule.duration_days.source_page,
    } == {22}
    assert all(
        "目标工期" in (item.source_text or "") and "施工周期65天" in (item.source_text or "")
        for item in (
            schedule.plan_start_date,
            schedule.plan_completion_date,
            schedule.duration_days,
        )
    )


def test_body_decimal_heading_survives_mismatched_contents_numbering() -> None:
    pages = [
        PDFPageText(
            2,
            "目录\n5.3 钢板桩施工方案……30\n5.4 旋挖灌注桩施工方案……36",
        ),
        PDFPageText(30, "5.3、钢板桩施工方案\n钢板桩施工要求。"),
        PDFPageText(33, "5.4、放坡支护施工方案\n放坡施工要求。"),
        PDFPageText(36, "5.5、旋挖灌注桩施工方案\n灌注桩施工要求。"),
        PDFPageText(40, "5.6、基坑开挖方案\n分层开挖。"),
    ]

    chapters = ChapterDetectionService().detect(pages)
    by_title = {chapter.title: chapter for chapter in chapters}
    assert by_title["钢板桩施工方案"].start_page == 30
    assert by_title["放坡支护施工方案"].start_page == 33
    assert by_title["旋挖灌注桩施工方案"].start_page == 36
    assert by_title["旋挖灌注桩施工方案"].level == 2
    assert by_title["基坑开挖方案"].start_page == 40


def test_construction_chapters_fill_severely_incomplete_work_items() -> None:
    chapter_titles = [
        "钢板桩施工方案",
        "放坡支护施工方案",
        "旋挖灌注桩施工方案",
        "基坑开挖方案",
        "基坑维护措施",
        "基坑回填",
        "基坑排水及抗渗水处理",
    ]
    chapters = [
        {"title": "施工工艺技术", "start_page": 30, "end_page": 50, "level": 1}
    ]
    chapters.extend(
        {
            "title": title,
            "start_page": 31 + index,
            "end_page": 32 + index,
            "level": 2,
        }
        for index, title in enumerate(chapter_titles)
    )
    pages = [PDFPageText(30, "施工工艺技术\n主要支护形式包括桩支护和放坡支护，基坑土方分层开挖。")]
    pages.extend(
        PDFPageText(
            31 + index,
            f"5.{index + 3}、{title}\n按照专项方案完成施工。",
        )
        for index, title in enumerate(chapter_titles)
    )
    analysis = DocumentAnalysis.model_validate(
        {"chapters": chapters, "main_work_items": ["排水系统"]}
    )

    completed = DeterministicPostProcessingService().process(analysis, pages)
    items = {item.value: item for item in completed.main_work_items}
    expected = {
        "钢板桩支护",
        "放坡支护",
        "旋挖灌注桩支护",
        "基坑土方开挖",
        "基坑回填",
        "基坑排水及抗渗水处理",
    }
    assert expected.issubset(items)
    assert "基坑维护措施" not in items
    assert all(items[name].source_page is not None for name in expected)
    assert all(items[name].source_text for name in expected)


def test_specific_repeated_pdf_title_beats_generic_llm_type_without_guessing_category() -> None:
    pages = [
        PDFPageText(1, "某项目支护及开挖专项施工方案\n编制单位：某建设有限公司"),
        PDFPageText(2, "深基坑支护及开挖专项施工方案\n1.1、验收标准"),
        PDFPageText(
            3,
            "深基坑支护及开挖专项施工方案\n2.1、施工安全保障措施\n"
            "2.2、扬尘治理措施\n2.3、应急组织架构及职责分工",
        ),
    ]

    class GenericDocumentTypeLLM:
        def analyze_document(self, _pages: list[PDFPageText]) -> DocumentAnalysis:
            return DocumentAnalysis.model_validate(
                {
                    "document_type": "专项施工方案",
                    "engineering_category": None,
                    "section_presence": {
                        "quality": False,
                        "safety": False,
                        "environment": False,
                        "emergency": False,
                    },
                }
            )

    analysis = DocumentAnalysisService(
        GenericDocumentTypeLLM(), pdf_service=StaticPDFService(pages)
    ).analyze(Path("specific-title.pdf"))
    assert analysis.document_type.value == "深基坑支护及开挖专项施工方案"
    assert analysis.document_type.source_text == "深基坑支护及开挖专项施工方案"
    assert analysis.engineering_category is None
    assert analysis.section_presence.model_dump() == {
        "quality": True,
        "safety": True,
        "environment": True,
        "emergency": True,
    }


def test_multi_row_schedule_table_uses_earliest_start_and_latest_completion() -> None:
    pages = [PDFPageText(12, "施工进度计划表\n任务名称 | 工期 | 开始时间 | 完成时间\n基础施工 | 8天 | 2025-03-10 | 2025-03-17\n主体安装 | 12天 | 2025-03-20 | 2025-03-31")]

    _, schedule = MetadataExtractionService().extract(pages)
    assert schedule.plan_start_date.value == "2025-03-10"
    assert schedule.plan_completion_date.value == "2025-03-31"
    assert schedule.duration_days is None
    assert schedule.plan_start_date.source_text.startswith("基础施工")
    assert schedule.plan_completion_date.source_text.startswith("主体安装")


def test_summary_schedule_row_takes_priority_over_child_rows() -> None:
    pages = [PDFPageText(8, "专项施工进度计划表\n总体施工 | 45天 | 2025/04/01 | 2025/05/15\n准备作业 | 5天 | 2025/03/20 | 2025/03/24\n收尾作业 | 3天 | 2025/05/20 | 2025/05/22")]

    _, schedule = MetadataExtractionService().extract(pages)
    assert schedule.plan_start_date.value == "2025-04-01"
    assert schedule.plan_completion_date.value == "2025-05-15"
    assert schedule.duration_days.value == 45
    assert schedule.duration_days.source_text.startswith("总体施工")


def test_schedule_without_explicit_total_duration_keeps_duration_null() -> None:
    pages = [PDFPageText(9, "施工进度计划表\n工作面一 | 4天 | 2025年6月1日 | 2025年6月4日\n工作面二 | 7天 | 2025年6月8日 | 2025年6月14日")]

    _, schedule = MetadataExtractionService().extract(pages)
    assert schedule.plan_start_date.value == "2025-06-01"
    assert schedule.plan_completion_date.value == "2025-06-14"
    assert schedule.duration_days is None


def test_discontinuous_schedule_work_faces_are_not_summed() -> None:
    pages = [PDFPageText(10, "施工进度计划表\n区域一施工 | 10天 | 2025-07-01 | 2025-07-10\n区域二施工 | 10天 | 2025-08-01 | 2025-08-10")]

    _, schedule = MetadataExtractionService().extract(pages)
    assert schedule.plan_start_date.value == "2025-07-01"
    assert schedule.plan_completion_date.value == "2025-08-10"
    assert schedule.duration_days is None


@pytest.mark.parametrize("interference", ["编制日期", "专家论证日期"])
def test_schedule_table_rejects_non_construction_date_interference(interference: str) -> None:
    pages = [PDFPageText(11, f"施工进度计划表\n{interference} | 2024-01-01 | 2024-01-02\n安装作业 | 6天 | 2025-09-01 | 2025-09-06")]

    _, schedule = MetadataExtractionService().extract(pages)
    assert schedule.plan_start_date.value == "2025-09-01"
    assert schedule.plan_completion_date.value == "2025-09-06"


def test_metadata_boundaries_include_participants_and_building_fields() -> None:
    pages = [PDFPageText(4, "工程地点：某市新区 建筑面积：32000平方米\n施工单位：某建设有限公司 监控单位：某监测有限公司\n建设单位：某投资有限公司 勘察、设计单位：某设计研究院有限公司")]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.project_location.value == "某市新区"
    assert project.construction_company.value == "某建设有限公司"
    assert project.construction_unit.value == "某投资有限公司"
    assert project.design_unit.value == "某设计研究院有限公司"


def test_multiline_company_name_is_preserved_but_following_section_stops_value() -> None:
    pages = [PDFPageText(5, "施工单位：某大型建设\n集团有限公司\n（二）结构概况\n主体结构说明。")]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.construction_company.value == "某大型建设集团有限公司"
    assert "结构概况" not in project.construction_company.value


def test_explicit_project_location_label_is_high_confidence() -> None:
    pages = [PDFPageText(3, "建设地点：某省某市产业大道北侧\n建筑高度：88米")]

    project, _ = MetadataExtractionService().extract(pages)
    assert project.project_location.value == "某省某市产业大道北侧"
    assert project.project_location.source_page == 3


@pytest.mark.parametrize(
    "text",
    [
        "区域地质构造上位于断裂带东缘，岩层倾角较大。",
        "梁体位置位于第三跨中部，受力位置需要加强。",
    ],
)
def test_project_location_rejects_geological_and_component_positions(text: str) -> None:
    project, _ = MetadataExtractionService().extract([PDFPageText(16, text)])
    assert project.project_location is None


def test_cover_document_type_occurrence_has_priority_over_late_header() -> None:
    pages = [
        PDFPageText(1, "悬挑构件安装工程专项施工方案\n某建设有限公司"),
        PDFPageText(96, "悬挑构件安装工程专项施工方案\n附录"),
    ]

    result = MetadataExtractionService().extract_document_type(pages)
    assert result.value == "悬挑构件安装工程专项施工方案"
    assert result.source_page == 1
    assert result.source_text == "悬挑构件安装工程专项施工方案"


def test_document_type_removes_project_name_and_preserves_specific_qualifier() -> None:
    pages = [PDFPageText(1, "工程名称：某产业园建设项目\n某产业园建设项目\n悬挑构件安装工程专项施工方案")]
    service = MetadataExtractionService()
    project, _ = service.extract(pages)

    result = service.extract_document_type(pages, project.project_name)
    assert result.value == "悬挑构件安装工程专项施工方案"
    assert result.source_page == 1


def test_document_type_rejects_contents_and_structural_section_titles() -> None:
    pages = [
        PDFPageText(1, "给排水管道施工方案\n某建设有限公司"),
        PDFPageText(2, "目录\n二、编制依据及主要施工方案........ 2"),
        PDFPageText(8, "二、编制依据及主要施工方案\n现行规范。"),
    ]

    result = MetadataExtractionService().extract_document_type(pages)
    assert result.value == "给排水管道施工方案"
    assert result.source_page == 1


def test_compilation_company_stops_before_compilation_date() -> None:
    project, _ = MetadataExtractionService().extract(
        [PDFPageText(1, "编制单位：某工程设备有限公司\n编制日期：2025年8月")]
    )
    assert project.construction_company.value == "某工程设备有限公司"


def test_cover_specific_title_beats_repeated_generic_title() -> None:
    pages = [
        PDFPageText(1, "某中心建设项目\n装配支撑体系专项施工方案"),
        PDFPageText(2, "某中心高大模板专项施工方案"),
        PDFPageText(20, "某中心高大模板专项施工方案"),
    ]
    service = MetadataExtractionService()
    project, _ = service.extract(pages)
    result = service.extract_document_type(pages, project.project_name)
    assert result.value == "装配支撑体系专项施工方案"
    assert result.source_page == 1


def test_document_type_preserves_numbered_structure_identifier() -> None:
    result = MetadataExtractionService().extract_document_type(
        [PDFPageText(1, "4#、5#、6#墩承台围堰专项施工方案")]
    )
    assert result.value == "4#、5#、6#墩承台围堰专项施工方案"


def test_document_type_removes_verified_project_and_contract_segment_prefix() -> None:
    pages = [
        PDFPageText(
            1,
            "工程名称：某通道跨线工程\n"
            "某通道跨线工程施工№.A1标段\n"
            "4#、5#构筑物专项施工方案",
        )
    ]
    service = MetadataExtractionService()
    project, _ = service.extract(pages)
    result = service.extract_document_type(pages, project.project_name)
    assert result.value == "4#、5#构筑物专项施工方案"


def test_document_type_removes_verified_project_suffix_split_across_cover_lines() -> None:
    pages = [
        PDFPageText(
            1,
            "工程名称：某地区跨线桥梁工程\n"
            "跨线桥梁工程施工№.A1标段\n"
            "4#构筑物专项施工方案",
        )
    ]
    service = MetadataExtractionService()
    project, _ = service.extract(pages)
    result = service.extract_document_type(pages, project.project_name)
    assert result.value == "4#构筑物专项施工方案"


@pytest.mark.parametrize(
    ("project_name", "cover_title", "expected"),
    [
        (
            "某住宅工程总承包（EPC+F）项目",
            "某住宅工程总承包（EPC+F）项目给排、水管道施工方案",
            "给排、水管道施工方案",
        ),
        (
            "某住宅工程总承包（EPC+F）项目安装工程",
            "工程总承包（EPC+F）项目给排、水管道施工方案",
            "给排、水管道施工方案",
        ),
        (
            "某更新建设项目",
            "某更新建设项目悬挑脚手架工程专项施工方案",
            "悬挑脚手架工程专项施工方案",
        ),
        (
            "某跨线工程",
            "某跨线工程QL7标段4#、5#、6#墩承台围堰专项施工方案",
            "4#、5#、6#墩承台围堰专项施工方案",
        ),
    ],
)
def test_generic_document_title_decomposition_preserves_specific_qualifier(
    project_name: str, cover_title: str, expected: str
) -> None:
    pages = [PDFPageText(1, f"工程名称：{project_name}\n{cover_title}")]
    service = MetadataExtractionService()
    project, _ = service.extract(pages)
    result = service.extract_document_type(pages, project.project_name)
    assert result.value == expected
    assert result.source_page == 1


def test_main_method_semantic_gate_rejects_non_method_scope_classes() -> None:
    pages = [
        PDFPageText(4, "1 编制依据及主要施工方案\n1.1 施工图纸及合同"),
        PDFPageText(5, "2 工程概况\n2.1 安装工程系统简介"),
        PDFPageText(6, "3 材料与设备计划\n3.1 安装工程材料选型及连接方式"),
        PDFPageText(10, "4 施工工艺技术\n4.1 套管的设置\n4.2 套管制作"),
    ]
    chapters = ChapterDetectionService().detect(pages)
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [chapter.model_dump() for chapter in chapters],
            "main_methods": [
                {"name": "施工图纸及合同", "source_page": 4, "source_text": "1.1 施工图纸及合同"},
                {"name": "安装工程系统简介", "source_page": 5, "source_text": "2.1 安装工程系统简介"},
                {"name": "安装工程材料选型及连接方式", "source_page": 6, "source_text": "3.1 安装工程材料选型及连接方式"},
            ],
        }
    )

    located = SourceLocatorService().locate(analysis, pages)
    assert [method.name for method in located.main_methods] == ["套管的设置", "套管制作"]


@pytest.mark.parametrize(
    "raw_name",
    [
        "（2）、《建筑给水排水及采暖工程施工质量验收规范",
        "(2) 建筑给水排水及采暖工程施工质量验收规范》；",
        "② “建筑给水排水及采暖工程施工质量验收规范”",
    ],
)
def test_standard_name_normalization_removes_list_and_book_title_markers(
    raw_name: str,
) -> None:
    assert DocumentAnalysisService._clean_standard_name(raw_name) == (
        "建筑给水排水及采暖工程施工质量验收规范"
    )


def test_mixed_enterprise_standard_code_is_tokenized_generically() -> None:
    text = "《建筑给水排水工程施工技术标准》ZJQ08-SGJB242-2017"
    extracted = PDFExtractionResult(1, len(text), text, [PDFPageText(6, text)])
    standards = DocumentAnalysisService(FakeLLMService())._extract_standards(extracted)
    assert [(item.name, item.code) for item in standards] == [
        ("建筑给水排水工程施工技术标准", "ZJQ08-SGJB242-2017")
    ]


@pytest.mark.parametrize(
    "value",
    ["550E8400-E29B-41D4-A716-446655440000", "2025-08-19", "Q355B"],
)
def test_standard_code_validator_rejects_uuid_date_and_material_model(value: str) -> None:
    assert DocumentAnalysisService._is_standard_code(value) is False


def test_single_numbered_process_titles_are_methods_inside_explicit_process_region() -> None:
    pages = [
        PDFPageText(9, "四、施工工艺技术\n（一）方案设计\n技术参数"),
        PDFPageText(14, "（三）构件工艺流程与施工方法\n1、构件定位安装"),
        PDFPageText(15, "2、可调节拉杆卸荷"),
        PDFPageText(16, "3、立杆设置\n4、纵、横向水平杆"),
        PDFPageText(17, "5、剪刀撑和横向斜撑设置\n6、脚手板铺设"),
    ]
    chapters = ChapterDetectionService().detect(pages)
    analysis = DocumentAnalysis.model_validate(
        {"chapters": [chapter.model_dump() for chapter in chapters]}
    )
    completed = DeterministicPostProcessingService().process(analysis, pages)
    assert [method.name for method in completed.main_methods] == [
        "构件定位安装",
        "可调节拉杆卸荷",
        "立杆设置",
        "纵、横向水平杆",
        "剪刀撑和横向斜撑设置",
        "脚手板铺设",
    ]


def test_decimal_measurement_without_heading_separator_is_not_a_method() -> None:
    pages = [PDFPageText(6, "3 施工方法\n0.30m，内立杆与墙体之间保持间距。")]
    chapters = ChapterDetectionService().detect(pages)
    analysis = DocumentAnalysis.model_validate(
        {"chapters": [chapter.model_dump() for chapter in chapters]}
    )
    completed = DeterministicPostProcessingService().process(analysis, pages)
    assert completed.main_methods == []


@pytest.mark.parametrize(
    "heading",
    [
        "10.2 M=25.4kN.m",
        "10.3 300mm×500mm",
    ],
)
def test_formula_and_unit_expressions_are_rejected_as_headings(heading: str) -> None:
    chapters = ChapterDetectionService().detect(
        [PDFPageText(20, f"10 设计计算\n{heading}\n10.4 施工荷载验算")]
    )
    assert all(chapter.title not in {"M=25.4kN.m", "300mm×500mm"} for chapter in chapters)


def test_calculation_section_isolates_formula_and_standard_lines() -> None:
    pages = [
        PDFPageText(30, "第十章 计算书"),
        PDFPageText(31, "10.1 N=18.2kN\n10.2 GB/T 50000-2024\n10.3 梁模板计算书"),
    ]

    chapters = ChapterDetectionService().detect(pages)
    assert [chapter.title for chapter in chapters] == ["计算书"]


def test_embedded_subdocument_reopens_structured_extraction_after_calculation() -> None:
    pages = [
        PDFPageText(40, "第十章 计算书"),
        PDFPageText(41, "10.1 q=12.5kN/m"),
        PDFPageText(42, "构件安装工程专项施工方案"),
        PDFPageText(43, "1 编制依据\n设计图纸及现行标准"),
        PDFPageText(44, "2 工程概况\n本工程包含构件安装。"),
        PDFPageText(45, "3 施工工艺技术"),
        PDFPageText(46, "3.1 悬挑构件定位安装\n按测量控制线完成定位安装。"),
    ]

    chapters = ChapterDetectionService().detect(pages)
    titles = [chapter.title for chapter in chapters]
    assert "q=12.5kN/m" not in titles
    assert "编制依据" in titles
    assert "工程概况" in titles
    assert "施工工艺技术" in titles
    assert "悬挑构件定位安装" in titles

    analysis = DocumentAnalysis.model_validate(
        {"chapters": [chapter.model_dump() for chapter in chapters]}
    )
    completed = DeterministicPostProcessingService().process(analysis, pages)
    assert [method.name for method in completed.main_methods] == ["悬挑构件定位安装"]
    assert completed.main_methods[0].source_page == 46


def test_calculation_region_methods_are_excluded_but_embedded_methods_remain() -> None:
    pages = [
        PDFPageText(50, "8 计算书"),
        PDFPageText(51, "8.1 支架安装计算\nF=20kN"),
        PDFPageText(52, "构件工程专项施工方案"),
        PDFPageText(53, "1 编制依据"),
        PDFPageText(54, "2 工程概况"),
        PDFPageText(55, "3 施工方法"),
        PDFPageText(56, "3.1 构件吊装\n采用起重设备完成吊装。"),
    ]
    chapters = ChapterDetectionService().detect(pages)
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [chapter.model_dump() for chapter in chapters],
            "main_methods": [
                {"name": "支架安装计算", "source_page": 51, "source_text": "8.1 支架安装计算"}
            ],
        }
    )

    completed = DeterministicPostProcessingService().final_cleanup(
        DeterministicPostProcessingService().process(analysis, pages), pages
    )
    assert [method.name for method in completed.main_methods] == ["构件吊装"]


def test_main_work_items_require_action_or_valid_construction_semantics() -> None:
    analysis = DocumentAnalysis.model_validate(
        {
            "main_work_items": [
                "三层服务用房",
                "十六层共享大厅",
                "构件模板计算书",
                "悬挑构件定位安装",
                "剪刀撑",
            ]
        }
    )

    cleaned = DeterministicPostProcessingService().final_cleanup(analysis, [])
    assert [item.value for item in cleaned.main_work_items] == [
        "悬挑构件定位安装",
        "剪刀撑",
    ]


def test_calculation_title_is_excluded_from_work_items() -> None:
    analysis = DocumentAnalysis.model_validate(
        {"main_work_items": ["某构件模板计算书", "构件拆除"]}
    )
    cleaned = DeterministicPostProcessingService().final_cleanup(analysis, [])
    assert [item.value for item in cleaned.main_work_items] == ["构件拆除"]


@pytest.mark.parametrize(
    ("row", "expected_name", "expected_code"),
    [
        ("建筑结构荷载规范 | GB 50009-2012", "建筑结构荷载规范", "GB50009-2012"),
        ("JGJ 130-2011 | 建筑施工扣件式钢管脚手架安全技术规范", "建筑施工扣件式钢管脚手架安全技术规范", "JGJ130-2011"),
        ("《公路桥涵施工技术规范》 JTG/T 3650-2020", "公路桥涵施工技术规范", "JTG/T3650-2020"),
        ("地方工程施工技术规程 | DBJ 01-80-2003", "地方工程施工技术规程", "DBJ01-80-2003"),
    ],
)
def test_standards_table_pairs_name_and_code_in_either_order(
    row: str, expected_name: str, expected_code: str
) -> None:
    extracted = PDFExtractionResult(1, len(row), row, [PDFPageText(6, row)])
    standards = DocumentAnalysisService(FakeLLMService())._extract_standards(extracted)
    selected = next(item for item in standards if item.code == expected_code)
    assert selected.name == expected_name
    assert selected.source_text == row


def test_standards_table_pairs_wrapped_name_and_code_without_using_header() -> None:
    text = "国家及\n标准\n建筑施工安全检查标准\nJGJ 59-2011"
    extracted = PDFExtractionResult(1, len(text), text, [PDFPageText(7, text)])

    standards = DocumentAnalysisService(FakeLLMService())._extract_standards(extracted)
    selected = next(item for item in standards if item.code == "JGJ59-2011")
    assert selected.name == "建筑施工安全检查标准"
    assert selected.source_text == "建筑施工安全检查标准\nJGJ 59-2011"


def test_compilation_basis_is_restricted_to_explicit_basis_region() -> None:
    pages = [
        PDFPageText(3, "1 编制说明及依据\n1.1 设计图纸\n现行施工技术规范"),
        PDFPageText(4, "2 工程概况\n验收应符合现行施工技术规范\n作业人员资格应符合标准要求"),
        PDFPageText(5, "3 计算书\n计算采用结构设计规范"),
    ]
    chapters = ChapterDetectionService().detect(pages)

    basis = DocumentAnalysisService._extract_compilation_basis(pages, chapters)
    values = [item.value for item in basis]
    assert any("设计图纸" in value for value in values)
    assert any("施工技术规范" in value for value in values)
    assert all("验收应符合" not in value for value in values)
    assert all("作业人员资格" not in value for value in values)
    assert all("计算采用" not in value for value in values)


def test_llm_body_reference_does_not_expand_deterministic_compilation_basis() -> None:
    deterministic = [
        SourceValue(value="设计图纸", source_page=3, source_text="1.1 设计图纸")
    ]
    merged = DocumentAnalysisService._merge_compilation_basis(
        deterministic,
        ["验收应符合现行规范", "施工人员资格标准"],
    )
    assert [item.value for item in merged] == ["设计图纸"]


def test_provenance_prefers_basis_and_material_plan_over_calculation() -> None:
    pages = [
        PDFPageText(
            2,
            "1 编制依据\n建筑结构荷载规范 | GB 50009-2012\n"
            "主要材料计划表\n构件钢板 Q355B",
        ),
        PDFPageText(
            20,
            "第十章 计算书\n《建筑结构荷载规范》GB 50009-2012\n"
            "构件钢板 Q355B\nM=25kN.m",
        ),
    ]
    extracted = PDFExtractionResult(
        2, sum(len(page.text) for page in pages), "\n".join(page.text for page in pages), pages
    )
    standards = DocumentAnalysisService(FakeLLMService())._extract_standards(extracted)
    assert next(item for item in standards if item.code == "GB50009-2012").source_page == 2

    analysis = DocumentAnalysis.model_validate(
        {
            "standards": [item.model_dump() for item in standards],
            "main_materials": [{"name": "构件钢板", "specification": "Q355B"}],
        }
    )
    located = SourceLocatorService().locate(analysis, pages)
    assert located.main_materials[0].source_page == 2
    assert located.main_materials[0].source_text == "构件钢板 Q355B"


def test_engineering_category_uses_strong_building_overview_evidence() -> None:
    pages = [
        PDFPageText(1, "某建设项目专项施工方案"),
        PDFPageText(
            5,
            "二、工程概况\n（一）建筑概况\n建筑面积：56000m²\n"
            "地下一层，地上16层，中间设三层裙房\n"
            "结构形式：框架剪力墙结构\n（二）结构概况",
        ),
    ]
    service = MetadataExtractionService()
    project, _ = service.extract(pages)

    category = service.extract_engineering_category(pages, project)

    assert category.value == "房屋建筑"
    assert category.source_page == 5
    assert "建筑概况" in category.source_text
    assert "建筑面积" in category.source_text


def test_engineering_category_rejects_weak_building_toc_evidence() -> None:
    pages = [PDFPageText(3, "目录\n（一）建筑概况……2\n（二）结构概况……2")]
    service = MetadataExtractionService()
    project, _ = service.extract(pages)

    assert service.extract_engineering_category(pages, project) is None


def test_calculation_region_actions_do_not_become_work_items() -> None:
    pages = [
        PDFPageText(10, "3 施工方法\n3.1 立杆设置\n按定位线设置立杆。"),
        PDFPageText(90, "9 计算书"),
        PDFPageText(91, "混凝土达到C15时即可安装拉杆、搭设架体。"),
        PDFPageText(93, "满足计算条件后安装钢梁。"),
    ]
    analysis = DocumentAnalysis.model_validate(
        {
            "main_work_items": [
                {"value": "立杆设置", "source_page": 10, "source_text": "3.1 立杆设置"},
                {"value": "安装拉杆", "source_page": 91, "source_text": "混凝土达到C15时即可安装拉杆、搭设架体。"},
                {"value": "搭设架体", "source_page": 91, "source_text": "混凝土达到C15时即可安装拉杆、搭设架体。"},
                {"value": "安装钢梁", "source_page": 93, "source_text": "满足计算条件后安装钢梁。"},
            ]
        }
    )

    cleaned = DeterministicPostProcessingService().final_cleanup(analysis, pages)

    assert [item.value for item in cleaned.main_work_items] == ["立杆设置"]


def test_method_gate_rejects_unheaded_construction_types_but_keeps_actions() -> None:
    pages = [
        PDFPageText(
            10,
            "3 施工方法\n3.1 悬挑构件定位、安装\n3.2 立杆设置\n"
            "3.3 剪刀撑设置\n3.4 构件拆除",
        )
    ]
    chapters = ChapterDetectionService().detect(pages)
    analysis = DocumentAnalysis.model_validate(
        {
            "chapters": [chapter.model_dump() for chapter in chapters],
            "main_methods": [
                {"name": "悬挑构件定位、安装", "source_page": 10, "source_text": "3.1 悬挑构件定位、安装"},
                {"name": "立杆设置", "source_page": 10, "source_text": "3.2 立杆设置"},
                {"name": "剪刀撑设置", "source_page": 10, "source_text": "3.3 剪刀撑设置"},
                {"name": "构件拆除", "source_page": 10, "source_text": "3.4 构件拆除"},
                {"name": "上拉式承力架", "source_page": 10, "source_text": "采用一种承力架。"},
                {"name": "组合支撑体系", "source_page": 10, "source_text": "可采用组合支撑体系。"},
                {"name": "框架结构形式", "source_page": 10, "source_text": "结构形式为框架结构。"},
            ],
        }
    )

    cleaned = DeterministicPostProcessingService().final_cleanup(analysis, pages)

    assert [method.name for method in cleaned.main_methods] == [
        "悬挑构件定位、安装",
        "立杆设置",
        "剪刀撑设置",
        "构件拆除",
    ]
