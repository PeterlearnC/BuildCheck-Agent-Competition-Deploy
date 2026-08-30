from app.schemas.standards import StandardPage
from app.services.standards.article_merge_service import ArticleMergeService
from app.services.standards.standard_chapter_service import StandardChapterService


def _merge(*texts: str):
    pages = [StandardPage(page_number=index + 1, text=text) for index, text in enumerate(texts)]
    _, positions = StandardChapterService().detect(
        "standard-1", pages, standard_checksum="a" * 64
    )
    return ArticleMergeService().merge("standard-1", pages, positions)


def test_single_page_multiline_article_is_joined() -> None:
    articles = _merge("第6章 构造要求\n6.3.4 支撑构件应\n设置可靠连接\n并进行检查。")
    assert len(articles) == 1
    assert articles[0].content == "支撑构件应设置可靠连接并进行检查。"
    assert articles[0].source_page_start == articles[0].source_page_end == 1


def test_number_only_line_uses_following_body() -> None:
    article = _merge("6.3.4\n构件安装前应进行检查。")[0]
    assert article.article_number == "6.3.4"
    assert article.content == "构件安装前应进行检查。"


def test_consecutive_articles_stay_separate() -> None:
    articles = _merge("6.3.4 第一条。\n6.3.5 第二条。\n6.3.6 第三条。")
    assert [item.article_number for item in articles] == ["6.3.4", "6.3.5", "6.3.6"]


def test_cross_page_continuation_is_merged() -> None:
    articles = _merge(
        "6.3.4 支撑架应设置剪刀撑，并符合下列规定：\n1 构件可靠连接；\n2 基础稳定；",
        "3 节点牢固；\n4 检查合格。\n6.3.5 连墙件应可靠连接。",
    )
    first = articles[0]
    assert first.source_page_start == 1
    assert first.source_page_end == 2
    assert "3 节点牢固" in first.content
    assert articles[1].article_number == "6.3.5"


def test_new_article_at_next_page_stops_merge() -> None:
    articles = _merge("6.3.4 第一条正文。", "6.3.5 第二条正文。")
    assert articles[0].source_page_end == 1
    assert "第二条" not in articles[0].content


def test_new_chapter_stops_active_article() -> None:
    articles = _merge("6.3.4 第一条正文。\n第7章 安全管理\n7.1.1 第二章条文。")
    assert [item.article_number for item in articles] == ["6.3.4", "7.1.1"]
    assert articles[0].content == "第一条正文。"


def test_numbered_subitems_remain_in_parent_article() -> None:
    articles = _merge("6.3.4 应符合下列规定：\n1 构件可靠；\n2 节点牢固；\n3 基础稳定。")
    assert len(articles) == 1
    assert all(fragment in articles[0].content for fragment in ["1 构件", "2 节点", "3 基础"])


def test_unpunctuated_numbered_requirements_remain_in_parent_article() -> None:
    articles = _merge("6.3.4 应符合下列规定：\n1 安装位置应准确\n2 连接节点应牢固")
    assert len(articles) == 1
    assert "1 安装位置应准确" in articles[0].content
    assert "2 连接节点应牢固" in articles[0].content


def test_plain_paragraph_does_not_create_article() -> None:
    assert _merge("本章说明施工的一般要求。\n构件安装前进行检查。") == []


def test_provenance_is_one_based_and_source_text_preserved() -> None:
    article = _merge("6.1.1 支撑构件应设置可靠连接。") [0]
    assert article.source_page_start == 1
    assert article.source_page_end == 1
    assert article.source_text == "6.1.1 支撑构件应设置可靠连接。"
    assert article.is_mandatory is None
