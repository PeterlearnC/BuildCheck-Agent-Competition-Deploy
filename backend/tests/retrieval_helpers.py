from datetime import datetime, timezone

from app.schemas.standards import (
    ArticleParseConfidence,
    ArticleType,
    DocumentRegionType,
    MetadataConfidence,
    StandardArticle,
    StandardDocument,
    StandardParseStatus,
)
from app.services.retrieval.embedding_provider import EmbeddingProvider
from app.services.retrieval.models import RetrievalRecord


def make_record(
    standard_id: str,
    article_number: str,
    content: str,
    *,
    sequence: int = 1,
    standard_code: str | None = None,
    standard_name: str | None = None,
    chapter_number: str | None = "6.1",
    chapter_title: str | None = "构造要求",
    page_start: int = 3,
    page_end: int | None = None,
) -> RetrievalRecord:
    now = datetime.now(timezone.utc)
    document = StandardDocument(
        standard_id=standard_id,
        standard_code=standard_code,
        standard_name=standard_name,
        source_filename=f"{standard_id}.pdf",
        source_checksum=(standard_id.encode("utf-8").hex() + "0" * 64)[:64],
        page_count=10,
        parse_status=StandardParseStatus.PARSED,
        metadata_confidence=MetadataConfidence.HIGH,
        created_at=now,
        updated_at=now,
    )
    article = StandardArticle(
        article_id=f"{standard_id}-{article_number}",
        standard_id=standard_id,
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        article_number=article_number,
        content=content,
        source_page_start=page_start,
        source_page_end=page_end or page_start,
        source_text=f"{article_number} {content}",
        sequence=sequence,
        parse_confidence=ArticleParseConfidence.HIGH,
        article_type=ArticleType.NORMATIVE,
        region_type=DocumentRegionType.NORMATIVE_BODY,
    )
    return RetrievalRecord(document=document, article=article)


def basic_records() -> list[RetrievalRecord]:
    return [
        make_record("std-a", "6.1.1", "支撑结构应设置可靠连接。", sequence=1, standard_code="JGJ 999-2024", standard_name="架体工程测试规范"),
        make_record("std-a", "6.1.2", "连墙件应按规定设置。", sequence=2, standard_code="JGJ 999-2024", standard_name="架体工程测试规范"),
        make_record("std-a", "6.1.3", "作业人员进入现场前应检查安全防护用品。", sequence=3, standard_code="JGJ 999-2024", standard_name="架体工程测试规范"),
        make_record("std-b", "6.1.2", "给水管道安装完成后应进行冲洗。", sequence=1, standard_code="GB 888-2023", standard_name="管道工程测试规范"),
    ]


class MappingEmbeddingProvider(EmbeddingProvider):
    """Readable fake: the first configured substring selects its explicit vector."""

    def __init__(self, mappings: list[tuple[str, list[float]]], dimension: int = 3) -> None:
        self.mappings = mappings
        self._dimension = dimension

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "deterministic-test-mapping"

    @property
    def provider_version(self) -> str:
        return "1"

    @property
    def embedding_model_id(self) -> str:
        return "deterministic-test-model-v1"

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        return next((vector[:] for phrase, vector in self.mappings if phrase in text), [0.0] * self._dimension)
