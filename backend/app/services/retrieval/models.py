"""Internal immutable corpus records shared by retrieval strategies."""

from dataclasses import dataclass

from app.schemas.standards import StandardArticle, StandardDocument


@dataclass(frozen=True)
class RetrievalRecord:
    document: StandardDocument
    article: StandardArticle

    @property
    def embedding_text(self) -> str:
        return "\n".join(
            value
            for value in (
                self.document.standard_name,
                self.article.chapter_title,
                self.article.article_number,
                self.article.content,
            )
            if value
        )
