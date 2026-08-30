"""Deterministic identities for parsed standards and retrieval evidence."""

import hashlib
import json
import re
import unicodedata


PARSER_VERSION = "v0.4-b.3a.1"
PARSER_SEMANTICS_VERSION = "v0.4-b.3c.3-s1-r1-parser"


class StableIdentityService:
    """Create content-addressed public identifiers without random state."""

    @staticmethod
    def normalize_text(value: str | None) -> str:
        normalized = unicodedata.normalize("NFKC", value or "")
        return re.sub(r"\s+", " ", normalized).strip()

    def chapter_id(
        self,
        *,
        standard_checksum: str,
        chapter_number: str | None,
        chapter_title: str,
    ) -> str:
        return self._identifier(
            "chapter",
            standard_checksum,
            self.normalize_text(chapter_number),
            self.normalize_text(chapter_title),
        )

    def article_id(
        self,
        *,
        standard_source_checksum: str,
        standard_code: str | None,
        chapter_number: str | None,
        article_number: str,
        article_text: str,
        source_page_start: int,
        source_page_end: int | None = None,
        region_type: str | None = None,
        article_type: str | None = None,
        source_kind: str = "PDF_TEXT",
        accepted_source_line_ids: list[str] | None = None,
    ) -> str:
        values: list[object] = [
            standard_source_checksum,
            self.normalize_text(standard_code).upper(),
            self.normalize_text(chapter_number),
            self.normalize_text(article_number).upper(),
            self.normalize_text(region_type).upper(),
            self.normalize_text(article_type).upper(),
            source_page_start,
            source_page_end if source_page_end is not None else source_page_start,
            self.normalize_text(article_text),
        ]
        if source_kind.upper() == "OCR_TEXT":
            values.extend(
                [
                    "OCR_TEXT",
                    list(accepted_source_line_ids or []),
                ]
            )
        return self._identifier("article", *values)

    def chunk_id(self, *, article_id: str, chunk_index: int, chunk_text: str) -> str:
        return self._identifier(
            "chunk",
            article_id,
            chunk_index,
            self.normalize_text(chunk_text),
        )

    @staticmethod
    def _identifier(kind: str, *values: object) -> str:
        payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{kind}_{digest}"
