"""Versioned JSON cache for deterministic completeness reviews."""

import hashlib
import json

from app.core.config import Settings, get_settings
from app.schemas.analysis import DocumentAnalysis
from app.schemas.completeness_review import (
    CompletenessReview,
    CompletenessReviewCache,
)


class CompletenessReviewCacheError(Exception):
    pass


class CompletenessReviewCacheService:
    REVIEW_VERSION = "v0.3"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @staticmethod
    def analysis_fingerprint(analysis: DocumentAnalysis) -> str:
        serialized = json.dumps(
            analysis.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def load(
        self, document_id: str, analysis: DocumentAnalysis
    ) -> CompletenessReview | None:
        path = self.settings.completeness_review_dir / f"{document_id}.json"
        if not path.is_file():
            return None
        try:
            cached = CompletenessReviewCache.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            # A damaged or obsolete review is disposable derived data.
            return None
        if cached.review_version != self.REVIEW_VERSION:
            return None
        if cached.document_id != document_id:
            return None
        if cached.analysis_fingerprint != self.analysis_fingerprint(analysis):
            return None
        return cached.review

    def save(
        self,
        document_id: str,
        analysis: DocumentAnalysis,
        review: CompletenessReview,
    ) -> None:
        payload = CompletenessReviewCache(
            review_version=self.REVIEW_VERSION,
            document_id=document_id,
            analysis_fingerprint=self.analysis_fingerprint(analysis),
            review=review,
        )
        try:
            self.settings.completeness_review_dir.mkdir(parents=True, exist_ok=True)
            target = self.settings.completeness_review_dir / f"{document_id}.json"
            temporary = self.settings.completeness_review_dir / f".{document_id}.tmp"
            temporary.write_text(
                json.dumps(payload.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(target)
        except OSError as exc:
            raise CompletenessReviewCacheError(
                "Failed to save completeness review cache."
            ) from exc
