"""Trusted checksum-indexed official-standard identity authority."""

from collections.abc import Iterable

from app.schemas.ocr import OfficialSourceBinding, TrustedOfficialSourceRecord


GB55023_2022_SOURCE_SHA256 = (
    "58414579d5d6b70f24c0380d632ce659985043b3bb1121dbdbd026452942d567"
)


class TrustedOfficialSourceConflictError(ValueError):
    pass


class TrustedOfficialSourceRegistry:
    """Resolve caller requests against immutable application-owned records."""

    def __init__(
        self, records: Iterable[TrustedOfficialSourceRecord] | None = None
    ) -> None:
        values = tuple(records) if records is not None else self.default_records()
        by_checksum: dict[str, TrustedOfficialSourceRecord] = {}
        by_identity: dict[tuple[str, str], TrustedOfficialSourceRecord] = {}
        for record in values:
            checksum = record.source_checksum.lower()
            identity = (
                record.canonical_standard_code.casefold(),
                record.standard_name.casefold(),
            )
            if checksum in by_checksum and by_checksum[checksum] != record:
                raise TrustedOfficialSourceConflictError(
                    "Multiple trusted identities claim the same source checksum."
                )
            if identity in by_identity and by_identity[identity].source_checksum != checksum:
                raise TrustedOfficialSourceConflictError(
                    "One trusted standard identity claims multiple source checksums."
                )
            by_checksum[checksum] = record
            by_identity[identity] = record
        self._by_checksum = by_checksum
        self._by_identity = by_identity

    @staticmethod
    def default_records() -> tuple[TrustedOfficialSourceRecord, ...]:
        return (
            TrustedOfficialSourceRecord(
                source_checksum=GB55023_2022_SOURCE_SHA256,
                canonical_standard_code="GB55023-2022",
                display_standard_code="GB 55023-2022",
                standard_name="施工脚手架通用规范",
                authority_id="b3b-q1-gb55023-2022-official-source-v1",
                authority_provenance=(
                    "Application-owned exact-checksum authority recovered from "
                    "the qualified government attachment."
                ),
            ),
        )

    def resolve(
        self,
        *,
        source_checksum: str,
        requested: OfficialSourceBinding | None = None,
    ) -> TrustedOfficialSourceRecord | None:
        checksum = source_checksum.lower()
        record = self._by_checksum.get(checksum)
        if record is None:
            if requested is not None:
                identity = (
                    requested.canonical_standard_code.casefold(),
                    requested.standard_name.casefold(),
                )
                known = self._by_identity.get(identity)
                if known is not None:
                    raise TrustedOfficialSourceConflictError(
                        "Requested trusted identity does not match its qualified checksum."
                    )
            return None

        if requested is not None:
            requested_values = (
                requested.source_checksum.lower(),
                requested.canonical_standard_code,
                requested.display_standard_code,
                requested.standard_name,
            )
            trusted_values = (
                checksum,
                record.canonical_standard_code,
                record.display_standard_code,
                record.standard_name,
            )
            if requested_values != trusted_values:
                raise TrustedOfficialSourceConflictError(
                    "Caller-requested official identity conflicts with trusted authority."
                )

        return record

    @staticmethod
    def binding(record: TrustedOfficialSourceRecord) -> OfficialSourceBinding:
        return OfficialSourceBinding(
            source_checksum=record.source_checksum,
            canonical_standard_code=record.canonical_standard_code,
            display_standard_code=record.display_standard_code,
            standard_name=record.standard_name,
            binding_reason="Resolved from trusted checksum-indexed authority.",
            binding_provenance=(
                f"{record.authority_id}: {record.authority_provenance}"
            ),
            confirmed=True,
        )
