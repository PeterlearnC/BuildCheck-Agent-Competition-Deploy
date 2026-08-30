"""Run the controlled three-page OCR integration against an explicit local worker."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import fitz

from app.core.config import Settings
from app.schemas.ocr import (
    OCRModelArtifact,
    OCRProviderIdentity,
    OfficialSourceBinding,
)
from app.schemas.standards import StandardDocument
from app.schemas.standards_retrieval import RetrievalMethod
from app.services.ocr.ocr_repository import OCRRepository
from app.services.ocr.page_renderer import OCRPageRenderer
from app.services.ocr.pilot_service import ControlledOCRPilotService
from app.services.ocr.provider import RapidOCRSubprocessProvider, RapidOCRWorkerConfig
from app.services.retrieval.hit_factory import make_hit
from app.services.retrieval.models import RetrievalRecord
from app.services.standards.standard_repository import StandardRepository


OFFICIAL_SHA256 = "58414579d5d6b70f24c0380d632ce659985043b3bb1121dbdbd026452942d567"
DET_SHA256 = "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"
REC_SHA256 = "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"
CLS_SHA256 = "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    checksum = _checksum(arguments.pdf)
    if checksum != OFFICIAL_SHA256:
        raise SystemExit("Official PDF checksum mismatch.")
    provider_identity = OCRProviderIdentity(
        provider="rapidocr",
        provider_version="3.9.2",
        runtime="onnxruntime",
        runtime_version="1.29.0",
        device="CPUExecutionProvider",
        detector=OCRModelArtifact(model_id="PP-OCRv6_det_small.onnx", sha256=DET_SHA256),
        recognizer=OCRModelArtifact(model_id="PP-OCRv6_rec_small.onnx", sha256=REC_SHA256),
        classifier=OCRModelArtifact(
            model_id="ch_ppocr_mobile_v2.0_cls_mobile.onnx", sha256=CLS_SHA256
        ),
        classification_enabled=False,
    )
    worker = Path(__file__).resolve().parents[1] / "ocr_worker" / "rapidocr_worker.py"
    provider = RapidOCRSubprocessProvider(
        RapidOCRWorkerConfig(
            python_executable=arguments.worker_python,
            worker_script=worker,
            detector_model_path=arguments.model_dir / "PP-OCRv6_det_small.onnx",
            recognizer_model_path=arguments.model_dir / "PP-OCRv6_rec_small.onnx",
            classifier_model_path=arguments.model_dir / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            identity=provider_identity,
        )
    )
    standards = StandardRepository(Settings(standards_dir=arguments.output_dir / "standards"))
    repository = OCRRepository(standards)
    now = datetime.now(timezone.utc)
    with fitz.open(arguments.pdf) as pdf:
        page_count = pdf.page_count
    document = StandardDocument(
        standard_id="55023b3b-1b00-4000-8000-000000000001",
        source_filename=arguments.pdf.name,
        source_checksum=checksum,
        page_count=page_count,
        created_at=now,
        updated_at=now,
    )
    binding = OfficialSourceBinding(
        source_checksum=checksum,
        canonical_standard_code="GB55023-2022",
        display_standard_code="GB 55023-2022",
        standard_name="施工脚手架通用规范",
        binding_reason="Official source checksum verified during the controlled OCR pilot.",
        binding_provenance="B.3B.1B curated official-source binding",
        confirmed=True,
    )
    result = ControlledOCRPilotService(
        renderer=OCRPageRenderer(),
        provider=provider,
        repository=repository,
    ).run(
        document=document,
        pdf_path=arguments.pdf,
        page_numbers=[1, 4, 10],
        binding=binding,
    )

    assessment_by_page = {page.page_number: page for page in result.assessment.pages}
    accepted_by_page = {page.page_number: page for page in result.accepted_pages}
    report = {
        "official_checksum": checksum,
        "ocr_run_id": result.run.ocr_run_id,
        "execution_id": result.run.execution_id,
        "run_state": result.assessment.state.value,
        "pages": [],
        "articles": [],
    }
    for raw_page in result.run.pages:
        assessment = assessment_by_page[raw_page.page_number]
        accepted = accepted_by_page.get(raw_page.page_number)
        report["pages"].append(
            {
                "page_number": raw_page.page_number,
                "image_sha256": raw_page.render.image_sha256,
                "line_count": len(raw_page.lines),
                "raw_lines": [line.model_dump(mode="json") for line in raw_page.lines],
                "quality_state": assessment.state.value,
                "issues": [issue.model_dump(mode="json") for issue in assessment.issues],
                "accepted_raw_line_ids": assessment.accepted_line_ids,
                "accepted_parser_text": (
                    [span.normalized_text for span in accepted.spans] if accepted else []
                ),
            }
        )
    if result.parse_result is not None:
        for article in result.parse_result.articles:
            hit = make_hit(
                RetrievalRecord(result.parse_result.document, article),
                rank=1,
                methods=[RetrievalMethod.KEYWORD],
            )
            report["articles"].append(
                {
                    "article_number": article.article_number,
                    "article_id": article.article_id,
                    "article_type": article.article_type.value,
                    "region_type": article.region_type.value,
                    "source_page_start": article.source_page_start,
                    "source_page_end": article.source_page_end,
                    "source_text": article.source_text,
                    "source_line_ids": [
                        mapping.raw_line_id for mapping in article.source_mappings
                    ],
                    "source_polygons": [
                        mapping.raw_polygon for mapping in article.source_mappings
                    ],
                    "evidence": hit.evidence.model_dump(mode="json"),
                }
            )
    report_path = arguments.output_dir / "controlled-validation-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "run_state": report["run_state"],
                "page_states": {
                    page["page_number"]: page["quality_state"] for page in report["pages"]
                },
                "issue_codes": {
                    page["page_number"]: [issue["code"] for issue in page["issues"]]
                    for page in report["pages"]
                },
                "article_numbers": [item["article_number"] for item in report["articles"]],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
