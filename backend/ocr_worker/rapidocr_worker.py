"""One-shot RapidOCR worker. This file runs only in the isolated OCR environment."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    request = json.load(sys.stdin)
    expected = request["expected_identity"]
    if (
        expected["provider"] != "rapidocr"
        or expected["runtime"] != "onnxruntime"
        or expected["device"] != "CPUExecutionProvider"
        or expected.get("classification_enabled") is not False
    ):
        _fail("Unsupported OCR provider execution identity.")
    if version("rapidocr") != expected["provider_version"]:
        _fail("RapidOCR package version mismatch.")
    if version("onnxruntime") != expected["runtime_version"]:
        _fail("ONNX Runtime package version mismatch.")

    model_pairs = (
        ("detector", Path(request["detector_model_path"])),
        ("recognizer", Path(request["recognizer_model_path"])),
        ("classifier", Path(request["classifier_model_path"])),
    )
    for name, path in model_pairs:
        artifact = expected.get(name)
        if artifact is None or not path.is_file():
            _fail(f"Required {name} model is missing.")
        if artifact["model_id"] != path.name:
            _fail(f"Required {name} model identity mismatch.")
        if _sha256(path).lower() != artifact["sha256"].lower():
            _fail(f"Required {name} model hash mismatch.")

    from rapidocr import RapidOCR

    params = {
        "Global.use_cls": False,
        "Global.use_preprocess_img": False,
        "Global.return_word_box": True,
        "Global.return_single_char_box": False,
        "Global.log_level": "warning",
        "EngineConfig.onnxruntime.intra_op_num_threads": request["intra_op_threads"],
        "EngineConfig.onnxruntime.inter_op_num_threads": request["inter_op_threads"],
        "EngineConfig.onnxruntime.enable_cpu_mem_arena": False,
        "Det.model_path": request["detector_model_path"],
        "Det.limit_side_len": request["detector_long_side_limit"],
        "Det.limit_type": "max",
        "Rec.model_path": request["recognizer_model_path"],
        "Cls.model_path": request["classifier_model_path"],
    }
    engine = RapidOCR(params=params)
    result = engine(request["image_path"], return_word_box=True)
    lines = []
    boxes = [] if result.boxes is None else result.boxes.tolist()
    texts = [] if result.txts is None else list(result.txts)
    scores = [] if result.scores is None else list(result.scores)
    word_results = list(result.word_results or [])
    for index, (polygon, text, score) in enumerate(zip(boxes, texts, scores)):
        characters = []
        if index < len(word_results) and word_results[index]:
            for char_text, confidence, char_polygon in word_results[index]:
                if char_polygon is not None:
                    characters.append(
                        {
                            "text": char_text,
                            "confidence": float(confidence),
                            "polygon": char_polygon,
                        }
                    )
        lines.append(
            {
                "text": text,
                "confidence": float(score),
                "polygon": polygon,
                "characters": characters,
            }
        )
    print(
        json.dumps(
            {
                "provider_identity": expected,
                "lines": lines,
                "elapsed_seconds": float(result.elapse),
                "warnings": [],
                "error": None,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
