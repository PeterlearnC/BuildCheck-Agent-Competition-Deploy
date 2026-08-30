"""Deterministic raster evidence derived before rendered OCR pixels are discarded."""

from dataclasses import dataclass
import math
from statistics import median

import fitz

from app.schemas.ocr import (
    OCRLineResult,
    OCRVisualSupportDatum,
    OCRVisualSupportGeometrySource,
)


OCR_RASTER_SUPPORT_RULE_VERSION = "v0.4-b.3c.2-f1-r1-raster-support"

# The two luminance cutoffs have distinct purposes on an 8-bit rendered page:
# 245 excludes near-white background/antialias noise, while 128 requires a
# pixel to be at least half-dark before it counts as printed-ink support.
BACKGROUND_LUMINANCE_CUTOFF = 245
DEEP_INK_LUMINANCE_CUTOFF = 128

# Reference bands are deliberately narrower than the Quality Gate outlier
# band. They model ordinary, near-horizontal body text without relying on its
# language, lexical content, or document identity.
REFERENCE_MAX_ANGLE_DEGREES = 5.0
REFERENCE_MIN_HEIGHT_RATIO = 0.70
REFERENCE_MAX_HEIGHT_RATIO = 1.30
REFERENCE_MIN_ASPECT_RATIO = 1.0
REFERENCE_MIN_DARK_SUPPORT = 0.20


@dataclass(frozen=True)
class _MeasuredLine:
    line: OCRLineResult
    geometry_source: OCRVisualSupportGeometrySource
    dark_support: float
    height_ratio: float
    aspect_ratio: float
    angle: float


class OCRRasterSupportAnalyzer:
    """Attach small, auditable visual measurements to immutable OCR lines."""

    def annotate(
        self, png_bytes: bytes, lines: list[OCRLineResult]
    ) -> list[OCRLineResult]:
        if not lines:
            return []
        pixmap = fitz.Pixmap(png_bytes)
        heights = [self._height(line.polygon) for line in lines]
        positive_heights = [value for value in heights if value > 0]
        typical_height = median(positive_heights) if positive_heights else 1.0
        measured = [
            self._measure_line(pixmap, line, typical_height) for line in lines
        ]
        reference_values = [
            item.dark_support
            for item in measured
            if (
                abs(item.angle) <= REFERENCE_MAX_ANGLE_DEGREES
                and REFERENCE_MIN_HEIGHT_RATIO
                <= item.height_ratio
                <= REFERENCE_MAX_HEIGHT_RATIO
                and item.aspect_ratio >= REFERENCE_MIN_ASPECT_RATIO
                and item.dark_support >= REFERENCE_MIN_DARK_SUPPORT
            )
        ]
        page_reference = median(reference_values) if reference_values else 0.0
        result = []
        for item in measured:
            relative = (
                item.dark_support / page_reference if page_reference > 0 else 0.0
            )
            result.append(
                item.line.model_copy(
                    update={
                        "visual_support": OCRVisualSupportDatum(
                            geometry_source=item.geometry_source,
                            dark_support=self._stable(item.dark_support),
                            page_reference_support=self._stable(page_reference),
                            relative_support_ratio=self._stable(relative),
                            height_ratio=self._stable(item.height_ratio),
                            aspect_ratio=self._stable(item.aspect_ratio),
                            rule_version=OCR_RASTER_SUPPORT_RULE_VERSION,
                        )
                    }
                )
            )
        return result

    def _measure_line(
        self, pixmap: fitz.Pixmap, line: OCRLineResult, typical_height: float
    ) -> _MeasuredLine:
        if line.characters:
            polygons = [character.polygon for character in line.characters]
            source = OCRVisualSupportGeometrySource.CHARACTER_POLYGONS
        else:
            polygons = [line.polygon]
            source = OCRVisualSupportGeometrySource.LINE_POLYGON
        dark_support = self._dark_support(pixmap, polygons)
        height = self._height(line.polygon)
        width = self._width(line.polygon)
        return _MeasuredLine(
            line=line,
            geometry_source=source,
            dark_support=dark_support,
            height_ratio=height / typical_height,
            aspect_ratio=width / height if height > 0 else 0.0,
            angle=self._top_edge_angle(line.polygon),
        )

    def _dark_support(
        self, pixmap: fitz.Pixmap, polygons: list[list[list[float]]]
    ) -> float:
        pixel_indexes: set[int] = set()
        for polygon in polygons:
            left = max(0, math.floor(min(point[0] for point in polygon)))
            right = min(
                pixmap.width, math.ceil(max(point[0] for point in polygon))
            )
            top = max(0, math.floor(min(point[1] for point in polygon)))
            bottom = min(
                pixmap.height, math.ceil(max(point[1] for point in polygon))
            )
            for y in range(top, bottom):
                for x in range(left, right):
                    if self._inside_convex_polygon(x + 0.5, y + 0.5, polygon):
                        pixel_indexes.add(y * pixmap.width + x)
        samples = memoryview(pixmap.samples)
        non_background = 0
        deep_ink = 0
        for index in pixel_indexes:
            y, x = divmod(index, pixmap.width)
            luminance = self._luminance(samples, pixmap, x, y)
            if luminance < BACKGROUND_LUMINANCE_CUTOFF:
                non_background += 1
                if luminance < DEEP_INK_LUMINANCE_CUTOFF:
                    deep_ink += 1
        return deep_ink / non_background if non_background else 0.0

    @staticmethod
    def _luminance(
        samples: memoryview, pixmap: fitz.Pixmap, x: int, y: int
    ) -> int:
        offset = y * pixmap.stride + x * pixmap.n
        if pixmap.n <= 2:
            return int(samples[offset])
        red, green, blue = (
            int(samples[offset]),
            int(samples[offset + 1]),
            int(samples[offset + 2]),
        )
        return (299 * red + 587 * green + 114 * blue + 500) // 1000

    @staticmethod
    def _inside_convex_polygon(
        x: float, y: float, polygon: list[list[float]]
    ) -> bool:
        sign = 0
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            cross = (second[0] - first[0]) * (y - first[1]) - (
                second[1] - first[1]
            ) * (x - first[0])
            if abs(cross) <= 1e-9:
                continue
            current = 1 if cross > 0 else -1
            if sign and current != sign:
                return False
            sign = current
        return True

    @staticmethod
    def _height(polygon: list[list[float]]) -> float:
        values = [point[1] for point in polygon]
        return max(values) - min(values)

    @staticmethod
    def _width(polygon: list[list[float]]) -> float:
        values = [point[0] for point in polygon]
        return max(values) - min(values)

    @staticmethod
    def _top_edge_angle(polygon: list[list[float]]) -> float:
        dx = polygon[1][0] - polygon[0][0]
        dy = polygon[1][1] - polygon[0][1]
        return math.degrees(math.atan2(dy, dx)) if dx or dy else 0.0

    @staticmethod
    def _stable(value: float) -> float:
        return round(float(value), 8)
