"""Deterministic normalization and lightweight mixed CJK/ASCII tokenization."""

import re
import unicodedata


class QueryNormalizationService:
    ASCII_TOKEN = re.compile(r"[a-z]+(?:/[a-z]+)?|\d+(?:[.-]\d+)*", re.IGNORECASE)
    CJK_RUN = re.compile(r"[\u3400-\u9fff]+")

    def normalize(self, text: str) -> str:
        value = unicodedata.normalize("NFKC", text or "").lower()
        value = " ".join(value.split())
        value = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", value)
        return value.strip()

    def tokenize(self, text: str) -> list[str]:
        normalized = self.normalize(text)
        tokens = self.ASCII_TOKEN.findall(normalized)
        for match in self.CJK_RUN.finditer(normalized):
            value = match.group(0)
            tokens.append(value)
            for size in range(2, min(4, len(value)) + 1):
                tokens.extend(value[index: index + size] for index in range(len(value) - size + 1))
        return list(dict.fromkeys(token for token in tokens if token))

    def cjk_phrases(self, text: str) -> list[str]:
        return [match.group(0) for match in self.CJK_RUN.finditer(self.normalize(text)) if len(match.group(0)) >= 2]
