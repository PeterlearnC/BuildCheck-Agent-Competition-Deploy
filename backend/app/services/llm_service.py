"""Minimal OpenAI-compatible LLM client with guarded JSON parsing."""

import json
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings, get_settings
from app.schemas.analysis import DocumentAnalysis
from app.services.pdf_service import PDFPageText


class LLMServiceError(Exception):
    """Safe base error for LLM configuration, transport, or output failures."""


class LLMConfigurationError(LLMServiceError):
    pass


class LLMCallError(LLMServiceError):
    pass


class LLMResponseError(LLMServiceError):
    pass


ModelT = TypeVar("ModelT", bound=BaseModel)


SYSTEM_PROMPT = """你是建筑工程施工方案信息提取助手。
你的唯一任务是从用户提供的施工方案原文提取明确事实。
严格规则：
1. 只能使用提供的施工方案原文；禁止使用常识补充或猜测。
2. 禁止进行规范审查、评价方案质量、风险评级或生成整改建议。
3. 文档不存在或无法确认的信息返回 null；数组不存在时返回 []。
4. 保留原文术语。每个重要事实尽量给出 source_page 和简短 source_text。
5. 页码仅可使用原文中的“[PDF第N页]”标记，不能制造来源。
6. 必须输出一个严格 JSON 对象，不输出 Markdown 或解释文字。
7. 提取 document_type、engineering_category、specialty、project、schedule、
chapters、compilation_basis、main_work_items、main_materials、main_methods、section_presence。
8. 只判断质量、安全、环境、应急相关章节是否存在，不评价其内容。
9. 原文日期明确时可规范化为 YYYY-MM-DD；无法明确判断时返回 null。
字段结构遵循：可追溯单值为 {"value":值,"source_page":页码或null,"source_text":原文片段或null}；
材料包含 name/specification/usage/source_page/source_text；方法包含 name/source_page/source_text。
chapters 必须严格使用以下结构，禁止使用 name、chapter_name、page 或 page_number 等替代字段：
"chapters": [{"title":"工程概况","start_page":4,"end_page":5,"level":1}]
"""


class LLMService:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self.client = client

    def analyze_document(self, pages: list[PDFPageText]) -> DocumentAnalysis:
        chunks = self._build_chunks(pages, self.settings.llm_max_input_chars)
        partials: list[DocumentAnalysis] = []
        for index, chunk in enumerate(chunks, start=1):
            prompt = (
                f"以下是施工方案第 {index}/{len(chunks)} 个文本分段。只提取本段明确出现的信息；"
                "未出现的字段返回 null 或空数组。\n\n" + chunk
            )
            raw = self.call_chat_completion(SYSTEM_PROMPT, prompt)
            partials.append(self.parse_json_response(raw, DocumentAnalysis))
        return self._merge_partials(partials)

    def call_chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        missing = [
            name
            for name, value in (
                ("LLM_API_KEY", self.settings.llm_api_key),
                ("LLM_BASE_URL", self.settings.llm_base_url),
                ("LLM_MODEL", self.settings.llm_model),
            )
            if not value
        ]
        if missing:
            raise LLMConfigurationError(
                "LLM is not configured. Missing environment variables: " + ", ".join(missing)
            )

        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            if self.client is not None:
                response = self.client.post(url, headers=self._headers(), json=payload)
            else:
                response = httpx.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.settings.llm_timeout,
                )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise KeyError("empty message content")
            return content
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMCallError("LLM request failed or returned an unsupported response.") from exc

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def clean_llm_json_response(raw: str) -> dict:
        cleaned = raw.strip().replace("```json", "").replace("```JSON", "").replace("```", "")
        decoder = json.JSONDecoder()
        for position, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(cleaned[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise LLMResponseError("LLM response did not contain a valid JSON object.")

    @classmethod
    def parse_json_response(
        cls, raw: str, model: type[ModelT] = DocumentAnalysis
    ) -> ModelT:
        payload = cls.clean_llm_json_response(raw)
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
                for item in exc.errors()[:5]
            )
            raise LLMResponseError(f"LLM JSON failed schema validation: {errors}") from exc

    @staticmethod
    def _build_chunks(pages: list[PDFPageText], limit: int) -> list[str]:
        if limit < 1000:
            raise LLMConfigurationError("LLM_MAX_INPUT_CHARS must be at least 1000.")
        chunks: list[str] = []
        current = ""
        for page in pages:
            marker = f"[PDF第{page.page_number}页]\n"
            remaining = page.text
            while remaining:
                available = limit - len(current) - len(marker)
                if available <= 0:
                    chunks.append(current.rstrip())
                    current = ""
                    available = limit - len(marker)
                part = remaining[:available]
                if current:
                    current += "\n\n"
                current += marker + part
                remaining = remaining[available:]
                marker = f"[PDF第{page.page_number}页续]\n"
                if remaining:
                    chunks.append(current.rstrip())
                    current = ""
        if current.strip():
            chunks.append(current.rstrip())
        return chunks or ["（无文本）"]

    @staticmethod
    def _merge_partials(partials: list[DocumentAnalysis]) -> DocumentAnalysis:
        if not partials:
            return DocumentAnalysis()
        merged = partials[0].model_dump()
        for partial in partials[1:]:
            incoming = partial.model_dump()
            for key in ("document_type", "engineering_category", "specialty"):
                if not merged.get(key) and incoming.get(key):
                    merged[key] = incoming[key]
            for group in ("project", "schedule"):
                for key, value in incoming[group].items():
                    if not merged[group].get(key) and value:
                        merged[group][key] = value
            for key in (
                "chapters", "compilation_basis", "standards", "main_work_items",
                "main_materials", "main_methods"
            ):
                for item in incoming[key]:
                    if item not in merged[key]:
                        merged[key].append(item)
            for key, value in incoming["section_presence"].items():
                merged["section_presence"][key] = merged["section_presence"][key] or value
        return DocumentAnalysis.model_validate(merged)


def get_llm_service() -> LLMService:
    return LLMService()
