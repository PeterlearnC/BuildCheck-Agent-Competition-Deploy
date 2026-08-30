"""Application configuration loaded from environment variables and project .env."""

from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel


BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseModel):
    app_name: str = "BuildCheck-Agent API"
    service_name: str = "BuildCheck-Agent"
    version: str = "0.2.1"
    # Keep the V0.1 health payload stable for existing API consumers.
    health_version: str = "0.1.0"
    api_v1_prefix: str = "/api/v1"
    upload_dir: Path = BACKEND_DIR / "data" / "uploads"
    max_upload_size: int = 30 * 1024 * 1024
    upload_chunk_size: int = 1024 * 1024
    text_preview_length: int = 1000
    analysis_dir: Path = BACKEND_DIR / "data" / "analysis"
    completeness_review_dir: Path = BACKEND_DIR / "data" / "reviews" / "completeness"
    standards_dir: Path = BACKEND_DIR / "data" / "standards"
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout: float = 60.0
    llm_max_input_chars: int = 60000


@lru_cache
def get_settings() -> Settings:
    # Resolve from this file, never from the process working directory. Existing
    # environment variables intentionally take precedence over values in .env.
    load_dotenv(dotenv_path=ENV_FILE, override=False)

    timeout = os.getenv("LLM_TIMEOUT", "60")
    max_input_chars = os.getenv("LLM_MAX_INPUT_CHARS", "60000")
    try:
        parsed_timeout = float(timeout)
        parsed_max_input_chars = int(max_input_chars)
    except ValueError as exc:
        raise RuntimeError("LLM_TIMEOUT and LLM_MAX_INPUT_CHARS must be numeric.") from exc

    return Settings(
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        llm_model=os.getenv("LLM_MODEL") or None,
        llm_timeout=parsed_timeout,
        llm_max_input_chars=parsed_max_input_chars,
    )
