"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.analysis import router as analysis_router
from app.api.routes.competition import router as competition_router
from app.api.routes.completeness_review import router as completeness_review_router
from app.api.routes.documents import router as documents_router
from app.api.routes.review import router as review_router
from app.api.routes.standards import router as standards_router
from app.core.config import get_settings
from app.schemas.document import HealthResponse


settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.version)
app.include_router(documents_router, prefix=settings.api_v1_prefix)
app.include_router(analysis_router, prefix=settings.api_v1_prefix)
app.include_router(completeness_review_router, prefix=settings.api_v1_prefix)
app.include_router(review_router, prefix=settings.api_v1_prefix)
app.include_router(standards_router, prefix=settings.api_v1_prefix)
app.include_router(competition_router, prefix=settings.api_v1_prefix)

competition_static_dir = Path(__file__).resolve().parent / "static" / "competition"
app.mount(
    "/competition",
    StaticFiles(directory=competition_static_dir, html=True),
    name="competition-ui",
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.health_version,
    )
