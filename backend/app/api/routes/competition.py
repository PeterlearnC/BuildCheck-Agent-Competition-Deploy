"""Same-origin API for the fixed, qualified competition demonstration."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from app.schemas.competition_demo import CompetitionCaseRunResult, CompetitionDemoMetadata
from app.services.competition_demo_service import (
    CompetitionDemoAuthorityDriftError,
    CompetitionDemoCaseNotFoundError,
    CompetitionDemoExecutionError,
    CompetitionDemoInternalError,
    CompetitionDemoNotReadyError,
    CompetitionDemoService,
    get_competition_demo_service,
)


router = APIRouter(prefix="/competition/demo", tags=["competition-demo"])


@router.get("", response_model=CompetitionDemoMetadata)
def get_demo_metadata(
    service: CompetitionDemoService = Depends(get_competition_demo_service),
) -> CompetitionDemoMetadata:
    return service.metadata()


@router.post("/cases/{case_id}/run", response_model=CompetitionCaseRunResult)
async def run_demo_case(
    case_id: str,
    request: Request,
    service: CompetitionDemoService = Depends(get_competition_demo_service),
) -> CompetitionCaseRunResult:
    if await request.body():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="该固定演示接口不接受请求体或权威覆盖字段。",
        )
    try:
        return await run_in_threadpool(service.run_case, case_id)
    except CompetitionDemoCaseNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="未找到该固定局部审查目标。",
        ) from exc
    except (CompetitionDemoNotReadyError, CompetitionDemoAuthorityDriftError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="系统已安全停止，本次未生成局部审查结果。请检查资格化比赛环境。",
        ) from exc
    except CompetitionDemoExecutionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="系统已安全停止，本次未生成局部审查结果。局部证据重建不可用。",
        ) from exc
    except CompetitionDemoInternalError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="系统已安全停止，本次未生成局部审查结果。",
        ) from exc
