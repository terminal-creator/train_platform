"""
Config Diff API Router (Phase 2)

提供配置对比的 REST API 接口。
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from sqlmodel import Session

from ...core.config_diff import (
    compare_configs,
    compare_recipes,
    compare_jobs,
    format_diff_report,
    comparison_result_to_dict,
    ConfigComparisonResult,
)
from ...core.database import get_session

router = APIRouter(prefix="/config-diff", tags=["Config Diff"])


# ============== Request/Response Models ==============

class CompareConfigsRequest(BaseModel):
    """对比两个配置请求"""
    config_a: Dict[str, Any] = Field(..., description="配置 A")
    config_b: Dict[str, Any] = Field(..., description="配置 B")
    name_a: str = Field(default="Config A", description="配置 A 的名称")
    name_b: str = Field(default="Config B", description="配置 B 的名称")


class CompareRecipesRequest(BaseModel):
    """对比两个配方请求"""
    recipe_id_a: str = Field(..., description="配方 A 的 ID")
    recipe_id_b: str = Field(..., description="配方 B 的 ID")


class CompareJobsRequest(BaseModel):
    """对比两个训练任务请求"""
    job_uuid_a: str = Field(..., description="任务 A 的 UUID")
    job_uuid_b: str = Field(..., description="任务 B 的 UUID")


class ConfigComparisonResponse(BaseModel):
    """配置对比响应"""
    diffs: List[Dict[str, Any]]
    added_count: int
    removed_count: int
    modified_count: int
    unchanged_count: int
    has_critical_changes: bool
    summary: str
    report: Optional[str] = None  # 可选的文本报告


# ============== API Endpoints ==============

@router.post("/compare", response_model=ConfigComparisonResponse)
async def compare_two_configs(request: CompareConfigsRequest) -> ConfigComparisonResponse:
    """
    对比两个配置字典

    这是通用的配置对比接口，可以对比任意两个配置字典。

    Args:
        request: 对比请求

    Returns:
        对比结果
    """
    result = compare_configs(
        config_a=request.config_a,
        config_b=request.config_b,
        name_a=request.name_a,
        name_b=request.name_b,
    )

    # 生成文本报告
    report = format_diff_report(result)

    response_dict = comparison_result_to_dict(result)
    response_dict["report"] = report

    return ConfigComparisonResponse(**response_dict)


@router.post("/compare/recipes", response_model=ConfigComparisonResponse)
async def compare_recipe_configs(request: CompareRecipesRequest) -> ConfigComparisonResponse:
    """
    对比两个配方的默认配置

    用于查看不同配方之间的配置差异。

    Args:
        request: 对比请求

    Returns:
        对比结果
    """
    result = compare_recipes(
        recipe_id_a=request.recipe_id_a,
        recipe_id_b=request.recipe_id_b,
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"One or both recipes not found: {request.recipe_id_a}, {request.recipe_id_b}"
        )

    # 生成文本报告
    report = format_diff_report(result)

    response_dict = comparison_result_to_dict(result)
    response_dict["report"] = report

    return ConfigComparisonResponse(**response_dict)


@router.post("/compare/jobs", response_model=ConfigComparisonResponse)
async def compare_job_configs(
    request: CompareJobsRequest,
    session: Session = Depends(get_session),
) -> ConfigComparisonResponse:
    """
    对比两个训练任务的配置

    用于对比实验（Experiment Comparison）。
    查看两个训练任务的配置差异，帮助理解不同实验的区别。

    Args:
        request: 对比请求
        session: 数据库会话

    Returns:
        对比结果
    """
    result = compare_jobs(
        job_uuid_a=request.job_uuid_a,
        job_uuid_b=request.job_uuid_b,
        session=session,
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"One or both jobs not found: {request.job_uuid_a}, {request.job_uuid_b}"
        )

    # 生成文本报告
    report = format_diff_report(result)

    response_dict = comparison_result_to_dict(result)
    response_dict["report"] = report

    return ConfigComparisonResponse(**response_dict)


@router.get("/compare/jobs/{job_uuid_a}/vs/{job_uuid_b}", response_model=ConfigComparisonResponse)
async def compare_jobs_by_uuid(
    job_uuid_a: str,
    job_uuid_b: str,
    session: Session = Depends(get_session),
) -> ConfigComparisonResponse:
    """
    对比两个训练任务的配置（GET 方法）

    提供更简洁的 URL 格式用于直接访问和分享。

    Args:
        job_uuid_a: 任务 A 的 UUID
        job_uuid_b: 任务 B 的 UUID
        session: 数据库会话

    Returns:
        对比结果
    """
    result = compare_jobs(
        job_uuid_a=job_uuid_a,
        job_uuid_b=job_uuid_b,
        session=session,
    )

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"One or both jobs not found: {job_uuid_a}, {job_uuid_b}"
        )

    # 生成文本报告
    report = format_diff_report(result)

    response_dict = comparison_result_to_dict(result)
    response_dict["report"] = report

    return ConfigComparisonResponse(**response_dict)
