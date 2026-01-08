"""
Experience Reuse API Router (Phase 2)

提供经验复用的 REST API 接口。
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from sqlmodel import Session

from ...core.experience_reuse import (
    clone_job_config,
    recommend_successful_recipes,
    get_best_practices,
    suggest_config_adjustments,
    find_similar_successful_jobs,
)
from ...core.database import (
    get_session,
    JobRepository,
)

router = APIRouter(prefix="/experience", tags=["Experience Reuse"])


# ============== Request/Response Models ==============

class CloneJobRequest(BaseModel):
    """克隆任务配置请求"""
    source_job_uuid: str = Field(..., description="源任务 UUID")
    new_name: Optional[str] = Field(None, description="新任务名称")
    new_description: Optional[str] = Field(None, description="新任务描述")
    overrides: Optional[Dict[str, Any]] = Field(None, description="覆盖参数")


class CloneJobResponse(BaseModel):
    """克隆任务配置响应"""
    config: Dict[str, Any]
    source_job_uuid: str
    source_job_name: str


class RecommendRecipesRequest(BaseModel):
    """推荐配方请求"""
    task_type: Optional[str] = Field(None, description="任务类型")
    algorithm: Optional[str] = Field(None, description="算法")
    min_success_count: int = Field(1, description="最少成功次数")
    limit: int = Field(5, description="返回数量")


class RecipeRecommendation(BaseModel):
    """配方推荐"""
    recipe_id: str
    recipe_name: str
    description: str
    algorithm: str
    success_rate: float
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    avg_learning_rate: Optional[float]
    avg_batch_size: Optional[float]


class RecommendRecipesResponse(BaseModel):
    """推荐配方响应"""
    recommendations: List[RecipeRecommendation]


class BestPracticesRequest(BaseModel):
    """最佳实践请求"""
    recipe_id: str = Field(..., description="配方 ID")
    metric: str = Field("reward_mean", description="评估指标")
    top_k: int = Field(3, description="返回前 K 个")


class BestPractice(BaseModel):
    """最佳实践"""
    job_uuid: str
    job_name: str
    metric_value: float
    learning_rate: float
    batch_size: int
    num_epochs: int
    kl_coef: float
    lora_rank: Optional[int]
    config: Dict[str, Any]
    created_at: str


class BestPracticesResponse(BaseModel):
    """最佳实践响应"""
    recipe_id: str
    metric: str
    best_practices: List[BestPractice]


class ConfigSuggestion(BaseModel):
    """配置建议"""
    parameter: str
    current_value: str
    suggested_value: str
    reason: str


class SuggestAdjustmentsRequest(BaseModel):
    """配置调整建议请求"""
    recipe_id: str = Field(..., description="配方 ID")
    current_config: Dict[str, Any] = Field(..., description="当前配置")
    metric: str = Field("reward_mean", description="评估指标")


class SuggestAdjustmentsResponse(BaseModel):
    """配置调整建议响应"""
    suggestions: List[ConfigSuggestion]
    best_practices: List[BestPractice]


class FindSimilarJobsRequest(BaseModel):
    """查找相似任务请求"""
    reference_config: Dict[str, Any] = Field(..., description="参考配置")
    algorithm: Optional[str] = Field(None, description="算法筛选")
    limit: int = Field(5, description="返回数量")


class SimilarJob(BaseModel):
    """相似任务"""
    job_uuid: str
    job_name: str
    algorithm: str
    recipe_id: Optional[str]
    similarity_score: float
    learning_rate: float
    batch_size: int
    kl_coef: float
    created_at: str


class FindSimilarJobsResponse(BaseModel):
    """查找相似任务响应"""
    similar_jobs: List[SimilarJob]


# ============== API Endpoints ==============

@router.post("/clone-job", response_model=CloneJobResponse)
async def clone_job(
    request: CloneJobRequest,
    session: Session = Depends(get_session),
) -> CloneJobResponse:
    """
    从现有任务克隆配置

    复制一个成功的训练任务配置，用于创建新的训练。
    可以指定覆盖参数来调整部分配置。

    Args:
        request: 克隆请求
        session: 数据库会话

    Returns:
        克隆的配置
    """
    repo = JobRepository(session)

    # 获取源任务
    source_job = repo.get_by_uuid(request.source_job_uuid)
    if not source_job:
        raise HTTPException(
            status_code=404,
            detail=f"Source job '{request.source_job_uuid}' not found"
        )

    # 克隆配置
    cloned_config = clone_job_config(
        source_job=source_job,
        overrides=request.overrides,
        new_name=request.new_name,
        new_description=request.new_description,
    )

    return CloneJobResponse(
        config=cloned_config,
        source_job_uuid=source_job.uuid,
        source_job_name=source_job.name,
    )


@router.post("/recommend-recipes", response_model=RecommendRecipesResponse)
async def recommend_recipes(
    request: RecommendRecipesRequest,
    session: Session = Depends(get_session),
) -> RecommendRecipesResponse:
    """
    推荐成功的配方

    基于历史训练任务的成功率推荐配方。
    帮助用户选择经过验证的配方开始新训练。

    Args:
        request: 推荐请求
        session: 数据库会话

    Returns:
        推荐配方列表
    """
    recommendations = recommend_successful_recipes(
        session=session,
        task_type=request.task_type,
        algorithm=request.algorithm,
        min_success_count=request.min_success_count,
        limit=request.limit,
    )

    return RecommendRecipesResponse(
        recommendations=[
            RecipeRecommendation(**rec) for rec in recommendations
        ]
    )


@router.post("/best-practices", response_model=BestPracticesResponse)
async def get_recipe_best_practices(
    request: BestPracticesRequest,
    session: Session = Depends(get_session),
) -> BestPracticesResponse:
    """
    获取配方的最佳实践

    查找使用该配方并且指标表现最好的训练任务。
    帮助用户了解该配方的最优配置。

    Args:
        request: 最佳实践请求
        session: 数据库会话

    Returns:
        最佳实践列表
    """
    best_practices = get_best_practices(
        session=session,
        recipe_id=request.recipe_id,
        metric=request.metric,
        top_k=request.top_k,
    )

    return BestPracticesResponse(
        recipe_id=request.recipe_id,
        metric=request.metric,
        best_practices=[
            BestPractice(**bp) for bp in best_practices
        ],
    )


@router.post("/suggest-adjustments", response_model=SuggestAdjustmentsResponse)
async def suggest_adjustments(
    request: SuggestAdjustmentsRequest,
    session: Session = Depends(get_session),
) -> SuggestAdjustmentsResponse:
    """
    建议配置调整

    对比当前配置与最佳实践，给出调整建议。
    帮助用户优化训练配置。

    Args:
        request: 调整建议请求
        session: 数据库会话

    Returns:
        调整建议列表
    """
    # 获取最佳实践
    best_practices = get_best_practices(
        session=session,
        recipe_id=request.recipe_id,
        metric=request.metric,
        top_k=3,
    )

    # 生成建议
    suggestions = suggest_config_adjustments(
        current_config=request.current_config,
        best_practices=best_practices,
    )

    return SuggestAdjustmentsResponse(
        suggestions=[ConfigSuggestion(**s) for s in suggestions],
        best_practices=[BestPractice(**bp) for bp in best_practices],
    )


@router.post("/find-similar", response_model=FindSimilarJobsResponse)
async def find_similar(
    request: FindSimilarJobsRequest,
    session: Session = Depends(get_session),
) -> FindSimilarJobsResponse:
    """
    查找相似的成功任务

    基于配置相似度查找历史成功任务。
    帮助用户找到参考案例。

    Args:
        request: 查找请求
        session: 数据库会话

    Returns:
        相似任务列表
    """
    similar_jobs = find_similar_successful_jobs(
        session=session,
        reference_config=request.reference_config,
        algorithm=request.algorithm,
        limit=request.limit,
    )

    return FindSimilarJobsResponse(
        similar_jobs=[SimilarJob(**job) for job in similar_jobs]
    )


@router.get("/recommend-recipes", response_model=RecommendRecipesResponse)
async def recommend_recipes_get(
    task_type: Optional[str] = None,
    algorithm: Optional[str] = None,
    min_success_count: int = 1,
    limit: int = 5,
    session: Session = Depends(get_session),
) -> RecommendRecipesResponse:
    """
    推荐成功的配方（GET 方法）

    提供更简洁的 URL 格式用于快速访问。

    Args:
        task_type: 任务类型筛选
        algorithm: 算法筛选
        min_success_count: 最少成功次数
        limit: 返回数量
        session: 数据库会话

    Returns:
        推荐配方列表
    """
    recommendations = recommend_successful_recipes(
        session=session,
        task_type=task_type,
        algorithm=algorithm,
        min_success_count=min_success_count,
        limit=limit,
    )

    return RecommendRecipesResponse(
        recommendations=[
            RecipeRecommendation(**rec) for rec in recommendations
        ]
    )
