"""
Pipelines API Router (Phase 3)

提供训练流水线管理的 REST API 接口。
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from sqlmodel import Session
import uuid as uuid_lib

from ...core.database import (
    get_session,
    Pipeline,
    PipelineStage,
    PipelineStatus,
    PipelineStageStatus,
    PipelineRepository,
)
from ...core.celery_tasks import run_training_pipeline, cancel_task

router = APIRouter(prefix="/pipelines", tags=["Pipelines"])


# ============== Request/Response Models ==============

class StageConfig(BaseModel):
    """Pipeline stage configuration"""
    name: str = Field(..., description="阶段名称")
    task: str = Field(..., description="Celery 任务名")
    params: Dict[str, Any] = Field(default_factory=dict, description="任务参数")
    depends_on: List[str] = Field(default_factory=list, description="依赖的阶段")
    max_retries: int = Field(3, description="最大重试次数")
    retry_delay: int = Field(60, description="重试延迟（秒）")


class CreatePipelineRequest(BaseModel):
    """创建 Pipeline 请求"""
    name: str = Field(..., description="Pipeline 名称")
    description: Optional[str] = Field(None, description="描述")
    stages: List[StageConfig] = Field(..., description="阶段配置")
    priority: int = Field(5, ge=1, le=10, description="优先级 (1-10)")
    max_retries: int = Field(3, description="Pipeline 最大重试次数")


class PipelineDetail(BaseModel):
    """Pipeline 详情"""
    uuid: str
    name: str
    description: Optional[str]
    status: str
    celery_task_id: Optional[str]
    stage_tasks: Dict[str, str]
    results: Dict[str, Any]
    priority: int
    max_retries: int
    retry_count: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]


class PipelineListResponse(BaseModel):
    """Pipeline 列表响应"""
    pipelines: List[PipelineDetail]
    total: int


class StageDetail(BaseModel):
    """Stage 详情"""
    stage_name: str
    stage_order: int
    task_name: str
    status: str
    celery_task_id: Optional[str]
    depends_on: List[str]
    result: Dict[str, Any]
    error_message: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


class PipelineStatusResponse(BaseModel):
    """Pipeline 状态响应"""
    pipeline: PipelineDetail
    stages: List[StageDetail]


# ============== API Endpoints ==============

@router.post("", response_model=PipelineDetail)
async def create_pipeline(
    request: CreatePipelineRequest,
    session: Session = Depends(get_session),
) -> PipelineDetail:
    """
    创建新的 Pipeline

    创建一个多阶段训练流水线。Pipeline 创建后处于 PENDING 状态，
    需要调用 start 接口启动执行。

    Args:
        request: Pipeline 创建请求
        session: 数据库会话

    Returns:
        创建的 Pipeline 详情
    """
    repo = PipelineRepository(session)

    # 生成 UUID
    pipeline_uuid = str(uuid_lib.uuid4())

    # 创建 Pipeline
    pipeline = Pipeline(
        uuid=pipeline_uuid,
        name=request.name,
        description=request.description,
        stages=[stage.dict() for stage in request.stages],
        status=PipelineStatus.PENDING,
        priority=request.priority,
        max_retries=request.max_retries,
    )

    pipeline = repo.create(pipeline)

    # 创建 Stages
    for idx, stage_config in enumerate(request.stages):
        stage = PipelineStage(
            pipeline_uuid=pipeline_uuid,
            stage_name=stage_config.name,
            stage_order=idx,
            task_name=stage_config.task,
            task_params=stage_config.params,
            depends_on=stage_config.depends_on,
            max_retries=stage_config.max_retries,
            retry_delay=stage_config.retry_delay,
        )
        repo.create_stage(stage)

    return PipelineDetail(
        uuid=pipeline.uuid,
        name=pipeline.name,
        description=pipeline.description,
        status=pipeline.status.value,
        celery_task_id=pipeline.celery_task_id,
        stage_tasks=pipeline.stage_tasks,
        results=pipeline.results,
        priority=pipeline.priority,
        max_retries=pipeline.max_retries,
        retry_count=pipeline.retry_count,
        created_at=pipeline.created_at.isoformat(),
        started_at=pipeline.started_at.isoformat() if pipeline.started_at else None,
        completed_at=pipeline.completed_at.isoformat() if pipeline.completed_at else None,
        error_message=pipeline.error_message,
    )


@router.get("", response_model=PipelineListResponse)
async def list_pipelines(
    status: Optional[str] = None,
    offset: int = 0,
    limit: int = 20,
    session: Session = Depends(get_session),
) -> PipelineListResponse:
    """
    列出所有 Pipelines

    可以按状态筛选 Pipelines。

    Args:
        status: 状态筛选 (pending, running, completed, failed, cancelled)
        offset: 分页偏移
        limit: 分页大小
        session: 数据库会话

    Returns:
        Pipeline 列表
    """
    repo = PipelineRepository(session)

    # 转换状态
    status_enum = None
    if status:
        try:
            status_enum = PipelineStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    pipelines, total = repo.list_pipelines(
        status=status_enum,
        offset=offset,
        limit=limit,
    )

    return PipelineListResponse(
        pipelines=[
            PipelineDetail(
                uuid=p.uuid,
                name=p.name,
                description=p.description,
                status=p.status.value,
                celery_task_id=p.celery_task_id,
                stage_tasks=p.stage_tasks,
                results=p.results,
                priority=p.priority,
                max_retries=p.max_retries,
                retry_count=p.retry_count,
                created_at=p.created_at.isoformat(),
                started_at=p.started_at.isoformat() if p.started_at else None,
                completed_at=p.completed_at.isoformat() if p.completed_at else None,
                error_message=p.error_message,
            )
            for p in pipelines
        ],
        total=total,
    )


@router.get("/{pipeline_uuid}", response_model=PipelineDetail)
async def get_pipeline(
    pipeline_uuid: str,
    session: Session = Depends(get_session),
) -> PipelineDetail:
    """
    获取 Pipeline 详情

    Args:
        pipeline_uuid: Pipeline UUID
        session: 数据库会话

    Returns:
        Pipeline 详情
    """
    repo = PipelineRepository(session)
    pipeline = repo.get_by_uuid(pipeline_uuid)

    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_uuid} not found")

    return PipelineDetail(
        uuid=pipeline.uuid,
        name=pipeline.name,
        description=pipeline.description,
        status=pipeline.status.value,
        celery_task_id=pipeline.celery_task_id,
        stage_tasks=pipeline.stage_tasks,
        results=pipeline.results,
        priority=pipeline.priority,
        max_retries=pipeline.max_retries,
        retry_count=pipeline.retry_count,
        created_at=pipeline.created_at.isoformat(),
        started_at=pipeline.started_at.isoformat() if pipeline.started_at else None,
        completed_at=pipeline.completed_at.isoformat() if pipeline.completed_at else None,
        error_message=pipeline.error_message,
    )


@router.get("/{pipeline_uuid}/status", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    pipeline_uuid: str,
    session: Session = Depends(get_session),
) -> PipelineStatusResponse:
    """
    获取 Pipeline 状态（包括所有阶段）

    Args:
        pipeline_uuid: Pipeline UUID
        session: 数据库会话

    Returns:
        Pipeline 和所有阶段的状态
    """
    repo = PipelineRepository(session)

    # 获取 Pipeline
    pipeline = repo.get_by_uuid(pipeline_uuid)
    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_uuid} not found")

    # 获取所有阶段
    stages = repo.get_stages(pipeline_uuid)

    return PipelineStatusResponse(
        pipeline=PipelineDetail(
            uuid=pipeline.uuid,
            name=pipeline.name,
            description=pipeline.description,
            status=pipeline.status.value,
            celery_task_id=pipeline.celery_task_id,
            stage_tasks=pipeline.stage_tasks,
            results=pipeline.results,
            priority=pipeline.priority,
            max_retries=pipeline.max_retries,
            retry_count=pipeline.retry_count,
            created_at=pipeline.created_at.isoformat(),
            started_at=pipeline.started_at.isoformat() if pipeline.started_at else None,
            completed_at=pipeline.completed_at.isoformat() if pipeline.completed_at else None,
            error_message=pipeline.error_message,
        ),
        stages=[
            StageDetail(
                stage_name=s.stage_name,
                stage_order=s.stage_order,
                task_name=s.task_name,
                status=s.status.value,
                celery_task_id=s.celery_task_id,
                depends_on=s.depends_on,
                result=s.result,
                error_message=s.error_message,
                created_at=s.created_at.isoformat(),
                started_at=s.started_at.isoformat() if s.started_at else None,
                completed_at=s.completed_at.isoformat() if s.completed_at else None,
            )
            for s in stages
        ],
    )


@router.post("/{pipeline_uuid}/start")
async def start_pipeline(
    pipeline_uuid: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    启动 Pipeline 执行

    Args:
        pipeline_uuid: Pipeline UUID
        session: 数据库会话

    Returns:
        启动结果
    """
    repo = PipelineRepository(session)
    pipeline = repo.get_by_uuid(pipeline_uuid)

    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_uuid} not found")

    if pipeline.status != PipelineStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline is not in PENDING status (current: {pipeline.status})"
        )

    # 构建 pipeline_config
    stages = repo.get_stages(pipeline_uuid)
    pipeline_config = {
        "pipeline_uuid": pipeline_uuid,
        "stages": [
            {
                "name": s.stage_name,
                "task": s.task_name,
                "params": s.task_params,
                "depends_on": s.depends_on,
            }
            for s in stages
        ],
    }

    # 提交到 Celery
    from datetime import datetime
    result = run_training_pipeline.delay(pipeline_config)

    # 更新 Pipeline 状态
    pipeline.status = PipelineStatus.RUNNING
    pipeline.celery_task_id = result.id
    pipeline.started_at = datetime.utcnow()
    repo.update(pipeline)

    return {
        "pipeline_uuid": pipeline_uuid,
        "celery_task_id": result.id,
        "status": "started",
    }


@router.post("/{pipeline_uuid}/cancel")
async def cancel_pipeline(
    pipeline_uuid: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    取消 Pipeline 执行

    Args:
        pipeline_uuid: Pipeline UUID
        session: 数据库会话

    Returns:
        取消结果
    """
    repo = PipelineRepository(session)
    pipeline = repo.get_by_uuid(pipeline_uuid)

    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_uuid} not found")

    if pipeline.status not in [PipelineStatus.PENDING, PipelineStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline cannot be cancelled (current status: {pipeline.status})"
        )

    # 取消 Celery 任务
    if pipeline.celery_task_id:
        cancel_task.delay(pipeline.celery_task_id)

    # 取消所有阶段任务
    stages = repo.get_stages(pipeline_uuid)
    for stage in stages:
        if stage.celery_task_id and stage.status == PipelineStageStatus.RUNNING:
            cancel_task.delay(stage.celery_task_id)
            stage.status = PipelineStageStatus.SKIPPED
            repo.update_stage(stage)

    # 更新 Pipeline 状态
    from datetime import datetime
    pipeline.status = PipelineStatus.CANCELLED
    pipeline.completed_at = datetime.utcnow()
    repo.update(pipeline)

    return {
        "pipeline_uuid": pipeline_uuid,
        "status": "cancelled",
    }


@router.delete("/{pipeline_uuid}")
async def delete_pipeline(
    pipeline_uuid: str,
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    删除 Pipeline

    只能删除已完成、失败或取消的 Pipeline。

    Args:
        pipeline_uuid: Pipeline UUID
        session: 数据库会话

    Returns:
        删除结果
    """
    repo = PipelineRepository(session)
    pipeline = repo.get_by_uuid(pipeline_uuid)

    if not pipeline:
        raise HTTPException(status_code=404, detail=f"Pipeline {pipeline_uuid} not found")

    if pipeline.status in [PipelineStatus.PENDING, PipelineStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete Pipeline in {pipeline.status} status. Cancel it first."
        )

    # 删除 Pipeline（级联删除 stages）
    success = repo.delete(pipeline_uuid)

    if success:
        return {
            "pipeline_uuid": pipeline_uuid,
            "status": "deleted",
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to delete Pipeline")
