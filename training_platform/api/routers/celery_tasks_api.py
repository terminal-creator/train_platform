"""
Celery Tasks API Router (Phase 3)

提供 Celery 任务管理的 REST API 接口。
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from ...core.celery_config import app
from ...core.celery_tasks import cancel_task, retry_failed_task

router = APIRouter(prefix="/celery-tasks", tags=["Celery Tasks"])


# ============== Request/Response Models ==============

class TaskDetail(BaseModel):
    """Celery 任务详情"""
    task_id: str
    name: Optional[str]
    state: str
    result: Optional[Any]
    traceback: Optional[str]
    args: Optional[List]
    kwargs: Optional[Dict]


class TaskListResponse(BaseModel):
    """任务列表响应"""
    tasks: List[TaskDetail]
    total: int


class CancelTaskRequest(BaseModel):
    """取消任务请求"""
    task_id: str = Field(..., description="Celery 任务 ID")


class RetryTaskRequest(BaseModel):
    """重试任务请求"""
    task_id: str = Field(..., description="Celery 任务 ID")


# ============== API Endpoints ==============

@router.get("/{task_id}", response_model=TaskDetail)
async def get_task_status(task_id: str) -> TaskDetail:
    """
    获取 Celery 任务状态

    Args:
        task_id: Celery 任务 ID

    Returns:
        任务详情
    """
    result = app.AsyncResult(task_id)

    return TaskDetail(
        task_id=task_id,
        name=result.name,
        state=result.state,
        result=result.result if result.successful() else None,
        traceback=result.traceback if result.failed() else None,
        args=result.args if hasattr(result, 'args') else None,
        kwargs=result.kwargs if hasattr(result, 'kwargs') else None,
    )


@router.get("", response_model=TaskListResponse)
async def list_tasks(
    state: Optional[str] = None,
    limit: int = 50,
) -> TaskListResponse:
    """
    列出所有 Celery 任务

    注意：这个接口需要 Celery 配置启用 task events。

    Args:
        state: 状态筛选 (PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED)
        limit: 返回数量限制

    Returns:
        任务列表
    """
    tasks = []

    try:
        # 获取活跃任务，设置超时
        inspect = app.control.inspect(timeout=2.0)

        # 获取各种状态的任务
        active_tasks = inspect.active() or {}

        # 收集所有任务
        for worker, worker_tasks in active_tasks.items():
            for task in worker_tasks[:limit]:
                if state and task.get('type', '').split('.')[-1] != state.lower():
                    continue
                tasks.append(TaskDetail(
                    task_id=task.get('id', ''),
                    name=task.get('name', ''),
                    state='STARTED',
                    result=None,
                    traceback=None,
                    args=task.get('args'),
                    kwargs=task.get('kwargs'),
                ))
    except Exception as e:
        # Celery not available, return empty list
        pass

    return TaskListResponse(
        tasks=tasks,
        total=len(tasks),
    )


@router.post("/{task_id}/cancel")
async def cancel_celery_task(task_id: str) -> Dict[str, Any]:
    """
    取消 Celery 任务

    Args:
        task_id: Celery 任务 ID

    Returns:
        取消结果
    """
    result = cancel_task.delay(task_id)

    return {
        "task_id": task_id,
        "cancel_task_id": result.id,
        "status": "cancelling",
        "message": "Task cancellation requested",
    }


@router.post("/{task_id}/retry")
async def retry_celery_task(task_id: str) -> Dict[str, Any]:
    """
    重试失败的 Celery 任务

    Args:
        task_id: Celery 任务 ID

    Returns:
        重试结果
    """
    result = retry_failed_task.delay(task_id)

    return {
        "task_id": task_id,
        "retry_task_id": result.id,
        "status": "retrying",
        "message": "Task retry requested",
    }


@router.get("/{task_id}/result")
async def get_task_result(task_id: str) -> Dict[str, Any]:
    """
    获取任务执行结果

    Args:
        task_id: Celery 任务 ID

    Returns:
        任务结果
    """
    result = app.AsyncResult(task_id)

    if result.ready():
        if result.successful():
            return {
                "task_id": task_id,
                "state": result.state,
                "result": result.result,
            }
        else:
            return {
                "task_id": task_id,
                "state": result.state,
                "error": str(result.result),
                "traceback": result.traceback,
            }
    else:
        return {
            "task_id": task_id,
            "state": result.state,
            "message": "Task not ready yet",
        }


@router.get("/stats/overview")
async def get_tasks_stats() -> Dict[str, Any]:
    """
    获取任务统计概览

    Returns:
        任务统计信息
    """
    # Default response when Celery is not available
    default_response = {
        "workers": [],
        "worker_count": 0,
        "active_tasks": 0,
        "scheduled_tasks": 0,
        "reserved_tasks": 0,
        "registered_tasks": 0,
        "stats": {},
    }

    try:
        inspect = app.control.inspect(timeout=2.0)

        # 获取各种统计
        stats = inspect.stats() or {}
        active_tasks = inspect.active() or {}
        scheduled_tasks = inspect.scheduled() or {}
        reserved_tasks = inspect.reserved() or {}
        registered_tasks = inspect.registered() or {}

        # 计算总数
        total_active = sum(len(tasks) for tasks in active_tasks.values())
        total_scheduled = sum(len(tasks) for tasks in scheduled_tasks.values())
        total_reserved = sum(len(tasks) for tasks in reserved_tasks.values())

        # 获取 worker 信息
        workers = list(stats.keys()) if stats else []

        return {
            "workers": workers,
            "worker_count": len(workers),
            "active_tasks": total_active,
            "scheduled_tasks": total_scheduled,
            "reserved_tasks": total_reserved,
            "registered_tasks": len(list(registered_tasks.values())[0]) if registered_tasks else 0,
            "stats": stats,
        }
    except Exception as e:
        # Celery not available
        return default_response


@router.post("/purge")
async def purge_tasks() -> Dict[str, Any]:
    """
    清除所有待处理的任务

    警告：这会删除队列中所有等待执行的任务！

    Returns:
        清除结果
    """
    purged = app.control.purge()

    return {
        "status": "purged",
        "tasks_purged": purged,
        "message": f"Purged {purged} tasks from queue",
    }
