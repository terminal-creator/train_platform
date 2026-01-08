"""
Pipeline DAG Executor - 真正的可恢复异步编排

实现特性：
1. 每个 stage 都是独立的 Celery task
2. 真正的依赖关系解析和执行（支持线性 + 简单依赖）
3. 每个 stage 的 task_id 记录到 DB
4. 支持 Pipeline 失败恢复
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from celery import signature, chain, group, chord
from celery.result import AsyncResult

from .celery_config import app
from .database import (
    engine,
    Session,
    PipelineRepository,
    PipelineStatus,
    PipelineStageStatus,
)

logger = logging.getLogger(__name__)


class DagExecutionError(Exception):
    """DAG execution error"""
    pass


@dataclass
class StageNode:
    """DAG 中的 stage 节点"""
    name: str
    task_name: str
    params: Dict[str, Any]
    depends_on: List[str]

    # Runtime state
    task_id: Optional[str] = None
    status: Optional[PipelineStageStatus] = None
    result: Optional[Dict[str, Any]] = None


class DagResolver:
    """
    DAG 依赖关系解析器

    支持：
    1. 线性依赖（A -> B -> C）
    2. 简单并行（A -> [B, C] -> D）
    3. 循环检测
    """

    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages
        self.nodes: Dict[str, StageNode] = {}
        self._build_nodes()

    def _build_nodes(self):
        """构建 stage 节点"""
        for stage in self.stages:
            node = StageNode(
                name=stage["name"],
                task_name=stage["task"],
                params=stage.get("params", {}),
                depends_on=stage.get("depends_on", []),
            )
            self.nodes[node.name] = node

    def validate(self) -> bool:
        """
        验证 DAG 有效性

        检查：
        1. 所有依赖的 stage 都存在
        2. 没有循环依赖
        """
        # Check all dependencies exist
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise DagExecutionError(
                        f"Stage '{node.name}' depends on non-existent stage '{dep}'"
                    )

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node_name: str) -> bool:
            visited.add(node_name)
            rec_stack.add(node_name)

            node = self.nodes[node_name]
            for dep in node.depends_on:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node_name)
            return False

        for node_name in self.nodes:
            if node_name not in visited:
                if has_cycle(node_name):
                    raise DagExecutionError("Cyclic dependency detected in pipeline")

        return True

    def get_execution_layers(self) -> List[List[str]]:
        """
        获取执行层级（拓扑排序）

        Returns:
            List[List[str]]: 每一层可以并行执行的 stage names

        Example:
            Input: A -> B, A -> C, B -> D, C -> D
            Output: [[A], [B, C], [D]]
        """
        # Calculate in-degree for each node
        in_degree = {name: len(node.depends_on) for name, node in self.nodes.items()}

        layers = []
        remaining = set(self.nodes.keys())

        while remaining:
            # Find nodes with no dependencies in current layer
            current_layer = [
                name for name in remaining
                if in_degree[name] == 0
            ]

            if not current_layer:
                raise DagExecutionError("Cannot resolve dependencies (possible cycle)")

            layers.append(current_layer)

            # Remove current layer and update in-degrees
            for name in current_layer:
                remaining.remove(name)

                # Decrease in-degree for nodes that depend on current node
                for other_name in remaining:
                    other_node = self.nodes[other_name]
                    if name in other_node.depends_on:
                        in_degree[other_name] -= 1

        return layers


class PipelineExecutor:
    """
    Pipeline DAG 执行器

    负责：
    1. 解析依赖关系
    2. 构建 Celery Canvas (chain/group/chord)
    3. 提交异步任务并记录 task_id
    4. 监控执行状态
    """

    # Mapping of task names to Celery task signatures and queues
    TASK_REGISTRY = {
        "preprocess_dataset": {
            "task": "training_platform.core.celery_tasks.preprocess_dataset",
            "queue": "preprocessing",
        },
        "train_model": {
            "task": "training_platform.core.celery_tasks.train_model",
            "queue": "training",
        },
        "run_evaluation": {
            "task": "training_platform.core.celery_tasks.run_evaluation",
            "queue": "evaluation",
        },
        "cleanup_checkpoints": {
            "task": "training_platform.core.celery_tasks.cleanup_checkpoints",
            "queue": "maintenance",
        },
    }

    def __init__(self, pipeline_uuid: str):
        self.pipeline_uuid = pipeline_uuid
        self.resolver: Optional[DagResolver] = None

    def execute(self, stages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        执行 Pipeline DAG

        Args:
            stages: Stage 配置列表，每个包含 name, task, params, depends_on

        Returns:
            Dict with execution info
        """
        logger.info(f"Starting DAG execution for pipeline {self.pipeline_uuid}")

        # 1. Validate and resolve DAG
        self.resolver = DagResolver(stages)
        self.resolver.validate()

        # 2. Get execution layers
        layers = self.resolver.get_execution_layers()
        logger.info(f"Execution plan: {len(layers)} layers")
        for i, layer in enumerate(layers):
            logger.info(f"  Layer {i}: {layer}")

        # 3. Build Celery Canvas
        canvas = self._build_canvas(layers)

        # 4. Submit pipeline
        result = canvas.apply_async()

        # 5. Update DB with root task_id
        with Session(engine) as session:
            repo = PipelineRepository(session)
            pipeline = repo.get_by_uuid(self.pipeline_uuid)
            if pipeline:
                pipeline.celery_task_id = result.id
                pipeline.status = PipelineStatus.RUNNING
                pipeline.started_at = datetime.utcnow()
                repo.update(pipeline)

        logger.info(f"Pipeline submitted with root task_id: {result.id}")

        return {
            "success": True,
            "pipeline_uuid": self.pipeline_uuid,
            "root_task_id": result.id,
            "layers": len(layers),
        }

    def _build_canvas(self, layers: List[List[str]]) -> Any:
        """
        构建 Celery Canvas

        根据执行层级构建 chain 和 group：
        - 单个 stage：直接调用
        - 同层多个 stage：使用 group 并行
        - 跨层：使用 chain 串联

        Args:
            layers: 执行层级

        Returns:
            Celery Canvas (chain/group)
        """
        layer_tasks = []

        for layer in layers:
            if len(layer) == 1:
                # Single task
                stage_name = layer[0]
                task_sig = self._create_stage_task(stage_name)
                layer_tasks.append(task_sig)
            else:
                # Multiple tasks in parallel (group)
                parallel_tasks = [
                    self._create_stage_task(stage_name)
                    for stage_name in layer
                ]
                layer_tasks.append(group(*parallel_tasks))

        # Chain all layers together
        if len(layer_tasks) == 1:
            return layer_tasks[0]
        else:
            return chain(*layer_tasks)

    def _create_stage_task(self, stage_name: str) -> signature:
        """
        创建 stage 的 Celery signature（真正的异步派发）

        关键改进（生产级）：
        1. Stage = 真实的 Celery task（不是同步调用）
        2. 使用独立队列（training/evaluation/preprocessing 分离）
        3. 使用 link/link_error 回调更新 DB 状态
        4. 支持 stage 粒度的 kill/timeout/retry

        Args:
            stage_name: Stage name

        Returns:
            Celery signature with callbacks
        """
        node = self.resolver.nodes[stage_name]

        # Get Celery task config
        task_config = self.TASK_REGISTRY.get(node.task_name)
        if not task_config:
            raise DagExecutionError(f"Unknown task type: {node.task_name}")

        celery_task_name = task_config["task"]
        task_queue = task_config.get("queue", "default")

        # Create signature for the actual task
        # 这是真正的异步 task，会被派发到专用队列
        # 注意：使用 .si() (immutable) 避免 chain 传递前序结果作为参数
        from celery import signature as sig

        # 在 params 中注入 pipeline_uuid 和 stage_name
        # 这样 task 可以在开始时调用 mark_stage_running
        task_params = dict(node.params)
        task_params['_pipeline_uuid'] = self.pipeline_uuid
        task_params['_stage_name'] = stage_name

        task_sig = sig(
            celery_task_name,
            kwargs=task_params,
            immutable=True,  # ✅ 关键：避免接收 chain 前序结果
        ).set(
            queue=task_queue,  # ✅ 队列隔离
            link=sig(
                "training_platform.core.pipeline_executor.on_stage_success",
                args=(self.pipeline_uuid, stage_name),
                # 不设置 immutable，让它接收 task 的 result
            ),
            link_error=sig(
                "training_platform.core.pipeline_executor.on_stage_error",
                args=(self.pipeline_uuid, stage_name),
                # errback 不需要 immutable
            ),
        )

        # 在 task 派发前记录 stage 状态（使用独立 task）
        init_stage_sig = sig(
            "training_platform.core.pipeline_executor.init_stage_status",
            args=(self.pipeline_uuid, stage_name),
            immutable=True,  # ✅ 关键：避免跨 layer 的 chain/group 结果注入
        )

        # 组合：先初始化状态，再执行实际 task
        # 两个都是 immutable，彻底阻止参数传递
        return chain(init_stage_sig, task_sig)

    def resume(self) -> Dict[str, Any]:
        """
        恢复中断的 Pipeline 执行

        核心逻辑（真正的恢复，不是重跑）：
        1. 从 DB 读取所有 stages 及其状态
        2. 构建 completed_stages 集合
        3. 从原始 stages 配置中过滤掉已完成的
        4. 调整依赖关系（移除已完成的依赖）
        5. 重新执行剩余 stages

        Returns:
            Dict with resume info
        """
        logger.info(f"[Resume] Pipeline {self.pipeline_uuid}")

        with Session(engine) as session:
            repo = PipelineRepository(session)
            pipeline = repo.get_by_uuid(self.pipeline_uuid)

            if not pipeline:
                raise DagExecutionError(f"Pipeline {self.pipeline_uuid} not found")

            if pipeline.status not in [PipelineStatus.FAILED, PipelineStatus.RUNNING]:
                return {
                    "success": False,
                    "error": f"Pipeline status is {pipeline.status}, cannot resume"
                }

            # 1. 获取所有 stages 及其状态
            db_stages = repo.get_stages(self.pipeline_uuid)

            # 2. 找出已完成的 stages（只有 COMPLETED 算完成）
            completed_stages = {
                stage.stage_name
                for stage in db_stages
                if stage.status == PipelineStageStatus.COMPLETED
            }

            logger.info(f"[Resume] Completed stages: {completed_stages}")

            # 3. 重建 stages 配置（从 DB 中读取）
            # 注意：这里我们从 DB 的 task_name 和 task_params 字段读取
            all_stages = []
            for db_stage in db_stages:
                stage_config = {
                    "name": db_stage.stage_name,
                    "task": db_stage.task_name,  # 从 DB 读取
                    "params": db_stage.task_params,  # 从 DB 读取
                    "depends_on": db_stage.depends_on,  # 从 DB 读取
                }
                all_stages.append(stage_config)

            # 4. 过滤出需要执行的 stages
            remaining_stages = [
                stage for stage in all_stages
                if stage["name"] not in completed_stages
            ]

            if not remaining_stages:
                # 所有 stages 都完成了
                logger.info(f"[Resume] All stages completed, marking pipeline as COMPLETED")

                pipeline.status = PipelineStatus.COMPLETED
                pipeline.completed_at = datetime.utcnow()
                repo.update(pipeline)

                return {
                    "success": True,
                    "message": "All stages already completed",
                    "completed_stages": len(completed_stages),
                    "remaining_stages": 0,
                }

            # 5. 调整依赖关系（移除已完成的依赖）
            for stage in remaining_stages:
                stage["depends_on"] = [
                    dep for dep in stage.get("depends_on", [])
                    if dep not in completed_stages
                ]

            logger.info(f"[Resume] Remaining stages: {[s['name'] for s in remaining_stages]}")

            # 6. 重置 pipeline 状态为 RUNNING
            pipeline.status = PipelineStatus.RUNNING
            repo.update(pipeline)

            # 7. 重新执行剩余 stages
            return self.execute(remaining_stages)


# ============== Stage Status Tracking ==============

def mark_stage_running(pipeline_uuid: str, stage_name: str, task_id: str):
    """
    标记 stage 为 RUNNING（由 task 自己调用）

    这个函数在每个 stage task 开始执行时调用，记录：
    1. celery_task_id (真实的 task_id)
    2. status = RUNNING
    3. started_at

    Args:
        pipeline_uuid: Pipeline UUID
        stage_name: Stage name
        task_id: Celery task ID (from self.request.id)

    使用方式：
    ```python
    @app.task(bind=True)
    def train_model(self, **kwargs):
        # 从 kwargs 中提取 pipeline_uuid 和 stage_name
        pipeline_uuid = kwargs.pop('_pipeline_uuid', None)
        stage_name = kwargs.pop('_stage_name', None)

        if pipeline_uuid and stage_name:
            mark_stage_running(pipeline_uuid, stage_name, self.request.id)

        # 执行实际的训练逻辑
        ...
    ```
    """
    logger.info(f"[Pipeline {pipeline_uuid}] Stage '{stage_name}' starting (task_id={task_id})")

    try:
        with Session(engine) as session:
            repo = PipelineRepository(session)
            stages = repo.get_stages(pipeline_uuid)

            stage = next((s for s in stages if s.stage_name == stage_name), None)
            if not stage:
                logger.warning(f"Stage {stage_name} not found in pipeline {pipeline_uuid}")
                return

            # 记录真实的 task_id 和状态
            stage.celery_task_id = task_id
            stage.status = PipelineStageStatus.RUNNING
            stage.started_at = datetime.utcnow()
            repo.update_stage(stage)

        logger.info(f"[Stage {stage_name}] Marked as RUNNING with task_id={task_id}")

    except Exception as e:
        logger.error(f"Failed to mark stage as running: {e}", exc_info=True)


# Callback tasks for stage status tracking
@app.task(name="training_platform.core.pipeline_executor.init_stage_status")
def init_stage_status(pipeline_uuid: str, stage_name: str):
    """
    在 stage 执行前初始化状态

    这个 task 在实际训练 task 之前执行（通过 chain 串联）
    此时还无法获取真实训练 task 的 task_id，但可以标记 stage 为 PENDING

    Args:
        pipeline_uuid: Pipeline UUID
        stage_name: Stage name
    """
    logger.info(f"[Pipeline {pipeline_uuid}] Initializing stage '{stage_name}'")

    with Session(engine) as session:
        repo = PipelineRepository(session)
        stages = repo.get_stages(pipeline_uuid)

        stage = next((s for s in stages if s.stage_name == stage_name), None)
        if stage:
            stage.status = PipelineStageStatus.PENDING
            repo.update_stage(stage)

    logger.info(f"[Stage {stage_name}] Status set to PENDING")


@app.task(name="training_platform.core.pipeline_executor.on_stage_success")
def on_stage_success(result, pipeline_uuid: str, stage_name: str):
    """
    Stage 成功回调（link callback）

    Celery link callback 会自动传递：
    1. result: 前一个 task 的返回值
    2. 我们手动传递的 args: (pipeline_uuid, stage_name)

    注意：
    - 不使用 bind=True，因为我们不需要 self.request
    - stage 的 celery_task_id 已经在 task 执行前记录（通过 mark_stage_running）
    - 这里只负责更新状态为 COMPLETED 和记录结果

    Args:
        result: 前一个 task 的返回值（训练结果）
        pipeline_uuid: Pipeline UUID
        stage_name: Stage name
    """
    logger.info(f"[Pipeline {pipeline_uuid}] Stage '{stage_name}' succeeded")

    try:
        with Session(engine) as session:
            repo = PipelineRepository(session)
            stages = repo.get_stages(pipeline_uuid)

            stage = next((s for s in stages if s.stage_name == stage_name), None)
            if not stage:
                logger.error(f"Stage {stage_name} not found in DB")
                return

            # 只更新状态和结果，task_id 已经在 task 开始时记录
            stage.status = PipelineStageStatus.COMPLETED
            stage.completed_at = datetime.utcnow()
            stage.result = result if isinstance(result, dict) else {"value": str(result)}
            repo.update_stage(stage)

        logger.info(f"[Stage {stage_name}] Marked as COMPLETED")

    except Exception as e:
        logger.error(f"Failed to update stage success status: {e}", exc_info=True)


@app.task(name="training_platform.core.pipeline_executor.on_stage_error")
def on_stage_error(request, exc, traceback, pipeline_uuid: str, stage_name: str):
    """
    Stage 失败回调（link_error callback / errback）

    Celery 5.x errback 的标准签名：
    def errback(request, exc, traceback, *args)

    Args:
        request: Celery request 对象（包含 task_id 等信息）
        exc: 异常对象
        traceback: Traceback 字符串
        pipeline_uuid: Pipeline UUID (我们手动传递的 args)
        stage_name: Stage name (我们手动传递的 args)

    注意：
    - 不要假设第一个参数是 uuid，而是 request 对象
    - 从 request.id 或 request.task_id 获取真实 task_id
    - 直接使用 exc 获取异常信息，不需要 AsyncResult
    """
    # 获取真实的 task_id
    task_id = getattr(request, "id", None) or getattr(request, "task_id", None) or "unknown"

    # 获取异常信息
    error_message = str(exc) if exc else "Unknown error"

    logger.error(
        f"[Pipeline {pipeline_uuid}] Stage '{stage_name}' failed "
        f"(task_id={task_id}, error={error_message})"
    )

    try:

        with Session(engine) as session:
            repo = PipelineRepository(session)
            stages = repo.get_stages(pipeline_uuid)

            stage = next((s for s in stages if s.stage_name == stage_name), None)
            if stage:
                # task_id 已经在 task 开始时记录，这里只更新失败状态
                stage.status = PipelineStageStatus.FAILED
                stage.completed_at = datetime.utcnow()
                stage.error_message = error_message
                repo.update_stage(stage)

        # 标记整个 pipeline 为 FAILED（使用原子性更新，避免竞态）
        with Session(engine) as session:
            repo = PipelineRepository(session)
            success = repo.update_pipeline_status_atomic(
                pipeline_uuid=pipeline_uuid,
                new_status=PipelineStatus.FAILED,
                error_message=f"Stage {stage_name} failed: {error_message}",
                allowed_current_statuses=[PipelineStatus.RUNNING],  # ✅ 只有 RUNNING 才能变 FAILED
            )

            if success:
                logger.info(f"[Pipeline {pipeline_uuid}] Marked as FAILED due to stage {stage_name}")
            else:
                logger.warning(
                    f"[Pipeline {pipeline_uuid}] Already marked as FAILED by another stage, "
                    f"stage {stage_name} failure recorded"
                )

    except Exception as e:
        logger.error(f"Failed to update stage error status: {e}", exc_info=True)


def get_pipeline_status(pipeline_uuid: str) -> Dict[str, Any]:
    """
    获取 Pipeline 的实时状态

    Args:
        pipeline_uuid: Pipeline UUID

    Returns:
        Dict with pipeline and stage statuses
    """
    with Session(engine) as session:
        repo = PipelineRepository(session)
        pipeline = repo.get_by_uuid(pipeline_uuid)

        if not pipeline:
            return {"error": "Pipeline not found"}

        stages = repo.get_stages(pipeline_uuid)

        # Get Celery task status if running
        task_status = None
        if pipeline.celery_task_id:
            async_result = AsyncResult(pipeline.celery_task_id, app=app)
            task_status = {
                "state": async_result.state,
                "info": str(async_result.info) if async_result.info else None,
            }

        return {
            "pipeline_uuid": pipeline_uuid,
            "status": pipeline.status.value,
            "celery_task_id": pipeline.celery_task_id,
            "task_status": task_status,
            "started_at": pipeline.started_at.isoformat() if pipeline.started_at else None,
            "completed_at": pipeline.completed_at.isoformat() if pipeline.completed_at else None,
            "stages": [
                {
                    "name": stage.stage_name,
                    "status": stage.status.value,
                    "celery_task_id": stage.celery_task_id,
                    "started_at": stage.started_at.isoformat() if stage.started_at else None,
                    "completed_at": stage.completed_at.isoformat() if stage.completed_at else None,
                    "result": stage.result,
                }
                for stage in stages
            ],
        }
