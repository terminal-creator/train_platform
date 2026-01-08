"""
Metrics Persister - Phase 1.2

从 PlatformCallback 生成的 JSON Lines 文件读取指标并持久化到数据库。

工作流程：
1. 读取 {job_id}_metrics.jsonl 文件（本地或通过 SSH）
2. 解析 JSON 格式的指标
3. 转换为 TrainingMetric 数据模型
4. 批量插入数据库（支持增量同步）
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlmodel import Session, select

from .database import TrainingMetric, MetricsRepository

logger = logging.getLogger(__name__)


def parse_platform_metric(raw_metric: Dict[str, Any], job_uuid: str) -> TrainingMetric:
    """
    将 PlatformCallback 输出的 JSON 格式指标转换为 TrainingMetric 模型

    Args:
        raw_metric: PlatformCallback 输出的指标字典，格式如下：
            {
                "step": 1,
                "timestamp": 1704672000.123,
                "loss": {"actor_loss": 2.5, "critic_loss": 1.2, "total_loss": 3.7},
                "reward": {"mean": 0.5, "std": 0.1, "max": 0.8, "min": 0.2},
                "kl": {"mean": 0.1, "max": 0.15},
                "gradient": {"actor_norm": 1.5, "critic_norm": 0.8},
                "performance": {"tokens_per_second": 1000, "step_time": 2.5, "gpu_memory_allocated": 40.5},
                "epoch": 0,
                "raw": {...}  # 完整原始指标
            }
        job_uuid: 训练任务的 UUID

    Returns:
        TrainingMetric 实例

    为什么这么设计：
    - 使用嵌套字典获取值，避免 KeyError
    - 将 actor_loss 映射到 policy_loss（兼容现有字段名）
    - 将完整原始数据存储到 extra_metrics JSON 字段
    - 时间戳转换为 datetime 对象
    """

    def safe_get(d: Dict, *keys, default=None):
        """安全地从嵌套字典获取值"""
        for key in keys:
            if isinstance(d, dict):
                d = d.get(key, {})
            else:
                return default
        return d if d != {} else default

    # 提取基础信息
    step = raw_metric.get("step", 0)
    epoch = raw_metric.get("epoch", 0)
    timestamp_float = raw_metric.get("timestamp")
    timestamp = datetime.fromtimestamp(timestamp_float) if timestamp_float else datetime.utcnow()

    # 提取 loss 指标（actor_loss -> policy_loss, critic_loss -> value_loss）
    loss = raw_metric.get("loss", {})
    policy_loss = safe_get(loss, "actor_loss")  # actor_loss for PPO
    value_loss = safe_get(loss, "critic_loss")  # critic_loss for PPO
    total_loss = safe_get(loss, "total_loss")

    # 提取 reward 指标
    reward = raw_metric.get("reward", {})
    reward_mean = safe_get(reward, "mean")
    reward_std = safe_get(reward, "std")
    reward_max = safe_get(reward, "max")
    reward_min = safe_get(reward, "min")

    # 提取 KL 指标
    kl = raw_metric.get("kl", {})
    kl_divergence = safe_get(kl, "mean")
    kl_divergence_max = safe_get(kl, "max")

    # 提取梯度指标
    gradient = raw_metric.get("gradient", {})
    grad_norm_actor = safe_get(gradient, "actor_norm")
    grad_norm_critic = safe_get(gradient, "critic_norm")

    # 提取性能指标
    performance = raw_metric.get("performance", {})
    tokens_per_second = safe_get(performance, "tokens_per_second")
    step_time = safe_get(performance, "step_time")
    gpu_memory_allocated_gib = safe_get(performance, "gpu_memory_allocated")

    # 存储完整原始数据到 extra_metrics
    # 为什么：保留所有信息，未来可能需要
    extra_metrics = raw_metric.get("raw", {})

    return TrainingMetric(
        job_uuid=job_uuid,
        step=step,
        epoch=epoch,
        timestamp=timestamp,
        # Loss metrics
        policy_loss=policy_loss,
        value_loss=value_loss,
        total_loss=total_loss,
        # Reward metrics
        reward_mean=reward_mean,
        reward_std=reward_std,
        reward_max=reward_max,
        reward_min=reward_min,
        # KL metrics
        kl_divergence=kl_divergence,
        kl_divergence_max=kl_divergence_max,
        # Gradient metrics
        grad_norm_actor=grad_norm_actor,
        grad_norm_critic=grad_norm_critic,
        # Performance metrics
        tokens_per_second=tokens_per_second,
        step_time=step_time,
        gpu_memory_allocated_gib=gpu_memory_allocated_gib,
        # 异常标记暂时为空，后续从状态文件读取
        has_anomaly=False,
        anomaly_type=None,
        anomaly_message=None,
        # 完整原始数据
        extra_metrics=extra_metrics,
    )


def sync_metrics_from_file(
    job_uuid: str,
    metrics_file: Path,
    session: Session,
    batch_size: int = 100,
    last_offset: int = 0,
) -> Dict[str, Any]:
    """
    从本地 JSON Lines 文件增量同步指标到数据库（使用 offset）

    Args:
        job_uuid: 训练任务的 UUID
        metrics_file: 指标文件路径（{job_id}_metrics.jsonl）
        session: 数据库 session
        batch_size: 批量插入的大小
        last_offset: 上次读取的文件 offset（字节位置）

    Returns:
        Dict with sync result:
            - new_metrics_count: 新增指标数量
            - new_offset: 新的文件 offset
            - file_size: 当前文件大小

    工作流程：
    1. 从 last_offset 开始读取文件（而不是从头读）
    2. 解析新增的行
    3. 批量插入数据库
    4. 返回新的 offset

    性能优化：
    - 使用 seek() 跳到 last_offset，避免重复读取
    - 只读取新增的行，文件越大优势越明显
    - 适合高频轮询场景
    """

    if not metrics_file.exists():
        logger.warning(f"Metrics file not found: {metrics_file}")
        return {
            "new_metrics_count": 0,
            "new_offset": last_offset,
            "file_size": 0,
        }

    file_size = metrics_file.stat().st_size

    # 如果文件没有增长，直接返回
    if file_size <= last_offset:
        logger.debug(f"No new data in metrics file (size={file_size}, offset={last_offset})")
        return {
            "new_metrics_count": 0,
            "new_offset": last_offset,
            "file_size": file_size,
        }

    repository = MetricsRepository(session)
    new_metrics = []

    # 从 last_offset 开始读取
    with open(metrics_file, 'r') as f:
        # 跳到上次读取的位置
        f.seek(last_offset)

        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                raw_metric = json.loads(line)
                metric = parse_platform_metric(raw_metric, job_uuid)
                new_metrics.append(metric)

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process line: {e}")
                continue

        # 记录新的 offset
        new_offset = f.tell()

    if not new_metrics:
        logger.debug(f"No valid metrics in new data")
        return {
            "new_metrics_count": 0,
            "new_offset": new_offset,
            "file_size": file_size,
        }

    # 批量插入数据库
    logger.info(f"Syncing {len(new_metrics)} new metrics (offset: {last_offset} -> {new_offset})...")

    for i in range(0, len(new_metrics), batch_size):
        batch = new_metrics[i:i + batch_size]
        repository.add_metrics_batch(batch)

    logger.info(f"✓ Synced {len(new_metrics)} metrics to database")

    return {
        "new_metrics_count": len(new_metrics),
        "new_offset": new_offset,
        "file_size": file_size,
    }


def sync_anomaly_from_status_file(
    job_uuid: str,
    status_file: Path,
    session: Session,
) -> bool:
    """
    从状态文件读取异常信息并更新数据库中的对应步骤

    Args:
        job_uuid: 训练任务的 UUID
        status_file: 状态文件路径（{job_id}_status.json）
        session: 数据库 session

    Returns:
        是否成功更新

    工作流程：
    1. 读取状态文件
    2. 如果有异常，查找对应 step 的 metric 记录
    3. 更新 has_anomaly、anomaly_type、anomaly_message 字段

    为什么需要这个函数：
    - PlatformCallback 在检测到异常时会更新状态文件
    - 需要将这个异常信息同步到数据库的对应步骤
    - 便于前端查询和展示异常告警
    """

    if not status_file.exists():
        logger.warning(f"Status file not found: {status_file}")
        return False

    try:
        with open(status_file, 'r') as f:
            status = json.load(f)

        if not status.get("anomaly_detected"):
            # 没有异常，无需更新
            return True

        current_step = status.get("current_step", 0)
        anomaly_reason = status.get("anomaly_reason", "Unknown anomaly")

        # 解析异常类型
        # 为什么：根据异常原因字符串识别异常类型
        anomaly_type = "unknown"
        if "NaN" in anomaly_reason or "Inf" in anomaly_reason:
            anomaly_type = "nan"
        elif "KL" in anomaly_reason:
            anomaly_type = "kl_explosion"
        elif "not improving" in anomaly_reason:
            anomaly_type = "loss_plateau"

        # 查找对应步骤的 metric 记录
        statement = select(TrainingMetric).where(
            TrainingMetric.job_uuid == job_uuid,
            TrainingMetric.step == current_step
        )
        metric = session.exec(statement).first()

        if metric:
            # 更新异常标记
            metric.has_anomaly = True
            metric.anomaly_type = anomaly_type
            metric.anomaly_message = anomaly_reason
            session.add(metric)
            session.commit()

            logger.info(f"✓ Updated anomaly for step {current_step}: {anomaly_type}")
            return True
        else:
            logger.warning(f"Metric not found for step {current_step}")
            return False

    except Exception as e:
        logger.error(f"Failed to sync anomaly: {e}")
        return False


def sync_metrics_for_job(
    job_uuid: str,
    job_id: str,
    metrics_dir: Path,
    session: Session,
) -> Dict[str, Any]:
    """
    为指定任务同步所有指标（便捷函数）

    Args:
        job_uuid: 训练任务的 UUID
        job_id: 训练任务的 ID（用于定位文件）
        metrics_dir: 指标文件所在目录
        session: 数据库 session

    Returns:
        同步结果摘要

    为什么提供这个函数：
    - 一次调用完成指标和异常的同步
    - 便于在定时任务中使用
    - 返回摘要信息便于日志记录
    """

    metrics_file = metrics_dir / f"{job_id}_metrics.jsonl"
    status_file = metrics_dir / f"{job_id}_status.json"

    # 同步指标
    new_count = sync_metrics_from_file(job_uuid, metrics_file, session)

    # 同步异常
    anomaly_synced = sync_anomaly_from_status_file(job_uuid, status_file, session)

    return {
        "job_uuid": job_uuid,
        "job_id": job_id,
        "new_metrics_count": new_count,
        "anomaly_synced": anomaly_synced,
        "metrics_file": str(metrics_file),
        "status_file": str(status_file),
    }


# ============== Enhanced: Real-time Metrics Buffer ==============

class MetricsBuffer:
    """
    实时 Metrics 缓冲区（用于训练侧直接推送）

    与文件方式相比的优势：
    - 实时性：无需轮询文件
    - 低延迟：直接写入数据库
    - 批量优化：自动批处理提升性能
    """

    def __init__(self, max_size: int = 100, max_age_seconds: int = 30):
        self.max_size = max_size
        self.max_age_seconds = max_age_seconds
        self.buffer: List[TrainingMetric] = []
        self.last_flush = datetime.utcnow()

    def add(self, metric: TrainingMetric):
        """添加 metric 到缓冲区"""
        self.buffer.append(metric)

        # 自动刷新条件：缓冲区满或超时
        current_time = datetime.utcnow()
        if len(self.buffer) >= self.max_size or \
           (current_time - self.last_flush).total_seconds() >= self.max_age_seconds:
            self.flush()

    def flush(self) -> int:
        """刷新缓冲区到数据库"""
        if not self.buffer:
            return 0

        try:
            from .database import engine
            with Session(engine) as session:
                repo = MetricsRepository(session)
                repo.add_metrics_batch(self.buffer)

            count = len(self.buffer)
            logger.info(f"[MetricsBuffer] Flushed {count} metrics to database")

            self.buffer.clear()
            self.last_flush = datetime.utcnow()

            return count

        except Exception as e:
            logger.error(f"[MetricsBuffer] Failed to flush: {e}")
            return 0


# 全局缓冲区实例
_global_buffer = MetricsBuffer()


def persist_metric_realtime(job_uuid: str, step: int, epoch: int, **metrics):
    """
    实时持久化 metric（供训练侧 callback 调用）

    这是增强版的实时接口，与文件方式互补：
    - 文件方式：适合远程训练、需要容错
    - 实时方式：适合本地训练、需要低延迟

    Args:
        job_uuid: Job UUID
        step: Training step
        epoch: Epoch number
        **metrics: Metric key-value pairs
    """
    metric = parse_platform_metric(
        {
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.utcnow().timestamp(),
            "loss": {
                "actor_loss": metrics.get("policy_loss"),
                "critic_loss": metrics.get("value_loss"),
                "total_loss": metrics.get("total_loss"),
            },
            "reward": {
                "mean": metrics.get("reward_mean"),
                "std": metrics.get("reward_std"),
                "max": metrics.get("reward_max"),
                "min": metrics.get("reward_min"),
            },
            "kl": {
                "mean": metrics.get("kl_divergence"),
                "max": metrics.get("kl_divergence_max"),
            },
            "gradient": {
                "actor_norm": metrics.get("grad_norm_actor"),
                "critic_norm": metrics.get("grad_norm_critic"),
            },
            "performance": {
                "tokens_per_second": metrics.get("tokens_per_second"),
                "step_time": metrics.get("step_time"),
                "gpu_memory_allocated": metrics.get("gpu_memory_allocated_gib"),
            },
            "raw": metrics,
        },
        job_uuid
    )

    _global_buffer.add(metric)


def create_training_callback(job_uuid: str):
    """
    创建训练侧 callback 函数

    使用示例：
    ```python
    callback = create_training_callback(job_uuid)

    for step in range(num_steps):
        loss = train_step()
        callback(step=step, policy_loss=loss, learning_rate=lr)
    ```

    Args:
        job_uuid: Job UUID

    Returns:
        Callback function
    """
    def callback(step: int, epoch: int = 0, **metrics):
        try:
            persist_metric_realtime(job_uuid, step, epoch, **metrics)
        except Exception as e:
            logger.error(f"[TrainingCallback] Failed: {e}")

    return callback


# ============== Celery Tasks for Metrics Processing ==============

from .celery_config import app


@app.task(name="training_platform.core.metrics_persister.periodic_flush")
def periodic_flush():
    """
    定期刷新 metrics 缓冲区（Celery Beat 任务）

    建议每 30 秒执行一次
    """
    count = _global_buffer.flush()
    return {"flushed": count}


@app.task(name="training_platform.core.metrics_persister.diagnose_metrics")
def diagnose_metrics(job_uuid: str, step: int):
    """
    诊断指标异常（Celery 任务）

    检查项：
    - Loss NaN/Inf
    - KL divergence 爆炸
    - Reward 异常
    - 梯度消失/爆炸

    Args:
        job_uuid: Job UUID
        step: Training step

    Returns:
        Diagnostic result
    """
    from .database import engine
    from .diagnostics import DiagnosticService

    with Session(engine) as session:
        repo = MetricsRepository(session)
        metric = repo.get_metric_by_step(job_uuid, step)

        if not metric:
            return {"error": "Metric not found"}

        diagnostics = DiagnosticService(session)
        result = diagnostics.diagnose_step(job_uuid, step)

        return result
