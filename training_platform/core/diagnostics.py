"""
Diagnostics Module - Phase 1.4

平台层异常检测和诊断功能

与 PlatformCallback 的区别：
- PlatformCallback：训练过程中实时检测，记录到文件
- Diagnostics：从数据库分析历史数据，触发告警，自动操作

工作流程：
1. DiagnosticService 定期扫描运行中的任务
2. AnomalyDetector 从数据库读取指标并分析
3. 检测到异常时触发告警
4. 根据配置自动标记任务状态或执行其他操作

为什么需要平台层诊断：
- 补充检测：训练层可能漏检或未配置检测
- 历史分析：分析整个训练过程，发现长期趋势
- 告警通知：向用户发送通知（邮件、Webhook 等）
- 自动操作：自动暂停、标记失败等
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from sqlmodel import Session, select

from .database import TrainingJob, TrainingMetric, JobStatus

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """异常类型"""
    NAN_INF = "nan_inf"                    # NaN/Inf 值
    KL_EXPLOSION = "kl_explosion"          # KL 散度爆炸
    LOSS_PLATEAU = "loss_plateau"          # Loss 长期不下降
    REWARD_COLLAPSE = "reward_collapse"    # Reward 崩溃
    GRADIENT_EXPLOSION = "gradient_explosion"  # 梯度爆炸
    GRADIENT_VANISHING = "gradient_vanishing"  # 梯度消失


class AnomalySeverity(str, Enum):
    """异常严重程度"""
    CRITICAL = "critical"  # 严重：需要立即停止训练
    HIGH = "high"          # 高：可能导致训练失败
    MEDIUM = "medium"      # 中：需要关注，可能影响效果
    LOW = "low"            # 低：警告，不影响训练


@dataclass
class AnomalyResult:
    """异常检测结果"""
    detected: bool                          # 是否检测到异常
    anomaly_type: Optional[AnomalyType]     # 异常类型
    severity: Optional[AnomalySeverity]     # 严重程度
    message: str                            # 详细消息
    step: Optional[int] = None              # 异常发生的步骤
    metrics_snapshot: Optional[Dict[str, Any]] = None  # 异常时的指标快照
    suggestion: Optional[str] = None        # 诊断建议

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "detected": self.detected,
            "anomaly_type": self.anomaly_type.value if self.anomaly_type else None,
            "severity": self.severity.value if self.severity else None,
            "message": self.message,
            "step": self.step,
            "metrics_snapshot": self.metrics_snapshot,
            "suggestion": self.suggestion,
        }


class AnomalyDetector:
    """
    异常检测器（平台层）

    从数据库读取指标并执行各种异常检测。
    """

    def __init__(self, session: Session):
        self.session = session

    def detect_nan_inf(
        self,
        job_uuid: str,
        recent_steps: int = 10
    ) -> AnomalyResult:
        """
        检测 NaN/Inf 值（Phase 1.4.1）

        检查最近 N 个步骤的指标中是否有 NaN/Inf。

        Args:
            job_uuid: 任务 UUID
            recent_steps: 检查最近多少步

        Returns:
            AnomalyResult

        为什么这么设计：
        - 检查最近步骤：避免处理全部历史数据
        - 检测多个字段：loss, reward, kl, gradient 都可能出现 NaN
        - 提供诊断建议：告诉用户可能的原因和解决方法
        """
        # 查询最近的指标
        statement = select(TrainingMetric).where(
            TrainingMetric.job_uuid == job_uuid
        ).order_by(TrainingMetric.step.desc()).limit(recent_steps)

        metrics = self.session.exec(statement).all()

        if not metrics:
            return AnomalyResult(
                detected=False,
                anomaly_type=None,
                severity=None,
                message="No metrics found"
            )

        # 检查每个指标
        for metric in reversed(metrics):  # 从旧到新检查
            nan_fields = []

            def is_nan_or_inf(value):
                """检查值是否为 NaN/Inf（包括 None，因为 SQLite 不支持 NaN）"""
                if value is None:
                    return False  # None 不算异常，可能就是没有这个字段
                try:
                    return (float('inf') == abs(value) or value != value)
                except:
                    return False

            # 检查 loss（这些字段如果训练正常应该都有值）
            # 如果在训练中途突然变成 None，可能表示计算出了 NaN
            if metric.policy_loss is not None and is_nan_or_inf(metric.policy_loss):
                nan_fields.append("policy_loss")
            if metric.value_loss is not None and is_nan_or_inf(metric.value_loss):
                nan_fields.append("value_loss")

            # 检查 reward
            if metric.reward_mean is not None and is_nan_or_inf(metric.reward_mean):
                nan_fields.append("reward_mean")

            # 检查 KL
            if metric.kl_divergence is not None and is_nan_or_inf(metric.kl_divergence):
                nan_fields.append("kl_divergence")

            # 检查梯度
            if metric.grad_norm_actor is not None and is_nan_or_inf(metric.grad_norm_actor):
                nan_fields.append("grad_norm_actor")

            if nan_fields:
                return AnomalyResult(
                    detected=True,
                    anomaly_type=AnomalyType.NAN_INF,
                    severity=AnomalySeverity.CRITICAL,
                    message=f"NaN/Inf detected in fields: {', '.join(nan_fields)} at step {metric.step}",
                    step=metric.step,
                    metrics_snapshot={
                        "policy_loss": metric.policy_loss,
                        "value_loss": metric.value_loss,
                        "reward_mean": metric.reward_mean,
                        "kl_divergence": metric.kl_divergence,
                        "grad_norm_actor": metric.grad_norm_actor,
                    },
                    suggestion=(
                        "NaN/Inf 通常由以下原因导致：\n"
                        "1. 学习率过高 -> 降低学习率\n"
                        "2. 梯度爆炸 -> 添加梯度裁剪\n"
                        "3. 数值不稳定 -> 检查 reward scaling\n"
                        "4. 数据问题 -> 检查数据预处理"
                    )
                )

        return AnomalyResult(
            detected=False,
            anomaly_type=None,
            severity=None,
            message="No NaN/Inf detected"
        )

    def detect_kl_explosion(
        self,
        job_uuid: str,
        kl_threshold: float = 1.0,
        recent_steps: int = 10
    ) -> AnomalyResult:
        """
        检测 KL 散度爆炸（Phase 1.4.2）

        检查 KL 散度是否超过阈值。

        Args:
            job_uuid: 任务 UUID
            kl_threshold: KL 散度阈值（默认 1.0）
            recent_steps: 检查最近多少步

        Returns:
            AnomalyResult

        为什么需要这个检测：
        - KL 散度过大：策略更新过于激进，可能导致性能崩溃
        - PPO 算法特性：KL 散度应该保持在小范围内
        - 早期发现：在性能崩溃前发出警告
        """
        # 查询最近的指标
        statement = select(TrainingMetric).where(
            TrainingMetric.job_uuid == job_uuid
        ).order_by(TrainingMetric.step.desc()).limit(recent_steps)

        metrics = self.session.exec(statement).all()

        if not metrics:
            return AnomalyResult(
                detected=False,
                anomaly_type=None,
                severity=None,
                message="No metrics found"
            )

        # 检查 KL 散度
        for metric in reversed(metrics):
            if metric.kl_divergence is not None and metric.kl_divergence > kl_threshold:
                # 判断严重程度
                if metric.kl_divergence > kl_threshold * 2:
                    severity = AnomalySeverity.CRITICAL
                elif metric.kl_divergence > kl_threshold * 1.5:
                    severity = AnomalySeverity.HIGH
                else:
                    severity = AnomalySeverity.MEDIUM

                return AnomalyResult(
                    detected=True,
                    anomaly_type=AnomalyType.KL_EXPLOSION,
                    severity=severity,
                    message=f"KL divergence explosion: {metric.kl_divergence:.4f} > {kl_threshold} at step {metric.step}",
                    step=metric.step,
                    metrics_snapshot={
                        "kl_divergence": metric.kl_divergence,
                        "kl_divergence_max": metric.kl_divergence_max,
                        "policy_loss": metric.policy_loss,
                        "reward_mean": metric.reward_mean,
                    },
                    suggestion=(
                        "KL 散度爆炸的解决方法：\n"
                        f"1. 降低学习率（当前可能过高）\n"
                        "2. 增大 KL penalty 系数\n"
                        "3. 减小 PPO clip range\n"
                        "4. 检查 reward scaling 是否合理"
                    )
                )

        return AnomalyResult(
            detected=False,
            anomaly_type=None,
            severity=None,
            message=f"KL divergence within threshold (<{kl_threshold})"
        )

    def detect_loss_plateau(
        self,
        job_uuid: str,
        patience: int = 50,
        min_improvement: float = 0.01
    ) -> AnomalyResult:
        """
        检测 Loss 不下降（Phase 1.4.3）

        检查 Loss 是否长期停滞不下降。

        Args:
            job_uuid: 任务 UUID
            patience: 容忍多少步没有改善
            min_improvement: 最小改善幅度（相对值）

        Returns:
            AnomalyResult

        为什么需要这个检测：
        - 训练停滞：Loss 不下降可能表示训练陷入局部最优
        - 浪费资源：继续训练可能无效，应该早停
        - 超参调整：可能需要调整学习率或其他超参
        """
        # 查询最近 patience * 2 步的指标
        statement = select(TrainingMetric).where(
            TrainingMetric.job_uuid == job_uuid,
            TrainingMetric.total_loss != None
        ).order_by(TrainingMetric.step.desc()).limit(patience * 2)

        metrics = list(reversed(self.session.exec(statement).all()))

        if len(metrics) < patience:
            return AnomalyResult(
                detected=False,
                anomaly_type=None,
                severity=None,
                message=f"Not enough data (need {patience} steps)"
            )

        # 找到最近 patience 步内的最小 loss
        recent_losses = [m.total_loss for m in metrics[-patience:] if m.total_loss is not None]
        if not recent_losses:
            return AnomalyResult(
                detected=False,
                anomaly_type=None,
                severity=None,
                message="No loss data available"
            )

        min_recent_loss = min(recent_losses)

        # 找到之前的最小 loss（作为基准）
        if len(metrics) > patience:
            previous_losses = [m.total_loss for m in metrics[:-patience] if m.total_loss is not None]
            if previous_losses:
                min_previous_loss = min(previous_losses)

                # 计算改善幅度
                improvement = (min_previous_loss - min_recent_loss) / abs(min_previous_loss)

                if improvement < min_improvement:
                    # 判断严重程度
                    if patience >= 100:
                        severity = AnomalySeverity.HIGH
                    elif patience >= 50:
                        severity = AnomalySeverity.MEDIUM
                    else:
                        severity = AnomalySeverity.LOW

                    return AnomalyResult(
                        detected=True,
                        anomaly_type=AnomalyType.LOSS_PLATEAU,
                        severity=severity,
                        message=f"Loss plateau detected: no improvement in last {patience} steps (improvement: {improvement:.4%})",
                        step=metrics[-1].step,
                        metrics_snapshot={
                            "current_loss": min_recent_loss,
                            "best_previous_loss": min_previous_loss,
                            "improvement": improvement,
                        },
                        suggestion=(
                            "Loss 不下降的解决方法：\n"
                            "1. 降低学习率（可能过大错过最优点）\n"
                            "2. 尝试学习率调度（cosine decay, warm restart）\n"
                            "3. 检查是否陷入局部最优\n"
                            "4. 考虑 early stopping"
                        )
                    )

        return AnomalyResult(
            detected=False,
            anomaly_type=None,
            severity=None,
            message="Loss is improving normally"
        )

    def detect_reward_collapse(
        self,
        job_uuid: str,
        recent_steps: int = 20,
        drop_threshold: float = 0.5
    ) -> AnomalyResult:
        """
        检测 Reward 崩溃

        检查 Reward 是否突然大幅下降。

        Args:
            job_uuid: 任务 UUID
            recent_steps: 检查最近多少步
            drop_threshold: 下降幅度阈值（相对值，0.5 = 下降 50%）

        Returns:
            AnomalyResult

        为什么需要这个检测：
        - 性能崩溃：Reward 突然下降通常表示策略崩溃
        - 早期发现：在完全崩溃前发出警告
        - PPO 特有问题：策略更新过大可能导致性能崩溃
        """
        # 查询最近的指标
        statement = select(TrainingMetric).where(
            TrainingMetric.job_uuid == job_uuid,
            TrainingMetric.reward_mean != None
        ).order_by(TrainingMetric.step.desc()).limit(recent_steps * 2)

        metrics = list(reversed(self.session.exec(statement).all()))

        if len(metrics) < recent_steps:
            return AnomalyResult(
                detected=False,
                anomaly_type=None,
                severity=None,
                message="Not enough data"
            )

        # 计算最近和之前的平均 reward
        recent_rewards = [m.reward_mean for m in metrics[-recent_steps:]]
        previous_rewards = [m.reward_mean for m in metrics[:-recent_steps]]

        if not recent_rewards or not previous_rewards:
            return AnomalyResult(
                detected=False,
                anomaly_type=None,
                severity=None,
                message="Not enough data for comparison"
            )

        avg_recent = sum(recent_rewards) / len(recent_rewards)
        avg_previous = sum(previous_rewards) / len(previous_rewards)

        # 计算下降幅度
        if avg_previous > 0:
            drop_ratio = (avg_previous - avg_recent) / avg_previous
        else:
            drop_ratio = 0

        if drop_ratio > drop_threshold:
            severity = AnomalySeverity.CRITICAL if drop_ratio > 0.7 else AnomalySeverity.HIGH

            return AnomalyResult(
                detected=True,
                anomaly_type=AnomalyType.REWARD_COLLAPSE,
                severity=severity,
                message=f"Reward collapse detected: dropped {drop_ratio:.1%} in last {recent_steps} steps",
                step=metrics[-1].step,
                metrics_snapshot={
                    "current_reward": avg_recent,
                    "previous_reward": avg_previous,
                    "drop_ratio": drop_ratio,
                },
                suggestion=(
                    "Reward 崩溃的解决方法：\n"
                    "1. 立即停止训练，从之前的检查点恢复\n"
                    "2. 降低学习率\n"
                    "3. 减小 PPO clip range\n"
                    "4. 增大 KL penalty"
                )
            )

        return AnomalyResult(
            detected=False,
            anomaly_type=None,
            severity=None,
            message="Reward is stable"
        )

    def detect_all(
        self,
        job_uuid: str,
        config: Optional[Dict[str, Any]] = None
    ) -> List[AnomalyResult]:
        """
        执行所有异常检测

        Args:
            job_uuid: 任务 UUID
            config: 检测配置（阈值等）

        Returns:
            所有检测到的异常列表

        为什么提供这个方法：
        - 便捷：一次调用执行所有检测
        - 完整：不会漏掉任何检测项
        - 灵活：可以通过 config 自定义阈值
        """
        config = config or {}

        results = []

        # 1. NaN/Inf 检测
        result = self.detect_nan_inf(job_uuid)
        if result.detected:
            results.append(result)

        # 2. KL 爆炸检测
        kl_threshold = config.get("kl_threshold", 1.0)
        result = self.detect_kl_explosion(job_uuid, kl_threshold=kl_threshold)
        if result.detected:
            results.append(result)

        # 3. Loss 不下降检测
        patience = config.get("loss_patience", 50)
        result = self.detect_loss_plateau(job_uuid, patience=patience)
        if result.detected:
            results.append(result)

        # 4. Reward 崩溃检测
        result = self.detect_reward_collapse(job_uuid)
        if result.detected:
            results.append(result)

        return results


class DiagnosticService:
    """
    诊断服务（Phase 1.4.4）

    定期扫描运行中的任务，执行异常检测，触发告警。
    """

    def __init__(self, session: Session):
        self.session = session
        self.detector = AnomalyDetector(session)

    def diagnose_job(
        self,
        job_uuid: str,
        auto_mark_failed: bool = True
    ) -> Dict[str, Any]:
        """
        诊断单个任务

        Args:
            job_uuid: 任务 UUID
            auto_mark_failed: 是否自动标记失败（检测到 CRITICAL 异常时）

        Returns:
            诊断结果

        为什么需要自动标记：
        - 及时响应：检测到严重异常时立即停止训练
        - 节省资源：避免继续浪费 GPU 资源
        - 用户体验：用户可以立即看到任务失败原因
        """
        from .database import JobRepository

        job_repo = JobRepository(self.session)
        job = job_repo.get_by_uuid(job_uuid)

        if not job:
            return {
                "success": False,
                "error": f"Job {job_uuid} not found"
            }

        # 执行所有检测
        anomalies = self.detector.detect_all(job_uuid)

        # 检查是否有严重异常
        critical_anomalies = [a for a in anomalies if a.severity == AnomalySeverity.CRITICAL]

        # 自动标记失败
        if critical_anomalies and auto_mark_failed and job.status == JobStatus.RUNNING:
            # 标记任务为失败
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()

            # 记录失败原因（注：TrainingJob 模型暂无 error_message 字段）
            anomaly = critical_anomalies[0]  # 取第一个严重异常

            self.session.add(job)
            self.session.commit()

            logger.warning(f"[Diagnostics] Auto-marked job {job_uuid} as FAILED: {anomaly.message}")

        return {
            "success": True,
            "job_uuid": job_uuid,
            "job_name": job.name,
            "status": job.status.value,
            "anomalies_count": len(anomalies),
            "critical_count": len(critical_anomalies),
            "anomalies": [a.to_dict() for a in anomalies],
            "auto_marked_failed": len(critical_anomalies) > 0 and auto_mark_failed,
        }

    def diagnose_all_running_jobs(self) -> Dict[str, Any]:
        """
        诊断所有运行中的任务

        用于定期扫描（如每分钟执行一次）。

        Returns:
            诊断结果汇总

        为什么需要定期扫描：
        - 主动监控：不依赖用户手动检查
        - 及时发现：异常发生后几分钟内就能检测到
        - 自动化：减少人工干预
        """
        from .database import JobRepository

        job_repo = JobRepository(self.session)

        # 查询所有运行中的任务
        statement = select(TrainingJob).where(TrainingJob.status == JobStatus.RUNNING)
        running_jobs = self.session.exec(statement).all()

        results = []
        total_anomalies = 0
        auto_failed_count = 0

        for job in running_jobs:
            result = self.diagnose_job(job.uuid, auto_mark_failed=True)
            results.append(result)
            total_anomalies += result["anomalies_count"]
            if result.get("auto_marked_failed"):
                auto_failed_count += 1

        logger.info(
            f"[Diagnostics] Scanned {len(running_jobs)} running jobs, "
            f"found {total_anomalies} anomalies, "
            f"auto-failed {auto_failed_count} jobs"
        )

        return {
            "scanned_jobs": len(running_jobs),
            "total_anomalies": total_anomalies,
            "auto_failed_count": auto_failed_count,
            "results": results,
        }
