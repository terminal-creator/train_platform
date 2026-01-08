"""
Experience Reuse (Phase 2)

提供经验复用功能，帮助用户从历史成功训练中学习。

主要功能：
1. 从历史任务创建新训练（Clone Job）
2. 推荐成功的配方和配置
3. 基于历史数据的最佳实践建议
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .database import (
    TrainingJob,
    JobStatus,
    JobRepository,
    MetricsRepository,
    Session,
)
from .recipes import RecipeRegistry


def clone_job_config(
    source_job: TrainingJob,
    overrides: Optional[Dict[str, Any]] = None,
    new_name: Optional[str] = None,
    new_description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从现有任务克隆配置

    Args:
        source_job: 源训练任务
        overrides: 覆盖参数
        new_name: 新任务名称
        new_description: 新任务描述

    Returns:
        克隆的配置字典
    """
    # 复制基础配置
    cloned_config = {
        "name": new_name or f"{source_job.name} (Clone)",
        "description": new_description or f"Cloned from {source_job.name}",
        "algorithm": source_job.algorithm,
        "recipe_id": source_job.recipe_id,

        # Model config
        "model_path": source_job.model_path,
        "lora_enabled": source_job.lora_enabled,
        "lora_rank": source_job.lora_rank,

        # Resource allocation
        "num_gpus": source_job.num_gpus,
        "gpu_type": source_job.gpu_type,

        # Training hyperparameters
        "learning_rate": source_job.learning_rate,
        "batch_size": source_job.batch_size,
        "num_epochs": source_job.num_epochs,
        "max_steps": source_job.max_steps,
        "context_length": source_job.context_length,
        "kl_coef": source_job.kl_coef,
        "rollout_n": source_job.rollout_n,

        # Full config (from JSON field)
        "config": source_job.config.copy() if source_job.config else {},

        # Metadata
        "cloned_from": source_job.uuid,
        "cloned_at": datetime.utcnow().isoformat(),
    }

    # Apply overrides
    if overrides:
        cloned_config.update(overrides)

    return cloned_config


def recommend_successful_recipes(
    session: Session,
    task_type: Optional[str] = None,
    algorithm: Optional[str] = None,
    min_success_count: int = 1,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    推荐成功的配方

    基于历史训练任务的成功率推荐配方。

    Args:
        session: 数据库会话
        task_type: 任务类型筛选
        algorithm: 算法筛选
        min_success_count: 最少成功次数
        limit: 返回数量限制

    Returns:
        推荐配方列表
    """
    repo = JobRepository(session)

    # 获取所有已完成的任务
    all_jobs, _ = repo.list_jobs(status=JobStatus.COMPLETED, limit=1000)

    # 统计每个配方的使用情况
    recipe_stats = {}

    for job in all_jobs:
        if not job.recipe_id:
            continue

        # 筛选条件
        if algorithm and job.algorithm != algorithm:
            continue

        recipe_id = job.recipe_id

        if recipe_id not in recipe_stats:
            recipe_stats[recipe_id] = {
                "recipe_id": recipe_id,
                "total_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "avg_learning_rate": [],
                "avg_batch_size": [],
            }

        recipe_stats[recipe_id]["total_jobs"] += 1

        if job.status == JobStatus.COMPLETED:
            recipe_stats[recipe_id]["completed_jobs"] += 1
        elif job.status == JobStatus.FAILED:
            recipe_stats[recipe_id]["failed_jobs"] += 1

        # 收集参数统计
        recipe_stats[recipe_id]["avg_learning_rate"].append(job.learning_rate)
        recipe_stats[recipe_id]["avg_batch_size"].append(job.batch_size)

    # 计算成功率和平均参数
    recommendations = []

    for recipe_id, stats in recipe_stats.items():
        if stats["completed_jobs"] < min_success_count:
            continue

        recipe = RecipeRegistry.get(recipe_id)
        if not recipe:
            continue

        # 筛选任务类型
        if task_type and hasattr(recipe, 'task_type'):
            if recipe.task_type.value != task_type:
                continue

        success_rate = stats["completed_jobs"] / stats["total_jobs"] if stats["total_jobs"] > 0 else 0

        recommendations.append({
            "recipe_id": recipe_id,
            "recipe_name": recipe.name,
            "description": recipe.description,
            "algorithm": recipe.recommended_algorithm,
            "success_rate": round(success_rate * 100, 2),
            "total_jobs": stats["total_jobs"],
            "completed_jobs": stats["completed_jobs"],
            "failed_jobs": stats["failed_jobs"],
            "avg_learning_rate": sum(stats["avg_learning_rate"]) / len(stats["avg_learning_rate"]) if stats["avg_learning_rate"] else None,
            "avg_batch_size": sum(stats["avg_batch_size"]) / len(stats["avg_batch_size"]) if stats["avg_batch_size"] else None,
        })

    # 按成功率排序
    recommendations.sort(key=lambda x: (x["success_rate"], x["completed_jobs"]), reverse=True)

    return recommendations[:limit]


def get_best_practices(
    session: Session,
    recipe_id: str,
    metric: str = "reward_mean",
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    获取配方的最佳实践

    查找使用该配方并且指标表现最好的训练任务。

    Args:
        session: 数据库会话
        recipe_id: 配方 ID
        metric: 评估指标
        top_k: 返回前 K 个

    Returns:
        最佳实践列表
    """
    repo = JobRepository(session)
    metrics_repo = MetricsRepository(session)

    # 获取使用该配方的所有已完成任务
    all_jobs, _ = repo.list_jobs(status=JobStatus.COMPLETED, limit=1000)

    recipe_jobs = [job for job in all_jobs if job.recipe_id == recipe_id]

    # 获取每个任务的最终指标
    job_performances = []

    for job in recipe_jobs:
        latest_metric = metrics_repo.get_latest_metric(job.uuid)

        if not latest_metric:
            continue

        # 获取指标值
        metric_value = getattr(latest_metric, metric, None)

        if metric_value is None:
            # 尝试从 extra_metrics 获取
            metric_value = latest_metric.extra_metrics.get(metric)

        if metric_value is not None:
            job_performances.append({
                "job_uuid": job.uuid,
                "job_name": job.name,
                "metric_value": metric_value,
                "learning_rate": job.learning_rate,
                "batch_size": job.batch_size,
                "num_epochs": job.num_epochs,
                "kl_coef": job.kl_coef,
                "lora_rank": job.lora_rank if job.lora_enabled else None,
                "config": job.config,
                "created_at": job.created_at.isoformat(),
            })

    # 按指标值排序（假设越高越好，如 reward_mean）
    job_performances.sort(key=lambda x: x["metric_value"], reverse=True)

    return job_performances[:top_k]


def suggest_config_adjustments(
    current_config: Dict[str, Any],
    best_practices: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """
    建议配置调整

    对比当前配置与最佳实践，给出调整建议。

    Args:
        current_config: 当前配置
        best_practices: 最佳实践列表

    Returns:
        调整建议列表
    """
    if not best_practices:
        return []

    suggestions = []

    # 获取最佳实践的平均值
    best_lr = sum(bp["learning_rate"] for bp in best_practices) / len(best_practices)
    best_bs = sum(bp["batch_size"] for bp in best_practices) / len(best_practices)
    best_kl = sum(bp["kl_coef"] for bp in best_practices) / len(best_practices)

    current_lr = current_config.get("learning_rate", 0)
    current_bs = current_config.get("batch_size", 0)
    current_kl = current_config.get("kl_coef", 0)

    # 学习率建议
    if current_lr and abs(current_lr - best_lr) / best_lr > 0.3:  # 相差 30% 以上
        suggestions.append({
            "parameter": "learning_rate",
            "current_value": str(current_lr),
            "suggested_value": str(round(best_lr, 8)),
            "reason": f"最佳实践的平均学习率是 {best_lr:.2e}，与当前值相差较大"
        })

    # Batch size 建议
    if current_bs and abs(current_bs - best_bs) / best_bs > 0.3:
        suggestions.append({
            "parameter": "batch_size",
            "current_value": str(current_bs),
            "suggested_value": str(int(best_bs)),
            "reason": f"最佳实践的平均 batch size 是 {int(best_bs)}"
        })

    # KL 系数建议
    if current_kl and abs(current_kl - best_kl) / best_kl > 0.3:
        suggestions.append({
            "parameter": "kl_coef",
            "current_value": str(current_kl),
            "suggested_value": str(round(best_kl, 4)),
            "reason": f"最佳实践的平均 KL 系数是 {best_kl:.4f}"
        })

    return suggestions


def find_similar_successful_jobs(
    session: Session,
    reference_config: Dict[str, Any],
    algorithm: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    查找相似的成功任务

    基于配置相似度查找历史成功任务。

    Args:
        session: 数据库会话
        reference_config: 参考配置
        algorithm: 算法筛选
        limit: 返回数量

    Returns:
        相似任务列表
    """
    repo = JobRepository(session)

    # 获取所有已完成任务
    all_jobs, _ = repo.list_jobs(status=JobStatus.COMPLETED, limit=1000)

    if algorithm:
        all_jobs = [job for job in all_jobs if job.algorithm == algorithm]

    # 计算相似度
    similarities = []

    ref_lr = reference_config.get("learning_rate", 0)
    ref_bs = reference_config.get("batch_size", 0)
    ref_kl = reference_config.get("kl_coef", 0)

    for job in all_jobs:
        # 简单的相似度计算（可以改进为更复杂的算法）
        lr_diff = abs(job.learning_rate - ref_lr) / ref_lr if ref_lr else 0
        bs_diff = abs(job.batch_size - ref_bs) / ref_bs if ref_bs else 0
        kl_diff = abs(job.kl_coef - ref_kl) / ref_kl if ref_kl else 0

        # 相似度分数（越小越相似）
        similarity_score = (lr_diff + bs_diff + kl_diff) / 3

        similarities.append({
            "job_uuid": job.uuid,
            "job_name": job.name,
            "algorithm": job.algorithm,
            "recipe_id": job.recipe_id,
            "similarity_score": similarity_score,
            "learning_rate": job.learning_rate,
            "batch_size": job.batch_size,
            "kl_coef": job.kl_coef,
            "created_at": job.created_at.isoformat(),
        })

    # 按相似度排序
    similarities.sort(key=lambda x: x["similarity_score"])

    return similarities[:limit]
