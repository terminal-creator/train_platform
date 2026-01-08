"""
Configuration Diff Tool (Phase 2)

提供训练配置对比功能，帮助用户理解不同实验的配置差异。

主要功能：
1. 深度对比两个配置字典
2. 识别新增、删除、修改的配置项
3. 标记重要参数的变化
4. 生成用户友好的对比报告
"""

from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import json


class DiffType(str, Enum):
    """配置差异类型"""
    ADDED = "added"  # 新增
    REMOVED = "removed"  # 删除
    MODIFIED = "modified"  # 修改
    UNCHANGED = "unchanged"  # 未变化


class ParamImportance(str, Enum):
    """参数重要性级别"""
    CRITICAL = "critical"  # 关键参数（如学习率、batch size）
    IMPORTANT = "important"  # 重要参数（如 KL 系数、warmup steps）
    NORMAL = "normal"  # 普通参数


# 定义关键参数和重要参数
CRITICAL_PARAMS = {
    "learning_rate",
    "batch_size",
    "num_epochs",
    "max_steps",
    "algorithm",
}

IMPORTANT_PARAMS = {
    "kl_coef",
    "warmup_steps",
    "weight_decay",
    "gradient_accumulation_steps",
    "context_length",
    "rollout_n",
    "lora_rank",
    "lr_scheduler",
}


@dataclass
class ConfigDiff:
    """配置差异项"""
    path: str  # 配置路径（如 "learning_rate" 或 "optimizer.lr"）
    diff_type: DiffType  # 差异类型
    old_value: Any  # 旧值（删除或修改时）
    new_value: Any  # 新值（新增或修改时）
    importance: ParamImportance  # 重要性级别


@dataclass
class ConfigComparisonResult:
    """配置对比结果"""
    diffs: List[ConfigDiff]  # 所有差异项
    added_count: int  # 新增数量
    removed_count: int  # 删除数量
    modified_count: int  # 修改数量
    unchanged_count: int  # 未变化数量
    has_critical_changes: bool  # 是否有关键参数变化
    summary: str  # 对比摘要


def get_param_importance(param_name: str) -> ParamImportance:
    """
    获取参数的重要性级别

    Args:
        param_name: 参数名称

    Returns:
        参数重要性级别
    """
    if param_name in CRITICAL_PARAMS:
        return ParamImportance.CRITICAL
    elif param_name in IMPORTANT_PARAMS:
        return ParamImportance.IMPORTANT
    else:
        return ParamImportance.NORMAL


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    将嵌套字典扁平化

    例如：
    {"optimizer": {"lr": 0.001, "weight_decay": 0.01}}
    -> {"optimizer.lr": 0.001, "optimizer.weight_decay": 0.01}

    Args:
        d: 嵌套字典
        parent_key: 父键名
        sep: 分隔符

    Returns:
        扁平化的字典
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def compare_configs(
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
    name_a: str = "Config A",
    name_b: str = "Config B",
) -> ConfigComparisonResult:
    """
    对比两个配置

    Args:
        config_a: 配置 A（通常是旧配置）
        config_b: 配置 B（通常是新配置）
        name_a: 配置 A 的名称
        name_b: 配置 B 的名称

    Returns:
        配置对比结果
    """
    # 扁平化配置
    flat_a = _flatten_dict(config_a)
    flat_b = _flatten_dict(config_b)

    # 获取所有键
    all_keys = set(flat_a.keys()) | set(flat_b.keys())

    diffs = []
    added_count = 0
    removed_count = 0
    modified_count = 0
    unchanged_count = 0
    has_critical_changes = False

    for key in sorted(all_keys):
        value_a = flat_a.get(key)
        value_b = flat_b.get(key)

        # 获取参数重要性
        importance = get_param_importance(key)

        if key not in flat_a:
            # 新增
            diff = ConfigDiff(
                path=key,
                diff_type=DiffType.ADDED,
                old_value=None,
                new_value=value_b,
                importance=importance,
            )
            diffs.append(diff)
            added_count += 1
            if importance == ParamImportance.CRITICAL:
                has_critical_changes = True

        elif key not in flat_b:
            # 删除
            diff = ConfigDiff(
                path=key,
                diff_type=DiffType.REMOVED,
                old_value=value_a,
                new_value=None,
                importance=importance,
            )
            diffs.append(diff)
            removed_count += 1
            if importance == ParamImportance.CRITICAL:
                has_critical_changes = True

        elif value_a != value_b:
            # 修改
            diff = ConfigDiff(
                path=key,
                diff_type=DiffType.MODIFIED,
                old_value=value_a,
                new_value=value_b,
                importance=importance,
            )
            diffs.append(diff)
            modified_count += 1
            if importance == ParamImportance.CRITICAL:
                has_critical_changes = True

        else:
            # 未变化（通常不记录）
            unchanged_count += 1

    # 生成摘要
    summary_parts = []
    if added_count > 0:
        summary_parts.append(f"{added_count} 个新增")
    if removed_count > 0:
        summary_parts.append(f"{removed_count} 个删除")
    if modified_count > 0:
        summary_parts.append(f"{modified_count} 个修改")
    if unchanged_count > 0:
        summary_parts.append(f"{unchanged_count} 个未变化")

    summary = f"{name_a} vs {name_b}: " + "、".join(summary_parts)

    return ConfigComparisonResult(
        diffs=diffs,
        added_count=added_count,
        removed_count=removed_count,
        modified_count=modified_count,
        unchanged_count=unchanged_count,
        has_critical_changes=has_critical_changes,
        summary=summary,
    )


def format_diff_report(result: ConfigComparisonResult, include_unchanged: bool = False) -> str:
    """
    格式化对比报告为人类可读的文本

    Args:
        result: 配置对比结果
        include_unchanged: 是否包含未变化的配置项

    Returns:
        格式化的对比报告
    """
    lines = []
    lines.append("=" * 80)
    lines.append("配置对比报告")
    lines.append("=" * 80)
    lines.append(result.summary)
    lines.append("")

    if result.has_critical_changes:
        lines.append("⚠️  警告：检测到关键参数变化！")
        lines.append("")

    # 按重要性分组
    critical_diffs = [d for d in result.diffs if d.importance == ParamImportance.CRITICAL]
    important_diffs = [d for d in result.diffs if d.importance == ParamImportance.IMPORTANT]
    normal_diffs = [d for d in result.diffs if d.importance == ParamImportance.NORMAL]

    def format_diff_section(title: str, diffs: List[ConfigDiff]):
        if not diffs:
            return []

        section_lines = []
        section_lines.append(f"\n{title}")
        section_lines.append("-" * 40)

        for diff in diffs:
            if diff.diff_type == DiffType.ADDED:
                section_lines.append(f"  [+] {diff.path}: {diff.new_value}")
            elif diff.diff_type == DiffType.REMOVED:
                section_lines.append(f"  [-] {diff.path}: {diff.old_value}")
            elif diff.diff_type == DiffType.MODIFIED:
                section_lines.append(f"  [~] {diff.path}: {diff.old_value} → {diff.new_value}")

        return section_lines

    # 关键参数变化
    if critical_diffs:
        lines.extend(format_diff_section("🔴 关键参数变化", critical_diffs))

    # 重要参数变化
    if important_diffs:
        lines.extend(format_diff_section("🟡 重要参数变化", important_diffs))

    # 普通参数变化
    if normal_diffs:
        lines.extend(format_diff_section("⚪ 普通参数变化", normal_diffs))

    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


def diff_to_dict(diff: ConfigDiff) -> Dict[str, Any]:
    """
    将 ConfigDiff 转换为字典（用于 API 响应）

    Args:
        diff: 配置差异项

    Returns:
        字典表示
    """
    return {
        "path": diff.path,
        "type": diff.diff_type.value,
        "old_value": diff.old_value,
        "new_value": diff.new_value,
        "importance": diff.importance.value,
    }


def comparison_result_to_dict(result: ConfigComparisonResult) -> Dict[str, Any]:
    """
    将 ConfigComparisonResult 转换为字典（用于 API 响应）

    Args:
        result: 配置对比结果

    Returns:
        字典表示
    """
    return {
        "diffs": [diff_to_dict(d) for d in result.diffs],
        "added_count": result.added_count,
        "removed_count": result.removed_count,
        "modified_count": result.modified_count,
        "unchanged_count": result.unchanged_count,
        "has_critical_changes": result.has_critical_changes,
        "summary": result.summary,
    }


# ============== 配方对比辅助函数 ==============

def compare_recipes(recipe_id_a: str, recipe_id_b: str) -> Optional[ConfigComparisonResult]:
    """
    对比两个配方的默认配置

    Args:
        recipe_id_a: 配方 A 的 ID
        recipe_id_b: 配方 B 的 ID

    Returns:
        配置对比结果（如果配方不存在则返回 None）
    """
    from .recipes import RecipeRegistry

    recipe_a = RecipeRegistry.get(recipe_id_a)
    recipe_b = RecipeRegistry.get(recipe_id_b)

    if not recipe_a or not recipe_b:
        return None

    return compare_configs(
        recipe_a.default_config,
        recipe_b.default_config,
        name_a=recipe_id_a,
        name_b=recipe_id_b,
    )


def compare_jobs(job_uuid_a: str, job_uuid_b: str, session) -> Optional[ConfigComparisonResult]:
    """
    对比两个训练任务的配置

    Args:
        job_uuid_a: 任务 A 的 UUID
        job_uuid_b: 任务 B 的 UUID
        session: 数据库会话

    Returns:
        配置对比结果（如果任务不存在则返回 None）
    """
    from .database import JobRepository

    repo = JobRepository(session)
    job_a = repo.get_by_uuid(job_uuid_a)
    job_b = repo.get_by_uuid(job_uuid_b)

    if not job_a or not job_b:
        return None

    return compare_configs(
        job_a.config or {},
        job_b.config or {},
        name_a=f"Job {job_a.name}",
        name_b=f"Job {job_b.name}",
    )
