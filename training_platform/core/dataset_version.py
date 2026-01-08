"""
Dataset Versioning (Phase 2)

数据版本化和血缘追踪系统。

主要功能：
1. 计算数据集文件的 hash（数据指纹）
2. 记录数据集的版本信息
3. 检测数据集是否发生变化
4. 提供数据血缘追溯功能

设计理念：
- 数据的 hash 是其唯一标识（内容相同 hash 就相同）
- 每个训练任务都关联一个数据版本快照
- 可以追溯任何训练使用的确切数据版本
"""

import hashlib
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import json


def calculate_file_hash(
    file_path: str,
    algorithm: str = "sha256",
    chunk_size: int = 8192
) -> str:
    """
    计算文件的 hash 值（数据指纹）

    使用分块读取，避免大文件占用过多内存。

    Args:
        file_path: 文件路径
        algorithm: Hash 算法（md5, sha1, sha256）
        chunk_size: 读取块大小（字节）

    Returns:
        文件的 hash 值（十六进制字符串）

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的 hash 算法
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 创建 hash 对象
    if algorithm == "md5":
        hasher = hashlib.md5()
    elif algorithm == "sha1":
        hasher = hashlib.sha1()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"不支持的 hash 算法: {algorithm}")

    # 分块读取文件并计算 hash
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    获取文件的元数据

    Args:
        file_path: 文件路径

    Returns:
        文件元数据字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    stat = os.stat(file_path)

    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size": stat.st_size,  # 字节
        "file_size_mb": round(stat.st_size / (1024 * 1024), 2),  # MB
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def count_dataset_samples(file_path: str, format: str = "jsonl") -> Optional[int]:
    """
    统计数据集样本数量

    Args:
        file_path: 数据集文件路径
        format: 数据格式（jsonl, parquet, csv）

    Returns:
        样本数量（如果无法统计则返回 None）
    """
    try:
        if format == "jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)

        elif format == "parquet":
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(file_path)
                return table.num_rows
            except ImportError:
                return None

        elif format == "csv":
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f) - 1  # 减去 header

        else:
            return None

    except Exception as e:
        print(f"统计样本数量失败: {e}")
        return None


def create_dataset_snapshot(
    file_path: str,
    dataset_name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    创建数据集快照

    记录数据集在某个时间点的完整状态。

    Args:
        file_path: 数据集文件路径
        dataset_name: 数据集名称（可选，默认使用文件名）
        description: 数据集描述
        tags: 标签列表

    Returns:
        数据集快照字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 计算 hash
    file_hash = calculate_file_hash(file_path, algorithm="sha256")

    # 获取元数据
    metadata = get_file_metadata(file_path)

    # 推测数据格式
    ext = Path(file_path).suffix.lower()
    if ext == ".jsonl":
        format_type = "jsonl"
    elif ext == ".parquet":
        format_type = "parquet"
    elif ext == ".csv":
        format_type = "csv"
    else:
        format_type = "unknown"

    # 统计样本数量
    num_samples = count_dataset_samples(file_path, format_type)

    # 创建快照
    snapshot = {
        "dataset_name": dataset_name or metadata["file_name"],
        "file_path": file_path,
        "file_hash": file_hash,
        "hash_algorithm": "sha256",
        "file_size": metadata["file_size"],
        "file_size_mb": metadata["file_size_mb"],
        "format": format_type,
        "num_samples": num_samples,
        "description": description,
        "tags": tags or [],
        "created_at": datetime.utcnow().isoformat(),
        "modified_at": metadata["modified_at"],
    }

    return snapshot


def compare_dataset_versions(snapshot_a: Dict[str, Any], snapshot_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    对比两个数据集版本

    Args:
        snapshot_a: 数据集快照 A
        snapshot_b: 数据集快照 B

    Returns:
        对比结果
    """
    hash_changed = snapshot_a["file_hash"] != snapshot_b["file_hash"]
    size_changed = snapshot_a["file_size"] != snapshot_b["file_size"]
    samples_changed = snapshot_a.get("num_samples") != snapshot_b.get("num_samples")

    return {
        "identical": not hash_changed,
        "hash_changed": hash_changed,
        "size_changed": size_changed,
        "samples_changed": samples_changed,
        "hash_a": snapshot_a["file_hash"],
        "hash_b": snapshot_b["file_hash"],
        "size_diff_mb": snapshot_b["file_size_mb"] - snapshot_a["file_size_mb"],
        "samples_diff": (snapshot_b.get("num_samples") or 0) - (snapshot_a.get("num_samples") or 0),
    }


class DatasetVersionManager:
    """
    数据集版本管理器

    负责管理数据集的版本信息和血缘追踪。
    """

    def __init__(self, session):
        """
        初始化数据集版本管理器

        Args:
            session: 数据库会话
        """
        self.session = session

    def snapshot_dataset(
        self,
        file_path: str,
        dataset_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        为数据集创建版本快照并保存到数据库

        Args:
            file_path: 数据集文件路径
            dataset_name: 数据集名称
            description: 描述
            tags: 标签

        Returns:
            数据集快照
        """
        # 创建快照
        snapshot = create_dataset_snapshot(
            file_path=file_path,
            dataset_name=dataset_name,
            description=description,
            tags=tags,
        )

        # TODO: 保存到数据库（需要 DatasetVersion 表）
        # 这里暂时返回快照，等数据库表创建后再实现

        return snapshot

    def get_dataset_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        根据 hash 查找数据集版本

        Args:
            file_hash: 文件 hash

        Returns:
            数据集快照（如果找到）
        """
        # TODO: 从数据库查询
        # 这里暂时返回 None，等数据库表创建后再实现
        return None

    def list_dataset_versions(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        列出数据集的所有版本

        Args:
            dataset_name: 数据集名称

        Returns:
            版本列表（按时间倒序）
        """
        # TODO: 从数据库查询
        # 这里暂时返回空列表，等数据库表创建后再实现
        return []


# ============== SSH 数据同步验证 ==============

def verify_remote_dataset(
    local_path: str,
    remote_path: str,
    ssh_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    验证远程数据集是否与本地一致

    Args:
        local_path: 本地文件路径
        remote_path: 远程文件路径
        ssh_config: SSH 配置

    Returns:
        验证结果
    """
    # 计算本地文件 hash
    local_hash = calculate_file_hash(local_path)

    # TODO: SSH 连接到远程服务器计算 hash
    # 这里暂时返回模拟结果
    # 实际实现需要使用 SSHRunner 连接远程服务器

    return {
        "verified": False,  # 暂时返回 False
        "local_hash": local_hash,
        "remote_hash": None,
        "message": "远程验证功能待实现（需要 SSHRunner）",
    }


# ============== 数据血缘追踪 ==============

def trace_dataset_lineage(dataset_hash: str, session) -> Dict[str, Any]:
    """
    追溯数据集的血缘关系

    查找使用了该数据集版本的所有训练任务。

    Args:
        dataset_hash: 数据集 hash
        session: 数据库会话

    Returns:
        血缘信息
    """
    # TODO: 查询使用了该数据集的所有任务
    # SELECT * FROM training_jobs WHERE dataset_version_hash = ?

    return {
        "dataset_hash": dataset_hash,
        "used_by_jobs": [],  # 待实现
        "num_jobs": 0,
    }
