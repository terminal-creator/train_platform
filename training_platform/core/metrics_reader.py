"""
Metrics Reader - Phase 1.3

从本地或远程读取训练指标文件（实时监控用）

工作流程：
1. 根据运行模式选择 LocalMetricsReader 或 SSHMetricsReader
2. 读取 {job_id}_metrics.jsonl 文件（支持增量读取）
3. 读取 {job_id}_status.json 文件
4. 返回解析后的指标数据

为什么这么设计：
- 抽象基类：统一本地和远程的读取接口
- 增量读取：只读取新增的指标，避免重复处理
- 错误容忍：文件不存在或读取失败返回空列表，不中断监控
- 支持 tail：实时监控场景下，持续读取最新数据
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsReader(ABC):
    """
    指标读取器基类

    为什么抽象：
    - 本地和远程读取方式不同，但接口相同
    - 便于测试和扩展（未来可能支持 S3、NFS 等）
    """

    def __init__(self, job_id: str, metrics_dir: str):
        """
        Args:
            job_id: 训练任务 ID
            metrics_dir: 指标文件目录
        """
        self.job_id = job_id
        self.metrics_dir = metrics_dir
        self.metrics_file = f"{job_id}_metrics.jsonl"
        self.status_file = f"{job_id}_status.json"

        # 增量读取：记录已读取的行数
        self._last_read_line = 0

    @abstractmethod
    def read_file(self, filename: str) -> Optional[str]:
        """
        读取文件内容

        Returns:
            文件内容（字符串），失败返回 None
        """
        pass

    @abstractmethod
    def file_exists(self, filename: str) -> bool:
        """检查文件是否存在"""
        pass

    def read_metrics(self, from_step: Optional[int] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        读取指标数据（支持增量读取）

        Args:
            from_step: 从哪个 step 开始读取（None 表示从头读取）
            limit: 最多读取多少条（None 表示全部读取）

        Returns:
            指标列表，每个元素是一个 JSON 对象

        为什么这么设计：
        - 增量读取：避免重复处理已读取的数据
        - 限制条数：防止一次读取过多数据导致内存占用过高
        - 错误容忍：解析失败的行会跳过，不影响其他数据
        """
        content = self.read_file(self.metrics_file)
        if not content:
            return []

        metrics = []
        lines = content.strip().split('\n')

        for line_num, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            try:
                metric = json.loads(line)

                # 过滤：只返回 step >= from_step 的数据
                if from_step is not None and metric.get('step', 0) < from_step:
                    continue

                metrics.append(metric)

                # 限制条数
                if limit and len(metrics) >= limit:
                    break

            except json.JSONDecodeError as e:
                logger.warning(f"[MetricsReader] Failed to parse line {line_num}: {e}")
                continue

        return metrics

    def read_metrics_incremental(self, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], int]:
        """
        增量读取指标（只返回新增的数据）

        Returns:
            (新增指标列表, 新的读取位置)

        为什么需要这个方法：
        - WebSocket 实时推送场景：每次调用只返回新增的指标
        - 记录读取位置：下次调用时从上次的位置继续读取
        - 高效：不需要每次都从头读取整个文件
        """
        content = self.read_file(self.metrics_file)
        if not content:
            return [], self._last_read_line

        lines = content.strip().split('\n')
        new_metrics = []

        # 从上次读取的位置开始
        for i in range(self._last_read_line, len(lines)):
            line = lines[i]
            if not line.strip():
                continue

            try:
                metric = json.loads(line)
                new_metrics.append(metric)

                # 限制条数
                if limit and len(new_metrics) >= limit:
                    break

            except json.JSONDecodeError as e:
                logger.warning(f"[MetricsReader] Failed to parse line {i+1}: {e}")
                continue

        # 更新读取位置
        self._last_read_line = min(len(lines), self._last_read_line + len(new_metrics))

        return new_metrics, self._last_read_line

    def read_status(self) -> Optional[Dict[str, Any]]:
        """
        读取状态文件

        Returns:
            状态字典，失败返回 None

        状态文件格式：
        {
            "job_id": "xxx",
            "status": "running",
            "current_step": 100,
            "total_steps": 1000,
            "anomaly_detected": false,
            "anomaly_reason": null
        }
        """
        content = self.read_file(self.status_file)
        if not content:
            return None

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"[MetricsReader] Failed to parse status file: {e}")
            return None

    def get_latest_metric(self) -> Optional[Dict[str, Any]]:
        """
        获取最新的一条指标

        为什么需要：
        - 快速获取当前训练状态（不需要读取整个文件）
        - WebSocket 连接建立时，发送初始状态
        """
        metrics = self.read_metrics(limit=None)  # 读取全部
        return metrics[-1] if metrics else None

    def reset_position(self):
        """重置读取位置（重新从头读取）"""
        self._last_read_line = 0


class LocalMetricsReader(MetricsReader):
    """
    本地指标读取器

    直接从本地文件系统读取指标文件
    """

    def __init__(self, job_id: str, metrics_dir: str):
        super().__init__(job_id, metrics_dir)
        self.metrics_dir_path = Path(metrics_dir)

    def read_file(self, filename: str) -> Optional[str]:
        """从本地文件系统读取文件"""
        file_path = self.metrics_dir_path / filename

        if not file_path.exists():
            logger.debug(f"[LocalMetricsReader] File not found: {file_path}")
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"[LocalMetricsReader] Failed to read {file_path}: {e}")
            return None

    def file_exists(self, filename: str) -> bool:
        """检查本地文件是否存在"""
        file_path = self.metrics_dir_path / filename
        return file_path.exists()


class SSHMetricsReader(MetricsReader):
    """
    SSH 指标读取器

    通过 SSH 从远程服务器读取指标文件

    为什么需要：
    - 训练在远程 GPU 服务器运行，指标文件在远程
    - 通过 SSH 读取文件内容，传输到本地处理
    - 复用 SSHManager 的连接池和错误处理
    """

    def __init__(self, job_id: str, metrics_dir: str, ssh_manager):
        """
        Args:
            job_id: 训练任务 ID
            metrics_dir: 远程服务器上的指标文件目录
            ssh_manager: SSHManager 实例
        """
        super().__init__(job_id, metrics_dir)
        self.ssh_manager = ssh_manager

    def read_file(self, filename: str) -> Optional[str]:
        """通过 SSH 读取远程文件"""
        remote_path = f"{self.metrics_dir}/{filename}"

        try:
            content = self.ssh_manager.read_file(remote_path)
            return content
        except Exception as e:
            logger.error(f"[SSHMetricsReader] Failed to read {remote_path}: {e}")
            return None

    def file_exists(self, filename: str) -> bool:
        """检查远程文件是否存在"""
        remote_path = f"{self.metrics_dir}/{filename}"

        try:
            # 使用 test -f 检查文件是否存在
            result = self.ssh_manager.execute_command(f"test -f {remote_path} && echo 'exists' || echo 'not_found'")
            return 'exists' in result
        except Exception as e:
            logger.error(f"[SSHMetricsReader] Failed to check file existence: {e}")
            return False


def create_metrics_reader(
    job_id: str,
    metrics_dir: str,
    run_mode: str = "local",
    ssh_manager = None
) -> MetricsReader:
    """
    工厂函数：根据运行模式创建对应的指标读取器

    Args:
        job_id: 训练任务 ID
        metrics_dir: 指标文件目录
        run_mode: 运行模式（"local" 或 "ssh"）
        ssh_manager: SSHManager 实例（仅在 SSH 模式需要）

    Returns:
        MetricsReader 实例

    为什么提供工厂函数：
    - 简化调用：调用方不需要知道具体的读取器类型
    - 便于扩展：未来添加新的读取器类型只需修改这个函数
    """
    if run_mode == "ssh":
        if not ssh_manager:
            raise ValueError("ssh_manager is required for SSH mode")
        return SSHMetricsReader(job_id, metrics_dir, ssh_manager)
    else:
        return LocalMetricsReader(job_id, metrics_dir)
