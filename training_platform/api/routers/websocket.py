"""
WebSocket Router for Real-time Metrics Streaming

Provides real-time updates for:
- Training metrics (loss, reward, KL divergence)
- GPU utilization
- Training logs
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import asyncio
import json
import logging

from sqlmodel import Session
from ...core.database import (
    get_session,
    JobRepository,
    MetricsRepository,
    TrainingMetric,
    TrainingLog,
    GpuUsageRecord,
)
from ...core.ray_runner import get_default_runner

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


class ConnectionManager:
    """Manages WebSocket connections for job monitoring"""

    def __init__(self):
        # job_uuid -> set of websocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Global broadcast connections
        self.broadcast_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, job_uuid: Optional[str] = None):
        """Accept and track a new WebSocket connection"""
        await websocket.accept()

        if job_uuid:
            if job_uuid not in self.active_connections:
                self.active_connections[job_uuid] = set()
            self.active_connections[job_uuid].add(websocket)
        else:
            self.broadcast_connections.add(websocket)

        logger.info(f"WebSocket connected for job: {job_uuid or 'broadcast'}")

    def disconnect(self, websocket: WebSocket, job_uuid: Optional[str] = None):
        """Remove a WebSocket connection"""
        if job_uuid and job_uuid in self.active_connections:
            self.active_connections[job_uuid].discard(websocket)
            if not self.active_connections[job_uuid]:
                del self.active_connections[job_uuid]
        else:
            self.broadcast_connections.discard(websocket)

        logger.info(f"WebSocket disconnected for job: {job_uuid or 'broadcast'}")

    async def send_to_job(self, job_uuid: str, message: Dict[str, Any]):
        """Send message to all connections watching a specific job"""
        connections = self.active_connections.get(job_uuid, set())
        dead_connections = set()

        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.active_connections[job_uuid].discard(conn)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connections"""
        dead_connections = set()

        for connection in self.broadcast_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.add(connection)

        # Clean up dead connections
        for conn in dead_connections:
            self.broadcast_connections.discard(conn)

        # Also send to all job-specific connections
        for job_uuid, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/jobs/{job_uuid}")
async def job_metrics_websocket(
    websocket: WebSocket,
    job_uuid: str,
):
    """
    WebSocket endpoint for real-time job metrics.

    Streams:
    - Training metrics (every step)
    - GPU usage (every 5 seconds)
    - Status updates
    """
    await manager.connect(websocket, job_uuid)

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "job_uuid": job_uuid,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (ping/pong, requests)
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0,  # Heartbeat timeout
                )

                # Handle client requests
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "request_metrics":
                    # Client requesting latest metrics
                    await send_latest_metrics(websocket, job_uuid)
                elif data.get("type") == "request_gpu":
                    await send_gpu_usage(websocket, job_uuid)

            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break

    except WebSocketDisconnect:
        manager.disconnect(websocket, job_uuid)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, job_uuid)


@router.websocket("/ws/jobs/{job_uuid}/logs")
async def job_logs_websocket(
    websocket: WebSocket,
    job_uuid: str,
):
    """
    WebSocket endpoint for streaming training logs.
    """
    await manager.connect(websocket, job_uuid)

    try:
        runner = get_default_runner()

        # Get Ray job ID if available
        # Note: In real implementation, look up from database

        await websocket.send_json({
            "type": "connected",
            "job_uuid": job_uuid,
        })

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0,
                )

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, job_uuid)
    except Exception as e:
        logger.error(f"Log WebSocket error: {e}")
        manager.disconnect(websocket, job_uuid)


@router.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for dashboard-wide updates.

    Streams:
    - Job status changes
    - Alert notifications
    - System status
    """
    await manager.connect(websocket)

    try:
        await websocket.send_json({
            "type": "connected",
            "timestamp": datetime.utcnow().isoformat(),
        })

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0,
                )

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
        manager.disconnect(websocket)


async def send_latest_metrics(websocket: WebSocket, job_uuid: str):
    """Send latest metrics to client"""
    # This would fetch from database in production
    await websocket.send_json({
        "type": "metrics",
        "job_uuid": job_uuid,
        "data": {
            "step": 0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "reward_mean": 0.0,
            "kl_divergence": 0.0,
        },
        "timestamp": datetime.utcnow().isoformat(),
    })


async def send_gpu_usage(websocket: WebSocket, job_uuid: str):
    """Send GPU usage data to client"""
    await websocket.send_json({
        "type": "gpu_usage",
        "job_uuid": job_uuid,
        "data": [],
        "timestamp": datetime.utcnow().isoformat(),
    })


# Functions to push updates to connected clients

async def push_metrics_update(job_uuid: str, metrics: Dict[str, Any]):
    """Push metrics update to all clients watching this job"""
    await manager.send_to_job(job_uuid, {
        "type": "metrics",
        "job_uuid": job_uuid,
        "data": metrics,
        "timestamp": datetime.utcnow().isoformat(),
    })


async def push_status_update(job_uuid: str, status: str, message: str = None):
    """Push status update to all clients watching this job"""
    await manager.send_to_job(job_uuid, {
        "type": "status",
        "job_uuid": job_uuid,
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Also broadcast to dashboard
    await manager.broadcast({
        "type": "job_status_change",
        "job_uuid": job_uuid,
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
    })


async def push_log_entry(job_uuid: str, level: str, message: str):
    """Push log entry to all clients watching this job"""
    await manager.send_to_job(job_uuid, {
        "type": "log",
        "job_uuid": job_uuid,
        "level": level,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    })


async def push_gpu_update(job_uuid: str, gpu_data: List[Dict[str, Any]]):
    """Push GPU usage update"""
    await manager.send_to_job(job_uuid, {
        "type": "gpu_usage",
        "job_uuid": job_uuid,
        "data": gpu_data,
        "timestamp": datetime.utcnow().isoformat(),
    })


async def push_alert(job_uuid: str, severity: str, message: str):
    """Push alert to dashboard"""
    await manager.broadcast({
        "type": "alert",
        "job_uuid": job_uuid,
        "severity": severity,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    })


# Background task for collecting and pushing metrics
class LogParser:
    """
    Parser for verl training logs.

    Supports two formats:
    1. Console format: step:100 - train/loss:0.5 - train/reward:0.8
    2. JSONL format: {"step": 100, "data": {"train/loss": 0.5, ...}}
    """

    import re

    # Regex patterns for console log format
    STEP_PATTERN = re.compile(r'step[:\s]*(\d+)')
    METRIC_PATTERN = re.compile(r'([a-zA-Z0-9_/]+)[:\s]+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

    # Known metric names for mapping
    METRIC_MAPPING = {
        'train/loss': 'total_loss',
        'train/policy_loss': 'policy_loss',
        'train/value_loss': 'value_loss',
        'train/reward': 'reward_mean',
        'train/reward_mean': 'reward_mean',
        'train/reward_std': 'reward_std',
        'train/kl': 'kl_divergence',
        'train/kl_divergence': 'kl_divergence',
        'train/entropy': 'entropy',
        'critic/loss': 'value_loss',
        'actor/loss': 'policy_loss',
        'reward/mean': 'reward_mean',
        'reward/std': 'reward_std',
        'kl/mean': 'kl_divergence',
    }

    @classmethod
    def parse_console_line(cls, line: str) -> Optional[Dict[str, Any]]:
        """Parse a console log line"""
        # Skip empty lines or non-metric lines
        if not line.strip() or 'step' not in line.lower():
            return None

        # Extract step number
        step_match = cls.STEP_PATTERN.search(line)
        if not step_match:
            return None

        step = int(step_match.group(1))
        metrics = {'step': step}

        # Extract all metric values
        for match in cls.METRIC_PATTERN.finditer(line):
            key = match.group(1)
            try:
                value = float(match.group(2))
                # Map to standard metric name
                standard_key = cls.METRIC_MAPPING.get(key, key)
                metrics[standard_key] = value
            except ValueError:
                continue

        return metrics if len(metrics) > 1 else None

    @classmethod
    def parse_jsonl_line(cls, line: str) -> Optional[Dict[str, Any]]:
        """Parse a JSONL log line"""
        try:
            data = json.loads(line.strip())
            if 'step' not in data:
                return None

            step = data['step']
            metrics = {'step': step}

            # Extract metrics from data field
            raw_data = data.get('data', {})
            for key, value in raw_data.items():
                if isinstance(value, (int, float)):
                    standard_key = cls.METRIC_MAPPING.get(key, key)
                    metrics[standard_key] = value

            return metrics if len(metrics) > 1 else None
        except json.JSONDecodeError:
            return None

    @classmethod
    def parse_logs(cls, logs: str) -> List[Dict[str, Any]]:
        """Parse multiple log lines and return list of metrics"""
        results = []
        seen_steps = set()

        for line in logs.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Try JSONL format first
            metrics = cls.parse_jsonl_line(line)
            if not metrics:
                # Fall back to console format
                metrics = cls.parse_console_line(line)

            if metrics and metrics['step'] not in seen_steps:
                seen_steps.add(metrics['step'])
                results.append(metrics)

        return results

    @classmethod
    def extract_latest_metrics(cls, logs: str, last_step: int = 0) -> List[Dict[str, Any]]:
        """Extract only new metrics since last_step"""
        all_metrics = cls.parse_logs(logs)
        return [m for m in all_metrics if m['step'] > last_step]


class MetricsCollector:
    """
    Background task for collecting metrics from running jobs.

    This class polls Ray for job logs, parses metrics, stores them in DB,
    and pushes updates to WebSocket clients.
    """

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_steps: Dict[str, int] = {}  # job_uuid -> last processed step
        self._log_offsets: Dict[str, int] = {}  # job_uuid -> last log offset

    async def start(self):
        """Start the metrics collection loop"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collector started")

    async def stop(self):
        """Stop the metrics collection loop"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collector stopped")

    async def _collection_loop(self):
        """Main collection loop"""
        while self._running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(2)  # Collect every 2 seconds
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def _collect_all_metrics(self):
        """
        Collect metrics from all running jobs.

        1. Query database for running jobs
        2. Query Ray for job status/metrics
        3. Parse training logs for metrics
        4. Store in database
        5. Push updates to WebSocket clients
        """
        from sqlmodel import Session
        from ...core.database import (
            engine,
            JobRepository,
            MetricsRepository,
            TrainingJob,
            TrainingMetric,
            JobStatus,
        )
        from ...core.ray_runner import get_default_runner

        runner = get_default_runner()

        with Session(engine) as session:
            job_repo = JobRepository(session)
            metrics_repo = MetricsRepository(session)

            # Get all running jobs
            running_jobs, _ = job_repo.list_jobs(status=JobStatus.RUNNING, limit=100)

            for job in running_jobs:
                try:
                    await self._collect_job_metrics(
                        job, runner, job_repo, metrics_repo, session
                    )
                except Exception as e:
                    logger.error(f"Error collecting metrics for job {job.uuid}: {e}")

    async def _collect_job_metrics(
        self,
        job: "TrainingJob",
        runner,
        job_repo: "JobRepository",
        metrics_repo: "MetricsRepository",
        session: "Session",
    ):
        """Collect metrics for a single job"""
        from ...core.database import TrainingMetric, JobStatus

        # Get Ray job status if available
        if job.ray_job_id:
            status_info = runner.get_job_status(job.ray_job_id)
            ray_status = status_info.get('status', 'unknown')

            # Update job status if changed
            if ray_status == 'SUCCEEDED' and job.status != JobStatus.COMPLETED:
                job.status = JobStatus.COMPLETED
                from datetime import datetime
                job.completed_at = datetime.utcnow()
                job_repo.update(job)
                await push_status_update(job.uuid, "completed", "训练完成")
            elif ray_status == 'FAILED' and job.status != JobStatus.FAILED:
                job.status = JobStatus.FAILED
                job_repo.update(job)
                await push_status_update(job.uuid, "failed", "训练失败")

            # Get logs and parse metrics
            logs = runner.get_job_logs(job.ray_job_id)
            if logs:
                last_step = self._last_steps.get(job.uuid, job.current_step or 0)
                new_metrics = LogParser.extract_latest_metrics(logs, last_step)

                for metrics in new_metrics:
                    step = metrics.pop('step')

                    # Create metric record
                    metric = TrainingMetric(
                        job_uuid=job.uuid,
                        step=step,
                        epoch=job.current_epoch,
                        policy_loss=metrics.get('policy_loss'),
                        value_loss=metrics.get('value_loss'),
                        total_loss=metrics.get('total_loss'),
                        reward_mean=metrics.get('reward_mean'),
                        reward_std=metrics.get('reward_std'),
                        kl_divergence=metrics.get('kl_divergence'),
                        entropy=metrics.get('entropy'),
                        learning_rate=metrics.get('learning_rate'),
                        extra_metrics={k: v for k, v in metrics.items()
                                      if k not in ['policy_loss', 'value_loss', 'total_loss',
                                                  'reward_mean', 'reward_std', 'kl_divergence', 'entropy', 'learning_rate']},
                    )
                    metrics_repo.add_metric(metric)

                    # Update job progress
                    job.current_step = step
                    job_repo.update(job)

                    # Push to WebSocket
                    await push_metrics_update(job.uuid, {
                        'step': step,
                        'policy_loss': metric.policy_loss,
                        'value_loss': metric.value_loss,
                        'total_loss': metric.total_loss,
                        'reward_mean': metric.reward_mean,
                        'reward_std': metric.reward_std,
                        'kl_divergence': metric.kl_divergence,
                        'entropy': metric.entropy,
                        **metric.extra_metrics,
                    })

                    self._last_steps[job.uuid] = step
                    logger.debug(f"Job {job.uuid} step {step} metrics collected")

    async def report_metrics(
        self,
        job_uuid: str,
        step: int,
        metrics: Dict[str, Any],
    ):
        """
        Handle metrics reported via Push mode (HTTP callback).

        This is called when training script POSTs to /api/v1/monitoring/report
        """
        from sqlmodel import Session
        from ...core.database import (
            engine,
            JobRepository,
            MetricsRepository,
            TrainingMetric,
        )

        with Session(engine) as session:
            job_repo = JobRepository(session)
            metrics_repo = MetricsRepository(session)

            job = job_repo.get_by_uuid(job_uuid)
            if not job:
                logger.warning(f"Metrics reported for unknown job: {job_uuid}")
                return False

            # Create metric record
            metric = TrainingMetric(
                job_uuid=job_uuid,
                step=step,
                epoch=metrics.get('epoch', 0),
                policy_loss=metrics.get('policy_loss'),
                value_loss=metrics.get('value_loss'),
                total_loss=metrics.get('total_loss'),
                reward_mean=metrics.get('reward_mean'),
                reward_std=metrics.get('reward_std'),
                kl_divergence=metrics.get('kl_divergence'),
                entropy=metrics.get('entropy'),
                learning_rate=metrics.get('learning_rate'),
                extra_metrics={k: v for k, v in metrics.items()
                              if k not in ['epoch', 'policy_loss', 'value_loss', 'total_loss',
                                          'reward_mean', 'reward_std', 'kl_divergence', 'entropy', 'learning_rate']},
            )
            metrics_repo.add_metric(metric)

            # Update job progress
            job.current_step = step
            if 'epoch' in metrics:
                job.current_epoch = metrics['epoch']
            job_repo.update(job)

            # Push to WebSocket
            await push_metrics_update(job_uuid, {
                'step': step,
                **metrics,
            })

            self._last_steps[job_uuid] = step
            return True


# Global metrics collector
metrics_collector = MetricsCollector()
