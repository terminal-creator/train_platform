"""
Ray Job Submission Runner

Provides robust job management using Ray's Job Submission API.
This decouples the training workload from the FastAPI process.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from .verl_adapter import VerlTrainingConfig, create_ray_entrypoint

logger = logging.getLogger(__name__)

# Ray Job Submission API (optional import)
try:
    from ray.job_submission import JobSubmissionClient, JobStatus as RayJobStatus
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not installed. Ray job submission will be disabled.")


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass
class RayJobConfig:
    """Configuration for Ray job submission"""
    # Job identification
    job_id: str
    name: str

    # verl training configuration (preferred)
    verl_config: Optional[VerlTrainingConfig] = None

    # Legacy fields (for backwards compatibility)
    algorithm: str = ""
    config_path: str = ""

    # Resources
    num_gpus: int = 8
    num_cpus: int = 16
    memory_gb: int = 64

    # Runtime environment
    working_dir: Optional[str] = None
    pip_packages: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)

    # Verl specific (legacy)
    model_path: str = ""
    train_data_path: str = ""
    output_dir: str = "./outputs"

    def to_entrypoint(self) -> str:
        """Generate entrypoint command using verl's Hydra-style args"""
        # Use VerlTrainingConfig if available (preferred)
        if self.verl_config:
            return create_ray_entrypoint(self.verl_config)

        # Legacy fallback: build command from individual fields
        from .verl_adapter import VerlAlgorithm

        config = VerlTrainingConfig(
            model_path=self.model_path,
            algorithm=VerlAlgorithm(self.algorithm) if self.algorithm else VerlAlgorithm.GRPO,
            train_data_path=self.train_data_path,
            num_gpus=self.num_gpus,
            output_dir=self.output_dir,
        )
        return create_ray_entrypoint(config)

    def to_runtime_env(self) -> Dict[str, Any]:
        """Generate Ray runtime environment"""
        runtime_env = {}

        if self.working_dir:
            runtime_env["working_dir"] = self.working_dir

        if self.pip_packages:
            runtime_env["pip"] = self.pip_packages

        if self.env_vars:
            runtime_env["env_vars"] = self.env_vars

        return runtime_env


class RayJobRunner:
    """
    Manages training jobs via Ray Job Submission API.

    This provides:
    - Decoupled execution (jobs run independently of FastAPI)
    - Fault tolerance (Ray handles job recovery)
    - Log streaming
    - Resource management
    """

    def __init__(
        self,
        address: str = None,
        create_cluster_if_needed: bool = True,
    ):
        """
        Initialize Ray job runner.

        Args:
            address: Ray cluster address. If None, uses RAY_ADDRESS env var
                    or defaults to localhost.
            create_cluster_if_needed: Start local Ray cluster if not connected
        """
        self.address = address or os.getenv("RAY_ADDRESS", "http://127.0.0.1:8265")
        self._client: Optional["JobSubmissionClient"] = None
        self._connected = False

        if RAY_AVAILABLE and create_cluster_if_needed:
            self._ensure_connection()

    def _ensure_connection(self):
        """Ensure connection to Ray cluster"""
        if self._connected:
            return

        try:
            self._client = JobSubmissionClient(self.address)
            # Test connection
            self._client.list_jobs()
            self._connected = True
            logger.info(f"Connected to Ray cluster at {self.address}")
        except Exception as e:
            logger.warning(f"Failed to connect to Ray cluster: {e}")
            self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to Ray cluster"""
        return self._connected and self._client is not None

    def submit_job(self, config: RayJobConfig) -> Dict[str, Any]:
        """
        Submit a training job to Ray cluster.

        Args:
            config: Job configuration

        Returns:
            Dict with job_id and submission status
        """
        if not RAY_AVAILABLE:
            return {
                "success": False,
                "error": "Ray not installed",
                "message": "Install ray[default] to enable job submission",
            }

        self._ensure_connection()
        if not self.is_connected:
            return {
                "success": False,
                "error": "Not connected to Ray cluster",
                "message": f"Cannot connect to Ray at {self.address}",
            }

        try:
            entrypoint = config.to_entrypoint()
            runtime_env = config.to_runtime_env()

            # Submit job
            ray_job_id = self._client.submit_job(
                entrypoint=entrypoint,
                runtime_env=runtime_env,
                submission_id=config.job_id,
                metadata={
                    "name": config.name,
                    "algorithm": config.algorithm,
                    "submitted_at": datetime.utcnow().isoformat(),
                },
            )

            logger.info(f"Submitted job {config.job_id} as Ray job {ray_job_id}")

            return {
                "success": True,
                "ray_job_id": ray_job_id,
                "job_id": config.job_id,
                "entrypoint": entrypoint,
            }

        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_job_status(self, ray_job_id: str) -> Dict[str, Any]:
        """Get status of a Ray job"""
        if not self.is_connected:
            return {"status": "unknown", "error": "Not connected"}

        try:
            status = self._client.get_job_status(ray_job_id)
            info = self._client.get_job_info(ray_job_id)

            return {
                "status": status.value if hasattr(status, 'value') else str(status),
                "start_time": info.start_time if info else None,
                "end_time": info.end_time if info else None,
                "metadata": info.metadata if info else {},
            }
        except Exception as e:
            return {"status": "unknown", "error": str(e)}

    def get_job_logs(self, ray_job_id: str) -> str:
        """Get logs from a Ray job"""
        if not self.is_connected:
            return "Not connected to Ray cluster"

        try:
            return self._client.get_job_logs(ray_job_id)
        except Exception as e:
            return f"Error getting logs: {e}"

    async def stream_job_logs(
        self,
        ray_job_id: str,
        callback: Callable[[str], None],
    ):
        """Stream job logs asynchronously"""
        if not self.is_connected:
            callback("Not connected to Ray cluster")
            return

        try:
            async for lines in self._client.tail_job_logs(ray_job_id):
                callback(lines)
        except Exception as e:
            callback(f"Error streaming logs: {e}")

    def stop_job(self, ray_job_id: str) -> Dict[str, Any]:
        """Stop a running Ray job"""
        if not self.is_connected:
            return {"success": False, "error": "Not connected"}

        try:
            self._client.stop_job(ray_job_id)
            return {"success": True, "message": f"Job {ray_job_id} stop requested"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def delete_job(self, ray_job_id: str) -> Dict[str, Any]:
        """Delete a Ray job"""
        if not self.is_connected:
            return {"success": False, "error": "Not connected"}

        try:
            self._client.delete_job(ray_job_id)
            return {"success": True, "message": f"Job {ray_job_id} deleted"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all Ray jobs"""
        if not self.is_connected:
            return []

        try:
            jobs = self._client.list_jobs()
            return [
                {
                    "ray_job_id": job.submission_id,
                    "status": job.status.value if hasattr(job.status, 'value') else str(job.status),
                    "metadata": job.metadata,
                }
                for job in jobs
            ]
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []


class LocalJobRunner:
    """
    Fallback job runner using subprocess when Ray is not available.

    This is less robust but allows local development without Ray cluster.
    """

    def __init__(self):
        self._processes: Dict[str, Any] = {}
        self._logs: Dict[str, List[str]] = {}

    def submit_job(self, config: RayJobConfig) -> Dict[str, Any]:
        """Submit a job locally using subprocess"""
        import subprocess
        import threading

        try:
            entrypoint = config.to_entrypoint()
            cmd = entrypoint.split()

            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=config.working_dir,
                env={**os.environ, **(config.env_vars or {})},
            )

            self._processes[config.job_id] = process
            self._logs[config.job_id] = []

            # Stream logs in background
            def stream_output():
                for line in process.stdout:
                    self._logs[config.job_id].append(line)

            thread = threading.Thread(target=stream_output, daemon=True)
            thread.start()

            return {
                "success": True,
                "job_id": config.job_id,
                "pid": process.pid,
                "mode": "local",
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of local job"""
        process = self._processes.get(job_id)
        if not process:
            return {"status": "unknown", "error": "Job not found"}

        poll = process.poll()
        if poll is None:
            status = "RUNNING"
        elif poll == 0:
            status = "SUCCEEDED"
        else:
            status = "FAILED"

        return {"status": status, "returncode": poll}

    def get_job_logs(self, job_id: str) -> str:
        """Get logs from local job"""
        logs = self._logs.get(job_id, [])
        return "".join(logs)

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """Stop a local job"""
        process = self._processes.get(job_id)
        if not process:
            return {"success": False, "error": "Job not found"}

        try:
            process.terminate()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_job_runner(prefer_ray: bool = True) -> Any:
    """
    Get appropriate job runner.

    Args:
        prefer_ray: If True, try Ray first, fall back to local

    Returns:
        RayJobRunner if available and connected, else LocalJobRunner
    """
    if prefer_ray and RAY_AVAILABLE:
        runner = RayJobRunner()
        if runner.is_connected:
            return runner
        logger.info("Ray not available, using local job runner")

    return LocalJobRunner()


# Singleton instance
_job_runner: Optional[Any] = None


def get_default_runner() -> Any:
    """Get default job runner singleton"""
    global _job_runner
    if _job_runner is None:
        _job_runner = get_job_runner()
    return _job_runner
