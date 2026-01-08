"""
SSH Job Runner

Provides remote job execution via SSH connection to GPU servers.
Supports real-time log streaming and GPU monitoring.
"""

import os
import logging
import asyncio
import threading
import time
import re
import uuid
from typing import Dict, Any, Optional, List, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

try:
    import paramiko
    from paramiko import SSHClient, AutoAddPolicy, RSAKey, Ed25519Key
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None

from .verl_adapter import VerlTrainingConfig, create_ray_entrypoint
from .command_utils import SafeCommands, build_command, validate_path, validate_integer

logger = logging.getLogger(__name__)


class SSHJobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


@dataclass
class SSHConfig:
    """SSH connection configuration"""
    host: str
    port: int = 22
    username: str = ""
    password: Optional[str] = None
    key_path: Optional[str] = None
    key_passphrase: Optional[str] = None

    # Remote environment
    working_dir: str = "~/verl_jobs"
    conda_env: Optional[str] = None
    python_path: Optional[str] = None

    # Timeout settings
    connect_timeout: int = 30
    command_timeout: int = 3600  # 1 hour default for training

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "password": "***" if self.password else None,
            "key_path": self.key_path,
            "working_dir": self.working_dir,
            "conda_env": self.conda_env,
        }


@dataclass
class SSHJobConfig:
    """Configuration for SSH job submission"""
    job_id: str
    name: str

    # verl training configuration
    verl_config: Optional[VerlTrainingConfig] = None

    # Resources
    num_gpus: int = 8

    # Environment
    env_vars: Dict[str, str] = field(default_factory=dict)

    # Output
    log_file: Optional[str] = None

    def to_entrypoint(self) -> str:
        """Generate entrypoint command"""
        if self.verl_config:
            return create_ray_entrypoint(self.verl_config)
        return ""


class SSHConnection:
    """Manages a single SSH connection"""

    def __init__(self, config: SSHConfig):
        self.config = config
        self._client: Optional["SSHClient"] = None
        self._sftp = None
        self._connected = False
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Establish SSH connection"""
        if not PARAMIKO_AVAILABLE:
            logger.error("paramiko not installed. Run: pip install paramiko")
            return False

        with self._lock:
            if self._connected:
                return True

            try:
                self._client = SSHClient()
                self._client.set_missing_host_key_policy(AutoAddPolicy())

                connect_kwargs = {
                    "hostname": self.config.host,
                    "port": self.config.port,
                    "username": self.config.username,
                    "timeout": self.config.connect_timeout,
                }

                # Authentication
                if self.config.key_path:
                    key_path = os.path.expanduser(self.config.key_path)
                    if os.path.exists(key_path):
                        try:
                            # Try RSA first
                            pkey = RSAKey.from_private_key_file(
                                key_path,
                                password=self.config.key_passphrase
                            )
                        except:
                            # Try Ed25519
                            pkey = Ed25519Key.from_private_key_file(
                                key_path,
                                password=self.config.key_passphrase
                            )
                        connect_kwargs["pkey"] = pkey
                elif self.config.password:
                    connect_kwargs["password"] = self.config.password

                self._client.connect(**connect_kwargs)
                self._connected = True
                logger.info(f"Connected to {self.config.host}:{self.config.port}")
                return True

            except Exception as e:
                logger.error(f"SSH connection failed: {e}")
                self._connected = False
                return False

    def disconnect(self):
        """Close SSH connection"""
        with self._lock:
            if self._sftp:
                self._sftp.close()
                self._sftp = None
            if self._client:
                self._client.close()
                self._client = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def exec_command(
        self,
        command: str,
        timeout: Optional[int] = None
    ) -> tuple[int, str, str]:
        """
        Execute command on remote server.

        Returns:
            Tuple of (exit_code, stdout, stderr)
        """
        if not self.is_connected:
            if not self.connect():
                return -1, "", "Not connected"

        try:
            timeout = timeout or self.config.command_timeout
            stdin, stdout, stderr = self._client.exec_command(
                command,
                timeout=timeout
            )

            exit_code = stdout.channel.recv_exit_status()
            stdout_str = stdout.read().decode('utf-8', errors='replace')
            stderr_str = stderr.read().decode('utf-8', errors='replace')

            return exit_code, stdout_str, stderr_str

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return -1, "", str(e)

    def exec_command_stream(
        self,
        command: str,
        callback: Callable[[str], None],
        timeout: Optional[int] = None
    ) -> int:
        """
        Execute command with streaming output.

        Args:
            command: Command to execute
            callback: Function called with each line of output
            timeout: Command timeout

        Returns:
            Exit code
        """
        if not self.is_connected:
            if not self.connect():
                callback("Error: Not connected to SSH server")
                return -1

        try:
            transport = self._client.get_transport()
            channel = transport.open_session()
            channel.settimeout(timeout or self.config.command_timeout)
            channel.exec_command(command)

            # Stream stdout
            while True:
                if channel.recv_ready():
                    data = channel.recv(4096).decode('utf-8', errors='replace')
                    if data:
                        for line in data.splitlines(keepends=True):
                            callback(line)

                if channel.recv_stderr_ready():
                    data = channel.recv_stderr(4096).decode('utf-8', errors='replace')
                    if data:
                        for line in data.splitlines(keepends=True):
                            callback(f"[stderr] {line}")

                if channel.exit_status_ready():
                    # Drain remaining data
                    while channel.recv_ready():
                        data = channel.recv(4096).decode('utf-8', errors='replace')
                        if data:
                            callback(data)
                    break

                time.sleep(0.1)

            return channel.recv_exit_status()

        except Exception as e:
            callback(f"Error: {e}")
            return -1

    def get_sftp(self):
        """Get SFTP client"""
        if not self.is_connected:
            if not self.connect():
                return None

        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return self._sftp

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to remote server"""
        sftp = self.get_sftp()
        if not sftp:
            return False

        try:
            sftp.put(local_path, remote_path)
            return True
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from remote server"""
        sftp = self.get_sftp()
        if not sftp:
            return False

        try:
            sftp.get(remote_path, local_path)
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False


class SSHJobRunner:
    """
    Manages training jobs via SSH to remote GPU servers.

    Features:
    - Remote command execution
    - Real-time log streaming
    - GPU monitoring
    - Process management
    """

    def __init__(self, ssh_config: SSHConfig):
        self.ssh_config = ssh_config
        self._connection: Optional[SSHConnection] = None
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._log_threads: Dict[str, threading.Thread] = {}
        self._stop_flags: Dict[str, threading.Event] = {}

    def _get_connection(self) -> SSHConnection:
        """Get or create SSH connection"""
        if self._connection is None:
            self._connection = SSHConnection(self.ssh_config)
        return self._connection

    @property
    def is_connected(self) -> bool:
        conn = self._get_connection()
        return conn.is_connected or conn.connect()

    def test_connection(self) -> Dict[str, Any]:
        """Test SSH connection"""
        conn = self._get_connection()
        if not conn.connect():
            return {
                "success": False,
                "error": "Failed to connect",
            }

        # Test basic command
        exit_code, stdout, stderr = conn.exec_command("echo 'Connection OK' && hostname")

        if exit_code == 0:
            return {
                "success": True,
                "hostname": stdout.strip().split('\n')[-1],
                "message": "Connection successful",
            }
        else:
            return {
                "success": False,
                "error": stderr or "Unknown error",
            }

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information from remote server"""
        conn = self._get_connection()
        if not conn.connect():
            return {"success": False, "error": "Not connected"}

        # nvidia-smi query
        cmd = "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits"
        exit_code, stdout, stderr = conn.exec_command(cmd, timeout=30)

        if exit_code != 0:
            return {"success": False, "error": stderr or "nvidia-smi failed"}

        gpus = []
        for line in stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 7:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total": int(parts[2]),
                    "memory_used": int(parts[3]),
                    "memory_free": int(parts[4]),
                    "utilization": int(parts[5]) if parts[5] != '[N/A]' else 0,
                    "temperature": int(parts[6]) if parts[6] != '[N/A]' else 0,
                })

        return {
            "success": True,
            "gpus": gpus,
            "gpu_count": len(gpus),
        }

    def _build_training_command(self, config: SSHJobConfig) -> str:
        """Build the full training command with environment setup (safe)"""
        import shlex
        commands = []

        # Change to working directory (safe path quoting)
        working_dir = self.ssh_config.working_dir
        commands.append(f"mkdir -p {shlex.quote(working_dir)}")
        commands.append(f"cd {shlex.quote(working_dir)}")

        # Activate conda environment if specified
        if self.ssh_config.conda_env:
            # Try common conda installation paths (use ; for single-line if/elif/fi)
            conda_init = (
                "if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then "
                "source /root/miniconda3/etc/profile.d/conda.sh; "
                "elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then "
                "source ~/miniconda3/etc/profile.d/conda.sh; "
                "elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then "
                "source /opt/conda/etc/profile.d/conda.sh; "
                "elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then "
                "source ~/anaconda3/etc/profile.d/conda.sh; "
                "fi"
            )
            commands.append(conda_init)
            commands.append(f"conda activate {shlex.quote(self.ssh_config.conda_env)}")

        # Set environment variables (safe quoting)
        for key, value in config.env_vars.items():
            # Validate environment variable name
            if not re.match(r'^[A-Z_][A-Z0-9_]*$', key):
                logger.warning(f"Skipping invalid env var name: {key}")
                continue
            commands.append(f"export {key}={shlex.quote(str(value))}")

        # Set CUDA visible devices
        if config.num_gpus > 0:
            gpu_ids = ",".join(str(i) for i in range(config.num_gpus))
            commands.append(f"export CUDA_VISIBLE_DEVICES={gpu_ids}")

        # Get training command
        entrypoint = config.to_entrypoint()

        # Setup log file (safe path quoting)
        log_dir = f"{working_dir}/logs"
        commands.append(f"mkdir -p {shlex.quote(log_dir)}")
        # Ensure log_file is also safely constructed
        safe_job_id = shlex.quote(config.job_id)
        log_file = config.log_file or f"{log_dir}/{config.job_id}.log"

        # Run training with nohup and log to file (safe quoting)
        # Note: Use "; echo $!" instead of "&& echo $!" because & returns immediately
        # The entrypoint is assumed to be safe (generated by to_entrypoint())
        # but we still quote the log file path
        nohup_cmd = f"nohup {entrypoint} > {shlex.quote(log_file)} 2>&1 & echo $!"
        commands.append(nohup_cmd)

        return " && ".join(commands)

    def submit_job(self, config: SSHJobConfig) -> Dict[str, Any]:
        """
        Submit a training job to remote server.

        Args:
            config: Job configuration

        Returns:
            Dict with job_id and submission status
        """
        conn = self._get_connection()
        if not conn.connect():
            return {
                "success": False,
                "error": "SSH connection failed",
            }

        try:
            # Build and execute command
            command = self._build_training_command(config)
            exit_code, stdout, stderr = conn.exec_command(command, timeout=60)

            if exit_code != 0:
                return {
                    "success": False,
                    "error": f"Job submission failed: {stderr}",
                }

            # Extract PID from output
            pid = stdout.strip().split('\n')[-1].strip()

            # Store job info
            log_file = config.log_file or f"{self.ssh_config.working_dir}/logs/{config.job_id}.log"
            self._jobs[config.job_id] = {
                "pid": pid,
                "log_file": log_file,
                "status": SSHJobStatus.RUNNING,
                "started_at": datetime.utcnow().isoformat(),
                "config": config,
            }

            logger.info(f"Submitted job {config.job_id} with PID {pid}")

            return {
                "success": True,
                "job_id": config.job_id,
                "pid": pid,
                "log_file": log_file,
                "mode": "ssh",
            }

        except Exception as e:
            logger.error(f"Job submission failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a remote job"""
        job_info = self._jobs.get(job_id)
        if not job_info:
            return {"status": SSHJobStatus.UNKNOWN.value, "error": "Job not found"}

        pid = job_info.get("pid")
        if not pid:
            return {"status": SSHJobStatus.UNKNOWN.value, "error": "No PID"}

        conn = self._get_connection()
        if not conn.connect():
            return {"status": SSHJobStatus.UNKNOWN.value, "error": "Not connected"}

        # Check if process is running (safe command construction)
        try:
            ps_cmd = SafeCommands.ps_check(pid)
            exit_code, stdout, stderr = conn.exec_command(ps_cmd)
        except ValueError as e:
            logger.error(f"Invalid PID for process check: {e}")
            return {"status": SSHJobStatus.UNKNOWN.value, "error": str(e)}

        if exit_code == 0 and stdout.strip():
            status = SSHJobStatus.RUNNING
        else:
            # Check log for completion status (safe command)
            import shlex
            log_file = job_info.get("log_file")
            if log_file:
                check_cmd = f"tail -20 {shlex.quote(log_file)} 2>/dev/null | grep -E '(Error|Exception|Traceback|Successfully|Completed)'"
                exit_code, stdout, stderr = conn.exec_command(check_cmd)
            else:
                stdout = ""

            if "Error" in stdout or "Exception" in stdout or "Traceback" in stdout:
                status = SSHJobStatus.FAILED
            else:
                status = SSHJobStatus.SUCCEEDED

        job_info["status"] = status

        return {
            "status": status.value,
            "pid": pid,
            "log_file": job_info.get("log_file"),
            "started_at": job_info.get("started_at"),
        }

    def get_job_logs(self, job_id: str, lines: int = 100) -> str:
        """Get logs from a remote job"""
        job_info = self._jobs.get(job_id)
        if not job_info:
            return f"Job {job_id} not found"

        log_file = job_info.get("log_file")
        if not log_file:
            return "No log file"

        conn = self._get_connection()
        if not conn.connect():
            return "Not connected to server"

        # Safe tail command construction
        try:
            tail_cmd = SafeCommands.tail_file(log_file, lines)
            exit_code, stdout, stderr = conn.exec_command(tail_cmd)
        except ValueError as e:
            return f"Error: Invalid parameters - {e}"

        if exit_code != 0:
            return f"Error reading logs: {stderr}"

        return stdout

    def stream_job_logs(
        self,
        job_id: str,
        callback: Callable[[str], None],
        stop_event: Optional[threading.Event] = None
    ):
        """
        Stream job logs in real-time.

        Args:
            job_id: Job ID
            callback: Function called with each log line
            stop_event: Event to signal stop
        """
        job_info = self._jobs.get(job_id)
        if not job_info:
            callback(f"Job {job_id} not found\n")
            return

        log_file = job_info.get("log_file")
        if not log_file:
            callback("No log file\n")
            return

        conn = self._get_connection()
        if not conn.connect():
            callback("Not connected to server\n")
            return

        stop_event = stop_event or threading.Event()

        # Use tail -f to stream logs (safe command construction)
        try:
            tail_cmd = SafeCommands.tail_follow(log_file)
        except ValueError as e:
            logger.error(f"Invalid log file path: {e}")
            return

        try:
            transport = conn._client.get_transport()
            channel = transport.open_session()
            channel.exec_command(tail_cmd)

            while not stop_event.is_set():
                if channel.recv_ready():
                    data = channel.recv(4096).decode('utf-8', errors='replace')
                    if data:
                        callback(data)

                if channel.exit_status_ready():
                    break

                time.sleep(0.1)

            channel.close()

        except Exception as e:
            callback(f"Error streaming logs: {e}\n")

    def start_log_streaming(
        self,
        job_id: str,
        callback: Callable[[str], None]
    ) -> bool:
        """Start background log streaming for a job"""
        if job_id in self._log_threads:
            return True  # Already streaming

        stop_event = threading.Event()
        self._stop_flags[job_id] = stop_event

        thread = threading.Thread(
            target=self.stream_job_logs,
            args=(job_id, callback, stop_event),
            daemon=True
        )
        thread.start()
        self._log_threads[job_id] = thread

        return True

    def stop_log_streaming(self, job_id: str):
        """Stop log streaming for a job"""
        if job_id in self._stop_flags:
            self._stop_flags[job_id].set()
            del self._stop_flags[job_id]

        if job_id in self._log_threads:
            del self._log_threads[job_id]

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """Stop a running job"""
        job_info = self._jobs.get(job_id)
        if not job_info:
            return {"success": False, "error": "Job not found"}

        pid = job_info.get("pid")
        if not pid:
            return {"success": False, "error": "No PID"}

        conn = self._get_connection()
        if not conn.connect():
            return {"success": False, "error": "Not connected"}

        # Kill process and children (validate PID for safety)
        try:
            if not validate_integer(pid, min_val=1):
                raise ValueError(f"Invalid PID: {pid}")

            # Safe command construction
            kill_cmd = f"pkill -TERM -P {pid}; kill -TERM {pid} 2>/dev/null"
            exit_code, stdout, stderr = conn.exec_command(kill_cmd)
        except ValueError as e:
            logger.error(f"Invalid PID for kill command: {e}")
            return {"success": False, "error": str(e)}

        # Stop log streaming
        self.stop_log_streaming(job_id)

        job_info["status"] = SSHJobStatus.STOPPED

        return {
            "success": True,
            "message": f"Job {job_id} (PID {pid}) stopped",
        }

    def cleanup_job(self, job_id: str) -> Dict[str, Any]:
        """Clean up job files from remote server"""
        job_info = self._jobs.get(job_id)
        if not job_info:
            return {"success": False, "error": "Job not found"}

        log_file = job_info.get("log_file")

        conn = self._get_connection()
        if conn.connect() and log_file:
            # Safe rm command
            try:
                rm_cmd = SafeCommands.rm_file(log_file)
                conn.exec_command(rm_cmd)
            except ValueError as e:
                logger.warning(f"Cannot remove log file: {e}")

        # Remove from local tracking
        self._jobs.pop(job_id, None)
        self.stop_log_streaming(job_id)

        return {"success": True}

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all tracked jobs"""
        jobs = []
        for job_id, info in self._jobs.items():
            status = self.get_job_status(job_id)
            jobs.append({
                "job_id": job_id,
                "status": status.get("status"),
                "pid": info.get("pid"),
                "started_at": info.get("started_at"),
                "log_file": info.get("log_file"),
            })
        return jobs

    def disconnect(self):
        """Disconnect from remote server"""
        # Stop all log streaming
        for job_id in list(self._stop_flags.keys()):
            self.stop_log_streaming(job_id)

        if self._connection:
            self._connection.disconnect()
            self._connection = None


class SSHRunnerManager:
    """
    Manages multiple SSH runners for different remote servers.
    """

    def __init__(self):
        self._runners: Dict[str, SSHJobRunner] = {}
        self._default_config: Optional[SSHConfig] = None

    def set_default_config(self, config: SSHConfig):
        """Set default SSH configuration"""
        self._default_config = config

    def get_runner(self, config: Optional[SSHConfig] = None) -> Optional[SSHJobRunner]:
        """Get or create SSH runner for given config"""
        config = config or self._default_config
        if not config:
            return None

        key = f"{config.host}:{config.port}:{config.username}"

        if key not in self._runners:
            self._runners[key] = SSHJobRunner(config)

        return self._runners[key]

    def test_connection(self, config: SSHConfig) -> Dict[str, Any]:
        """Test connection with given config"""
        runner = SSHJobRunner(config)
        result = runner.test_connection()
        if not result.get("success"):
            runner.disconnect()
        return result

    def disconnect_all(self):
        """Disconnect all runners"""
        for runner in self._runners.values():
            runner.disconnect()
        self._runners.clear()


# Singleton instance
_ssh_manager: Optional[SSHRunnerManager] = None


def get_ssh_manager() -> SSHRunnerManager:
    """Get SSH runner manager singleton"""
    global _ssh_manager
    if _ssh_manager is None:
        _ssh_manager = SSHRunnerManager()
    return _ssh_manager


class DatasetSyncService:
    """
    Service for syncing datasets to remote SSH server.

    Handles uploading training datasets to the remote server
    and managing the remote paths.
    """

    def __init__(self, ssh_config: SSHConfig):
        self.ssh_config = ssh_config
        self._connection: Optional[SSHConnection] = None

    def _get_connection(self) -> SSHConnection:
        """Get or create SSH connection"""
        if self._connection is None:
            self._connection = SSHConnection(self.ssh_config)
        return self._connection

    def _ensure_remote_dir(self, remote_dir: str) -> bool:
        """Ensure remote directory exists"""
        conn = self._get_connection()
        if not conn.connect():
            return False

        # Safe mkdir command construction
        try:
            mkdir_cmd = SafeCommands.mkdir(remote_dir)
            exit_code, _, stderr = conn.exec_command(mkdir_cmd)
            return exit_code == 0
        except ValueError as e:
            logger.error(f"Invalid remote directory: {e}")
            return False

    def sync_dataset(
        self,
        local_path: str,
        dataset_name: str,
        dataset_uuid: str,
    ) -> Dict[str, Any]:
        """
        Sync a dataset file to the remote server.

        Args:
            local_path: Local path to the dataset file
            dataset_name: Name of the dataset (for remote filename)
            dataset_uuid: UUID of the dataset (for unique remote path)

        Returns:
            Dict with sync result:
            - success: bool
            - remote_path: str (if successful)
            - error: str (if failed)
        """
        conn = self._get_connection()
        if not conn.connect():
            return {
                "success": False,
                "error": "Failed to connect to SSH server"
            }

        try:
            # Create remote datasets directory
            remote_base_dir = f"{self.ssh_config.working_dir}/datasets"
            if not self._ensure_remote_dir(remote_base_dir):
                return {
                    "success": False,
                    "error": f"Failed to create remote directory: {remote_base_dir}"
                }

            # Determine remote filename (preserve extension)
            local_filename = os.path.basename(local_path)
            extension = os.path.splitext(local_filename)[1]
            # Use uuid to ensure uniqueness, but include name for readability
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name)
            remote_filename = f"{safe_name}_{dataset_uuid[:8]}{extension}"
            remote_path = f"{remote_base_dir}/{remote_filename}"

            # Expand ~ in remote path for display (safe command)
            try:
                echo_cmd = SafeCommands.echo_expand_path(remote_path)
                exit_code, stdout, _ = conn.exec_command(echo_cmd)
                if exit_code == 0:
                    expanded_remote_path = stdout.strip()
                else:
                    expanded_remote_path = remote_path
            except ValueError:
                expanded_remote_path = remote_path

            # Upload file
            logger.info(f"Uploading dataset to {remote_path}")
            if conn.upload_file(local_path, expanded_remote_path):
                # Verify upload (safe command)
                try:
                    ls_cmd = SafeCommands.ls_file(expanded_remote_path)
                    exit_code, stdout, _ = conn.exec_command(ls_cmd)
                except ValueError as e:
                    logger.warning(f"Cannot verify upload: {e}")
                    exit_code = 1
                if exit_code == 0:
                    logger.info(f"Dataset synced successfully to {expanded_remote_path}")
                    return {
                        "success": True,
                        "remote_path": expanded_remote_path,
                    }
                else:
                    return {
                        "success": False,
                        "error": "Upload completed but file verification failed"
                    }
            else:
                return {
                    "success": False,
                    "error": "Failed to upload file via SFTP"
                }

        except Exception as e:
            logger.error(f"Dataset sync failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def delete_remote_dataset(self, remote_path: str) -> Dict[str, Any]:
        """Delete a dataset from remote server (safe)"""
        conn = self._get_connection()
        if not conn.connect():
            return {"success": False, "error": "Not connected"}

        # Safe rm command
        try:
            rm_cmd = SafeCommands.rm_file(remote_path)
            exit_code, _, stderr = conn.exec_command(rm_cmd)

            if exit_code == 0:
                return {"success": True}
            else:
                return {"success": False, "error": stderr}
        except ValueError as e:
            return {"success": False, "error": f"Invalid path: {e}"}

    def check_remote_file(self, remote_path: str) -> Dict[str, Any]:
        """Check if remote file exists and get info (safe)"""
        conn = self._get_connection()
        if not conn.connect():
            return {"exists": False, "error": "Not connected"}

        # Safe ls command
        try:
            ls_cmd = SafeCommands.ls_file(remote_path)
            exit_code, stdout, _ = conn.exec_command(ls_cmd)
        except ValueError as e:
            return {"exists": False, "error": f"Invalid path: {e}"}

        if exit_code == 0 and stdout.strip():
            # Parse file info
            parts = stdout.strip().split()
            if len(parts) >= 5:
                return {
                    "exists": True,
                    "size": parts[4],
                    "remote_path": remote_path,
                }

        return {"exists": False}

    def disconnect(self):
        """Disconnect from server"""
        if self._connection:
            self._connection.disconnect()
            self._connection = None


def get_dataset_sync_service() -> Optional[DatasetSyncService]:
    """
    Get dataset sync service using current SSH configuration.

    Returns None if SSH mode is not configured.
    """
    from .run_mode import load_run_mode_config, RunMode

    config = load_run_mode_config()
    if config.mode != RunMode.SSH:
        return None

    ssh_config = config.to_ssh_config()
    if not ssh_config:
        return None

    return DatasetSyncService(ssh_config)
