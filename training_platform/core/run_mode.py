"""
Run Mode Configuration

Manages different execution modes for training jobs:
- Local: Run on local machine using Ray or subprocess
- SSH: Run on remote GPU server via SSH
"""

import os
import logging
from enum import Enum
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

from .ray_runner import RayJobConfig, RayJobRunner, LocalJobRunner, get_job_runner
from .ssh_runner import SSHConfig, SSHJobConfig, SSHJobRunner, get_ssh_manager
from .verl_adapter import VerlTrainingConfig
from .crypto_utils import encrypt_password, decrypt_password, CryptoError

logger = logging.getLogger(__name__)


class RunMode(str, Enum):
    """Training execution mode"""
    LOCAL = "local"      # Local execution (Ray or subprocess)
    SSH = "ssh"          # Remote execution via SSH


@dataclass
class RunModeConfig:
    """Configuration for run mode"""
    mode: RunMode = RunMode.LOCAL

    # SSH configuration (only used when mode == SSH)
    ssh_host: Optional[str] = None
    ssh_port: int = 22
    ssh_username: Optional[str] = None
    ssh_password: Optional[str] = None
    ssh_key_path: Optional[str] = None
    ssh_working_dir: str = "~/verl_jobs"
    ssh_conda_env: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (hide password)"""
        d = asdict(self)
        d['mode'] = self.mode.value
        if d.get('ssh_password'):
            d['ssh_password'] = '***'
        return d

    def to_ssh_config(self) -> Optional[SSHConfig]:
        """Convert to SSHConfig if SSH mode"""
        if self.mode != RunMode.SSH:
            return None

        if not self.ssh_host or not self.ssh_username:
            return None

        return SSHConfig(
            host=self.ssh_host,
            port=self.ssh_port,
            username=self.ssh_username,
            password=self.ssh_password,
            key_path=self.ssh_key_path,
            working_dir=self.ssh_working_dir,
            conda_env=self.ssh_conda_env,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunModeConfig":
        """Create from dictionary"""
        mode = data.get('mode', 'local')
        if isinstance(mode, str):
            mode = RunMode(mode)

        return cls(
            mode=mode,
            ssh_host=data.get('ssh_host'),
            ssh_port=data.get('ssh_port', 22),
            ssh_username=data.get('ssh_username'),
            ssh_password=data.get('ssh_password'),
            ssh_key_path=data.get('ssh_key_path'),
            ssh_working_dir=data.get('ssh_working_dir', '~/verl_jobs'),
            ssh_conda_env=data.get('ssh_conda_env'),
        )


class UnifiedJobRunner:
    """
    Unified job runner that handles both local and SSH execution modes.

    This provides a consistent interface regardless of execution mode.
    """

    def __init__(self, config: RunModeConfig):
        self.config = config
        self._local_runner: Optional[Union[RayJobRunner, LocalJobRunner]] = None
        self._ssh_runner: Optional[SSHJobRunner] = None

    def _get_local_runner(self) -> Union[RayJobRunner, LocalJobRunner]:
        """Get local job runner"""
        if self._local_runner is None:
            self._local_runner = get_job_runner(prefer_ray=True)
        return self._local_runner

    def _get_ssh_runner(self) -> Optional[SSHJobRunner]:
        """Get SSH job runner"""
        if self._ssh_runner is None:
            ssh_config = self.config.to_ssh_config()
            if ssh_config:
                manager = get_ssh_manager()
                self._ssh_runner = manager.get_runner(ssh_config)
        return self._ssh_runner

    @property
    def mode(self) -> RunMode:
        return self.config.mode

    @property
    def is_connected(self) -> bool:
        """Check if runner is ready"""
        if self.config.mode == RunMode.LOCAL:
            runner = self._get_local_runner()
            return hasattr(runner, 'is_connected') and runner.is_connected or True
        else:
            runner = self._get_ssh_runner()
            return runner is not None and runner.is_connected

    def test_connection(self) -> Dict[str, Any]:
        """Test connection for current mode"""
        if self.config.mode == RunMode.LOCAL:
            return {
                "success": True,
                "mode": "local",
                "message": "Local mode ready",
            }
        else:
            runner = self._get_ssh_runner()
            if not runner:
                return {
                    "success": False,
                    "mode": "ssh",
                    "error": "SSH not configured",
                }
            result = runner.test_connection()
            result["mode"] = "ssh"
            return result

    def submit_job(
        self,
        job_id: str,
        name: str,
        verl_config: VerlTrainingConfig,
        num_gpus: int = 8,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Submit a training job using the configured mode.

        Args:
            job_id: Unique job identifier
            name: Job name
            verl_config: Verl training configuration
            num_gpus: Number of GPUs to use
            env_vars: Environment variables

        Returns:
            Submission result dict
        """
        env_vars = env_vars or {}

        if self.config.mode == RunMode.LOCAL:
            # Local execution
            runner = self._get_local_runner()

            config = RayJobConfig(
                job_id=job_id,
                name=name,
                verl_config=verl_config,
                num_gpus=num_gpus,
                env_vars=env_vars,
            )

            result = runner.submit_job(config)
            result["mode"] = "local"
            return result

        else:
            # SSH execution
            runner = self._get_ssh_runner()
            if not runner:
                return {
                    "success": False,
                    "error": "SSH runner not available",
                    "mode": "ssh",
                }

            config = SSHJobConfig(
                job_id=job_id,
                name=name,
                verl_config=verl_config,
                num_gpus=num_gpus,
                env_vars=env_vars,
            )

            result = runner.submit_job(config)
            result["mode"] = "ssh"
            return result

    def get_job_status(self, job_id: str, ray_job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get job status"""
        if self.config.mode == RunMode.LOCAL:
            runner = self._get_local_runner()
            job_ref = ray_job_id or job_id
            return runner.get_job_status(job_ref)
        else:
            runner = self._get_ssh_runner()
            if not runner:
                return {"status": "unknown", "error": "SSH runner not available"}
            return runner.get_job_status(job_id)

    def get_job_logs(self, job_id: str, ray_job_id: Optional[str] = None, lines: int = 100) -> str:
        """Get job logs"""
        if self.config.mode == RunMode.LOCAL:
            runner = self._get_local_runner()
            job_ref = ray_job_id or job_id
            return runner.get_job_logs(job_ref)
        else:
            runner = self._get_ssh_runner()
            if not runner:
                return "SSH runner not available"
            return runner.get_job_logs(job_id, lines=lines)

    def stop_job(self, job_id: str, ray_job_id: Optional[str] = None) -> Dict[str, Any]:
        """Stop a running job"""
        if self.config.mode == RunMode.LOCAL:
            runner = self._get_local_runner()
            job_ref = ray_job_id or job_id
            return runner.stop_job(job_ref)
        else:
            runner = self._get_ssh_runner()
            if not runner:
                return {"success": False, "error": "SSH runner not available"}
            return runner.stop_job(job_id)

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        if self.config.mode == RunMode.LOCAL:
            # Local GPU info via nvidia-smi
            import subprocess
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    gpus = []
                    for line in result.stdout.strip().split('\n'):
                        if not line.strip():
                            continue
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 5:
                            gpus.append({
                                "index": int(parts[0]),
                                "name": parts[1],
                                "memory_total": int(parts[2]),
                                "memory_used": int(parts[3]),
                                "utilization": int(parts[4]) if parts[4] != '[N/A]' else 0,
                            })
                    return {"success": True, "gpus": gpus, "gpu_count": len(gpus)}
            except Exception as e:
                pass
            return {"success": False, "error": "No NVIDIA GPU available", "gpus": [], "gpu_count": 0}
        else:
            runner = self._get_ssh_runner()
            if not runner:
                return {"success": False, "error": "SSH runner not available"}
            return runner.get_gpu_info()

    def start_log_streaming(self, job_id: str, callback) -> bool:
        """Start real-time log streaming"""
        if self.config.mode == RunMode.SSH:
            runner = self._get_ssh_runner()
            if runner:
                return runner.start_log_streaming(job_id, callback)
        return False

    def stop_log_streaming(self, job_id: str):
        """Stop log streaming"""
        if self.config.mode == RunMode.SSH:
            runner = self._get_ssh_runner()
            if runner:
                runner.stop_log_streaming(job_id)

    def disconnect(self):
        """Disconnect all runners"""
        if self._ssh_runner:
            self._ssh_runner.disconnect()
            self._ssh_runner = None


# Global configuration storage
_CONFIG_FILE = Path.home() / ".train_platform" / "run_mode.json"
_current_config: Optional[RunModeConfig] = None
_current_runner: Optional[UnifiedJobRunner] = None


def load_run_mode_config() -> RunModeConfig:
    """Load run mode configuration from file"""
    global _current_config

    if _current_config is not None:
        return _current_config

    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, 'r') as f:
                data = json.load(f)

                # Decrypt password if encrypted
                if 'ssh_password_encrypted' in data and data['ssh_password_encrypted']:
                    try:
                        decrypted_password = decrypt_password(data['ssh_password_encrypted'])
                        data['ssh_password'] = decrypted_password
                        logger.info("SSH password decrypted from storage")
                    except CryptoError as e:
                        logger.error(f"Failed to decrypt password: {e}")
                        data['ssh_password'] = None

                _current_config = RunModeConfig.from_dict(data)
                return _current_config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    _current_config = RunModeConfig()
    return _current_config


def save_run_mode_config(config: RunModeConfig):
    """Save run mode configuration to file"""
    global _current_config, _current_runner

    _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving
    save_data = config.to_dict()

    # Encrypt password before saving
    if config.ssh_password:
        try:
            encrypted_password = encrypt_password(config.ssh_password)
            save_data['ssh_password_encrypted'] = encrypted_password
            # Remove plaintext password
            save_data.pop('ssh_password', None)
            logger.info("SSH password encrypted for storage")
        except CryptoError as e:
            logger.error(f"Failed to encrypt password: {e}")
            # Don't save password if encryption fails
            save_data.pop('ssh_password', None)

    with open(_CONFIG_FILE, 'w') as f:
        json.dump(save_data, f, indent=2)

    logger.info(f"Run mode config saved to {_CONFIG_FILE}")

    # Update global state
    _current_config = config
    _current_runner = None  # Reset runner


def get_current_runner() -> UnifiedJobRunner:
    """Get current unified job runner"""
    global _current_runner

    if _current_runner is None:
        config = load_run_mode_config()
        _current_runner = UnifiedJobRunner(config)

    return _current_runner


def set_run_mode(mode: RunMode, ssh_config: Optional[Dict[str, Any]] = None):
    """
    Set the current run mode.

    Args:
        mode: RunMode.LOCAL or RunMode.SSH
        ssh_config: SSH configuration dict (required for SSH mode)
    """
    global _current_config, _current_runner

    if mode == RunMode.SSH and not ssh_config:
        raise ValueError("SSH configuration required for SSH mode")

    config = RunModeConfig(mode=mode)

    if ssh_config:
        config.ssh_host = ssh_config.get('host')
        config.ssh_port = ssh_config.get('port', 22)
        config.ssh_username = ssh_config.get('username')
        config.ssh_password = ssh_config.get('password')
        config.ssh_key_path = ssh_config.get('key_path')
        config.ssh_working_dir = ssh_config.get('working_dir', '~/verl_jobs')
        config.ssh_conda_env = ssh_config.get('conda_env')

    save_run_mode_config(config)
    _current_runner = UnifiedJobRunner(config)

    return config


def execute_training(
    job_uuid: str,
    config: Dict[str, Any],
    run_mode: str = "local",
    ssh_config: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Execute a training job with the specified run mode.

    This is the main entry point called by Celery tasks to run training.

    Args:
        job_uuid: Training job UUID
        config: Training configuration dict
        run_mode: Execution mode ('local' or 'ssh')
        ssh_config: SSH configuration for remote execution
        progress_callback: Callback function for progress updates (current, total, status)

    Returns:
        Dict with training results including:
            - status: Job status
            - ray_job_id: Ray job ID (if applicable)
            - checkpoints: List of checkpoint paths
            - metrics: Final metrics
    """
    import time
    from .database import engine, Session, CheckpointRepository, MetricsRepository, JobRepository

    logger.info(f"Executing training job {job_uuid} in {run_mode} mode")

    # Create run mode config
    mode = RunMode.LOCAL if run_mode == "local" else RunMode.SSH
    run_config = RunModeConfig(mode=mode)

    if ssh_config:
        run_config.ssh_host = ssh_config.get('host')
        run_config.ssh_port = ssh_config.get('port', 22)
        run_config.ssh_username = ssh_config.get('username')
        run_config.ssh_password = ssh_config.get('password')
        run_config.ssh_key_path = ssh_config.get('key_path')
        run_config.ssh_working_dir = ssh_config.get('working_dir', '~/verl_jobs')
        run_config.ssh_conda_env = ssh_config.get('conda_env')

    # Create unified runner
    runner = UnifiedJobRunner(run_config)

    # Test connection
    if progress_callback:
        progress_callback(0, 100, "Testing connection...")

    connection_result = runner.test_connection()
    if not connection_result.get("success"):
        raise RuntimeError(f"Connection test failed: {connection_result.get('error')}")

    # Fetch job from database to get full configuration
    with Session(engine) as session:
        job_repo = JobRepository(session)
        job = job_repo.get_by_uuid(job_uuid)
        if not job:
            raise ValueError(f"Job {job_uuid} not found")

    # Convert config dict to VerlTrainingConfig
    if progress_callback:
        progress_callback(5, 100, "Preparing configuration...")

    # Convert algorithm: handle both enum and string
    from .verl_adapter import VerlAlgorithm
    algorithm_value = config.get("algorithm") or job.algorithm
    if isinstance(algorithm_value, str):
        # Convert string to VerlAlgorithm enum
        algorithm = VerlAlgorithm(algorithm_value.lower())
    else:
        # Already an enum, convert to VerlAlgorithm
        algorithm = VerlAlgorithm(algorithm_value.value.lower())

    # Unified num_gpus configuration (Priority: config > job > default)
    # Ensures verl_config and submit_job use the same value
    num_gpus = config.get("num_gpus") or job.num_gpus or 8

    verl_config = VerlTrainingConfig(
        model_path=config.get("model_path") or job.model_path,
        model_size=config.get("model_size"),
        algorithm=algorithm,
        train_data_path=config.get("train_data_path") or job.train_data_path or "",
        eval_data_path=config.get("eval_data_path") or job.eval_data_path,
        num_epochs=config.get("num_epochs") or job.num_epochs or 3,
        max_steps=config.get("max_steps"),
        learning_rate=config.get("learning_rate") or job.learning_rate or 1e-6,
        batch_size=config.get("batch_size") or job.batch_size or 256,
        micro_batch_size=config.get("micro_batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        max_prompt_length=config.get("max_prompt_length") or job.context_length or 512,
        max_response_length=config.get("max_response_length", 1024),
        kl_coef=config.get("kl_coef", 0.001),
        entropy_coef=config.get("entropy_coef", 0.0),
        clip_ratio=config.get("clip_ratio", 0.2),
        rollout_n=config.get("rollout_n", 5),
        use_kl_loss=config.get("use_kl_loss", True),
        reward_fn_type=config.get("reward_fn_type", "math_verify"),
        reward_fn_extract_answer=config.get("reward_fn_extract_answer", "boxed"),
        reward_fn_compare_method=config.get("reward_fn_compare_method", "exact"),
        reward_fn_answer_key=config.get("reward_fn_answer_key", "solution"),
        reward_fn_custom_path=config.get("reward_fn_custom_path"),
        reward_model_path=config.get("reward_model_path"),
        reward_script_path=config.get("reward_script_path"),
        lora_enabled=config.get("lora_enabled", False),
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16),
        num_gpus=num_gpus,
        num_nodes=config.get("num_nodes", 1),
        gpu_type=config.get("gpu_type", "A100-80G"),
        tensor_parallel_size=config.get("tensor_parallel_size", 1),
        gpu_memory_utilization=config.get("gpu_memory_utilization", 0.6),
    )

    # Submit job
    if progress_callback:
        progress_callback(10, 100, "Submitting training job...")

    job_name = f"train_{job_uuid[:8]}"

    submit_result = runner.submit_job(
        job_id=job_uuid,
        name=job_name,
        verl_config=verl_config,
        num_gpus=num_gpus,
    )

    if not submit_result.get("success"):
        raise RuntimeError(f"Job submission failed: {submit_result.get('error')}")

    ray_job_id = submit_result.get("ray_job_id") or submit_result.get("job_id")
    logger.info(f"Job submitted successfully, ray_job_id: {ray_job_id}")

    # Monitor job status
    if progress_callback:
        progress_callback(15, 100, "Training in progress...")

    poll_interval = 10  # seconds
    last_progress = 15
    max_no_change_iterations = 60  # 10 minutes without status change
    no_change_count = 0
    last_status = None
    job_stuck = False  # Flag to track if job is stuck

    checkpoints = []
    final_metrics = {}

    while True:
        time.sleep(poll_interval)

        # Get job status
        status_result = runner.get_job_status(job_uuid, ray_job_id)
        current_status = status_result.get("status", "unknown")

        # Check for status change
        if current_status == last_status:
            no_change_count += 1
            if no_change_count >= max_no_change_iterations:
                logger.error(f"Job {job_uuid} status unchanged for {max_no_change_iterations * poll_interval}s, assuming stuck")
                job_stuck = True
                break
        else:
            no_change_count = 0
            last_status = current_status

        # Update progress based on status
        if current_status == "RUNNING":
            # Gradually increment progress
            if last_progress < 85:
                last_progress += 1
            if progress_callback:
                progress_callback(last_progress, 100, "Training in progress...")

        elif current_status in ["SUCCEEDED", "COMPLETED", "SUCCESS"]:
            logger.info(f"Job {job_uuid} completed successfully")
            if progress_callback:
                progress_callback(95, 100, "Training completed, collecting results...")
            break

        elif current_status in ["FAILED", "STOPPED", "ERROR"]:
            logger.error(f"Job {job_uuid} failed with status: {current_status}")
            logs = runner.get_job_logs(job_uuid, ray_job_id, lines=50)
            raise RuntimeError(f"Training job failed with status {current_status}. Logs:\n{logs}")

        # Check for checkpoints in database
        try:
            with Session(engine) as session:
                checkpoint_repo = CheckpointRepository(session)
                job_checkpoints = checkpoint_repo.get_checkpoints(job_uuid)
                if job_checkpoints:
                    checkpoints = [
                        {
                            "path": cp.path,
                            "step": cp.step,
                            "metrics": cp.metrics,
                        }
                        for cp in job_checkpoints
                    ]
        except Exception as e:
            logger.warning(f"Failed to fetch checkpoints: {e}")

    # Collect final metrics
    if progress_callback:
        progress_callback(98, 100, "Collecting final metrics...")

    try:
        # Option C: Read directly from metrics file (most accurate, immediate)
        metrics_dir = Path(os.getenv("PLATFORM_METRICS_DIR", "./platform_metrics"))
        metrics_file = metrics_dir / f"{job_uuid}_metrics.jsonl"

        if metrics_file.exists():
            # Read the last line (latest metric)
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        final_metrics = json.loads(last_line)
                        logger.info(f"Loaded final metrics from file: step={final_metrics.get('step')}")

        # Fallback: Read from database if file doesn't exist or is empty
        if not final_metrics:
            with Session(engine) as session:
                metrics_repo = MetricsRepository(session)
                # Fix: Use get_latest_metric (singular) instead of get_latest_metrics (plural)
                latest_metric = metrics_repo.get_latest_metric(job_uuid)
                if latest_metric:
                    final_metrics = latest_metric.metrics
                    logger.info(f"Loaded final metrics from database: step={final_metrics.get('step')}")
    except Exception as e:
        logger.warning(f"Failed to fetch final metrics: {e}")

    # Check if job was stuck
    if job_stuck:
        logs = runner.get_job_logs(job_uuid, ray_job_id, lines=100)
        raise RuntimeError(
            f"Training job {job_uuid} appears to be stuck. "
            f"Status unchanged for {max_no_change_iterations * poll_interval} seconds (last status: {last_status}). "
            f"Recent logs:\n{logs}"
        )

    # Get final logs
    final_logs = runner.get_job_logs(job_uuid, ray_job_id, lines=100)

    if progress_callback:
        progress_callback(100, 100, "Training completed!")

    return {
        "status": "completed",
        "ray_job_id": ray_job_id,
        "checkpoints": checkpoints,
        "metrics": final_metrics,
        "logs_tail": final_logs,
        "mode": run_mode,
    }
