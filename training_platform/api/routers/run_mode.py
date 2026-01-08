"""
Run Mode API Router

Manages execution mode configuration (Local vs SSH).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from ...core.run_mode import (
    RunMode,
    RunModeConfig,
    load_run_mode_config,
    save_run_mode_config,
    get_current_runner,
    set_run_mode,
)
from ...core.ssh_runner import SSHConfig, get_ssh_manager

router = APIRouter(prefix="/run-mode", tags=["run-mode"])


# ============== Pydantic Models ==============

class SSHConfigRequest(BaseModel):
    """SSH configuration request"""
    host: str = Field(..., description="SSH host address")
    port: int = Field(22, description="SSH port")
    username: str = Field(..., description="SSH username")
    password: Optional[str] = Field(None, description="SSH password (optional if using key)")
    key_path: Optional[str] = Field(None, description="Path to SSH private key")
    working_dir: str = Field("~/verl_jobs", description="Remote working directory")
    conda_env: Optional[str] = Field(None, description="Conda environment name")


class RunModeRequest(BaseModel):
    """Run mode configuration request"""
    mode: str = Field(..., description="Run mode: 'local' or 'ssh'")
    ssh_config: Optional[SSHConfigRequest] = Field(None, description="SSH configuration (required for SSH mode)")


class RunModeResponse(BaseModel):
    """Run mode configuration response"""
    mode: str
    ssh_configured: bool
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_username: Optional[str] = None
    ssh_working_dir: Optional[str] = None
    ssh_conda_env: Optional[str] = None


class ConnectionTestResponse(BaseModel):
    """Connection test response"""
    success: bool
    mode: str
    message: Optional[str] = None
    error: Optional[str] = None
    hostname: Optional[str] = None


class GPUInfo(BaseModel):
    """GPU information"""
    index: int
    name: str
    memory_total: int
    memory_used: int
    memory_free: Optional[int] = None
    utilization: int
    temperature: Optional[int] = None


class GPUInfoResponse(BaseModel):
    """GPU information response"""
    success: bool
    mode: str
    gpu_count: int
    gpus: List[GPUInfo]
    error: Optional[str] = None


# ============== API Endpoints ==============

@router.get("/config", response_model=RunModeResponse)
async def get_run_mode_config():
    """Get current run mode configuration"""
    config = load_run_mode_config()

    return RunModeResponse(
        mode=config.mode.value,
        ssh_configured=config.ssh_host is not None,
        ssh_host=config.ssh_host,
        ssh_port=config.ssh_port,
        ssh_username=config.ssh_username,
        ssh_working_dir=config.ssh_working_dir,
        ssh_conda_env=config.ssh_conda_env,
    )


@router.post("/config", response_model=RunModeResponse)
async def set_run_mode_config(request: RunModeRequest):
    """Set run mode configuration"""
    try:
        mode = RunMode(request.mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")

    if mode == RunMode.SSH:
        if not request.ssh_config:
            raise HTTPException(status_code=400, detail="SSH configuration required for SSH mode")

        ssh_config = {
            "host": request.ssh_config.host,
            "port": request.ssh_config.port,
            "username": request.ssh_config.username,
            "password": request.ssh_config.password,
            "key_path": request.ssh_config.key_path,
            "working_dir": request.ssh_config.working_dir,
            "conda_env": request.ssh_config.conda_env,
        }

        config = set_run_mode(mode, ssh_config)
    else:
        config = set_run_mode(mode)

    return RunModeResponse(
        mode=config.mode.value,
        ssh_configured=config.ssh_host is not None,
        ssh_host=config.ssh_host,
        ssh_port=config.ssh_port,
        ssh_username=config.ssh_username,
        ssh_working_dir=config.ssh_working_dir,
        ssh_conda_env=config.ssh_conda_env,
    )


@router.post("/test-connection", response_model=ConnectionTestResponse)
async def test_connection():
    """Test connection for current run mode"""
    runner = get_current_runner()
    result = runner.test_connection()

    return ConnectionTestResponse(
        success=result.get("success", False),
        mode=result.get("mode", "unknown"),
        message=result.get("message"),
        error=result.get("error"),
        hostname=result.get("hostname"),
    )


@router.post("/test-ssh", response_model=ConnectionTestResponse)
async def test_ssh_connection(config: SSHConfigRequest):
    """Test SSH connection with given configuration (without saving)"""
    ssh_config = SSHConfig(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
        key_path=config.key_path,
        working_dir=config.working_dir,
        conda_env=config.conda_env,
    )

    manager = get_ssh_manager()
    result = manager.test_connection(ssh_config)

    return ConnectionTestResponse(
        success=result.get("success", False),
        mode="ssh",
        message=result.get("message"),
        error=result.get("error"),
        hostname=result.get("hostname"),
    )


@router.get("/gpu-info", response_model=GPUInfoResponse)
async def get_gpu_info():
    """Get GPU information from current execution target"""
    runner = get_current_runner()
    result = runner.get_gpu_info()

    gpus = []
    for gpu in result.get("gpus", []):
        gpus.append(GPUInfo(
            index=gpu.get("index", 0),
            name=gpu.get("name", "Unknown"),
            memory_total=gpu.get("memory_total", 0),
            memory_used=gpu.get("memory_used", 0),
            memory_free=gpu.get("memory_free"),
            utilization=gpu.get("utilization", 0),
            temperature=gpu.get("temperature"),
        ))

    return GPUInfoResponse(
        success=result.get("success", False),
        mode=runner.mode.value,
        gpu_count=result.get("gpu_count", 0),
        gpus=gpus,
        error=result.get("error"),
    )


@router.post("/ssh/gpu-info", response_model=GPUInfoResponse)
async def get_ssh_gpu_info(config: SSHConfigRequest):
    """Get GPU information from SSH server (without saving config)"""
    from ...core.ssh_runner import SSHJobRunner

    ssh_config = SSHConfig(
        host=config.host,
        port=config.port,
        username=config.username,
        password=config.password,
        key_path=config.key_path,
        working_dir=config.working_dir,
        conda_env=config.conda_env,
    )

    runner = SSHJobRunner(ssh_config)
    result = runner.get_gpu_info()
    runner.disconnect()

    gpus = []
    for gpu in result.get("gpus", []):
        gpus.append(GPUInfo(
            index=gpu.get("index", 0),
            name=gpu.get("name", "Unknown"),
            memory_total=gpu.get("memory_total", 0),
            memory_used=gpu.get("memory_used", 0),
            memory_free=gpu.get("memory_free"),
            utilization=gpu.get("utilization", 0),
            temperature=gpu.get("temperature"),
        ))

    return GPUInfoResponse(
        success=result.get("success", False),
        mode="ssh",
        gpu_count=result.get("gpu_count", 0),
        gpus=gpus,
        error=result.get("error"),
    )


@router.get("/status")
async def get_runner_status():
    """Get current runner status"""
    config = load_run_mode_config()
    runner = get_current_runner()

    return {
        "mode": config.mode.value,
        "is_connected": runner.is_connected,
        "config": config.to_dict(),
    }


@router.get("/remote/models")
async def list_remote_models():
    """List available models on remote server (SSH mode only)"""
    config = load_run_mode_config()
    if config.mode != RunMode.SSH:
        raise HTTPException(status_code=400, detail="Only available in SSH mode")

    ssh_config = config.to_ssh_config()
    if not ssh_config:
        raise HTTPException(status_code=400, detail="SSH not configured")

    from ...core.ssh_runner import SSHConnection

    conn = SSHConnection(ssh_config)
    if not conn.connect():
        raise HTTPException(status_code=500, detail="SSH connection failed")

    try:
        # Common model directories to scan
        model_dirs = [
            "~/autodl-tmp",
            "~/models",
            "/root/autodl-tmp",
            "/root/models",
            ssh_config.working_dir,
        ]

        models = []
        for base_dir in model_dirs:
            # Find directories containing config.json (likely model dirs)
            cmd = f"find {base_dir} -maxdepth 2 -name 'config.json' -type f 2>/dev/null | head -20"
            exit_code, stdout, stderr = conn.exec_command(cmd, timeout=10)

            if exit_code == 0 and stdout.strip():
                for config_path in stdout.strip().split('\n'):
                    if config_path:
                        model_dir = config_path.rsplit('/', 1)[0]
                        model_name = model_dir.rsplit('/', 1)[-1]

                        # Get directory size
                        size_cmd = f"du -sb {model_dir} 2>/dev/null | cut -f1"
                        _, size_out, _ = conn.exec_command(size_cmd, timeout=5)
                        try:
                            size_bytes = int(size_out.strip()) if size_out.strip() else 0
                            size_gb = round(size_bytes / (1024**3), 2)
                        except:
                            size_gb = 0

                        # Avoid duplicates
                        if not any(m['path'] == model_dir for m in models):
                            models.append({
                                "name": model_name,
                                "path": model_dir,
                                "size_gb": size_gb,
                                "has_config": True,
                            })

        return {"models": models, "mode": "ssh"}

    finally:
        conn.disconnect()


@router.get("/remote/datasets")
async def list_remote_datasets():
    """List available datasets on remote server (SSH mode only)"""
    config = load_run_mode_config()
    if config.mode != RunMode.SSH:
        raise HTTPException(status_code=400, detail="Only available in SSH mode")

    ssh_config = config.to_ssh_config()
    if not ssh_config:
        raise HTTPException(status_code=400, detail="SSH not configured")

    from ...core.ssh_runner import SSHConnection

    conn = SSHConnection(ssh_config)
    if not conn.connect():
        raise HTTPException(status_code=500, detail="SSH connection failed")

    try:
        # Common dataset directories to scan
        dataset_dirs = [
            f"{ssh_config.working_dir}/datasets",
            "~/datasets",
            "/root/verl_jobs/datasets",
            "/root/datasets",
        ]

        datasets = []
        for base_dir in dataset_dirs:
            # Find parquet, jsonl, json files
            cmd = f"find {base_dir} -type f \\( -name '*.parquet' -o -name '*.jsonl' -o -name '*.json' \\) 2>/dev/null | head -50"
            exit_code, stdout, stderr = conn.exec_command(cmd, timeout=10)

            if exit_code == 0 and stdout.strip():
                for file_path in stdout.strip().split('\n'):
                    if file_path:
                        file_name = file_path.rsplit('/', 1)[-1]
                        ext = file_name.rsplit('.', 1)[-1] if '.' in file_name else ''

                        # Get file size
                        size_cmd = f"stat -c%s {file_path} 2>/dev/null || stat -f%z {file_path} 2>/dev/null"
                        _, size_out, _ = conn.exec_command(size_cmd, timeout=5)
                        try:
                            size_bytes = int(size_out.strip()) if size_out.strip() else 0
                            size_mb = round(size_bytes / (1024**2), 2)
                        except:
                            size_mb = 0

                        # Avoid duplicates
                        if not any(d['path'] == file_path for d in datasets):
                            datasets.append({
                                "name": file_name,
                                "path": file_path,
                                "format": ext,
                                "size_mb": size_mb,
                            })

        return {"datasets": datasets, "mode": "ssh"}

    finally:
        conn.disconnect()


class RemoteCommandRequest(BaseModel):
    """Remote command execution request"""
    command: str = Field(..., description="Shell command to execute")


@router.post("/exec")
async def execute_remote_command(request: RemoteCommandRequest):
    """Execute a command on remote server (SSH mode only)"""
    config = load_run_mode_config()
    if config.mode != RunMode.SSH:
        raise HTTPException(status_code=400, detail="Only available in SSH mode")

    runner = get_current_runner()
    ssh_config = config.to_ssh_config()

    if not ssh_config:
        raise HTTPException(status_code=400, detail="SSH not configured")

    from ...core.ssh_runner import SSHConnection
    conn = SSHConnection(ssh_config)

    if not conn.connect():
        raise HTTPException(status_code=500, detail="SSH connection failed")

    try:
        exit_code, stdout, stderr = conn.exec_command(request.command, timeout=30)
        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }
    finally:
        conn.disconnect()
