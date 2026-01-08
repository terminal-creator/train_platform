# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Training Platform built on the [verl](https://github.com/volcengine/verl) framework. Provides training job management, compute configuration optimization, model surgery, and real-time monitoring for SFT, PPO, GRPO, DPO, and GSPO training algorithms.

**Current Version**: v1.0.0 (Phase 1 in progress)
**verl Version**: b12eb3b (v0.7.0-23) - included as git submodule in `environments/verl`

## Development Commands

### Backend (FastAPI)
```bash
# Start development server
uvicorn training_platform.api.main:app --reload --port 8000

# Run tests
pytest tests/

# Run specific test file
pytest tests/test_phase1_2.py -v

# Database migration (after schema changes)
python scripts/migrate_db_phase1_2.py
```

### Frontend (Vue 3)
```bash
cd frontend
npm install        # Install dependencies
npm run dev        # Start dev server (http://localhost:5173)
npm run build      # Production build
```

### Environment Setup
```bash
# Manager node (macOS/Linux, no GPU)
bash scripts/setup_local_env.sh manager

# Training node (Linux + GPU)
bash scripts/setup_local_env.sh training

# Remote GPU server setup
bash scripts/setup_remote_env.sh user@gpu-server

# Verify environment
python scripts/verify_env.py
```

### Docker Services (Milvus, Redis, MinIO)
```bash
docker-compose up -d    # Start services
docker-compose down     # Stop services
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Vue 3 + Pinia)                  │
│                    http://localhost:5173                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend (:8000)                    │
│  /api/v1/compute, /jobs, /monitoring, /surgery, /recipes    │
└──────────────────────────┬──────────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    ▼                      ▼                      ▼
┌────────┐          ┌────────────┐         ┌───────────┐
│ Local  │          │    SSH     │         │   verl    │
│ Runner │          │   Runner   │         │ Framework │
│(ray)   │          │(paramiko)  │         │(submodule)│
└────────┘          └────────────┘         └───────────┘
```

### Key Modules

**Backend (`training_platform/`)**:
- `api/main.py` - FastAPI app entry point
- `api/routers/` - API endpoints (jobs, monitoring, compute, surgery, etc.)
- `api/models/` - Pydantic schemas
- `core/verl_adapter.py` - verl command generation and execution
- `core/ssh_runner.py` - SSH remote execution (security hardened)
- `core/crypto_utils.py` - Fernet encryption for SSH passwords
- `core/command_utils.py` - Secure command construction
- `core/metrics_reader.py` - Read training metrics (local/SSH)
- `core/metrics_persister.py` - Persist metrics to database
- `core/database.py` - SQLModel ORM and SQLite database

**Frontend (`frontend/`)**:
- `src/views/` - Page components (JobsView, MonitoringView, etc.)
- `src/stores/` - Pinia stores for state management
- `src/api/index.js` - API client

**verl Integration (`environments/verl/`)**:
- `trainer/callbacks/platform_callback.py` - Platform metrics callback
- `trainer/ppo/ray_trainer.py` - Modified to support custom callbacks

## Run Modes

The platform supports two run modes:

1. **Local Mode**: Training runs on local machine (requires GPU)
2. **SSH Remote Mode**: Training runs on remote GPU server via SSH

Configuration stored in `~/.train_platform/run_mode.json` (passwords encrypted).

## Security Notes

- SSH passwords use Fernet symmetric encryption (`crypto_utils.py`)
- All command execution uses `shlex.quote()` to prevent injection
- Path validation via `validate_path()` in `command_utils.py`
- Never use `shell=True` in subprocess calls

## Database

SQLite database at `training_platform.db`. Key tables:
- `trainingjob` - Training job configurations and status
- `trainingmetric` - Time-series training metrics
- `trainingdataset` - Training dataset metadata

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing Real Training

Use small models for testing:
- Model: Qwen2.5-0.5B-Instruct or TinyLlama
- Dataset: 100-1000 samples in `datasets/` directory
- Test both Local and SSH modes

## Common Tasks Reference

See `TASKS.md` for detailed development roadmap (Phase 0 completed, Phase 1 in progress).
