# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Training Platform built on the [verl](https://github.com/volcengine/verl) framework. Provides training job management, compute configuration optimization, model surgery, and real-time monitoring for SFT, PPO, GRPO, DPO, and GSPO training algorithms.

**Current Version**: v1.1.0 (feature-driven development, see feature_list.json)
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



## MANDATORY: Agent Workflow

Every new agent session MUST follow this workflow:

### Step 1: Initialize Environment

```bash
./init.sh
```

This will:
- Install all dependencies
- Start the development server at http://localhost:3001

**DO NOT skip this step.** Ensure the server is running before proceeding.

### Step 2: Select Next Task

Read `feature_list.json` and select ONE task to work on.

Selection criteria (in order of priority):
1. Choose a task where `passes: false`
2. Consider dependencies — fundamental features should be done first
3. Pick the highest-priority incomplete task

### Step 3: Implement the Task

- Read the task description and steps carefully
- Implement the functionality to satisfy all steps
- Follow existing code patterns and conventions

### Step 4: Test Thoroughly

After implementation, verify ALL steps in the task:
- Write unit tests if applicable
- Use browser testing for UI features (MCP Puppeteer tools)这一步需要用mcp工具
- Run `cd frontend && npm run build` to verify zero TypeScript errors
- Fix any errors before proceeding

### Step 5: Update Progress

1. Only change `"passes": false` → `"passes": true` in `feature_list.json` after **verified** testing
2. Append a progress entry to `claude-progress.txt` with the following format:

```
## [Date] — Task: [task description]
- What was done
- Current state
- Known issues
- Next priority
```

- It is **unacceptable** to remove or edit existing feature descriptions in `feature_list.json` — only change the `passes` field

### Step 6: Git Commit

After updating progress, commit your changes:
- Stage all relevant changed files
- Write a clear commit message describing the feature implemented
- Do NOT push to remote unless explicitly asked
提交到self-sop这个分支

### Key Rules
- Never declare the project "done" — always check `feature_list.json` for remaining `passes: false` items
- If you find a bug in an existing feature, fix it before starting new work
- Leave the codebase in a clean, working state at the end of every session
- If a feature cannot be completed in one session, document progress in `claude-progress.txt` and leave the code compilable
- 在需要的地方都打上log可以帮助你进行debug
- Work on **one feature at a time** — do not attempt multiple features in a single session
- 如果需要人工协助，比如申请apikey等，可以停下来找我，但是只有在必要的情况下，在test.md 里面有autodl链接的ssh和qwen-max的apikey，在autodl里面有下载好的model

过程你可以用英文确保准确率，告诉我结果和与我交流的时候用中文