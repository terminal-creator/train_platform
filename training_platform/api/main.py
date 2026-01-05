"""
Training Platform API - Main Application

A comprehensive platform for LLM training based on verl framework.
Features:
- Compute Calculator: Optimize GPU and batch size configurations
- Model Surgery: Merge, average, and optimize model checkpoints
- Training Jobs: Manage and monitor training runs
- Real-time Monitoring: Track metrics, gradients, and resources
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import time
import logging

from .routers import compute_router, surgery_router, jobs_router, monitoring_router, evaluation_router, training_dataset_router
from .routers.websocket import router as websocket_router, metrics_collector
from ..core.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting Training Platform API...")

    # Initialize database
    logger.info("Initializing database...")
    init_db()

    # Start metrics collector
    logger.info("Starting metrics collector...")
    await metrics_collector.start()

    yield

    # Cleanup
    logger.info("Stopping metrics collector...")
    await metrics_collector.stop()
    logger.info("Shutting down Training Platform API...")


# Create FastAPI application
app = FastAPI(
    title="Training Platform API",
    description="""
## LLM Training Platform

A comprehensive platform for large language model training based on the verl framework.

### Features

- **Compute Calculator**: Automatically calculate optimal training configurations
  - Memory estimation
  - ZeRO stage recommendation
  - Batch size optimization

- **Model Surgery**: Post-training model optimization
  - Model merging (SLERP, TIES, DARE)
  - Checkpoint selection
  - Weight averaging (EMA, SWA)

- **Training Jobs**: Full training lifecycle management
  - Job creation and monitoring
  - Pause/resume support
  - Automatic evaluation

- **Real-time Monitoring**: Live training insights
  - Metrics streaming
  - Gradient heatmaps
  - Resource utilization
  - Alert management

### Supported Training Types

- **SFT**: Supervised Fine-Tuning
- **PPO**: Proximal Policy Optimization
- **GRPO**: Group Relative Policy Optimization
- **DPO**: Direct Preference Optimization
- **GSPO**: Group Self-Play Preference Optimization

### LoRA Support

All training types support LoRA (Low-Rank Adaptation) for efficient fine-tuning.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# Include routers
app.include_router(compute_router, prefix="/api/v1")
app.include_router(surgery_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(websocket_router, prefix="/api/v1")
app.include_router(evaluation_router, prefix="/api/v1")
app.include_router(training_dataset_router, prefix="/api/v1")

# Import dataset router (optional - requires Milvus)
try:
    from .routers.dataset import router as dataset_router
    app.include_router(dataset_router, prefix="/api/v1")
except ImportError:
    logger.warning("Dataset router not available (Milvus may not be installed)")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Training Platform API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "compute": "/api/v1/compute",
            "surgery": "/api/v1/surgery",
            "jobs": "/api/v1/jobs",
            "monitoring": "/api/v1/monitoring",
            "evaluation": "/api/v1/evaluation",
            "training_datasets": "/api/v1/training-datasets",
        },
    }


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
    }


# API info
@app.get("/api/v1")
async def api_info():
    """API version information"""
    return {
        "version": "1.0.0",
        "supported_algorithms": ["sft", "ppo", "grpo", "dpo", "gspo"],
        "supported_models": ["0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B"],
        "supported_gpus": [
            "A100-40G", "A100-80G", "H100-80G", "H100-SXM",
            "A800-80G", "H800-80G", "RTX4090", "L40S"
        ],
        "features": [
            "compute_calculator",
            "model_merger",
            "checkpoint_selector",
            "training_jobs",
            "real_time_monitoring",
            "gradient_heatmap",
            "alert_management",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
