# API Routers
from .compute import router as compute_router
from .surgery import router as surgery_router
from .jobs import router as jobs_router
from .monitoring import router as monitoring_router
from .websocket import router as websocket_router
from .dataset import router as dataset_router
from .evaluation import router as evaluation_router
from .training_dataset import router as training_dataset_router

__all__ = [
    "compute_router",
    "surgery_router",
    "jobs_router",
    "monitoring_router",
    "websocket_router",
    "dataset_router",
    "evaluation_router",
    "training_dataset_router",
]
