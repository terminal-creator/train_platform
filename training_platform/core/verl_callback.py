"""
VerlCallback - Deep Instrumentation for verl Training

This module provides a comprehensive callback system for verl training that collects:
1. Training metrics (loss, reward, KL, entropy)
2. Gradient statistics (norm, mean, std per layer)
3. Sample data (prompts, responses, rewards for data review)
4. GPU usage metrics
5. Timing information

The callback pushes data to the Training Platform's monitoring API in real-time.

Usage:
    from training_platform.core.verl_callback import VerlTrainingCallback, create_platform_logger

    # Create logger for verl's Tracking system
    logger = create_platform_logger(
        job_uuid="your-job-uuid",
        platform_url="http://localhost:8000"
    )

    # Or create callback for manual integration
    callback = VerlTrainingCallback(
        job_uuid="your-job-uuid",
        platform_url="http://localhost:8000"
    )
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json
import requests
from threading import Thread
from queue import Queue
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GradientStats:
    """Statistics for gradient analysis"""
    layer_name: str
    norm: float
    mean: float
    std: float
    max_abs: float
    min_abs: float
    sparsity: float  # Fraction of near-zero gradients


@dataclass
class SampleData:
    """Sample data for data review"""
    index: int
    prompt: str
    response: str
    reward: float
    kl: float
    log_prob: float
    ref_log_prob: float
    advantage: float
    label: Optional[str] = None  # Optional label field


@dataclass
class MetricsBuffer:
    """Buffer for batching metrics before sending"""
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    gradient_stats: List[Dict[str, Any]] = field(default_factory=list)
    samples: List[Dict[str, Any]] = field(default_factory=list)
    gpu_metrics: List[Dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([self.metrics, self.gradient_stats, self.samples, self.gpu_metrics])


class AsyncMetricsPusher:
    """Asynchronously push metrics to the platform API"""

    def __init__(self, platform_url: str, job_uuid: str, batch_size: int = 10):
        self.platform_url = platform_url.rstrip('/')
        self.job_uuid = job_uuid
        self.batch_size = batch_size
        self.queue = Queue()
        self._session = requests.Session()
        self._running = True
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _get_url(self, endpoint: str) -> str:
        return f"{self.platform_url}/api/v1/monitoring/{endpoint}"

    def _worker(self):
        """Background worker that sends buffered metrics"""
        buffer = MetricsBuffer()
        last_flush = time.time()
        flush_interval = 5.0  # Flush every 5 seconds

        while self._running:
            try:
                # Get items from queue with timeout
                try:
                    item_type, data = self.queue.get(timeout=1.0)

                    if item_type == "metrics":
                        buffer.metrics.append(data)
                    elif item_type == "gradients":
                        buffer.gradient_stats.append(data)
                    elif item_type == "sample":
                        buffer.samples.append(data)
                    elif item_type == "gpu":
                        buffer.gpu_metrics.append(data)

                    self.queue.task_done()
                except:
                    pass  # Timeout, continue

                # Flush if buffer is full or time elapsed
                should_flush = (
                    len(buffer.metrics) >= self.batch_size or
                    len(buffer.samples) >= self.batch_size or
                    (time.time() - last_flush) >= flush_interval
                )

                if should_flush and not buffer.is_empty():
                    self._flush(buffer)
                    buffer = MetricsBuffer()
                    last_flush = time.time()

            except Exception as e:
                logger.warning(f"Error in metrics pusher: {e}")

    def _flush(self, buffer: MetricsBuffer):
        """Send buffered metrics to platform"""
        try:
            # Send training metrics
            if buffer.metrics:
                self._session.post(
                    self._get_url("report/batch"),
                    json=buffer.metrics,
                    timeout=5.0
                )

            # Send gradient stats
            if buffer.gradient_stats:
                self._session.post(
                    self._get_url("gradients/batch"),
                    json={
                        "job_uuid": self.job_uuid,
                        "gradients": buffer.gradient_stats
                    },
                    timeout=5.0
                )

            # Send samples
            if buffer.samples:
                self._session.post(
                    self._get_url("samples/batch"),
                    json={
                        "job_uuid": self.job_uuid,
                        "samples": buffer.samples
                    },
                    timeout=5.0
                )

            # Send GPU metrics
            if buffer.gpu_metrics:
                for gpu_metric in buffer.gpu_metrics:
                    self._session.post(
                        self._get_url("gpu"),
                        json=gpu_metric,
                        timeout=5.0
                    )

        except Exception as e:
            logger.warning(f"Failed to flush metrics: {e}")

    def push_metrics(self, step: int, metrics: Dict[str, Any], epoch: int = 0):
        """Queue training metrics for sending"""
        self.queue.put(("metrics", {
            "job_uuid": self.job_uuid,
            "step": step,
            "epoch": epoch,
            **metrics
        }))

    def push_gradient_stats(self, step: int, stats: List[GradientStats]):
        """Queue gradient statistics"""
        self.queue.put(("gradients", {
            "step": step,
            "stats": [
                {
                    "layer_name": s.layer_name,
                    "norm": s.norm,
                    "mean": s.mean,
                    "std": s.std,
                    "max_abs": s.max_abs,
                    "min_abs": s.min_abs,
                    "sparsity": s.sparsity,
                }
                for s in stats
            ]
        }))

    def push_sample(self, step: int, sample: SampleData):
        """Queue sample data"""
        self.queue.put(("sample", {
            "step": step,
            "index": sample.index,
            "prompt": sample.prompt[:2000],  # Truncate long prompts
            "response": sample.response[:2000],
            "reward": sample.reward,
            "kl": sample.kl,
            "log_prob": sample.log_prob,
            "ref_log_prob": sample.ref_log_prob,
            "advantage": sample.advantage,
            "label": sample.label,
        }))

    def push_gpu_usage(self, gpu_index: int, utilization: float,
                       memory_used: float, memory_total: float,
                       temperature: Optional[float] = None):
        """Queue GPU usage metrics"""
        self.queue.put(("gpu", {
            "job_uuid": self.job_uuid,
            "gpu_index": gpu_index,
            "utilization": utilization,
            "memory_used": memory_used,
            "memory_total": memory_total,
            "temperature": temperature,
        }))

    def stop(self):
        """Stop the background worker"""
        self._running = False
        self._thread.join(timeout=10)
        self._session.close()


class VerlTrainingCallback:
    """
    Deep instrumentation callback for verl training.

    This callback collects comprehensive training data including:
    - Standard metrics (loss, reward, KL, etc.)
    - Gradient statistics per layer
    - Sample data for data review
    - GPU utilization

    Example:
        callback = VerlTrainingCallback(
            job_uuid="abc123",
            platform_url="http://localhost:8000",
            collect_gradients=True,
            collect_samples=True,
            sample_rate=0.01,  # Sample 1% of data
        )

        # In training loop
        callback.on_train_step(step, model, metrics, batch_data)
    """

    def __init__(
        self,
        job_uuid: str,
        platform_url: str = "http://localhost:8000",
        collect_gradients: bool = True,
        collect_samples: bool = True,
        sample_rate: float = 0.01,
        gradient_layers: Optional[List[str]] = None,
    ):
        self.job_uuid = job_uuid
        self.pusher = AsyncMetricsPusher(platform_url, job_uuid)
        self.collect_gradients = collect_gradients
        self.collect_samples = collect_samples
        self.sample_rate = sample_rate
        self.gradient_layers = gradient_layers or [
            "layers.0", "layers.15", "layers.31",  # Sample layers for 32-layer models
            "embed_tokens", "lm_head"
        ]

        self._step_times: List[float] = []
        self._last_step_time = time.time()
        self._samples_collected = 0

    def on_train_start(self, config: Dict[str, Any]):
        """Called at the start of training"""
        logger.info(f"VerlTrainingCallback started for job {self.job_uuid}")
        self._notify_status("running", "Training started")

    def on_train_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        model=None,  # PyTorch model for gradient collection
        batch_data=None,  # Batch data for sample collection
        epoch: int = 0,
    ):
        """Called after each training step"""
        # Track step timing
        current_time = time.time()
        step_time = current_time - self._last_step_time
        self._step_times.append(step_time)
        self._last_step_time = current_time

        # Calculate throughput
        if len(self._step_times) >= 10:
            avg_step_time = sum(self._step_times[-10:]) / 10
            metrics["throughput/steps_per_sec"] = 1.0 / avg_step_time

        # Normalize and push metrics
        normalized_metrics = self._normalize_metrics(metrics)
        self.pusher.push_metrics(step, normalized_metrics, epoch)

        # Collect gradient statistics
        if self.collect_gradients and model is not None and step % 10 == 0:
            grad_stats = self._collect_gradient_stats(model)
            if grad_stats:
                self.pusher.push_gradient_stats(step, grad_stats)

        # Collect sample data
        if self.collect_samples and batch_data is not None:
            self._collect_samples(step, batch_data)

        # Collect GPU usage periodically
        if step % 50 == 0:
            self._collect_gpu_usage()

    def on_eval_step(self, step: int, eval_results: Dict[str, float]):
        """Called after evaluation"""
        eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
        self.pusher.push_metrics(step, eval_metrics)

    def on_save_checkpoint(
        self,
        step: int,
        checkpoint_path: str,
        eval_results: Optional[Dict[str, float]] = None,
    ):
        """Called when saving a checkpoint"""
        self._notify_checkpoint(step, checkpoint_path, eval_results)

    def on_train_end(self, summary: Optional[Dict[str, Any]] = None):
        """Called at the end of training"""
        self._notify_status("completed", "Training completed")
        self.pusher.stop()

    def on_train_error(self, error: Exception):
        """Called when training fails"""
        self._notify_status("failed", str(error))
        self.pusher.stop()

    def _normalize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize metric names to platform standard"""
        mapping = {
            "train/loss": "total_loss",
            "train/policy_loss": "policy_loss",
            "train/value_loss": "value_loss",
            "train/reward": "reward_mean",
            "train/reward_mean": "reward_mean",
            "train/reward_std": "reward_std",
            "train/kl": "kl_divergence",
            "actor/kl": "kl_divergence",
            "train/entropy": "entropy",
            "actor/entropy": "entropy",
            "actor/loss": "policy_loss",
            "critic/loss": "value_loss",
            "loss": "total_loss",
            "reward": "reward_mean",
            "kl": "kl_divergence",
        }

        normalized = {}
        extra = {}

        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue

            # Check for NaN/Inf
            if np.isnan(value) or np.isinf(value):
                continue

            std_key = mapping.get(key)
            if std_key:
                normalized[std_key] = float(value)
            else:
                extra[key] = float(value)

        if extra:
            normalized["extra_metrics"] = extra

        return normalized

    def _collect_gradient_stats(self, model) -> List[GradientStats]:
        """Collect gradient statistics from model"""
        stats = []

        try:
            for name, param in model.named_parameters():
                # Filter to only selected layers
                if not any(layer in name for layer in self.gradient_layers):
                    continue

                if param.grad is None:
                    continue

                grad = param.grad.detach()
                grad_flat = grad.flatten().float()

                # Calculate statistics
                norm = grad.norm().item()
                mean = grad_flat.mean().item()
                std = grad_flat.std().item()
                abs_grad = grad_flat.abs()
                max_abs = abs_grad.max().item()
                min_abs = abs_grad.min().item()
                sparsity = (abs_grad < 1e-8).float().mean().item()

                stats.append(GradientStats(
                    layer_name=name,
                    norm=norm,
                    mean=mean,
                    std=std,
                    max_abs=max_abs,
                    min_abs=min_abs,
                    sparsity=sparsity,
                ))
        except Exception as e:
            logger.warning(f"Error collecting gradient stats: {e}")

        return stats

    def _collect_samples(self, step: int, batch_data):
        """Collect sample data for data review"""
        try:
            import random

            # batch_data is expected to be a DataProto or similar structure
            batch_size = self._get_batch_size(batch_data)
            if batch_size == 0:
                return

            # Sample based on rate
            num_samples = max(1, int(batch_size * self.sample_rate))
            indices = random.sample(range(batch_size), min(num_samples, batch_size))

            for idx in indices:
                sample = self._extract_sample(batch_data, idx, step)
                if sample:
                    self.pusher.push_sample(step, sample)
                    self._samples_collected += 1

        except Exception as e:
            logger.warning(f"Error collecting samples: {e}")

    def _get_batch_size(self, batch_data) -> int:
        """Get batch size from data"""
        if hasattr(batch_data, 'batch'):
            if hasattr(batch_data.batch, 'batch_size'):
                return batch_data.batch.batch_size[0]
        if hasattr(batch_data, '__len__'):
            return len(batch_data)
        return 0

    def _extract_sample(self, batch_data, idx: int, step: int) -> Optional[SampleData]:
        """Extract a single sample from batch"""
        try:
            batch = getattr(batch_data, 'batch', batch_data)

            # Get tokenizer for decoding (if available)
            tokenizer = getattr(batch_data, 'tokenizer', None)

            # Extract fields
            prompt = self._decode_tensor(batch.get('input_ids', None), idx, tokenizer)
            response = self._decode_tensor(batch.get('responses', None), idx, tokenizer)

            # Get scalar values
            reward = self._get_scalar(batch.get('token_level_scores', None), idx)
            kl = self._get_scalar(batch.get('kl', None), idx)
            log_prob = self._get_scalar(batch.get('old_log_probs', None), idx)
            ref_log_prob = self._get_scalar(batch.get('ref_log_prob', None), idx)
            advantage = self._get_scalar(batch.get('advantages', None), idx)

            return SampleData(
                index=self._samples_collected,
                prompt=prompt or "[unknown]",
                response=response or "[unknown]",
                reward=reward or 0.0,
                kl=kl or 0.0,
                log_prob=log_prob or 0.0,
                ref_log_prob=ref_log_prob or 0.0,
                advantage=advantage or 0.0,
            )
        except Exception as e:
            logger.debug(f"Error extracting sample: {e}")
            return None

    def _decode_tensor(self, tensor, idx: int, tokenizer) -> Optional[str]:
        """Decode tensor to string"""
        if tensor is None:
            return None

        try:
            if idx >= len(tensor):
                return None

            tokens = tensor[idx]
            if tokenizer:
                return tokenizer.decode(tokens, skip_special_tokens=True)
            else:
                return str(tokens.tolist()[:50])  # First 50 tokens as list
        except:
            return None

    def _get_scalar(self, tensor, idx: int) -> Optional[float]:
        """Get scalar value from tensor"""
        if tensor is None:
            return None

        try:
            if hasattr(tensor, '__getitem__'):
                val = tensor[idx]
                if hasattr(val, 'sum'):
                    return float(val.sum().item())
                return float(val)
        except:
            pass
        return None

    def _collect_gpu_usage(self):
        """Collect GPU usage metrics"""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_used = torch.cuda.memory_allocated(i) / 1024**3
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3

                    # Get utilization via nvidia-smi (if available)
                    utilization = self._get_gpu_utilization(i)

                    self.pusher.push_gpu_usage(
                        gpu_index=i,
                        utilization=utilization,
                        memory_used=memory_used,
                        memory_total=memory_total,
                    )
        except Exception as e:
            logger.debug(f"Error collecting GPU usage: {e}")

    def _get_gpu_utilization(self, gpu_index: int) -> float:
        """Get GPU utilization percentage"""
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', f'--id={gpu_index}', '--query-gpu=utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1.0
            )
            return float(result.stdout.strip())
        except:
            return 0.0

    def _notify_status(self, status: str, message: str):
        """Notify platform of status change"""
        try:
            requests.post(
                f"{self.pusher.platform_url}/api/v1/monitoring/status",
                json={
                    "job_uuid": self.job_uuid,
                    "status": status,
                    "message": message,
                },
                timeout=5.0
            )
        except Exception as e:
            logger.warning(f"Failed to notify status: {e}")

    def _notify_checkpoint(self, step: int, path: str, eval_results: Optional[Dict]):
        """Notify platform of checkpoint save"""
        try:
            requests.post(
                f"{self.pusher.platform_url}/api/v1/monitoring/checkpoint",
                json={
                    "job_uuid": self.job_uuid,
                    "step": step,
                    "path": path,
                    "eval_results": eval_results,
                },
                timeout=5.0
            )
        except Exception as e:
            logger.warning(f"Failed to notify checkpoint: {e}")


class PlatformLogger:
    """
    Logger compatible with verl's Tracking system.

    This can be used as a backend in verl's Tracking class.

    Example:
        # Add to verl's tracking.py supported_backend list:
        # "platform"

        # Then in trainer:
        from verl.utils.tracking import Tracking
        logger = Tracking(
            project_name="my-project",
            experiment_name="exp1",
            default_backend=["wandb", "platform"]
        )
    """

    def __init__(self, job_uuid: str, platform_url: str = "http://localhost:8000"):
        self.job_uuid = job_uuid
        self.pusher = AsyncMetricsPusher(platform_url, job_uuid)

    def log(self, data: Dict[str, Any], step: int):
        """Log metrics (compatible with verl's Tracking interface)"""
        # Normalize metric names
        normalized = {}
        for key, value in data.items():
            if isinstance(value, (int, float)):
                normalized[key] = float(value)

        self.pusher.push_metrics(step, normalized)

    def finish(self):
        """Finish logging"""
        self.pusher.stop()


def create_platform_logger(
    job_uuid: str,
    platform_url: str = "http://localhost:8000"
) -> PlatformLogger:
    """
    Create a logger for verl's Tracking system.

    This can be added to verl's supported backends.
    """
    return PlatformLogger(job_uuid, platform_url)


def create_verl_callback(
    job_uuid: str,
    platform_url: str = "http://localhost:8000",
    collect_gradients: bool = True,
    collect_samples: bool = True,
    sample_rate: float = 0.01,
) -> VerlTrainingCallback:
    """
    Create a VerlTrainingCallback for deep instrumentation.

    This is the main entry point for creating a callback.

    Args:
        job_uuid: Unique identifier for the training job
        platform_url: URL of the Training Platform API
        collect_gradients: Whether to collect gradient statistics
        collect_samples: Whether to collect sample data
        sample_rate: Fraction of samples to collect (0.01 = 1%)

    Returns:
        VerlTrainingCallback instance
    """
    return VerlTrainingCallback(
        job_uuid=job_uuid,
        platform_url=platform_url,
        collect_gradients=collect_gradients,
        collect_samples=collect_samples,
        sample_rate=sample_rate,
    )
