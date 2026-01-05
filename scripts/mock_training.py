#!/usr/bin/env python3
"""
Mock training script for testing the monitoring system.

Usage:
    python scripts/mock_training.py --job-uuid <job_uuid> --total-steps 100
"""

import argparse
import time
import random
import math
import requests
import sys

API_BASE = "http://localhost:8000/api/v1"


def report_metrics(job_uuid: str, step: int, epoch: int, metrics: dict):
    """Report metrics to the platform"""
    try:
        resp = requests.post(
            f"{API_BASE}/monitoring/report",
            json={
                "job_uuid": job_uuid,
                "step": step,
                "epoch": epoch,
                **metrics,
            },
            timeout=5,
        )
        if resp.status_code == 200:
            print(f"Step {step}: reported metrics")
        else:
            print(f"Step {step}: failed to report - {resp.text}")
    except Exception as e:
        print(f"Step {step}: error reporting - {e}")


def report_status(job_uuid: str, status: str, message: str = None):
    """Report status change"""
    try:
        requests.post(
            f"{API_BASE}/monitoring/status",
            json={
                "job_uuid": job_uuid,
                "status": status,
                "message": message,
            },
            timeout=5,
        )
    except Exception as e:
        print(f"Error reporting status: {e}")


def report_gpu_usage(job_uuid: str, num_gpus: int = 1):
    """Report GPU usage"""
    try:
        for i in range(num_gpus):
            requests.post(
                f"{API_BASE}/monitoring/gpu",
                json={
                    "job_uuid": job_uuid,
                    "gpu_index": i,
                    "utilization": random.uniform(85, 99),
                    "memory_used": random.uniform(35, 45),
                    "memory_total": 48.0,
                    "temperature": random.uniform(55, 75),
                },
                timeout=5,
            )
    except Exception as e:
        print(f"Error reporting GPU: {e}")


def simulate_training(job_uuid: str, total_steps: int = 100, num_epochs: int = 3):
    """Simulate a training run with realistic metrics"""
    print(f"Starting mock training for job {job_uuid}")
    print(f"Total steps: {total_steps}, Epochs: {num_epochs}")

    # Report running status
    report_status(job_uuid, "running", "Training started")

    steps_per_epoch = total_steps // num_epochs

    # Initial loss values
    policy_loss = 2.5
    value_loss = 1.8
    reward_mean = -0.5
    kl_divergence = 0.01

    for step in range(1, total_steps + 1):
        epoch = (step - 1) // steps_per_epoch + 1

        # Simulate loss decreasing with some noise
        progress = step / total_steps
        policy_loss = 2.5 * math.exp(-2 * progress) + random.uniform(-0.05, 0.05)
        value_loss = 1.8 * math.exp(-1.5 * progress) + random.uniform(-0.03, 0.03)
        reward_mean = -0.5 + 1.5 * progress + random.uniform(-0.1, 0.1)
        kl_divergence = 0.01 + 0.02 * progress + random.uniform(-0.005, 0.005)
        entropy = 2.0 - 0.5 * progress + random.uniform(-0.1, 0.1)

        # Learning rate with warmup and decay
        warmup_steps = total_steps * 0.1
        if step < warmup_steps:
            lr = 1e-4 * (step / warmup_steps)
        else:
            lr = 1e-4 * (1 - (step - warmup_steps) / (total_steps - warmup_steps))

        metrics = {
            "policy_loss": max(0, policy_loss),
            "value_loss": max(0, value_loss),
            "total_loss": max(0, policy_loss + value_loss),
            "reward_mean": reward_mean,
            "reward_std": random.uniform(0.1, 0.3),
            "kl_divergence": max(0, kl_divergence),
            "entropy": max(0, entropy),
            "learning_rate": lr,
        }

        # Report metrics
        report_metrics(job_uuid, step, epoch, metrics)

        # Report GPU usage every 5 steps
        if step % 5 == 0:
            report_gpu_usage(job_uuid, num_gpus=1)

        # Print progress
        if step % 10 == 0:
            print(f"Epoch {epoch} | Step {step}/{total_steps} | "
                  f"Loss: {policy_loss:.4f} | Reward: {reward_mean:.4f}")

        # Simulate training time
        time.sleep(0.5)

    # Report completed status
    report_status(job_uuid, "completed", "Training completed successfully")
    print(f"Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Mock training script")
    parser.add_argument("--job-uuid", required=True, help="Job UUID")
    parser.add_argument("--total-steps", type=int, default=100, help="Total training steps")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")

    args = parser.parse_args()

    try:
        simulate_training(args.job_uuid, args.total_steps, args.num_epochs)
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        report_status(args.job_uuid, "cancelled", "Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
