#!/usr/bin/env python
"""
测试 W&B 集成

用法:
    # 离线模式测试 (不需要登录)
    python scripts/test_wandb_integration.py --offline

    # 在线模式 (需要 WANDB_API_KEY)
    WANDB_API_KEY=xxx python scripts/test_wandb_integration.py
"""

import argparse
import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_dual_logger(offline: bool = True):
    """测试 DualLogger"""
    print("=" * 50)
    print("测试 DualLogger")
    print("=" * 50)

    if offline:
        os.environ["WANDB_MODE"] = "offline"
        print("模式: 离线 (不上传到 W&B 服务器)")
    else:
        print("模式: 在线")

    from training_platform.core.wandb_callback import DualLogger

    # 创建 logger (平台禁用因为没有后端运行)
    logger = DualLogger(
        job_uuid="test-job-001",
        platform_url="http://localhost:8000",
        platform_enabled=False,  # 禁用平台 (没有后端运行)
        wandb_project="test-integration",
        wandb_run_name="test-run",
        wandb_config={
            "model": "Qwen2.5-7B",
            "algorithm": "GRPO",
            "learning_rate": 1e-6,
        },
        wandb_tags=["test", "integration"],
        wandb_enabled=True,
    )

    print("\n开始模拟训练...")

    # 模拟训练 10 步
    for step in range(10):
        metrics = {
            "loss": 1.0 - step * 0.08,
            "reward_mean": 0.1 + step * 0.05,
            "kl_divergence": 0.01 + step * 0.001,
        }

        logger.log_metrics(step, metrics)
        print(f"  Step {step}: loss={metrics['loss']:.3f}, reward={metrics['reward_mean']:.3f}")
        time.sleep(0.1)

    # 记录汇总
    logger.log_summary({
        "best_reward": 0.55,
        "final_loss": 0.28,
    })

    logger.finish()
    print("\n测试完成!")

    if offline:
        print("\n离线模式下，日志保存在 ./wandb/ 目录")
        print("可以使用 'wandb sync wandb/offline-run-*' 上传到服务器")


def test_platform_logger():
    """测试仅平台 Logger (需要后端运行)"""
    print("=" * 50)
    print("测试 Platform Logger")
    print("=" * 50)

    from training_platform.core.wandb_callback import DualLogger
    import requests

    # 检查后端是否运行
    try:
        resp = requests.get("http://localhost:8000/api/v1/monitoring/health", timeout=2)
        backend_running = resp.status_code == 200
    except Exception:
        backend_running = False

    if not backend_running:
        print("警告: 后端未运行 (http://localhost:8000)")
        print("跳过平台 Logger 测试")
        return

    logger = DualLogger(
        job_uuid="test-platform-001",
        platform_url="http://localhost:8000",
        platform_enabled=True,
        wandb_enabled=False,
    )

    for step in range(5):
        logger.log_metrics(step, {"loss": 1.0 - step * 0.1})
        print(f"  Step {step}: sent to platform")

    logger.finish()
    print("平台 Logger 测试完成!")


def test_init_logging():
    """测试便捷函数 init_logging"""
    print("=" * 50)
    print("测试 init_logging 便捷函数")
    print("=" * 50)

    os.environ["WANDB_MODE"] = "offline"

    from training_platform.core.wandb_callback import init_logging

    logger = init_logging(
        job_uuid="quick-test",
        wandb_project="quick-project",
    )

    logger.log_metrics(0, {"loss": 0.5})
    logger.log_metrics(1, {"loss": 0.4})
    logger.finish()

    print("init_logging 测试完成!")


def test_verl_callback():
    """测试 verl 回调"""
    print("=" * 50)
    print("测试 verl 回调")
    print("=" * 50)

    os.environ["WANDB_MODE"] = "offline"

    from training_platform.core.wandb_callback import create_verl_callback

    callback = create_verl_callback(
        job_uuid="verl-test",
        wandb_project="verl-project",
        wandb_config={"model": "test"},
    )

    # 模拟 verl 训练生命周期
    callback.on_train_start({"epochs": 10, "batch_size": 256})

    for step in range(5):
        callback.on_train_step(step, {"loss": 1.0 - step * 0.1}, epoch=0)

    callback.on_eval_step(5, {"gsm8k": 0.45})
    callback.on_train_end({"best_score": 0.45})

    print("verl 回调测试完成!")


def main():
    parser = argparse.ArgumentParser(description="测试 W&B 集成")
    parser.add_argument("--offline", action="store_true", help="使用离线模式")
    parser.add_argument("--online", action="store_true", help="使用在线模式 (需要 API Key)")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("Training Platform - W&B 集成测试")
    print("=" * 50 + "\n")

    # 默认离线模式
    offline = not args.online

    try:
        # 测试 DualLogger
        test_dual_logger(offline=offline)
        print()

        if args.all:
            # 测试平台 Logger
            test_platform_logger()
            print()

            # 测试便捷函数
            test_init_logging()
            print()

            # 测试 verl 回调
            test_verl_callback()
            print()

        print("\n" + "=" * 50)
        print("所有测试通过!")
        print("=" * 50)

        if offline:
            print("\n提示: 使用 --online 参数可测试在线模式")
            print("需要先设置 WANDB_API_KEY 环境变量")

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
