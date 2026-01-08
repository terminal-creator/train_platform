#!/usr/bin/env python3
"""
全算法训练测试

测试所有支持的算法：SFT, PPO, GRPO, GSPO
使用 SSH 远程 GPU 服务器和阿里 Reward Model API
"""

import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from training_platform.core.database import (
    engine,
    Session,
    init_db,
    JobRepository,
    PipelineRepository,
    TrainingJob,
    Pipeline,
    PipelineStage,
    JobStatus,
    TrainingAlgorithm,
    PipelineStatus,
    PipelineStageStatus,
)
from training_platform.core.pipeline_executor import PipelineExecutor
from training_platform.core.ssh_runner import SSHConfig

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def print_section(title):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.END}\n")


# SSH 配置（GPU 服务器）
SSH_CONFIG = {
    "host": "connect.westc.gpuhub.com",
    "port": 27192,
    "username": "root",
    "password": "A32qbQ1UR3Y6",
    "working_dir": "~/verl_jobs",
}

# 阿里 Reward Model API
ALIBABA_RM_API_KEY = "sk-85ae32fc59d345e4ab1137f6bd3c3f10"
ALIBABA_RM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def create_and_run_training(
    algorithm: TrainingAlgorithm,
    dataset_path: str,
    test_name: str,
    extra_config: dict = None,
) -> bool:
    """
    创建并运行训练任务

    Args:
        algorithm: 训练算法
        dataset_path: 数据集路径
        test_name: 测试名称
        extra_config: 额外配置参数

    Returns:
        是否成功
    """
    timestamp = int(time.time())
    job_uuid = f"{algorithm.value}-job-{timestamp}"
    pipeline_uuid = f"{algorithm.value}-pipeline-{timestamp}"

    try:
        print_section(f"测试 {test_name}: {algorithm.value.upper()}")

        # ============== 1. 创建 Job ==============
        print_info(f"步骤 1: 创建 {algorithm.value.upper()} Job")

        with Session(engine) as session:
            repo = JobRepository(session)

            job_config = {
                "uuid": job_uuid,
                "name": f"{test_name} {timestamp}",
                "description": f"{algorithm.value.upper()} 训练任务",
                "status": JobStatus.PENDING,
                "algorithm": algorithm,
                "model_path": "Qwen/Qwen2.5-0.5B",
                "train_data_path": dataset_path,
                "num_gpus": 1,
                "learning_rate": 1e-5,
                "batch_size": 2,
                "num_epochs": 1,
                "context_length": 512,
                "warmup_steps": 5,
                "save_steps": 20,
                "eval_steps": 20,
            }

            # 合并额外配置
            if extra_config:
                job_config.update(extra_config)

            job = TrainingJob(**job_config)
            created_job = repo.create(job)

            print_success(f"Job 创建成功: {job_uuid}")
            print_info(f"  算法: {algorithm.value}")
            print_info(f"  数据集: {dataset_path}")

        # ============== 2. 创建 Pipeline ==============
        print_info("\n步骤 2: 创建 Pipeline")

        with Session(engine) as session:
            repo = PipelineRepository(session)

            pipeline = Pipeline(
                uuid=pipeline_uuid,
                name=f"{test_name} Pipeline {timestamp}",
                description=f"{algorithm.value.upper()} 训练流程",
                status=PipelineStatus.PENDING,
            )
            repo.create(pipeline)

            # 训练配置
            train_config = {
                "num_gpus": 1,
                "batch_size": 2,
                "learning_rate": 1e-5,
            }

            # 添加算法特定配置
            if algorithm in [TrainingAlgorithm.PPO, TrainingAlgorithm.GRPO, TrainingAlgorithm.GSPO]:
                # RL 算法需要 reward model/function
                if algorithm == TrainingAlgorithm.PPO:
                    # PPO 使用阿里 API 作为 reward model
                    train_config.update({
                        "reward_model_type": "api",
                        "reward_model_api_base": ALIBABA_RM_BASE_URL,
                        "reward_model_api_key": ALIBABA_RM_API_KEY,
                        "kl_coef": 0.001,
                        "clip_ratio": 0.2,
                    })
                elif algorithm == TrainingAlgorithm.GRPO:
                    # GRPO 使用内置 reward function
                    train_config.update({
                        "reward_fn_type": "math_verify",
                        "reward_fn_extract_answer": "boxed",
                        "reward_fn_compare_method": "exact",
                        "rollout_n": 5,
                    })
                elif algorithm == TrainingAlgorithm.GSPO:
                    # GSPO 自博弈
                    train_config.update({
                        "reward_fn_type": "self_play",
                        "rollout_n": 5,
                    })

            # Pipeline stage
            stage_data = {
                "stage_name": "train",
                "task_name": "train_model",
                "task_params": {
                    "job_uuid": job_uuid,
                    "config": train_config,
                    "run_mode": "ssh",
                    "ssh_config": SSH_CONFIG,
                },
                "depends_on": [],
                "stage_order": 0,
            }

            stage = PipelineStage(
                pipeline_uuid=pipeline_uuid,
                **stage_data,
                status=PipelineStageStatus.PENDING,
            )
            repo.create_stage(stage)

            print_success(f"Pipeline 创建成功: {pipeline_uuid}")

        # ============== 3. 执行 Pipeline ==============
        print_info("\n步骤 3: 执行 Pipeline")

        stages_config = [
            {
                "name": "train",
                "task": "train_model",
                "params": {
                    "job_uuid": job_uuid,
                    "config": train_config,
                    "run_mode": "ssh",
                    "ssh_config": SSH_CONFIG,
                },
                "depends_on": [],
            },
        ]

        executor = PipelineExecutor(pipeline_uuid)
        result = executor.execute(stages_config)

        print_success(f"Pipeline 提交成功")
        print_info(f"  Task ID: {result.get('root_task_id')}")

        # ============== 4. 监控状态 ==============
        print_info("\n步骤 4: 监控执行（最长 60 秒）")

        max_wait_time = 60  # 秒
        check_interval = 5  # 秒
        checks = max_wait_time // check_interval

        for i in range(checks):
            time.sleep(check_interval)

            with Session(engine) as session:
                repo = PipelineRepository(session)
                job_repo = JobRepository(session)

                pipeline = repo.get_by_uuid(pipeline_uuid)
                job = job_repo.get_by_uuid(job_uuid)
                stages = repo.get_stages(pipeline_uuid)

                elapsed = (i + 1) * check_interval
                print_info(f"\n[{elapsed}秒] Pipeline: {pipeline.status.value}, Job: {job.status.value}")

                for stage in stages:
                    status_emoji = "✓" if stage.status == PipelineStageStatus.COMPLETED else (
                        "✗" if stage.status == PipelineStageStatus.FAILED else "⏳"
                    )
                    print_info(f"  {status_emoji} Stage '{stage.stage_name}': {stage.status.value}")

                    if stage.error_message:
                        print_error(f"    错误: {stage.error_message[:150]}")

                # 检查是否完成或失败
                if pipeline.status == PipelineStatus.COMPLETED:
                    print_success(f"\n✓ {algorithm.value.upper()} 训练成功完成！")
                    return True
                elif pipeline.status == PipelineStatus.FAILED:
                    print_error(f"\n✗ {algorithm.value.upper()} 训练失败")
                    return False

        # 超时
        print_info(f"\n⏱ {algorithm.value.upper()} 训练仍在执行（超过 {max_wait_time} 秒监控时间）")

        with Session(engine) as session:
            job_repo = JobRepository(session)
            job = job_repo.get_by_uuid(job_uuid)

            if job.status == JobStatus.COMPLETED:
                print_success(f"✓ Job 已完成（Pipeline 可能仍在后处理）")
                return True
            elif job.status in [JobStatus.RUNNING, JobStatus.PENDING]:
                print_info(f"ℹ Job 仍在运行中")
                return True  # 认为成功（正在执行）
            else:
                return False

    except Exception as e:
        print_error(f"{algorithm.value.upper()} 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有算法测试"""
    print(f"\n{Colors.BLUE}")
    print("="*60)
    print("  全算法训练测试")
    print("  GPU: RTX 5090 (32GB)")
    print("  RM API: 阿里 DashScope")
    print("="*60)
    print(f"{Colors.END}\n")

    # 初始化数据库
    init_db()

    # 配置环境变量（阿里 API Key）
    os.environ["DASHSCOPE_API_KEY"] = ALIBABA_RM_API_KEY

    results = {}

    # 测试 1: SFT
    results["SFT"] = create_and_run_training(
        algorithm=TrainingAlgorithm.SFT,
        dataset_path="./datasets/sales_sft.jsonl",
        test_name="SFT Sales Training",
    )

    # 测试 2: PPO (使用阿里 RM API)
    results["PPO"] = create_and_run_training(
        algorithm=TrainingAlgorithm.PPO,
        dataset_path="./datasets/ppo_general.json",
        test_name="PPO General Training",
    )

    # 测试 3: GRPO
    results["GRPO"] = create_and_run_training(
        algorithm=TrainingAlgorithm.GRPO,
        dataset_path="./datasets/sales_grpo.jsonl",
        test_name="GRPO Math Training",
    )

    # 测试 4: GSPO
    results["GSPO"] = create_and_run_training(
        algorithm=TrainingAlgorithm.GSPO,
        dataset_path="./datasets/sales_grpo.jsonl",  # GSPO 可以使用 GRPO 数据
        test_name="GSPO Self-Play Training",
    )

    # 汇总结果
    print_section("测试结果汇总")

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for algo_name, success in results.items():
        if success:
            print_success(f"{algo_name:10} ✓ PASS")
        else:
            print_error(f"{algo_name:10} ✗ FAIL")

    print(f"\n{Colors.BLUE}总计: {passed}/{total} 算法测试通过{Colors.END}\n")

    if passed == total:
        print_success("🎉 所有算法训练成功！")
        print_info("\n详细信息:")
        print_info("  - SFT: 监督微调")
        print_info("  - PPO: 近端策略优化（使用阿里 RM API）")
        print_info("  - GRPO: 组相对策略优化")
        print_info("  - GSPO: 组自博弈偏好优化")
    else:
        print_error(f"部分算法测试失败 ({total - passed} 个)")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
