#!/usr/bin/env python3
"""
真实训练任务测试

测试完整的训练流程：
1. 创建真实训练 Job
2. 创建包含预处理、训练、评测的 Pipeline
3. 执行并监控状态
"""

import sys
import time
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


def test_sft_training_pipeline():
    """测试完整的 SFT 训练 Pipeline"""
    print_section("真实 SFT 训练 Pipeline 测试")

    timestamp = int(time.time())
    job_uuid = f"sft-job-{timestamp}"
    pipeline_uuid = f"sft-pipeline-{timestamp}"

    try:
        # ============== 1. 创建训练 Job ==============
        print_info("步骤 1: 创建训练 Job")

        with Session(engine) as session:
            repo = JobRepository(session)

            # 使用 sales_sft.jsonl 数据集
            job = TrainingJob(
                uuid=job_uuid,
                name=f"Sales SFT Training {timestamp}",
                description="销售对话 SFT 微调任务",
                status=JobStatus.PENDING,
                algorithm=TrainingAlgorithm.SFT,
                model_path="Qwen/Qwen2.5-0.5B",  # 小模型，快速测试
                train_data_path="./datasets/sales_sft.jsonl",
                num_gpus=1,
                learning_rate=1e-5,
                batch_size=2,  # 小 batch，快速迭代
                num_epochs=1,
                context_length=512,
                warmup_steps=10,
                save_steps=50,
                eval_steps=50,
            )

            created_job = repo.create(job)
            print_success(f"Job 创建成功: {job_uuid}")
            print_info(f"  算法: {created_job.algorithm}")
            print_info(f"  模型: {created_job.model_path}")
            print_info(f"  数据: {created_job.train_data_path}")

        # ============== 2. 创建 Pipeline ==============
        print_info("\n步骤 2: 创建 Pipeline")

        with Session(engine) as session:
            repo = PipelineRepository(session)

            # 创建 pipeline
            pipeline = Pipeline(
                uuid=pipeline_uuid,
                name=f"SFT Training Pipeline {timestamp}",
                description="完整的 SFT 训练流程",
                status=PipelineStatus.PENDING,
            )
            repo.create(pipeline)

            # 定义 stages
            stages_data = [
                {
                    "stage_name": "preprocess",
                    "task_name": "preprocess_dataset",
                    "task_params": {
                        "dataset_uuid": job_uuid,
                        "preprocessing_config": {
                            "deduplicate": False,  # 数据集较小，不去重
                            "max_length": 512,
                        },
                    },
                    "depends_on": [],
                    "stage_order": 0,
                },
                {
                    "stage_name": "train",
                    "task_name": "train_model",
                    "task_params": {
                        "job_uuid": job_uuid,
                        "config": {
                            "num_gpus": 1,
                            "batch_size": 2,
                            "learning_rate": 1e-5,
                        },
                        "run_mode": "local",  # 本地模式
                    },
                    "depends_on": ["preprocess"],
                    "stage_order": 1,
                },
                {
                    "stage_name": "evaluate",
                    "task_name": "run_evaluation",
                    "task_params": {
                        "job_uuid": job_uuid,
                        "eval_config": {
                            "metrics": ["loss", "accuracy"],
                        },
                    },
                    "depends_on": ["train"],
                    "stage_order": 2,
                },
            ]

            for stage_data in stages_data:
                stage = PipelineStage(
                    pipeline_uuid=pipeline_uuid,
                    **stage_data,
                    status=PipelineStageStatus.PENDING,
                )
                repo.create_stage(stage)

            print_success(f"Pipeline 创建成功: {pipeline_uuid}")
            print_info(f"  Stages: {len(stages_data)}")

        # ============== 3. 执行 Pipeline ==============
        print_info("\n步骤 3: 执行 Pipeline")

        stages_config = [
            {
                "name": "preprocess",
                "task": "preprocess_dataset",
                "params": {
                    "dataset_uuid": job_uuid,
                    "preprocessing_config": {
                        "deduplicate": False,
                        "max_length": 512,
                    },
                },
                "depends_on": [],
            },
            {
                "name": "train",
                "task": "train_model",
                "params": {
                    "job_uuid": job_uuid,
                    "config": {
                        "num_gpus": 1,
                        "batch_size": 2,
                        "learning_rate": 1e-5,
                    },
                    "run_mode": "local",
                },
                "depends_on": ["preprocess"],
            },
            {
                "name": "evaluate",
                "task": "run_evaluation",
                "params": {
                    "job_uuid": job_uuid,
                    "eval_config": {
                        "metrics": ["loss", "accuracy"],
                    },
                },
                "depends_on": ["train"],
            },
        ]

        executor = PipelineExecutor(pipeline_uuid)
        result = executor.execute(stages_config)

        print_success(f"Pipeline 提交成功")
        print_info(f"  Root task ID: {result.get('root_task_id')}")
        print_info(f"  执行层级: {result.get('layers')}")

        # ============== 4. 监控状态 ==============
        print_info("\n步骤 4: 监控执行状态（20秒）")

        for i in range(4):
            time.sleep(5)
            with Session(engine) as session:
                repo = PipelineRepository(session)
                pipeline = repo.get_by_uuid(pipeline_uuid)
                stages = repo.get_stages(pipeline_uuid)

                print_info(f"\n[{i*5+5}秒] Pipeline 状态: {pipeline.status.value}")

                for stage in stages:
                    status_color = Colors.GREEN if stage.status == PipelineStageStatus.COMPLETED else Colors.YELLOW
                    print_info(f"  {status_color}Stage '{stage.stage_name}': {stage.status.value}{Colors.END}")
                    if stage.celery_task_id:
                        print_info(f"    Task ID: {stage.celery_task_id}")
                    if stage.error_message:
                        print_error(f"    错误: {stage.error_message}")

        # ============== 5. 最终状态 ==============
        print_section("执行结果")

        with Session(engine) as session:
            repo = PipelineRepository(session)
            pipeline = repo.get_by_uuid(pipeline_uuid)
            stages = repo.get_stages(pipeline_uuid)

            final_status = pipeline.status
            if final_status == PipelineStatus.COMPLETED:
                print_success(f"Pipeline 执行成功！")
            elif final_status == PipelineStatus.RUNNING:
                print_info(f"Pipeline 仍在执行中（训练任务较长）")
            else:
                print_error(f"Pipeline 状态: {final_status.value}")

            print_info("\nStage 详情:")
            for stage in stages:
                print_info(f"  - {stage.stage_name}: {stage.status.value}")
                if stage.error_message:
                    print_error(f"    错误: {stage.error_message}")

        return True

    except Exception as e:
        print_error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ssh_training_pipeline():
    """测试 SSH 远程训练 Pipeline"""
    print_section("SSH 远程训练 Pipeline 测试")

    # SSH 配置（需要用户提供）
    print_info("SSH 模式需要提供以下配置:")
    print_info("  - ssh_host: 远程服务器地址")
    print_info("  - ssh_port: SSH 端口 (默认 22)")
    print_info("  - ssh_username: SSH 用户名")
    print_info("  - ssh_password 或 ssh_key_path: 认证方式")
    print_info("  - ssh_working_dir: 远程工作目录")

    print_info("\n请在代码中配置 SSH 参数后运行此测试")

    # 示例配置（需要替换为实际值）
    ssh_config = {
        "ssh_host": "your-gpu-server.com",
        "ssh_port": 22,
        "ssh_username": "user",
        "ssh_password": "password",  # 或使用 ssh_key_path
        "ssh_working_dir": "~/verl_jobs",
    }

    print_info(f"\n示例配置: {ssh_config}")

    return False  # 需要实际配置后才能运行


def main():
    """运行测试"""
    print(f"\n{Colors.BLUE}")
    print("="*60)
    print("  真实训练任务测试")
    print("="*60)
    print(f"{Colors.END}\n")

    # 初始化数据库
    init_db()

    results = {}

    # 测试 1: 本地 SFT 训练
    print_info("注意: 本地模式需要 NVIDIA GPU")
    print_info("      macOS 用户请使用 SSH 远程模式\n")

    results["sft_local"] = test_sft_training_pipeline()

    # 测试 2: SSH 远程训练
    # results["sft_ssh"] = test_ssh_training_pipeline()

    # 汇总结果
    print_section("测试结果汇总")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status:10}{Colors.END} {test_name}")

    print(f"\n{Colors.BLUE}总计: {passed}/{total} 测试通过{Colors.END}\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
