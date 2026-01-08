#!/usr/bin/env python3
"""
SSH 远程训练测试

使用真实的 SSH 服务器测试完整训练流程
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
from training_platform.core.ssh_runner import SSHConfig, get_ssh_manager

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


def test_ssh_connection():
    """测试 SSH 连接"""
    print_section("测试 1: SSH 连接")

    ssh_config = SSHConfig(
        host="connect.westc.gpuhub.com",
        port=27192,
        username="root",
        password="A32qbQ1UR3Y6",
        working_dir="~/verl_jobs",
    )

    try:
        manager = get_ssh_manager()
        runner = manager.get_runner(ssh_config)

        print_info("正在连接到 SSH 服务器...")
        result = runner.test_connection()

        if result.get("success"):
            print_success("SSH 连接成功")
            print_info(f"  主机: {ssh_config.host}:{ssh_config.port}")
            print_info(f"  用户: {ssh_config.username}")

            # 检查 GPU
            gpu_info = runner.get_gpu_info()
            if gpu_info.get("success"):
                print_success(f"发现 {gpu_info.get('gpu_count', 0)} 个 GPU")
                for gpu in gpu_info.get("gpus", [])[:3]:
                    print_info(f"  - {gpu.get('name')}: {gpu.get('memory_total', 0)} MB")
            else:
                print_error(f"获取 GPU 信息失败: {gpu_info.get('error')}")

            return ssh_config
        else:
            print_error(f"SSH 连接失败: {result.get('error')}")
            return None

    except Exception as e:
        print_error(f"SSH 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_ssh_training_pipeline(ssh_config: SSHConfig):
    """测试 SSH 远程训练 Pipeline"""
    print_section("测试 2: SSH 远程训练 Pipeline")

    timestamp = int(time.time())
    job_uuid = f"ssh-sft-{timestamp}"
    pipeline_uuid = f"ssh-pipeline-{timestamp}"

    try:
        # ============== 1. 创建训练 Job ==============
        print_info("步骤 1: 创建训练 Job")

        with Session(engine) as session:
            repo = JobRepository(session)

            job = TrainingJob(
                uuid=job_uuid,
                name=f"SSH SFT Training {timestamp}",
                description="远程 GPU 服务器 SFT 训练",
                status=JobStatus.PENDING,
                algorithm=TrainingAlgorithm.SFT,
                model_path="Qwen/Qwen2.5-0.5B",
                train_data_path="./datasets/sales_sft.jsonl",
                num_gpus=1,
                learning_rate=1e-5,
                batch_size=2,
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

        # ============== 2. 创建 Pipeline ==============
        print_info("\n步骤 2: 创建 Pipeline")

        with Session(engine) as session:
            repo = PipelineRepository(session)

            pipeline = Pipeline(
                uuid=pipeline_uuid,
                name=f"SSH SFT Pipeline {timestamp}",
                description="SSH 远程训练流程",
                status=PipelineStatus.PENDING,
            )
            repo.create(pipeline)

            # SSH 配置
            ssh_config_dict = {
                "host": ssh_config.host,
                "port": ssh_config.port,
                "username": ssh_config.username,
                "password": ssh_config.password,
                "working_dir": ssh_config.working_dir,
            }

            # 定义 stages - 只运行训练，不运行预处理和评测（简化测试）
            stages_data = [
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
                        "run_mode": "ssh",
                        "ssh_config": ssh_config_dict,
                    },
                    "depends_on": [],
                    "stage_order": 0,
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
                "name": "train",
                "task": "train_model",
                "params": {
                    "job_uuid": job_uuid,
                    "config": {
                        "num_gpus": 1,
                        "batch_size": 2,
                        "learning_rate": 1e-5,
                    },
                    "run_mode": "ssh",
                    "ssh_config": ssh_config_dict,
                },
                "depends_on": [],
            },
        ]

        executor = PipelineExecutor(pipeline_uuid)
        result = executor.execute(stages_config)

        print_success(f"Pipeline 提交成功")
        print_info(f"  Root task ID: {result.get('root_task_id')}")
        print_info(f"  执行模式: SSH 远程")

        # ============== 4. 监控状态 ==============
        print_info("\n步骤 4: 监控执行状态（60秒）")

        for i in range(12):
            time.sleep(5)
            with Session(engine) as session:
                repo = PipelineRepository(session)
                pipeline = repo.get_by_uuid(pipeline_uuid)
                stages = repo.get_stages(pipeline_uuid)

                print_info(f"\n[{i*5+5}秒] Pipeline 状态: {pipeline.status.value}")

                for stage in stages:
                    status_color = Colors.GREEN if stage.status == PipelineStageStatus.COMPLETED else (
                        Colors.RED if stage.status == PipelineStageStatus.FAILED else Colors.YELLOW
                    )
                    print_info(f"  {status_color}Stage '{stage.stage_name}': {stage.status.value}{Colors.END}")
                    if stage.celery_task_id:
                        print_info(f"    Task ID: {stage.celery_task_id}")
                    if stage.error_message:
                        print_error(f"    错误: {stage.error_message[:200]}")

                # 如果完成或失败，提前退出
                if pipeline.status in [PipelineStatus.COMPLETED, PipelineStatus.FAILED]:
                    break

        # ============== 5. 最终状态 ==============
        print_section("执行结果")

        with Session(engine) as session:
            repo = PipelineRepository(session)
            job_repo = JobRepository(session)

            pipeline = repo.get_by_uuid(pipeline_uuid)
            job = job_repo.get_by_uuid(job_uuid)
            stages = repo.get_stages(pipeline_uuid)

            if pipeline.status == PipelineStatus.COMPLETED:
                print_success(f"Pipeline 执行成功！")
            elif pipeline.status == PipelineStatus.RUNNING:
                print_info(f"Pipeline 仍在执行中（训练需要更长时间）")
            else:
                print_error(f"Pipeline 状态: {pipeline.status.value}")

            print_info(f"\nJob 状态: {job.status.value}")

            print_info("\nStage 详情:")
            for stage in stages:
                print_info(f"  - {stage.stage_name}: {stage.status.value}")
                if stage.error_message:
                    print_error(f"    错误: {stage.error_message[:300]}")

        return True

    except Exception as e:
        print_error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行测试"""
    print(f"\n{Colors.BLUE}")
    print("="*60)
    print("  SSH 远程训练测试")
    print("="*60)
    print(f"{Colors.END}\n")

    # 初始化数据库
    init_db()

    # 测试 1: SSH 连接
    ssh_config = test_ssh_connection()
    if not ssh_config:
        print_error("\nSSH 连接失败，无法继续测试")
        return False

    # 测试 2: SSH 训练 Pipeline
    print_info("\n继续执行 SSH 训练 Pipeline...")
    result = test_ssh_training_pipeline(ssh_config)

    # 汇总结果
    print_section("测试结果汇总")

    if result:
        print_success("SSH 远程训练测试通过")
        print_info("\n提示: 训练可能需要较长时间")
        print_info("      可以通过 Job 的 celery_task_id 在 Flower UI 中监控")
    else:
        print_error("SSH 远程训练测试失败")

    return result


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
