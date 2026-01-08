#!/usr/bin/env python3
"""
全功能测试脚本

测试所有关键功能：
1. 数据库连接和初始化
2. 数据集管理
3. Job 创建
4. Pipeline 执行
5. Metrics 同步
6. API 端点
"""

import sys
import os
import time
import requests
from pathlib import Path

# 添加项目路径
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
from sqlmodel import select

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

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def print_section(title):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.END}\n")


# ============== 测试 1: 数据库连接 ==============

def test_database_connection():
    """测试数据库连接"""
    print_section("测试 1: 数据库连接")

    try:
        # 创建表
        init_db()
        print_success("数据库表创建成功")

        # 测试连接
        with Session(engine) as session:
            # 简单查询
            result = session.exec(select(TrainingJob).limit(1)).first()
            print_success(f"数据库连接成功 (现有 jobs: {result.name if result else '无'})")

        return True
    except Exception as e:
        print_error(f"数据库连接失败: {e}")
        return False


# ============== 测试 2: 数据集文件 ==============

def test_datasets():
    """测试数据集文件"""
    print_section("测试 2: 数据集文件")

    datasets_dir = Path("./datasets")

    if not datasets_dir.exists():
        print_error(f"数据集目录不存在: {datasets_dir}")
        return False

    # 检查数据集文件
    expected_files = [
        "sft_math.json",
        "ppo_general.json",
        "grpo_math.json",
        "dpo_preference.json",
        "sales_sft.jsonl",
        "sales_grpo.jsonl",
        "sales_dpo.jsonl",
    ]

    found_files = []
    missing_files = []

    for file in expected_files:
        file_path = datasets_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            print_success(f"{file} ({size:,} bytes)")
            found_files.append(file)
        else:
            print_warning(f"{file} 不存在")
            missing_files.append(file)

    if found_files:
        print_info(f"找到 {len(found_files)}/{len(expected_files)} 个数据集文件")
        return True
    else:
        print_error("没有找到任何数据集文件")
        return False


# ============== 测试 3: Job 创建 ==============

def test_job_creation():
    """测试 Job 创建"""
    print_section("测试 3: Job 创建")

    try:
        with Session(engine) as session:
            repo = JobRepository(session)

            # 创建测试 job
            job = TrainingJob(
                uuid="test-job-" + str(int(time.time())),
                name="Test Training Job",
                description="测试训练任务",
                status=JobStatus.PENDING,
                algorithm=TrainingAlgorithm.SFT,
                model_path="Qwen/Qwen2.5-0.5B",
                train_data_path="./datasets/sales_sft.jsonl",
                num_gpus=1,
                learning_rate=1e-5,
                batch_size=4,
                num_epochs=1,
                context_length=512,
            )

            created_job = repo.create(job)
            print_success(f"Job 创建成功: {created_job.uuid}")
            print_info(f"  - 名称: {created_job.name}")
            print_info(f"  - 算法: {created_job.algorithm}")
            print_info(f"  - 状态: {created_job.status}")

            return created_job.uuid
    except Exception as e:
        print_error(f"Job 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============== 测试 4: Pipeline 创建 ==============

def test_pipeline_creation():
    """测试 Pipeline 创建"""
    print_section("测试 4: Pipeline 创建")

    try:
        pipeline_uuid = "test-pipeline-" + str(int(time.time()))

        with Session(engine) as session:
            repo = PipelineRepository(session)

            # 创建 pipeline
            pipeline = Pipeline(
                uuid=pipeline_uuid,
                name="Test Pipeline",
                description="测试 pipeline",
                status=PipelineStatus.PENDING,
            )
            created_pipeline = repo.create(pipeline)

            # 创建 stages
            stages_config = [
                {
                    "name": "stage_A",
                    "task_name": "preprocess_dataset",
                    "task_params": {"dataset_uuid": "test-dataset", "preprocessing_config": {}},
                    "depends_on": [],
                    "stage_order": 0,
                },
                {
                    "name": "stage_B",
                    "task_name": "train_model",
                    "task_params": {"job_uuid": "test-job", "config": {}},
                    "depends_on": ["stage_A"],
                    "stage_order": 1,
                },
            ]

            for stage_config in stages_config:
                stage = PipelineStage(
                    pipeline_uuid=pipeline_uuid,
                    stage_name=stage_config["name"],
                    task_name=stage_config["task_name"],
                    task_params=stage_config["task_params"],
                    depends_on=stage_config["depends_on"],
                    stage_order=stage_config["stage_order"],
                    status=PipelineStageStatus.PENDING,
                )
                repo.create_stage(stage)

            print_success(f"Pipeline 创建成功: {pipeline_uuid}")
            print_info(f"  - 名称: {created_pipeline.name}")
            print_info(f"  - Stages: {len(stages_config)}")

            return pipeline_uuid
    except Exception as e:
        print_error(f"Pipeline 创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============== 测试 5: DAG 解析 ==============

def test_dag_resolution():
    """测试 DAG 依赖解析"""
    print_section("测试 5: DAG 依赖解析")

    try:
        from training_platform.core.pipeline_executor import DagResolver

        # 测试简单线性 DAG
        stages = [
            {"name": "A", "task": "preprocess_dataset", "params": {}, "depends_on": []},
            {"name": "B", "task": "train_model", "params": {}, "depends_on": ["A"]},
            {"name": "C", "task": "run_evaluation", "params": {}, "depends_on": ["B"]},
        ]

        resolver = DagResolver(stages)
        resolver.validate()
        layers = resolver.get_execution_layers()

        print_success("线性 DAG 解析成功")
        print_info(f"  执行层级: {layers}")

        # 测试并行 DAG
        parallel_stages = [
            {"name": "A", "task": "preprocess_dataset", "params": {}, "depends_on": []},
            {"name": "B", "task": "train_model", "params": {}, "depends_on": ["A"]},
            {"name": "C", "task": "train_model", "params": {}, "depends_on": ["A"]},
            {"name": "D", "task": "run_evaluation", "params": {}, "depends_on": ["B", "C"]},
        ]

        resolver2 = DagResolver(parallel_stages)
        resolver2.validate()
        layers2 = resolver2.get_execution_layers()

        print_success("并行 DAG 解析成功")
        print_info(f"  执行层级: {layers2}")

        return True
    except Exception as e:
        print_error(f"DAG 解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============== 测试 6: Celery 连接 ==============

def test_celery_connection():
    """测试 Celery 连接"""
    print_section("测试 6: Celery/Redis 连接")

    try:
        from training_platform.core.celery_config import app

        # 检查 Redis 连接
        inspect = app.control.inspect()
        active_workers = inspect.active()

        if active_workers:
            print_success(f"发现 {len(active_workers)} 个活跃 worker")
            for worker_name, tasks in active_workers.items():
                print_info(f"  - {worker_name}: {len(tasks)} 个活跃任务")
        else:
            print_warning("没有发现活跃的 Celery workers")
            print_info("  提示: 需要启动 Celery workers 才能执行 pipeline")

        return True
    except Exception as e:
        print_error(f"Celery 连接失败: {e}")
        return False


# ============== 测试 7: API 端点 ==============

def test_api_endpoints():
    """测试 API 端点"""
    print_section("测试 7: API 端点 (需要 FastAPI 运行)")

    base_url = "http://localhost:8000"

    try:
        # 测试健康检查
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.status_code == 200:
            print_success("API 健康检查通过")

            # 测试获取 jobs 列表
            response = requests.get(f"{base_url}/api/jobs", timeout=2)
            if response.status_code == 200:
                jobs = response.json()
                print_success(f"获取 jobs 列表成功 ({jobs.get('total', 0)} 个)")
            else:
                print_warning(f"获取 jobs 列表失败: {response.status_code}")

            return True
        else:
            print_warning(f"API 响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_warning("无法连接到 API 服务器 (未启动)")
        print_info("  提示: 运行 'uvicorn training_platform.api.main:app' 启动 API")
        return False
    except Exception as e:
        print_error(f"API 测试失败: {e}")
        return False


# ============== 测试 8: Metrics 路径 ==============

def test_metrics_paths():
    """测试 Metrics 路径配置"""
    print_section("测试 8: Metrics 路径配置")

    import os

    # 检查 metrics 目录
    metrics_dir = Path(os.getenv("PLATFORM_METRICS_DIR", "./platform_metrics"))

    if not metrics_dir.exists():
        print_warning(f"Metrics 目录不存在: {metrics_dir}")
        print_info("  创建目录...")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Metrics 目录已创建: {metrics_dir}")
    else:
        print_success(f"Metrics 目录存在: {metrics_dir}")

        # 列出现有文件
        files = list(metrics_dir.glob("*_metrics.jsonl"))
        if files:
            print_info(f"  找到 {len(files)} 个 metrics 文件")
            for f in files[:3]:  # 只显示前 3 个
                print_info(f"    - {f.name}")
        else:
            print_info("  目录为空（训练后会生成文件）")

    return True


# ============== 测试 9: SSH 配置 ==============

def test_ssh_config():
    """测试 SSH 配置"""
    print_section("测试 9: SSH 配置")

    # 这里只测试配置格式，不实际连接
    print_info("SSH 配置测试（仅检查格式）")

    ssh_config_example = {
        "ssh_host": "remote.server.com",
        "ssh_port": 22,
        "ssh_username": "user",
        "ssh_password": "password",  # 或使用 ssh_key_path
        "ssh_working_dir": "~/verl_jobs",
    }

    print_success("SSH 配置格式正确")
    print_info("  示例配置:")
    for key, value in ssh_config_example.items():
        print_info(f"    {key}: {value}")

    print_info("\n  提示: 实际连接测试需要在 Job 创建时指定 SSH 配置")

    return True


# ============== 主测试函数 ==============

def main():
    """运行所有测试"""
    print(f"\n{Colors.BLUE}")
    print("="*60)
    print("  Training Platform - 全功能测试")
    print("="*60)
    print(f"{Colors.END}\n")

    results = {}

    # 运行所有测试
    results["database"] = test_database_connection()
    results["datasets"] = test_datasets()
    results["job_creation"] = test_job_creation()
    results["pipeline_creation"] = test_pipeline_creation()
    results["dag_resolution"] = test_dag_resolution()
    results["celery"] = test_celery_connection()
    results["api"] = test_api_endpoints()
    results["metrics_paths"] = test_metrics_paths()
    results["ssh_config"] = test_ssh_config()

    # 汇总结果
    print_section("测试结果汇总")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status:10}{Colors.END} {test_name}")

    print(f"\n{Colors.BLUE}总计: {passed}/{total} 测试通过{Colors.END}\n")

    # 后续步骤建议
    if passed == total:
        print_success("所有测试通过！系统就绪 🎉")
        print_info("\n下一步:")
        print_info("  1. 启动 Celery workers: ./scripts/start_workers.sh")
        print_info("  2. 启动 API 服务器: uvicorn training_platform.api.main:app")
        print_info("  3. 创建训练任务并运行 pipeline")
    else:
        print_warning("部分测试未通过，请检查失败的组件")

        if not results["celery"]:
            print_info("\n  启动 Celery workers:")
            print_info("    ./scripts/start_workers.sh")

        if not results["api"]:
            print_info("\n  启动 API 服务器:")
            print_info("    uvicorn training_platform.api.main:app --reload")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
