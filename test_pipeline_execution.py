#!/usr/bin/env python3
"""
Pipeline 执行测试

测试实际的 Pipeline 执行流程
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from training_platform.core.database import (
    engine,
    Session,
    PipelineRepository,
    Pipeline,
    PipelineStage,
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

def test_simple_pipeline():
    """测试简单的单层 Pipeline"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("  测试 1: 简单单层 Pipeline")
    print(f"{'='*60}{Colors.END}\n")

    pipeline_uuid = f"test-simple-{int(time.time())}"

    try:
        # 创建 pipeline
        with Session(engine) as session:
            repo = PipelineRepository(session)

            pipeline = Pipeline(
                uuid=pipeline_uuid,
                name="Simple Test Pipeline",
                description="单层测试 pipeline",
                status=PipelineStatus.PENDING,
            )
            repo.create(pipeline)

            # 创建 stages
            stage = PipelineStage(
                pipeline_uuid=pipeline_uuid,
                stage_name="preprocess",
                task_name="preprocess_dataset",
                task_params={
                    "dataset_uuid": "test-dataset-001",
                    "preprocessing_config": {"action": "test"},
                },
                depends_on=[],
                stage_order=0,
                status=PipelineStageStatus.PENDING,
            )
            repo.create_stage(stage)

        print_success(f"Pipeline 创建成功: {pipeline_uuid}")

        # 定义 stages 配置（用于执行）
        stages_config = [
            {
                "name": "preprocess",
                "task": "preprocess_dataset",
                "params": {
                    "dataset_uuid": "test-dataset-001",
                    "preprocessing_config": {"action": "test"},
                },
                "depends_on": [],
            }
        ]

        # 执行 pipeline
        print_info("开始执行 pipeline...")
        executor = PipelineExecutor(pipeline_uuid)
        result = executor.execute(stages_config)

        print_success(f"Pipeline 提交成功")
        print_info(f"  Root task ID: {result.get('root_task_id')}")
        print_info(f"  执行层级: {result.get('layers')}")

        # 等待一会儿让任务执行
        print_info("\n等待 5 秒让任务执行...")
        time.sleep(5)

        # 检查状态
        with Session(engine) as session:
            repo = PipelineRepository(session)
            pipeline = repo.get_by_uuid(pipeline_uuid)
            stages = repo.get_stages(pipeline_uuid)

            print_info(f"\nPipeline 状态: {pipeline.status.value}")

            for stage in stages:
                print_info(f"  Stage '{stage.stage_name}':")
                print_info(f"    - 状态: {stage.status.value}")
                print_info(f"    - Task ID: {stage.celery_task_id}")
                if stage.error_message:
                    print_info(f"    - 错误: {stage.error_message}")

        return True

    except Exception as e:
        print_error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_layer_pipeline():
    """测试多层 Pipeline（修复问题 1 后）"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("  测试 2: 多层 Pipeline (3层)")
    print(f"{'='*60}{Colors.END}\n")

    pipeline_uuid = f"test-multi-{int(time.time())}"

    try:
        # 创建 pipeline
        with Session(engine) as session:
            repo = PipelineRepository(session)

            pipeline = Pipeline(
                uuid=pipeline_uuid,
                name="Multi-layer Test Pipeline",
                description="3层测试 pipeline",
                status=PipelineStatus.PENDING,
            )
            repo.create(pipeline)

            # 创建 stages
            stages_data = [
                {
                    "stage_name": "layer1",
                    "task_name": "preprocess_dataset",
                    "task_params": {"dataset_uuid": "test-001", "preprocessing_config": {}},
                    "depends_on": [],
                    "stage_order": 0,
                },
                {
                    "stage_name": "layer2",
                    "task_name": "preprocess_dataset",
                    "task_params": {"dataset_uuid": "test-002", "preprocessing_config": {}},
                    "depends_on": ["layer1"],
                    "stage_order": 1,
                },
                {
                    "stage_name": "layer3",
                    "task_name": "preprocess_dataset",
                    "task_params": {"dataset_uuid": "test-003", "preprocessing_config": {}},
                    "depends_on": ["layer2"],
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

        # 定义 stages 配置（用于执行）
        stages_config = [
            {"name": "layer1", "task": "preprocess_dataset", "params": {"dataset_uuid": "test-001", "preprocessing_config": {}}, "depends_on": []},
            {"name": "layer2", "task": "preprocess_dataset", "params": {"dataset_uuid": "test-002", "preprocessing_config": {}}, "depends_on": ["layer1"]},
            {"name": "layer3", "task": "preprocess_dataset", "params": {"dataset_uuid": "test-003", "preprocessing_config": {}}, "depends_on": ["layer2"]},
        ]

        # 执行 pipeline
        print_info("开始执行 pipeline...")
        executor = PipelineExecutor(pipeline_uuid)
        result = executor.execute(stages_config)

        print_success(f"Pipeline 提交成功（修复问题 1 后应该不会 TypeError）")
        print_info(f"  Root task ID: {result.get('root_task_id')}")
        print_info(f"  执行层级: {result.get('layers')}")

        # 等待一会儿让任务执行
        print_info("\n等待 10 秒让任务执行...")
        time.sleep(10)

        # 检查状态
        with Session(engine) as session:
            repo = PipelineRepository(session)
            pipeline = repo.get_by_uuid(pipeline_uuid)
            stages = repo.get_stages(pipeline_uuid)

            print_info(f"\nPipeline 状态: {pipeline.status.value}")

            for stage in stages:
                print_info(f"  Stage '{stage.stage_name}':")
                print_info(f"    - 状态: {stage.status.value}")
                print_info(f"    - Task ID: {stage.celery_task_id}")

        return True

    except Exception as e:
        print_error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_pipeline():
    """测试并行 Pipeline"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("  测试 3: 并行 Pipeline (1->2->1)")
    print(f"{'='*60}{Colors.END}\n")

    pipeline_uuid = f"test-parallel-{int(time.time())}"

    try:
        # 创建 pipeline
        with Session(engine) as session:
            repo = PipelineRepository(session)

            pipeline = Pipeline(
                uuid=pipeline_uuid,
                name="Parallel Test Pipeline",
                description="并行测试 pipeline",
                status=PipelineStatus.PENDING,
            )
            repo.create(pipeline)

            # 创建 stages
            stages_data = [
                {
                    "stage_name": "A",
                    "task_name": "preprocess_dataset",
                    "task_params": {"dataset_uuid": "test-A", "preprocessing_config": {}},
                    "depends_on": [],
                    "stage_order": 0,
                },
                {
                    "stage_name": "B",
                    "task_name": "preprocess_dataset",
                    "task_params": {"dataset_uuid": "test-B", "preprocessing_config": {}},
                    "depends_on": ["A"],
                    "stage_order": 1,
                },
                {
                    "stage_name": "C",
                    "task_name": "preprocess_dataset",
                    "task_params": {"dataset_uuid": "test-C", "preprocessing_config": {}},
                    "depends_on": ["A"],
                    "stage_order": 1,
                },
                {
                    "stage_name": "D",
                    "task_name": "preprocess_dataset",
                    "task_params": {"dataset_uuid": "test-D", "preprocessing_config": {}},
                    "depends_on": ["B", "C"],
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

        # 定义 stages 配置（用于执行）
        stages_config = [
            {"name": "A", "task": "preprocess_dataset", "params": {"dataset_uuid": "test-A", "preprocessing_config": {}}, "depends_on": []},
            {"name": "B", "task": "preprocess_dataset", "params": {"dataset_uuid": "test-B", "preprocessing_config": {}}, "depends_on": ["A"]},
            {"name": "C", "task": "preprocess_dataset", "params": {"dataset_uuid": "test-C", "preprocessing_config": {}}, "depends_on": ["A"]},
            {"name": "D", "task": "preprocess_dataset", "params": {"dataset_uuid": "test-D", "preprocessing_config": {}}, "depends_on": ["B", "C"]},
        ]

        # 执行 pipeline
        print_info("开始执行 pipeline...")
        executor = PipelineExecutor(pipeline_uuid)
        result = executor.execute(stages_config)

        print_success(f"Pipeline 提交成功")
        print_info(f"  Root task ID: {result.get('root_task_id')}")
        print_info(f"  执行层级: {result.get('layers')}")

        # 等待一会儿让任务执行
        print_info("\n等待 15 秒让任务执行...")
        time.sleep(15)

        # 检查状态
        with Session(engine) as session:
            repo = PipelineRepository(session)
            pipeline = repo.get_by_uuid(pipeline_uuid)
            stages = repo.get_stages(pipeline_uuid)

            print_info(f"\nPipeline 状态: {pipeline.status.value}")

            for stage in stages:
                print_info(f"  Stage '{stage.stage_name}':")
                print_info(f"    - 状态: {stage.status.value}")
                print_info(f"    - Task ID: {stage.celery_task_id}")

        return True

    except Exception as e:
        print_error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有 Pipeline 测试"""
    print(f"\n{Colors.BLUE}")
    print("="*60)
    print("  Pipeline 执行测试")
    print("="*60)
    print(f"{Colors.END}\n")

    results = {}

    # 运行测试
    results["simple"] = test_simple_pipeline()
    results["multi_layer"] = test_multi_layer_pipeline()
    results["parallel"] = test_parallel_pipeline()

    # 汇总结果
    print(f"\n{Colors.BLUE}{'='*60}")
    print("  测试结果汇总")
    print(f"{'='*60}{Colors.END}\n")

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status:10}{Colors.END} {test_name}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\n{Colors.BLUE}总计: {passed}/{total} 测试通过{Colors.END}\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
