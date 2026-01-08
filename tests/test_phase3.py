"""
Phase 3 测试用例

测试 Celery 集成和 Pipeline 编排功能。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, create_engine
from sqlmodel.pool import StaticPool

from training_platform.api.main import app
from training_platform.core.database import (
    get_session,
    SQLModel,
    Pipeline,
    PipelineStage,
    PipelineStatus,
    PipelineStageStatus,
)


# 测试数据库
@pytest.fixture(name="session")
def session_fixture():
    """创建测试数据库会话"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(name="client")
def client_fixture(session: Session):
    """创建测试客户端"""
    def get_session_override():
        return session

    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# ============== Pipeline API 测试 ==============

def test_create_pipeline(client: TestClient):
    """测试创建 Pipeline"""
    request = {
        "name": "Test Training Pipeline",
        "description": "Complete training pipeline with all stages",
        "stages": [
            {
                "name": "preprocess",
                "task": "preprocess_dataset",
                "params": {"dataset_uuid": "test-dataset"},
                "depends_on": [],
                "max_retries": 3,
                "retry_delay": 60,
            },
            {
                "name": "train",
                "task": "train_model",
                "params": {"job_uuid": "test-job"},
                "depends_on": ["preprocess"],
                "max_retries": 2,
                "retry_delay": 120,
            },
            {
                "name": "evaluate",
                "task": "run_evaluation",
                "params": {"job_uuid": "test-job"},
                "depends_on": ["train"],
                "max_retries": 3,
                "retry_delay": 60,
            },
        ],
        "priority": 8,
        "max_retries": 3,
    }

    response = client.post("/api/v1/pipelines", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["name"] == "Test Training Pipeline"
    assert data["status"] == "pending"
    assert data["priority"] == 8
    assert data["max_retries"] == 3
    assert data["retry_count"] == 0
    assert "uuid" in data

    print(f"✓ 创建 Pipeline: {data['uuid']}")
    return data["uuid"]


def test_list_pipelines(client: TestClient):
    """测试列出 Pipelines"""
    # 先创建几个 Pipelines
    for i in range(3):
        request = {
            "name": f"Test Pipeline {i}",
            "description": f"Pipeline {i}",
            "stages": [
                {
                    "name": "stage1",
                    "task": "train_model",
                    "params": {},
                    "depends_on": [],
                    "max_retries": 3,
                    "retry_delay": 60,
                }
            ],
            "priority": 5,
            "max_retries": 3,
        }
        client.post("/api/v1/pipelines", json=request)

    # 列出所有 Pipelines
    response = client.get("/api/v1/pipelines")
    assert response.status_code == 200

    data = response.json()
    assert "pipelines" in data
    assert "total" in data
    assert data["total"] >= 3
    assert len(data["pipelines"]) >= 3

    print(f"✓ 列出 {data['total']} 个 Pipelines")


def test_get_pipeline(client: TestClient):
    """测试获取 Pipeline 详情"""
    # 先创建一个 Pipeline
    uuid = test_create_pipeline(client)

    # 获取详情
    response = client.get(f"/api/v1/pipelines/{uuid}")
    assert response.status_code == 200

    data = response.json()
    assert data["uuid"] == uuid
    assert data["name"] == "Test Training Pipeline"
    assert data["status"] == "pending"

    print(f"✓ 获取 Pipeline 详情: {uuid}")


def test_get_pipeline_status(client: TestClient):
    """测试获取 Pipeline 状态（包括所有阶段）"""
    # 先创建一个 Pipeline
    uuid = test_create_pipeline(client)

    # 获取状态
    response = client.get(f"/api/v1/pipelines/{uuid}/status")
    assert response.status_code == 200

    data = response.json()
    assert "pipeline" in data
    assert "stages" in data

    # 验证 Pipeline 信息
    assert data["pipeline"]["uuid"] == uuid
    assert data["pipeline"]["status"] == "pending"

    # 验证 Stages 信息
    stages = data["stages"]
    assert len(stages) == 3
    assert stages[0]["stage_name"] == "preprocess"
    assert stages[0]["status"] == "pending"
    assert stages[1]["stage_name"] == "train"
    assert stages[1]["depends_on"] == ["preprocess"]
    assert stages[2]["stage_name"] == "evaluate"

    print(f"✓ 获取 Pipeline 状态，包含 {len(stages)} 个阶段")


def test_filter_pipelines_by_status(client: TestClient):
    """测试按状态筛选 Pipelines"""
    # 创建 PENDING Pipeline
    uuid = test_create_pipeline(client)

    # 筛选 PENDING 状态
    response = client.get("/api/v1/pipelines?status=pending")
    assert response.status_code == 200

    data = response.json()
    assert data["total"] >= 1
    assert all(p["status"] == "pending" for p in data["pipelines"])

    print(f"✓ 筛选到 {data['total']} 个 PENDING Pipelines")


def test_cancel_pipeline(client: TestClient):
    """测试取消 Pipeline"""
    # 先创建一个 Pipeline
    uuid = test_create_pipeline(client)

    # 取消 Pipeline
    response = client.post(f"/api/v1/pipelines/{uuid}/cancel")
    assert response.status_code == 200

    data = response.json()
    assert data["pipeline_uuid"] == uuid
    assert data["status"] == "cancelled"

    # 验证状态已更新
    response = client.get(f"/api/v1/pipelines/{uuid}")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

    print(f"✓ 取消 Pipeline: {uuid}")


def test_delete_pipeline(client: TestClient):
    """测试删除 Pipeline"""
    # 先创建并取消一个 Pipeline
    uuid = test_create_pipeline(client)
    client.post(f"/api/v1/pipelines/{uuid}/cancel")

    # 删除 Pipeline
    response = client.delete(f"/api/v1/pipelines/{uuid}")
    assert response.status_code == 200

    data = response.json()
    assert data["pipeline_uuid"] == uuid
    assert data["status"] == "deleted"

    # 验证已删除
    response = client.get(f"/api/v1/pipelines/{uuid}")
    assert response.status_code == 404

    print(f"✓ 删除 Pipeline: {uuid}")


def test_pipeline_validation(client: TestClient):
    """测试 Pipeline 验证"""
    # 无效的优先级
    request = {
        "name": "Invalid Pipeline",
        "description": "Test validation",
        "stages": [
            {
                "name": "stage1",
                "task": "train_model",
                "params": {},
                "depends_on": [],
                "max_retries": 3,
                "retry_delay": 60,
            }
        ],
        "priority": 15,  # 超出范围 (1-10)
        "max_retries": 3,
    }

    response = client.post("/api/v1/pipelines", json=request)
    assert response.status_code == 422  # 验证错误

    print("✓ Pipeline 验证测试通过")


# ============== Celery Tasks API 测试 ==============

def test_celery_tasks_stats(client: TestClient):
    """测试获取 Celery 任务统计"""
    response = client.get("/api/v1/celery-tasks/stats/overview")
    assert response.status_code == 200

    data = response.json()
    assert "workers" in data
    assert "worker_count" in data
    assert "active_tasks" in data
    assert "scheduled_tasks" in data
    assert "reserved_tasks" in data

    print(f"✓ Celery 统计: {data['worker_count']} workers, {data['active_tasks']} active tasks")


# ============== 数据模型测试 ==============

def test_pipeline_model(session: Session):
    """测试 Pipeline 模型"""
    # 创建 Pipeline
    pipeline = Pipeline(
        uuid="test-pipeline-1",
        name="Test Pipeline",
        description="Test description",
        stages=[{"name": "stage1", "task": "train_model"}],
        dependencies={"stage1": []},
        status=PipelineStatus.PENDING,
        stage_tasks={},
        results={},
        priority=5,
        max_retries=3,
        retry_count=0,
    )

    session.add(pipeline)
    session.commit()
    session.refresh(pipeline)

    # 验证
    assert pipeline.uuid == "test-pipeline-1"
    assert pipeline.name == "Test Pipeline"
    assert pipeline.status == PipelineStatus.PENDING
    assert pipeline.priority == 5
    assert pipeline.created_at is not None

    print("✓ Pipeline 模型测试通过")


def test_pipeline_stage_model(session: Session):
    """测试 PipelineStage 模型"""
    # 先创建 Pipeline
    pipeline = Pipeline(
        uuid="test-pipeline-2",
        name="Test Pipeline",
        stages=[],
        dependencies={},
        status=PipelineStatus.PENDING,
        stage_tasks={},
        results={},
        priority=5,
        max_retries=3,
        retry_count=0,
    )
    session.add(pipeline)
    session.commit()

    # 创建 PipelineStage
    stage = PipelineStage(
        pipeline_uuid="test-pipeline-2",
        stage_name="train",
        stage_order=0,
        task_name="train_model",
        task_params={"job_uuid": "test-job"},
        depends_on=[],
        status=PipelineStageStatus.PENDING,
        result={},
        max_retries=3,
        retry_count=0,
        retry_delay=60,
    )

    session.add(stage)
    session.commit()
    session.refresh(stage)

    # 验证
    assert stage.pipeline_uuid == "test-pipeline-2"
    assert stage.stage_name == "train"
    assert stage.status == PipelineStageStatus.PENDING
    assert stage.max_retries == 3
    assert stage.created_at is not None

    print("✓ PipelineStage 模型测试通过")


# ============== 集成测试 ==============

def test_pipeline_workflow(client: TestClient):
    """测试完整的 Pipeline 工作流"""
    print("\n=== Pipeline 工作流测试 ===")

    # 1. 创建 Pipeline
    request = {
        "name": "Complete Workflow Test",
        "description": "Test complete workflow",
        "stages": [
            {
                "name": "preprocess",
                "task": "preprocess_dataset",
                "params": {"dataset_uuid": "test-dataset"},
                "depends_on": [],
                "max_retries": 3,
                "retry_delay": 60,
            },
            {
                "name": "train",
                "task": "train_model",
                "params": {"job_uuid": "test-job"},
                "depends_on": ["preprocess"],
                "max_retries": 2,
                "retry_delay": 120,
            },
        ],
        "priority": 7,
        "max_retries": 3,
    }

    response = client.post("/api/v1/pipelines", json=request)
    assert response.status_code == 200
    uuid = response.json()["uuid"]
    print(f"  1. 创建 Pipeline: {uuid}")

    # 2. 查看状态
    response = client.get(f"/api/v1/pipelines/{uuid}/status")
    assert response.status_code == 200
    assert response.json()["pipeline"]["status"] == "pending"
    print("  2. 状态: PENDING")

    # 3. 取消 Pipeline
    response = client.post(f"/api/v1/pipelines/{uuid}/cancel")
    assert response.status_code == 200
    print("  3. 取消 Pipeline")

    # 4. 验证已取消
    response = client.get(f"/api/v1/pipelines/{uuid}")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    print("  4. 状态: CANCELLED")

    # 5. 删除 Pipeline
    response = client.delete(f"/api/v1/pipelines/{uuid}")
    assert response.status_code == 200
    print("  5. 删除 Pipeline")

    # 6. 验证已删除
    response = client.get(f"/api/v1/pipelines/{uuid}")
    assert response.status_code == 404
    print("  6. Pipeline 已删除")

    print("✓ Pipeline 工作流测试通过")


# ============== 主测试函数 ==============

def main():
    """运行所有 Phase 3 测试"""
    print("\n" + "=" * 80)
    print("Phase 3 功能测试")
    print("=" * 80 + "\n")

    from sqlmodel import create_engine
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        def get_session_override():
            return session

        app.dependency_overrides[get_session] = get_session_override
        client = TestClient(app)

        # 运行测试
        tests = [
            ("Pipeline 创建", lambda: test_create_pipeline(client)),
            ("Pipeline 列表", lambda: test_list_pipelines(client)),
            ("Pipeline 详情", lambda: test_get_pipeline(client)),
            ("Pipeline 状态", lambda: test_get_pipeline_status(client)),
            ("状态筛选", lambda: test_filter_pipelines_by_status(client)),
            ("取消 Pipeline", lambda: test_cancel_pipeline(client)),
            ("删除 Pipeline", lambda: test_delete_pipeline(client)),
            ("Pipeline 验证", lambda: test_pipeline_validation(client)),
            ("Celery 统计", lambda: test_celery_tasks_stats(client)),
            ("Pipeline 模型", lambda: test_pipeline_model(session)),
            ("PipelineStage 模型", lambda: test_pipeline_stage_model(session)),
            ("完整工作流", lambda: test_pipeline_workflow(client)),
        ]

        passed = 0
        failed = 0

        for name, test_func in tests:
            try:
                test_func()
                passed += 1
            except Exception as e:
                print(f"✗ {name} 失败: {e}")
                failed += 1

        # 清理
        app.dependency_overrides.clear()

        # 总结
        print("\n" + "=" * 80)
        print(f"测试完成: {passed} 通过, {failed} 失败")
        print("=" * 80 + "\n")

        return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
