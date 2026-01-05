"""
Tests for FastAPI Endpoints
"""

import pytest
import sys
import os
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from fastapi.testclient import TestClient


# Import app - handle import errors gracefully
try:
    from training_platform.api.main import app
    from training_platform.core.database import init_db, get_session, engine
    from sqlmodel import Session, SQLModel
    APP_AVAILABLE = True
except ImportError as e:
    APP_AVAILABLE = False
    APP_ERROR = str(e)


@pytest.fixture(scope="function")
def client():
    if not APP_AVAILABLE:
        pytest.skip(f"App not available: {APP_ERROR}")

    # Use in-memory SQLite for tests
    os.environ["DATABASE_URL"] = "sqlite:///./test_db.db"

    # Create tables
    SQLModel.metadata.create_all(engine)

    # Override the get_session dependency
    def get_test_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session

    yield TestClient(app)

    # Cleanup
    app.dependency_overrides.clear()
    SQLModel.metadata.drop_all(engine)

    # Remove test database file
    if os.path.exists("./test_db.db"):
        os.remove("./test_db.db")


class TestRootEndpoints:
    """Test root and info endpoints"""

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Training Platform API"
        assert "endpoints" in data

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_api_info(self, client):
        response = client.get("/api/v1")
        assert response.status_code == 200
        data = response.json()
        assert "supported_algorithms" in data
        assert "grpo" in data["supported_algorithms"]
        assert "supported_models" in data
        assert "supported_gpus" in data


class TestComputeEndpoints:
    """Test compute calculator endpoints"""

    def test_calculate_config(self, client):
        response = client.post(
            "/api/v1/compute/calculate",
            json={
                "model_size": "7B",
                "gpu_type": "A100-80G",
                "num_gpus": 8,
                "context_length": 4096,
                "training_type": "grpo",
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert "memory_estimate" in data
        assert "summary" in data

    def test_calculate_config_with_lora(self, client):
        response = client.post(
            "/api/v1/compute/calculate",
            json={
                "model_size": "7B",
                "gpu_type": "A100-80G",
                "num_gpus": 8,
                "training_type": "grpo",
                "lora_enabled": True,
                "lora_rank": 16,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["config"]["actor"]["lora"]["enabled"] is True

    def test_estimate_memory(self, client):
        response = client.post(
            "/api/v1/compute/memory",
            json={
                "model_size": "7B",
                "gpu_type": "A100-80G",
                "num_gpus": 8,
                "batch_size": 4,
                "seq_len": 4096,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "breakdown" in data
        assert "recommended_zero_stage" in data

    def test_get_gpu_types(self, client):
        response = client.get("/api/v1/compute/gpu-types")
        assert response.status_code == 200
        data = response.json()
        assert "gpu_types" in data
        assert len(data["gpu_types"]) > 0

    def test_get_model_sizes(self, client):
        response = client.get("/api/v1/compute/model-sizes")
        assert response.status_code == 200
        data = response.json()
        assert "model_sizes" in data
        # model_sizes is a list of objects with 'id' field
        model_ids = [m["id"] for m in data["model_sizes"]]
        assert "7B" in model_ids


class TestJobsEndpoints:
    """Test training jobs endpoints"""

    def test_create_job(self, client):
        response = client.post(
            "/api/v1/jobs",
            json={
                "name": "test-job",
                "algorithm": "grpo",
                "model_path": "/path/to/model",
                "train_data_path": "/path/to/data",
                "num_gpus": 8,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == "test-job"
        assert data["status"] == "pending"

    def test_list_jobs(self, client):
        # Create a job first
        client.post(
            "/api/v1/jobs",
            json={
                "name": "test-job-list",
                "algorithm": "grpo",
                "model_path": "/path/to/model",
                "train_data_path": "/path/to/data",
                "num_gpus": 8,
            }
        )

        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data

    def test_get_job(self, client):
        # Create a job first
        create_response = client.post(
            "/api/v1/jobs",
            json={
                "name": "test-job-get",
                "algorithm": "grpo",
                "model_path": "/path/to/model",
                "train_data_path": "/path/to/data",
                "num_gpus": 8,
            }
        )
        job_id = create_response.json()["id"]

        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id

    def test_get_job_not_found(self, client):
        response = client.get("/api/v1/jobs/nonexistent")
        assert response.status_code == 404

    def test_update_job(self, client):
        # Create a job first
        create_response = client.post(
            "/api/v1/jobs",
            json={
                "name": "test-job-update",
                "algorithm": "grpo",
                "model_path": "/path/to/model",
                "train_data_path": "/path/to/data",
                "num_gpus": 8,
            }
        )
        job_id = create_response.json()["id"]

        response = client.patch(
            f"/api/v1/jobs/{job_id}",
            json={"name": "updated-name"}
        )
        assert response.status_code == 200
        assert response.json()["name"] == "updated-name"

    def test_delete_job(self, client):
        # Create a job first
        create_response = client.post(
            "/api/v1/jobs",
            json={
                "name": "test-job-delete",
                "algorithm": "grpo",
                "model_path": "/path/to/model",
                "train_data_path": "/path/to/data",
                "num_gpus": 8,
            }
        )
        job_id = create_response.json()["id"]

        response = client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200

        # Verify job is deleted
        get_response = client.get(f"/api/v1/jobs/{job_id}")
        assert get_response.status_code == 404


class TestSurgeryEndpoints:
    """Test model surgery endpoints"""

    def test_merge_models(self, client):
        response = client.post(
            "/api/v1/surgery/merge",
            json={
                "method": "linear",
                "models": ["model1", "model2"],
                "weights": [0.5, 0.5],
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "method" in data

    def test_merge_slerp(self, client):
        response = client.post(
            "/api/v1/surgery/merge",
            json={
                "method": "slerp",
                "models": ["model1", "model2"],
                "interpolation_t": 0.5,
            }
        )
        assert response.status_code == 200

    def test_select_checkpoint(self, client):
        response = client.post(
            "/api/v1/surgery/checkpoint/select",
            json={
                "experiment_path": "/path/to/experiment",
                "criteria": "balanced",
            }
        )
        assert response.status_code == 200

    def test_get_merge_methods(self, client):
        response = client.get("/api/v1/surgery/methods")
        assert response.status_code == 200
        data = response.json()
        assert "methods" in data
        assert "linear" in [m["id"] for m in data["methods"]]


class TestMonitoringEndpoints:
    """Test monitoring endpoints"""

    def test_get_dashboard(self, client):
        response = client.get("/api/v1/monitoring/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "active_jobs" in data
        assert "total_gpu_hours" in data

    def test_get_gradient_heatmap(self, client):
        response = client.get("/api/v1/monitoring/test-job/gradient-heatmap")
        assert response.status_code == 200
        data = response.json()
        assert "layers" in data
        assert "steps" in data
        assert "data" in data

    def test_get_evaluations(self, client):
        response = client.get("/api/v1/monitoring/test-job/evaluations")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_get_resources(self, client):
        response = client.get("/api/v1/monitoring/test-job/resources")
        assert response.status_code == 200
        data = response.json()
        assert "usage" in data

    def test_get_alerts(self, client):
        response = client.get("/api/v1/monitoring/test-job/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
