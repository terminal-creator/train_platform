"""
Tests for Dataset Deduplication and Upload Features

Tests the semantic data cleaning pipeline:
1. File upload
2. Similarity search
3. Deduplication detection
4. Duplicate deletion
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os


class TestDatasetAPI:
    """Tests for Dataset API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from training_platform.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test dataset health endpoint"""
        response = client.get("/api/v1/datasets/health")
        assert response.status_code == 200

        data = response.json()
        assert "milvus_available" in data
        assert "embedding_available" in data

    def test_list_tasks_empty(self, client):
        """Test listing tasks when empty"""
        # Clear any existing tasks
        from training_platform.api.routers.dataset import _dedup_tasks
        _dedup_tasks.clear()

        response = client.get("/api/v1/datasets/tasks")
        assert response.status_code == 200

        data = response.json()
        assert "tasks" in data
        assert "total" in data

    def test_task_not_found(self, client):
        """Test getting non-existent task"""
        response = client.get("/api/v1/datasets/tasks/nonexistent")
        assert response.status_code == 404

    def test_dedup_task_not_found(self, client):
        """Test getting non-existent dedup task"""
        response = client.get("/api/v1/datasets/deduplicate/nonexistent")
        assert response.status_code == 404


class TestFileUpload:
    """Tests for file upload functionality"""

    @pytest.fixture
    def client(self):
        from training_platform.api.main import app
        return TestClient(app)

    def test_upload_invalid_file_type(self, client):
        """Test uploading invalid file type"""
        # Create a fake file with wrong extension
        files = {"file": ("test.txt", b"some content", "text/plain")}
        response = client.post("/api/v1/datasets/upload", files=files)

        assert response.status_code == 400
        assert "jsonl" in response.json()["detail"].lower()

    def test_upload_valid_jsonl_structure(self, client):
        """Test upload endpoint accepts valid JSONL file structure"""
        # Create valid JSONL content
        content = '\n'.join([
            json.dumps({"id": "1", "text": "Hello world"}),
            json.dumps({"id": "2", "text": "Test data"}),
        ])

        files = {"file": ("test.jsonl", content.encode(), "application/json")}

        # The upload should work but processing may fail without Milvus
        response = client.post("/api/v1/datasets/upload", files=files)

        # Should return 200 with task_id (processing happens in background)
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "filename" in data
        assert data["success"] is True


class TestDeduplication:
    """Tests for deduplication functionality"""

    @pytest.fixture
    def client(self):
        from training_platform.api.main import app
        return TestClient(app)

    def test_start_deduplication_default_params(self, client):
        """Test starting deduplication with default parameters"""
        response = client.post("/api/v1/datasets/deduplicate", json={})

        # Should accept the request (actual processing may fail without Milvus)
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "processing"

    def test_start_deduplication_custom_params(self, client):
        """Test starting deduplication with custom parameters"""
        response = client.post("/api/v1/datasets/deduplicate", json={
            "threshold": 0.90,
            "sample_size": 500,
            "source_filter": "test_source",
        })

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data

    def test_deduplication_threshold_validation(self, client):
        """Test threshold validation"""
        # Threshold too low
        response = client.post("/api/v1/datasets/deduplicate", json={
            "threshold": 0.3,
        })
        assert response.status_code == 422  # Validation error

        # Threshold too high
        response = client.post("/api/v1/datasets/deduplicate", json={
            "threshold": 1.5,
        })
        assert response.status_code == 422

    def test_delete_duplicates_empty(self, client):
        """Test deleting with empty ID list"""
        response = client.post("/api/v1/datasets/deduplicate/delete", json={
            "ids": [],
        })
        assert response.status_code == 400

    def test_delete_task(self, client):
        """Test deleting a completed task"""
        from training_platform.api.routers.dataset import _dedup_tasks

        # Create a fake completed task
        task_id = "test123"
        _dedup_tasks[task_id] = {
            "type": "deduplicate",
            "status": "completed",
            "started_at": "2024-01-01T00:00:00",
            "completed_at": "2024-01-01T00:01:00",
            "duplicates_found": 0,
            "duplicates": [],
        }

        response = client.delete(f"/api/v1/datasets/tasks/{task_id}")
        assert response.status_code == 200
        assert task_id not in _dedup_tasks

    def test_delete_running_task(self, client):
        """Test that running tasks cannot be deleted"""
        from training_platform.api.routers.dataset import _dedup_tasks

        # Create a fake running task
        task_id = "running123"
        _dedup_tasks[task_id] = {
            "type": "deduplicate",
            "status": "processing",
            "started_at": "2024-01-01T00:00:00",
        }

        response = client.delete(f"/api/v1/datasets/tasks/{task_id}")
        assert response.status_code == 400

        # Cleanup
        del _dedup_tasks[task_id]


class TestBatchSearch:
    """Tests for batch search functionality"""

    @pytest.fixture
    def client(self):
        from training_platform.api.main import app
        return TestClient(app)

    def test_batch_search_too_many_queries(self, client):
        """Test batch search with too many queries"""
        queries = [f"query_{i}" for i in range(101)]

        response = client.post("/api/v1/datasets/search/batch", json={
            "queries": queries,
        })

        assert response.status_code == 400
        assert "100" in response.json()["detail"]


class TestDatasetManagerDedup:
    """Unit tests for DatasetManager deduplication logic"""

    def test_find_duplicates_returns_list(self):
        """Test that find_duplicates returns a list"""
        from training_platform.core.vector_store import DatasetManager

        # Create mock vector store
        mock_vector_store = Mock()
        mock_vector_store._ensure_collection = Mock()
        mock_vector_store._collection = Mock()
        mock_vector_store._collection.num_entities = 0

        manager = DatasetManager(vector_store=mock_vector_store)
        result = manager.find_duplicates(threshold=0.95, sample_size=100)

        assert isinstance(result, list)

    def test_get_samples_empty_collection(self):
        """Test _get_samples with empty collection"""
        from training_platform.core.vector_store import DatasetManager

        mock_vector_store = Mock()
        mock_vector_store._ensure_collection = Mock()
        mock_vector_store._collection = Mock()
        mock_vector_store._collection.num_entities = 0

        manager = DatasetManager(vector_store=mock_vector_store)
        result = manager._get_samples(sample_size=100)

        assert result == []

    def test_delete_duplicates_empty(self):
        """Test delete_duplicates with empty list"""
        from training_platform.core.vector_store import DatasetManager

        mock_vector_store = Mock()
        manager = DatasetManager(vector_store=mock_vector_store)

        result = manager.delete_duplicates([])
        assert result == 0


class TestDuplicatePairModel:
    """Tests for DuplicatePair model"""

    def test_duplicate_pair_creation(self):
        """Test creating a DuplicatePair"""
        from training_platform.api.routers.dataset import DuplicatePair

        pair = DuplicatePair(
            id1="abc123",
            id2="def456",
            similarity=0.97,
            text1="Hello world",
            text2="Hello world!",
            source1="source_a",
            source2="source_b",
        )

        assert pair.id1 == "abc123"
        assert pair.id2 == "def456"
        assert pair.similarity == 0.97

    def test_duplicate_pair_optional_sources(self):
        """Test DuplicatePair with optional sources"""
        from training_platform.api.routers.dataset import DuplicatePair

        pair = DuplicatePair(
            id1="abc",
            id2="def",
            similarity=0.95,
            text1="text",
            text2="text",
        )

        assert pair.source1 == ""
        assert pair.source2 == ""


class TestDeduplicationWorkflow:
    """Integration tests for deduplication workflow"""

    @pytest.fixture
    def client(self):
        from training_platform.api.main import app
        return TestClient(app)

    def test_full_dedup_workflow(self, client):
        """Test the full deduplication workflow"""
        from training_platform.api.routers.dataset import _dedup_tasks

        # 1. Start deduplication
        response = client.post("/api/v1/datasets/deduplicate", json={
            "threshold": 0.95,
            "sample_size": 100,
        })
        assert response.status_code == 200
        task_id = response.json()["task_id"]

        # 2. Check task status
        response = client.get(f"/api/v1/datasets/deduplicate/{task_id}")
        assert response.status_code == 200
        status = response.json()
        assert status["task_id"] == task_id
        assert status["status"] in ["processing", "completed", "failed"]

        # 3. List all tasks
        response = client.get("/api/v1/datasets/tasks")
        assert response.status_code == 200
        tasks = response.json()["tasks"]
        task_ids = [t["task_id"] for t in tasks]
        assert task_id in task_ids

        # 4. Get specific task status
        response = client.get(f"/api/v1/datasets/tasks/{task_id}")
        assert response.status_code == 200

        # Cleanup
        if task_id in _dedup_tasks:
            _dedup_tasks[task_id]["status"] = "completed"
            client.delete(f"/api/v1/datasets/tasks/{task_id}")


class TestSearchEndpoints:
    """Tests for search endpoints"""

    @pytest.fixture
    def client(self):
        from training_platform.api.main import app
        return TestClient(app)

    def test_search_endpoint_structure(self, client):
        """Test search endpoint accepts correct structure"""
        response = client.post("/api/v1/datasets/search", json={
            "query": "test query",
            "top_k": 5,
        })

        # May fail without Milvus, but structure should be valid
        assert response.status_code in [200, 500, 503]

    def test_search_batch_structure(self, client):
        """Test batch search endpoint accepts correct structure"""
        response = client.post("/api/v1/datasets/search/batch", json={
            "queries": ["query 1", "query 2"],
            "top_k": 3,
        })

        # May fail without Milvus
        assert response.status_code in [200, 500, 503]


class TestDataDistributionAnalysis:
    """Tests for data distribution analysis"""

    @pytest.fixture
    def client(self):
        from training_platform.api.main import app
        return TestClient(app)

    @pytest.fixture
    def sample_jsonl(self):
        """Create sample JSONL content"""
        records = [
            {"id": "1", "text": "Hello world", "score": 0.95, "category": "greeting"},
            {"id": "2", "text": "How are you?", "score": 0.88, "category": "question"},
            {"id": "3", "text": "Fine thanks", "score": 0.72, "category": "response"},
            {"id": "4", "text": "Goodbye!", "score": 0.91, "category": "greeting"},
            {"id": "5", "text": "What is AI?", "score": 0.85, "category": "question"},
        ]
        return '\n'.join(json.dumps(r) for r in records)

    def test_analyze_upload_string_field(self, client, sample_jsonl):
        """Test analyzing string field distribution"""
        files = {"file": ("test.jsonl", sample_jsonl.encode(), "application/json")}

        response = client.post(
            "/api/v1/datasets/analyze/upload?fields=text",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["total_records"] == 5
        assert len(data["fields"]) == 1

        field = data["fields"][0]
        assert field["field_name"] == "text"
        assert field["field_type"] == "string"
        assert field["total_count"] == 5
        assert field["null_count"] == 0
        assert "length_stats" in field
        assert "length_histogram" in field

    def test_analyze_upload_numeric_field(self, client, sample_jsonl):
        """Test analyzing numeric field distribution"""
        files = {"file": ("test.jsonl", sample_jsonl.encode(), "application/json")}

        response = client.post(
            "/api/v1/datasets/analyze/upload?fields=score",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()

        field = data["fields"][0]
        assert field["field_name"] == "score"
        assert field["field_type"] == "number"
        assert "numeric_stats" in field
        assert field["numeric_stats"]["min"] == 0.72
        assert field["numeric_stats"]["max"] == 0.95

    def test_analyze_two_fields_cross_stats(self, client, sample_jsonl):
        """Test analyzing two fields with cross-field statistics"""
        files = {"file": ("test.jsonl", sample_jsonl.encode(), "application/json")}

        response = client.post(
            "/api/v1/datasets/analyze/upload?fields=text,category",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["fields"]) == 2
        assert data["cross_field_stats"] is not None
        assert "field1" in data["cross_field_stats"]
        assert "field2" in data["cross_field_stats"]

    def test_analyze_invalid_file_type(self, client):
        """Test uploading invalid file type"""
        files = {"file": ("test.txt", b"not json", "text/plain")}

        response = client.post(
            "/api/v1/datasets/analyze/upload?fields=text",
            files=files,
        )

        assert response.status_code == 400

    def test_analyze_no_fields(self, client, sample_jsonl):
        """Test analyzing with no fields specified"""
        files = {"file": ("test.jsonl", sample_jsonl.encode(), "application/json")}

        response = client.post(
            "/api/v1/datasets/analyze/upload?fields=",
            files=files,
        )

        assert response.status_code == 400

    def test_analyze_nested_fields(self, client):
        """Test analyzing nested fields"""
        records = [
            {"id": "1", "metadata": {"source": "web", "score": 0.9}},
            {"id": "2", "metadata": {"source": "book", "score": 0.8}},
            {"id": "3", "metadata": {"source": "web", "score": 0.7}},
        ]
        content = '\n'.join(json.dumps(r) for r in records)
        files = {"file": ("test.jsonl", content.encode(), "application/json")}

        response = client.post(
            "/api/v1/datasets/analyze/upload?fields=metadata.source",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()

        field = data["fields"][0]
        assert field["field_name"] == "metadata.source"
        assert field["unique_count"] == 2

    def test_analyze_array_field(self, client):
        """Test analyzing array fields"""
        records = [
            {"id": "1", "tags": ["a", "b", "c"]},
            {"id": "2", "tags": ["x", "y"]},
            {"id": "3", "tags": ["p"]},
        ]
        content = '\n'.join(json.dumps(r) for r in records)
        files = {"file": ("test.jsonl", content.encode(), "application/json")}

        response = client.post(
            "/api/v1/datasets/analyze/upload?fields=tags",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()

        field = data["fields"][0]
        assert field["field_type"] == "array"
        assert "array_length_stats" in field
        assert field["array_length_stats"]["min"] == 1
        assert field["array_length_stats"]["max"] == 3

    def test_get_available_fields(self, client, sample_jsonl):
        """Test getting available fields from file"""
        files = {"file": ("test.jsonl", sample_jsonl.encode(), "application/json")}

        response = client.post(
            "/api/v1/datasets/analyze/fields",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "fields" in data
        assert "id" in data["fields"]
        assert "text" in data["fields"]
        assert "score" in data["fields"]
        assert "category" in data["fields"]

    def test_analyze_with_null_values(self, client):
        """Test analyzing fields with null values"""
        records = [
            {"id": "1", "text": "hello", "optional": "value"},
            {"id": "2", "text": "world", "optional": None},
            {"id": "3", "text": "test"},  # missing optional field
        ]
        content = '\n'.join(json.dumps(r) for r in records)
        files = {"file": ("test.jsonl", content.encode(), "application/json")}

        response = client.post(
            "/api/v1/datasets/analyze/upload?fields=optional",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()

        field = data["fields"][0]
        assert field["null_count"] == 2  # None + missing

    def test_analyze_numeric_correlation(self, client):
        """Test correlation between two numeric fields"""
        records = [
            {"x": 1, "y": 2},
            {"x": 2, "y": 4},
            {"x": 3, "y": 6},
            {"x": 4, "y": 8},
            {"x": 5, "y": 10},
        ]
        content = '\n'.join(json.dumps(r) for r in records)
        files = {"file": ("test.jsonl", content.encode(), "application/json")}

        response = client.post(
            "/api/v1/datasets/analyze/upload?fields=x,y",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()

        cross_stats = data["cross_field_stats"]
        assert cross_stats["type"] == "numeric"
        # Perfect positive correlation
        assert abs(cross_stats["correlation"] - 1.0) < 0.01

    def test_analyze_value_counts_with_percentage(self, client):
        """Test analyzing categorical field with value counts and percentages"""
        records = [
            {"id": "1", "node": "A"},
            {"id": "2", "node": "A"},
            {"id": "3", "node": "A"},
            {"id": "4", "node": "B"},
            {"id": "5", "node": "B"},
            {"id": "6", "node": "C"},
            {"id": "7", "node": "C"},
            {"id": "8", "node": "C"},
            {"id": "9", "node": "C"},
            {"id": "10", "node": "D"},
        ]
        content = '\n'.join(json.dumps(r) for r in records)
        files = {"file": ("test.jsonl", content.encode(), "application/json")}

        response = client.post(
            "/api/v1/datasets/analyze/upload?fields=node",
            files=files,
        )

        assert response.status_code == 200
        data = response.json()

        field = data["fields"][0]
        assert field["field_name"] == "node"
        assert field["unique_count"] == 4
        assert field["total_count"] == 10

        # Check value_counts
        value_counts = field["value_counts"]
        assert value_counts is not None
        assert len(value_counts) == 4

        # Values should be sorted by count descending
        # C: 4 (40%), A: 3 (30%), B: 2 (20%), D: 1 (10%)
        assert value_counts[0]["value"] == "C"
        assert value_counts[0]["count"] == 4
        assert value_counts[0]["percentage"] == 40.0

        assert value_counts[1]["value"] == "A"
        assert value_counts[1]["count"] == 3
        assert value_counts[1]["percentage"] == 30.0

        assert value_counts[2]["value"] == "B"
        assert value_counts[2]["count"] == 2
        assert value_counts[2]["percentage"] == 20.0

        assert value_counts[3]["value"] == "D"
        assert value_counts[3]["count"] == 1
        assert value_counts[3]["percentage"] == 10.0


class TestAnalysisHelperFunctions:
    """Tests for analysis helper functions"""

    def test_get_nested_value(self):
        """Test getting nested values"""
        from training_platform.api.routers.dataset import _get_nested_value

        record = {
            "a": 1,
            "b": {"c": 2, "d": {"e": 3}},
            "arr": [{"x": 10}, {"x": 20}],
        }

        assert _get_nested_value(record, "a") == 1
        assert _get_nested_value(record, "b.c") == 2
        assert _get_nested_value(record, "b.d.e") == 3
        assert _get_nested_value(record, "arr.0.x") == 10
        assert _get_nested_value(record, "arr.1.x") == 20
        assert _get_nested_value(record, "nonexistent") is None

    def test_compute_histogram(self):
        """Test histogram computation"""
        from training_platform.api.routers.dataset import _compute_histogram

        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        histogram = _compute_histogram(values, bins=5)

        assert len(histogram) == 5
        # Total count should equal number of values
        total = sum(bin["count"] for bin in histogram)
        assert total == 10

    def test_compute_top_values(self):
        """Test top values computation"""
        from training_platform.api.routers.dataset import _compute_top_values

        values = ["a", "a", "a", "b", "b", "c"]
        top = _compute_top_values(values, top_n=3)

        assert len(top) == 3
        assert top[0]["value"] == "a"
        assert top[0]["count"] == 3
        assert top[1]["value"] == "b"
        assert top[1]["count"] == 2

    def test_extract_field_paths(self):
        """Test field path extraction"""
        from training_platform.api.routers.dataset import _extract_field_paths

        obj = {
            "a": 1,
            "b": {"c": 2, "d": 3},
            "arr": [{"x": 1}],
        }

        paths = _extract_field_paths(obj)

        assert "a" in paths
        assert "b" in paths
        assert "b.c" in paths
        assert "b.d" in paths
        assert "arr" in paths
        assert "arr.0.x" in paths
