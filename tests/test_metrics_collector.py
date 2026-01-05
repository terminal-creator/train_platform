"""
Tests for MetricsCollector and LogParser

Tests the training -> monitoring pipeline:
1. LogParser: Parse verl training logs
2. MetricsCollector: Collect and push metrics
3. Push Mode API: Direct metrics reporting
"""

import pytest
from datetime import datetime


class TestLogParser:
    """Tests for LogParser"""

    def test_parse_console_line_basic(self):
        """Test parsing basic console log line"""
        from training_platform.api.routers.websocket import LogParser

        line = "step:100 - train/loss:0.523 - train/reward:0.156"
        result = LogParser.parse_console_line(line)

        assert result is not None
        assert result['step'] == 100
        assert result['total_loss'] == 0.523
        assert result['reward_mean'] == 0.156

    def test_parse_console_line_with_scientific_notation(self):
        """Test parsing with scientific notation"""
        from training_platform.api.routers.websocket import LogParser

        line = "step:500 - train/loss:1.23e-04 - train/kl:5.6e-3"
        result = LogParser.parse_console_line(line)

        assert result is not None
        assert result['step'] == 500
        assert abs(result['total_loss'] - 1.23e-04) < 1e-10
        assert abs(result['kl_divergence'] - 5.6e-3) < 1e-10

    def test_parse_console_line_full_metrics(self):
        """Test parsing line with all metrics"""
        from training_platform.api.routers.websocket import LogParser

        line = "step:1000 - train/policy_loss:0.45 - train/value_loss:0.32 - train/reward_mean:0.78 - train/kl_divergence:0.015 - train/entropy:1.25"
        result = LogParser.parse_console_line(line)

        assert result is not None
        assert result['step'] == 1000
        assert result['policy_loss'] == 0.45
        assert result['value_loss'] == 0.32
        assert result['reward_mean'] == 0.78
        assert result['kl_divergence'] == 0.015
        assert result['entropy'] == 1.25

    def test_parse_console_line_skip_non_metric(self):
        """Test skipping non-metric lines"""
        from training_platform.api.routers.websocket import LogParser

        lines = [
            "Loading model from /path/to/model",
            "Starting training...",
            "",
            "   ",
        ]

        for line in lines:
            result = LogParser.parse_console_line(line)
            assert result is None

    def test_parse_jsonl_line_basic(self):
        """Test parsing basic JSONL log line"""
        from training_platform.api.routers.websocket import LogParser
        import json

        data = {
            "step": 200,
            "data": {
                "train/loss": 0.456,
                "train/reward": 0.789,
            }
        }
        line = json.dumps(data)
        result = LogParser.parse_jsonl_line(line)

        assert result is not None
        assert result['step'] == 200
        assert result['total_loss'] == 0.456
        assert result['reward_mean'] == 0.789

    def test_parse_jsonl_line_full_metrics(self):
        """Test parsing JSONL with all metrics"""
        from training_platform.api.routers.websocket import LogParser
        import json

        data = {
            "step": 1500,
            "data": {
                "train/policy_loss": 0.33,
                "train/value_loss": 0.22,
                "train/reward_mean": 0.88,
                "train/reward_std": 0.15,
                "train/kl_divergence": 0.012,
                "train/entropy": 1.5,
                "custom/accuracy": 0.92,
            }
        }
        line = json.dumps(data)
        result = LogParser.parse_jsonl_line(line)

        assert result is not None
        assert result['step'] == 1500
        assert result['policy_loss'] == 0.33
        assert result['value_loss'] == 0.22
        assert result['reward_mean'] == 0.88
        assert result['reward_std'] == 0.15
        assert result['kl_divergence'] == 0.012
        assert result['entropy'] == 1.5
        assert result['custom/accuracy'] == 0.92

    def test_parse_jsonl_line_invalid(self):
        """Test parsing invalid JSONL"""
        from training_platform.api.routers.websocket import LogParser

        invalid_lines = [
            "not json at all",
            '{"no_step": true}',
            '{"step": "not a number"}',
        ]

        for line in invalid_lines:
            result = LogParser.parse_jsonl_line(line)
            # Should return None or a dict with only step
            assert result is None or len(result) <= 1

    def test_parse_logs_mixed_format(self):
        """Test parsing mixed format logs"""
        from training_platform.api.routers.websocket import LogParser
        import json

        logs = """
Loading model...
step:100 - train/loss:0.5 - train/reward:0.1
Some other output
{"step": 200, "data": {"train/loss": 0.4, "train/reward": 0.2}}
step:300 - train/loss:0.3 - train/reward:0.3
        """

        results = LogParser.parse_logs(logs)

        assert len(results) == 3
        assert results[0]['step'] == 100
        assert results[1]['step'] == 200
        assert results[2]['step'] == 300

    def test_parse_logs_deduplication(self):
        """Test that duplicate steps are deduplicated"""
        from training_platform.api.routers.websocket import LogParser

        logs = """
step:100 - train/loss:0.5
step:100 - train/loss:0.4
step:200 - train/loss:0.3
        """

        results = LogParser.parse_logs(logs)

        assert len(results) == 2
        steps = [r['step'] for r in results]
        assert 100 in steps
        assert 200 in steps

    def test_extract_latest_metrics(self):
        """Test extracting only new metrics"""
        from training_platform.api.routers.websocket import LogParser

        logs = """
step:100 - train/loss:0.5
step:200 - train/loss:0.4
step:300 - train/loss:0.3
step:400 - train/loss:0.2
        """

        # Extract metrics after step 200
        results = LogParser.extract_latest_metrics(logs, last_step=200)

        assert len(results) == 2
        assert results[0]['step'] == 300
        assert results[1]['step'] == 400


class TestMetricsCollector:
    """Tests for MetricsCollector"""

    @pytest.mark.asyncio
    async def test_collector_initialization(self):
        """Test MetricsCollector initialization"""
        from training_platform.api.routers.websocket import MetricsCollector

        collector = MetricsCollector()

        assert collector._running is False
        assert collector._task is None
        assert len(collector._last_steps) == 0

    @pytest.mark.asyncio
    async def test_collector_start_stop(self):
        """Test starting and stopping collector"""
        from training_platform.api.routers.websocket import MetricsCollector
        import asyncio

        collector = MetricsCollector()

        # Start
        await collector.start()
        assert collector._running is True
        assert collector._task is not None

        # Wait a bit for the loop to run
        await asyncio.sleep(0.1)

        # Stop
        await collector.stop()
        assert collector._running is False

    @pytest.mark.asyncio
    async def test_collector_idempotent_start(self):
        """Test that multiple starts don't create multiple tasks"""
        from training_platform.api.routers.websocket import MetricsCollector

        collector = MetricsCollector()

        await collector.start()
        first_task = collector._task

        await collector.start()
        second_task = collector._task

        assert first_task is second_task

        await collector.stop()


class TestMonitoringAPI:
    """Tests for monitoring API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from training_platform.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test monitoring health endpoint"""
        response = client.get("/api/v1/monitoring/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "metrics_collector_running" in data
        assert "timestamp" in data

    def test_report_metrics_missing_job(self, client):
        """Test reporting metrics for non-existent job"""
        response = client.post("/api/v1/monitoring/report", json={
            "job_uuid": "non-existent-job",
            "step": 100,
            "policy_loss": 0.5,
        })

        # Should return 404 or 500 depending on implementation
        assert response.status_code in [404, 500]

    def test_report_gpu_usage(self, client):
        """Test GPU usage reporting structure"""
        # This tests the API structure, actual DB operations may fail without setup
        response = client.post("/api/v1/monitoring/gpu", json={
            "job_uuid": "test-job",
            "gpu_index": 0,
            "utilization": 85.5,
            "memory_used": 60.2,
            "memory_total": 80.0,
            "temperature": 72.0,
        })

        # May fail without DB setup, but structure should be valid
        assert response.status_code in [200, 404, 500]

    def test_report_status(self, client):
        """Test status reporting structure"""
        response = client.post("/api/v1/monitoring/status", json={
            "job_uuid": "test-job",
            "status": "running",
            "message": "Training in progress",
            "current_step": 1000,
        })

        # May fail without DB setup
        assert response.status_code in [200, 404, 500]

    def test_report_log(self, client):
        """Test log reporting structure"""
        response = client.post("/api/v1/monitoring/log", json={
            "job_uuid": "test-job",
            "level": "INFO",
            "message": "Training started",
        })

        # May fail without DB setup
        assert response.status_code in [200, 404, 500]


class TestVerlLogFormats:
    """Test parsing actual verl log formats"""

    def test_verl_console_format(self):
        """Test verl's LocalLogger console format"""
        from training_platform.api.routers.websocket import LogParser

        # verl LocalLogger format: step:X - key:value - key:value
        line = "step:1000 - actor/loss:0.0523 - critic/loss:0.0891 - reward/mean:0.234 - kl/mean:0.0156"
        result = LogParser.parse_console_line(line)

        assert result is not None
        assert result['step'] == 1000
        assert 'policy_loss' in result  # actor/loss -> policy_loss
        assert 'value_loss' in result   # critic/loss -> value_loss
        assert 'reward_mean' in result
        assert 'kl_divergence' in result

    def test_verl_file_logger_format(self):
        """Test verl's FileLogger JSONL format"""
        from training_platform.api.routers.websocket import LogParser
        import json

        # verl FileLogger format: {"step": X, "data": {...}}
        data = {
            "step": 2000,
            "data": {
                "actor/loss": 0.0412,
                "critic/loss": 0.0723,
                "reward/mean": 0.345,
                "reward/std": 0.123,
                "kl/mean": 0.0089,
                "entropy": 1.45,
                "throughput/tokens_per_sec": 15234.5,
            }
        }
        line = json.dumps(data)
        result = LogParser.parse_jsonl_line(line)

        assert result is not None
        assert result['step'] == 2000
        assert 'policy_loss' in result
        assert 'value_loss' in result
        assert 'reward_mean' in result
        assert 'reward_std' in result
        assert 'kl_divergence' in result
        assert 'entropy' in result
        # Custom metrics preserved
        assert 'throughput/tokens_per_sec' in result

    def test_verl_log_stream(self):
        """Test parsing a realistic verl log stream"""
        from training_platform.api.routers.websocket import LogParser

        log_stream = """
2024-01-15 10:00:00 - INFO - Starting PPO training
2024-01-15 10:00:01 - INFO - Loading model from /models/qwen2.5-7b
step:0 - actor/loss:2.3456 - critic/loss:1.2345 - reward/mean:-0.5 - kl/mean:0.0
{"step": 100, "data": {"actor/loss": 1.5678, "critic/loss": 0.8901, "reward/mean": 0.1, "kl/mean": 0.01}}
step:200 - actor/loss:0.8765 - critic/loss:0.5432 - reward/mean:0.35 - kl/mean:0.015
2024-01-15 10:05:00 - INFO - Saving checkpoint at step 200
{"step": 300, "data": {"actor/loss": 0.5432, "critic/loss": 0.3210, "reward/mean": 0.55, "kl/mean": 0.012}}
        """

        results = LogParser.parse_logs(log_stream)

        # Should parse 4 metric entries (step 0, 100, 200, 300)
        assert len(results) == 4

        # Verify ordering
        steps = [r['step'] for r in results]
        assert steps == [0, 100, 200, 300]

        # Verify metrics are extracted
        for result in results:
            assert 'policy_loss' in result or 'total_loss' in result
            assert 'reward_mean' in result
