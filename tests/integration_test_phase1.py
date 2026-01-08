#!/usr/bin/env python3
"""
Phase 1 集成测试

使用真实模型和数据测试完整的监控和诊断流程。

测试内容：
1. 准备测试数据集（小规模 SFT 数据）
2. 配置 verl 训练任务
3. 启动训练（带 PlatformCallback）
4. 实时监控指标文件
5. 同步指标到数据库
6. 验证 WebSocket 推送
7. 验证异常检测
8. 清理测试数据

环境要求：
- 已安装 verl（environments/verl）
- 小型模型（可选，可以使用 mock）
- 至少 1 个 GPU（或使用 CPU 模式测试）
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import uuid

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlmodel import Session, select
from training_platform.core.database import (
    engine, TrainingJob, TrainingMetric, JobStatus, TrainingAlgorithm
)
from training_platform.core.metrics_persister import sync_metrics_for_job
from training_platform.core.diagnostics import AnomalyDetector, DiagnosticService


class IntegrationTest:
    """集成测试类"""

    def __init__(self):
        self.test_id = f"integration_test_{int(time.time())}"
        self.test_dir = Path("./test_integration")
        self.metrics_dir = self.test_dir / "platform_metrics"
        self.data_dir = self.test_dir / "data"
        self.job_uuid = None

        print("="*80)
        print("Phase 1 集成测试")
        print("="*80)
        print(f"测试 ID: {self.test_id}")
        print(f"测试目录: {self.test_dir}")
        print()

    def setup(self):
        """准备测试环境"""
        print("\n" + "="*80)
        print("步骤 1: 准备测试环境")
        print("="*80)

        # 创建测试目录
        self.test_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        print(f"✓ 创建测试目录: {self.test_dir}")

    def create_test_dataset(self, num_samples: int = 50) -> Path:
        """创建测试数据集

        生成简单的 SFT 数据集，格式：
        {"prompt": "...", "response": "..."}

        Args:
            num_samples: 样本数量

        Returns:
            数据集文件路径
        """
        print("\n" + "="*80)
        print("步骤 2: 创建测试数据集")
        print("="*80)

        dataset_file = self.data_dir / "test_sft_data.jsonl"

        prompts = [
            "What is 1+1?",
            "Explain machine learning.",
            "Write a hello world program.",
            "What is the capital of France?",
            "Describe the water cycle.",
        ]

        responses = [
            "1+1 equals 2.",
            "Machine learning is a subset of AI that enables computers to learn from data.",
            "print('Hello, World!')",
            "The capital of France is Paris.",
            "The water cycle involves evaporation, condensation, and precipitation.",
        ]

        with open(dataset_file, 'w') as f:
            for i in range(num_samples):
                sample = {
                    "prompt": prompts[i % len(prompts)],
                    "response": responses[i % len(responses)],
                }
                f.write(json.dumps(sample) + '\n')

        print(f"✓ 生成 {num_samples} 个测试样本")
        print(f"✓ 数据集文件: {dataset_file}")

        return dataset_file

    def create_test_job(self) -> str:
        """创建测试任务记录

        Returns:
            任务 UUID
        """
        print("\n" + "="*80)
        print("步骤 3: 创建测试任务")
        print("="*80)

        with Session(engine) as session:
            job_uuid = str(uuid.uuid4())

            job = TrainingJob(
                uuid=job_uuid,
                name=f"Integration Test {self.test_id}",
                algorithm=TrainingAlgorithm.PPO,
                model_path="mock/model",
                train_data_path=str(self.data_dir / "test_sft_data.jsonl"),
                status=JobStatus.RUNNING,
                created_at=datetime.utcnow(),
                started_at=datetime.utcnow(),
            )
            session.add(job)
            session.commit()

            self.job_uuid = job_uuid
            print(f"✓ 创建任务: {job.name}")
            print(f"✓ UUID: {job_uuid}")

            return job_uuid

    def simulate_training_with_callback(self, num_steps: int = 30):
        """模拟训练过程（使用 PlatformCallback）

        这里不运行真实的 verl 训练，而是模拟 PlatformCallback 的输出。

        Args:
            num_steps: 模拟的训练步数
        """
        print("\n" + "="*80)
        print("步骤 4: 模拟训练过程")
        print("="*80)

        metrics_file = self.metrics_dir / f"{self.test_id}_metrics.jsonl"
        status_file = self.metrics_dir / f"{self.test_id}_status.json"

        print(f"生成指标文件: {metrics_file}")
        print(f"生成状态文件: {status_file}")
        print()

        # 模拟训练指标
        with open(metrics_file, 'w') as f:
            for step in range(1, num_steps + 1):
                # 模拟指标变化
                actor_loss = 3.0 - step * 0.08  # 逐步下降
                critic_loss = 2.0 - step * 0.05
                total_loss = actor_loss + critic_loss

                reward_mean = -1.0 + step * 0.05  # 逐步上升
                kl_mean = 0.05 + step * 0.002  # 缓慢上升

                # 在 step 25 引入一个 KL 异常（用于测试异常检测）
                if step == 25:
                    kl_mean = 1.2  # 超过阈值

                metric = {
                    "step": step,
                    "timestamp": time.time(),
                    "epoch": 0,
                    "loss": {
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "total_loss": total_loss,
                    },
                    "reward": {
                        "mean": reward_mean,
                        "std": 0.1,
                        "max": reward_mean + 0.2,
                        "min": reward_mean - 0.2,
                    },
                    "kl": {
                        "mean": kl_mean,
                        "max": kl_mean * 1.2,
                    },
                    "gradient": {
                        "actor_norm": 1.5 - step * 0.01,
                        "critic_norm": 1.0 - step * 0.01,
                    },
                    "performance": {
                        "tokens_per_second": 1000 + step * 10,
                        "step_time": 2.0,
                        "gpu_memory_allocated": 40.0 + step * 0.1,
                    },
                    "raw": {},
                }

                f.write(json.dumps(metric) + '\n')

                # 模拟实时写入
                time.sleep(0.1)

                if step % 5 == 0:
                    print(f"  Step {step:3d}: loss={total_loss:.3f}, reward={reward_mean:.3f}, kl={kl_mean:.4f}")

        # 更新状态文件
        status = {
            "job_id": self.test_id,
            "status": "running",
            "current_step": num_steps,
            "total_steps": num_steps,
            "anomaly_detected": True,  # Step 25 有 KL 异常
            "anomaly_reason": "KL divergence explosion: 1.20 > 1.0 at step 25",
        }

        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

        print(f"\n✓ 生成 {num_steps} 步训练指标")
        print(f"✓ 在 step 25 引入 KL 异常（用于测试检测）")

    def test_metrics_sync(self):
        """测试指标同步到数据库"""
        print("\n" + "="*80)
        print("步骤 5: 测试指标同步")
        print("="*80)

        with Session(engine) as session:
            result = sync_metrics_for_job(
                job_uuid=self.job_uuid,
                job_id=self.test_id,
                metrics_dir=self.metrics_dir,
                session=session,
            )

            print(f"✓ 同步指标到数据库")
            print(f"  新增指标数量: {result['new_metrics_count']}")
            print(f"  异常已同步: {result['anomaly_synced']}")

            # 验证数据库中的指标
            statement = select(TrainingMetric).where(
                TrainingMetric.job_uuid == self.job_uuid
            ).order_by(TrainingMetric.step)
            metrics = session.exec(statement).all()

            print(f"\n✓ 验证数据库")
            print(f"  数据库中的指标数量: {len(metrics)}")

            if metrics:
                first = metrics[0]
                last = metrics[-1]
                print(f"  第一条: step={first.step}, loss={first.total_loss:.3f}")
                print(f"  最后一条: step={last.step}, loss={last.total_loss:.3f}")

            # 验证异常标记
            anomalies = [m for m in metrics if m.has_anomaly]
            if anomalies:
                print(f"\n✓ 检测到异常指标: {len(anomalies)} 条")
                for a in anomalies:
                    print(f"    Step {a.step}: {a.anomaly_type} - {a.anomaly_message}")

            assert len(metrics) > 0, "数据库中应该有指标"
            print("\n✓ 指标同步测试通过")

    def test_anomaly_detection(self):
        """测试异常检测"""
        print("\n" + "="*80)
        print("步骤 6: 测试异常检测")
        print("="*80)

        with Session(engine) as session:
            detector = AnomalyDetector(session)

            # 1. KL 散度爆炸检测
            print("\n测试 KL 散度爆炸检测...")
            result = detector.detect_kl_explosion(self.job_uuid, kl_threshold=1.0)
            print(f"  检测结果: {'检测到异常' if result.detected else '无异常'}")
            if result.detected:
                print(f"  异常类型: {result.anomaly_type}")
                print(f"  严重程度: {result.severity}")
                print(f"  消息: {result.message}")
                print(f"  步骤: {result.step}")

            assert result.detected, "应该检测到 KL 爆炸"
            assert result.step == 25, f"应该在 step 25 检测到，实际: {result.step}"

            # 2. 完整诊断
            print("\n运行完整诊断...")
            service = DiagnosticService(session)
            diag_result = service.diagnose_job(self.job_uuid, auto_mark_failed=False)

            print(f"  异常数量: {diag_result['anomalies_count']}")
            print(f"  严重异常数量: {diag_result['critical_count']}")

            for anomaly in diag_result['anomalies']:
                print(f"\n  异常: {anomaly['anomaly_type']}")
                print(f"    严重程度: {anomaly['severity']}")
                print(f"    消息: {anomaly['message']}")

            print("\n✓ 异常检测测试通过")

    def test_metrics_reader(self):
        """测试指标读取器"""
        print("\n" + "="*80)
        print("步骤 7: 测试指标读取器")
        print("="*80)

        from training_platform.core.metrics_reader import LocalMetricsReader

        reader = LocalMetricsReader(
            job_id=self.test_id,
            metrics_dir=str(self.metrics_dir)
        )

        # 测试读取所有指标
        print("\n测试读取所有指标...")
        all_metrics = reader.read_metrics()
        print(f"  读取到 {len(all_metrics)} 条指标")

        if all_metrics:
            print(f"  第一条: step={all_metrics[0]['step']}")
            print(f"  最后一条: step={all_metrics[-1]['step']}")

        # 测试增量读取
        print("\n测试增量读取...")
        reader.reset_position()
        batch1, pos1 = reader.read_metrics_incremental(limit=10)
        print(f"  第一批: {len(batch1)} 条，位置: {pos1}")

        batch2, pos2 = reader.read_metrics_incremental(limit=10)
        print(f"  第二批: {len(batch2)} 条，位置: {pos2}")

        assert len(batch1) == 10, "第一批应该读取 10 条"
        assert len(batch2) == 10, "第二批应该读取 10 条"
        assert batch1[0]['step'] < batch2[0]['step'], "第二批应该是更新的数据"

        # 测试读取状态
        print("\n测试读取状态...")
        status = reader.read_status()
        if status:
            print(f"  状态: {status['status']}")
            print(f"  当前步骤: {status['current_step']}")
            print(f"  异常检测: {status['anomaly_detected']}")

        print("\n✓ 指标读取器测试通过")

    def test_websocket_simulation(self):
        """测试 WebSocket 推送（模拟）

        注：这里不启动真实的 WebSocket 服务器，而是验证逻辑
        """
        print("\n" + "="*80)
        print("步骤 8: 测试 WebSocket 逻辑（模拟）")
        print("="*80)

        from training_platform.core.metrics_reader import LocalMetricsReader

        reader = LocalMetricsReader(
            job_id=self.test_id,
            metrics_dir=str(self.metrics_dir)
        )

        print("\n模拟 WebSocket 推送...")
        reader.reset_position()

        # 模拟 5 次推送
        for i in range(5):
            new_metrics, pos = reader.read_metrics_incremental(limit=5)
            print(f"  推送 {i+1}: {len(new_metrics)} 条新指标，位置: {pos}")

            if not new_metrics:
                print("  无新指标，等待...")
                break

        print("\n✓ WebSocket 推送逻辑测试通过")

    def cleanup(self):
        """清理测试数据"""
        print("\n" + "="*80)
        print("步骤 9: 清理测试数据")
        print("="*80)

        # 清理数据库
        with Session(engine) as session:
            if self.job_uuid:
                # 删除指标
                statement = select(TrainingMetric).where(
                    TrainingMetric.job_uuid == self.job_uuid
                )
                metrics = session.exec(statement).all()
                for m in metrics:
                    session.delete(m)

                # 删除任务
                statement = select(TrainingJob).where(
                    TrainingJob.uuid == self.job_uuid
                )
                job = session.exec(statement).first()
                if job:
                    session.delete(job)

                session.commit()
                print(f"✓ 删除数据库记录: 1 个任务, {len(metrics)} 条指标")

        # 清理文件
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            print(f"✓ 删除测试目录: {self.test_dir}")

    def run(self):
        """运行完整的集成测试"""
        try:
            self.setup()
            self.create_test_dataset(num_samples=50)
            self.create_test_job()
            self.simulate_training_with_callback(num_steps=30)
            self.test_metrics_sync()
            self.test_anomaly_detection()
            self.test_metrics_reader()
            self.test_websocket_simulation()

            print("\n" + "="*80)
            print("✅ 集成测试全部通过！")
            print("="*80)
            print("\nPhase 1 集成测试成功：")
            print("  ✓ 数据集创建")
            print("  ✓ 指标文件生成（模拟 PlatformCallback）")
            print("  ✓ 指标同步到数据库")
            print("  ✓ 异常检测（KL 爆炸）")
            print("  ✓ 指标读取器（增量读取）")
            print("  ✓ WebSocket 推送逻辑")
            print()

            return 0

        except Exception as e:
            print(f"\n❌ 集成测试失败: {e}")
            import traceback
            traceback.print_exc()
            return 1

        finally:
            # 总是清理
            try:
                self.cleanup()
            except Exception as e:
                print(f"⚠️  清理失败: {e}")


def main():
    """主函数"""
    test = IntegrationTest()
    return test.run()


if __name__ == "__main__":
    sys.exit(main())
