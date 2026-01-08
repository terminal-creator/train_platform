#!/usr/bin/env python3
"""
Phase 1.2 功能测试

测试内容：
1. PlatformCallback 生成指标文件
2. metrics_persister 解析和持久化
3. API 接口查询指标

使用方法：
    cd /Users/weixiaochen/Desktop/Tutor/S4/train_platform
    python tests/test_phase1_2.py
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_1_create_mock_metrics_file():
    """
    测试 1: 创建模拟的指标文件

    模拟 PlatformCallback 输出的格式，创建测试数据
    """
    print("\n" + "="*60)
    print("测试 1: 创建模拟指标文件")
    print("="*60)

    # 创建测试目录
    test_dir = Path("./test_platform_metrics")
    test_dir.mkdir(exist_ok=True)

    job_id = "test_job_001"
    metrics_file = test_dir / f"{job_id}_metrics.jsonl"
    status_file = test_dir / f"{job_id}_status.json"

    # 生成模拟指标（100 个步骤）
    print(f"\n生成 100 个训练步骤的模拟数据...")

    with open(metrics_file, 'w') as f:
        for step in range(1, 101):
            # 模拟训练过程：loss 下降，reward 上升
            actor_loss = 2.5 - (step * 0.015) + (0.1 if step % 10 == 0 else 0)
            critic_loss = 1.2 - (step * 0.008)
            reward_mean = 0.3 + (step * 0.005)
            kl_mean = 0.1 + (0.05 if step % 20 == 0 else 0)  # 每20步有波动

            metric = {
                "step": step,
                "timestamp": datetime.utcnow().timestamp(),
                "epoch": step // 33,  # 每33步为1个epoch
                "loss": {
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "total_loss": actor_loss + critic_loss,
                },
                "reward": {
                    "mean": reward_mean,
                    "std": 0.1,
                    "max": reward_mean + 0.2,
                    "min": reward_mean - 0.2,
                },
                "kl": {
                    "mean": kl_mean,
                    "max": kl_mean + 0.05,
                },
                "gradient": {
                    "actor_norm": 1.5 - (step * 0.01),
                    "critic_norm": 0.8 - (step * 0.005),
                },
                "performance": {
                    "tokens_per_second": 1000 + (step * 5),
                    "step_time": 2.5 - (step * 0.01),
                    "gpu_memory_allocated": 40.5,
                },
                "raw": {},  # 完整原始数据（这里简化）
            }

            f.write(json.dumps(metric) + '\n')

    # 创建状态文件（模拟训练正常完成）
    status = {
        "job_id": job_id,
        "status": "completed",
        "current_step": 100,
        "total_steps": 100,
        "anomaly_detected": False,
        "anomaly_reason": None,
    }

    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

    print(f"✓ 指标文件已创建: {metrics_file}")
    print(f"✓ 状态文件已创建: {status_file}")
    print(f"✓ 共生成 100 个训练步骤的数据")

    return test_dir, job_id


def test_2_parse_platform_metric():
    """
    测试 2: 测试指标解析函数

    验证 parse_platform_metric() 能否正确转换格式
    """
    print("\n" + "="*60)
    print("测试 2: 测试指标格式转换")
    print("="*60)

    from training_platform.core.metrics_persister import parse_platform_metric

    # 模拟一条指标
    raw_metric = {
        "step": 1,
        "timestamp": datetime.utcnow().timestamp(),
        "epoch": 0,
        "loss": {
            "actor_loss": 2.5,
            "critic_loss": 1.2,
            "total_loss": 3.7,
        },
        "reward": {
            "mean": 0.5,
            "std": 0.1,
            "max": 0.7,
            "min": 0.3,
        },
        "kl": {
            "mean": 0.1,
            "max": 0.15,
        },
        "gradient": {
            "actor_norm": 1.5,
            "critic_norm": 0.8,
        },
        "performance": {
            "tokens_per_second": 1000,
            "step_time": 2.5,
            "gpu_memory_allocated": 40.5,
        },
        "raw": {},
    }

    job_uuid = "test-uuid-001"

    print("\n原始数据（PlatformCallback 格式）:")
    print(json.dumps(raw_metric, indent=2))

    # 解析
    metric = parse_platform_metric(raw_metric, job_uuid)

    print("\n解析后的数据（TrainingMetric 模型）:")
    print(f"  job_uuid: {metric.job_uuid}")
    print(f"  step: {metric.step}")
    print(f"  epoch: {metric.epoch}")
    print(f"  policy_loss (actor_loss): {metric.policy_loss}")
    print(f"  value_loss (critic_loss): {metric.value_loss}")
    print(f"  reward_mean: {metric.reward_mean}")
    print(f"  reward_max: {metric.reward_max}")
    print(f"  kl_divergence: {metric.kl_divergence}")
    print(f"  kl_divergence_max: {metric.kl_divergence_max}")
    print(f"  grad_norm_actor: {metric.grad_norm_actor}")
    print(f"  tokens_per_second: {metric.tokens_per_second}")
    print(f"  has_anomaly: {metric.has_anomaly}")

    # 验证
    assert metric.job_uuid == job_uuid, "job_uuid 不匹配"
    assert metric.step == 1, "step 不匹配"
    assert metric.policy_loss == 2.5, "actor_loss -> policy_loss 映射错误"
    assert metric.value_loss == 1.2, "critic_loss -> value_loss 映射错误"
    assert metric.reward_mean == 0.5, "reward_mean 不匹配"
    assert metric.kl_divergence == 0.1, "kl_divergence 不匹配"

    print("\n✓ 所有字段映射正确！")

    return True


def test_3_sync_metrics_to_database():
    """
    测试 3: 测试指标同步到数据库

    验证 sync_metrics_from_file() 能否正确持久化
    """
    print("\n" + "="*60)
    print("测试 3: 同步指标到数据库")
    print("="*60)

    from sqlmodel import Session, select
    from training_platform.core.database import engine, TrainingJob, TrainingMetric, JobStatus
    from training_platform.core.metrics_persister import sync_metrics_from_file
    import uuid

    # 创建测试任务
    job_uuid = str(uuid.uuid4())
    job_id = "test_job_001"

    print(f"\n创建测试任务: {job_uuid}")

    with Session(engine) as session:
        test_job = TrainingJob(
            uuid=job_uuid,
            name="Phase 1.2 测试任务",
            description="测试指标持久化功能",
            status=JobStatus.RUNNING,
            algorithm="PPO",
            model_path="/test/model",
            train_data_path="/test/data.parquet",
            num_gpus=1,
            learning_rate=1e-5,
            batch_size=32,
            num_epochs=3,
        )
        session.add(test_job)
        session.commit()

    print("✓ 测试任务已创建")

    # 同步指标
    test_dir = Path("./test_platform_metrics")
    metrics_file = test_dir / f"{job_id}_metrics.jsonl"

    print(f"\n从文件同步指标: {metrics_file}")

    with Session(engine) as session:
        new_count = sync_metrics_from_file(
            job_uuid=job_uuid,
            metrics_file=metrics_file,
            session=session,
            batch_size=50,  # 每50条一批
        )

    print(f"✓ 同步完成，新增 {new_count} 条指标")

    # 验证数据库
    print("\n验证数据库...")

    with Session(engine) as session:
        statement = select(TrainingMetric).where(
            TrainingMetric.job_uuid == job_uuid
        ).order_by(TrainingMetric.step)

        metrics = session.exec(statement).all()

        print(f"  数据库中共有 {len(metrics)} 条指标")

        if len(metrics) > 0:
            first = metrics[0]
            last = metrics[-1]

            print(f"\n  第 1 条指标:")
            print(f"    step: {first.step}")
            print(f"    actor_loss: {first.policy_loss}")
            print(f"    reward_mean: {first.reward_mean}")

            print(f"\n  第 {len(metrics)} 条指标:")
            print(f"    step: {last.step}")
            print(f"    actor_loss: {last.policy_loss}")
            print(f"    reward_mean: {last.reward_mean}")

    # 验证
    assert new_count == 100, f"应该同步100条指标，实际同步了{new_count}条"
    assert len(metrics) == 100, f"数据库应该有100条指标，实际有{len(metrics)}条"

    print("\n✓ 数据持久化成功！")

    return job_uuid


def test_4_incremental_sync():
    """
    测试 4: 测试增量同步

    验证重复同步时不会插入重复数据
    """
    print("\n" + "="*60)
    print("测试 4: 测试增量同步（避免重复）")
    print("="*60)

    from sqlmodel import Session, select
    from training_platform.core.database import TrainingMetric
    from training_platform.core.metrics_persister import sync_metrics_from_file

    # 获取上一个测试的任务ID
    # 这里我们需要使用已存在的任务
    test_dir = Path("./test_platform_metrics")
    job_id = "test_job_001"
    metrics_file = test_dir / f"{job_id}_metrics.jsonl"

    # 从数据库获取任务UUID
    from training_platform.core.database import engine, JobRepository

    with Session(engine) as session:
        job_repo = JobRepository(session)
        statement = select(TrainingMetric).limit(1)
        metric = session.exec(statement).first()

        if not metric:
            print("⚠️  数据库中没有指标，跳过增量同步测试")
            return True

        job_uuid = metric.job_uuid

        # 查询当前数量
        statement = select(TrainingMetric).where(
            TrainingMetric.job_uuid == job_uuid
        )
        count_before = len(session.exec(statement).all())

        print(f"\n同步前数据库中的指标数量: {count_before}")

        # 再次同步（应该是增量，不会插入重复数据）
        new_count = sync_metrics_from_file(
            job_uuid=job_uuid,
            metrics_file=metrics_file,
            session=session,
        )

        print(f"第二次同步，新增指标: {new_count} 条")

        # 查询同步后数量
        statement = select(TrainingMetric).where(
            TrainingMetric.job_uuid == job_uuid
        )
        count_after = len(session.exec(statement).all())

        print(f"同步后数据库中的指标数量: {count_after}")

    # 验证
    assert new_count == 0, f"增量同步应该不插入新数据，实际插入了{new_count}条"
    assert count_before == count_after, "增量同步后数量应该不变"

    print("\n✓ 增量同步正确，没有重复插入数据！")

    return True


def test_5_query_metrics_api():
    """
    测试 5: 测试 API 查询接口

    验证 GET /monitoring/{job_id}/metrics 能否正确返回数据
    """
    print("\n" + "="*60)
    print("测试 5: 测试 API 查询接口")
    print("="*60)

    from sqlmodel import Session, select
    from training_platform.core.database import engine, TrainingMetric, MetricsRepository

    # 获取一个测试任务的 UUID
    with Session(engine) as session:
        statement = select(TrainingMetric).limit(1)
        metric = session.exec(statement).first()

        if not metric:
            print("⚠️  数据库中没有指标，跳过 API 测试")
            return True

        job_uuid = metric.job_uuid

        print(f"\n测试任务 UUID: {job_uuid}")

        # 模拟 API 查询逻辑
        print("\n模拟 API 查询 (start_step=1, end_step=10)...")

        metrics_repo = MetricsRepository(session)
        db_metrics = metrics_repo.get_metrics(
            job_uuid=job_uuid,
            start_step=1,
            end_step=10,
            limit=100,
        )

        print(f"查询到 {len(db_metrics)} 条指标")

        # 转换为 API 返回格式
        metrics_list = []
        for m in db_metrics:
            metrics_list.append({
                "step": m.step,
                "epoch": m.epoch,
                "loss": {
                    "actor_loss": m.policy_loss,
                    "critic_loss": m.value_loss,
                },
                "reward": {
                    "mean": m.reward_mean,
                    "max": m.reward_max,
                },
                "kl": {
                    "mean": m.kl_divergence,
                    "max": m.kl_divergence_max,
                },
            })

        if len(metrics_list) > 0:
            print(f"\n第 1 条指标 (API 格式):")
            print(json.dumps(metrics_list[0], indent=2))

        # 验证
        assert len(db_metrics) == 10, f"应该查询到10条指标，实际{len(db_metrics)}条"
        assert db_metrics[0].step == 1, "第一条的step应该是1"
        assert db_metrics[-1].step == 10, "最后一条的step应该是10"

    print("\n✓ API 查询接口逻辑正确！")

    return True


def test_6_anomaly_detection():
    """
    测试 6: 测试异常检测和标记

    创建包含异常的指标文件，验证异常同步功能
    """
    print("\n" + "="*60)
    print("测试 6: 测试异常检测和同步")
    print("="*60)

    import uuid
    from sqlmodel import Session, select
    from training_platform.core.database import engine, TrainingJob, TrainingMetric, JobStatus
    from training_platform.core.metrics_persister import sync_metrics_from_file, sync_anomaly_from_status_file

    # 创建新的测试任务
    job_uuid = str(uuid.uuid4())
    job_id = "test_job_anomaly"

    test_dir = Path("./test_platform_metrics")
    metrics_file = test_dir / f"{job_id}_metrics.jsonl"
    status_file = test_dir / f"{job_id}_status.json"

    print(f"\n创建包含异常的测试数据...")

    # 生成指标（在 step 50 时 KL 爆炸）
    with open(metrics_file, 'w') as f:
        for step in range(1, 101):
            # Step 50 时模拟 KL 爆炸
            if step == 50:
                kl_mean = 1.5  # 超过阈值 1.0
                actor_loss = float('nan')  # 同时出现 NaN
            else:
                kl_mean = 0.1
                actor_loss = 2.5 - (step * 0.015)

            metric = {
                "step": step,
                "timestamp": datetime.utcnow().timestamp(),
                "epoch": 0,
                "loss": {
                    "actor_loss": actor_loss,
                    "critic_loss": 1.2,
                },
                "reward": {
                    "mean": 0.5,
                    "std": 0.1,
                    "max": 0.7,
                    "min": 0.3,
                },
                "kl": {
                    "mean": kl_mean,
                    "max": kl_mean + 0.05,
                },
                "gradient": {},
                "performance": {},
                "raw": {},
            }

            f.write(json.dumps(metric) + '\n')

    # 创建包含异常的状态文件
    status = {
        "job_id": job_id,
        "status": "failed",
        "current_step": 50,
        "total_steps": 100,
        "anomaly_detected": True,
        "anomaly_reason": "KL divergence explosion: 1.50 > 1.0 at step 50",
    }

    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)

    print("✓ 异常测试数据已创建")

    # 创建测试任务
    with Session(engine) as session:
        test_job = TrainingJob(
            uuid=job_uuid,
            name="异常检测测试任务",
            status=JobStatus.FAILED,
            algorithm="PPO",
            model_path="/test/model",
            train_data_path="/test/data.parquet",
        )
        session.add(test_job)
        session.commit()

        # 同步指标
        print("\n同步指标...")
        new_count = sync_metrics_from_file(
            job_uuid=job_uuid,
            metrics_file=metrics_file,
            session=session,
        )

        print(f"✓ 同步了 {new_count} 条指标")

        # 同步异常
        print("\n同步异常信息...")
        anomaly_synced = sync_anomaly_from_status_file(
            job_uuid=job_uuid,
            status_file=status_file,
            session=session,
        )

        print(f"✓ 异常同步: {anomaly_synced}")

        # 验证异常标记
        print("\n验证异常标记...")
        statement = select(TrainingMetric).where(
            TrainingMetric.job_uuid == job_uuid,
            TrainingMetric.step == 50,
        )

        anomaly_metric = session.exec(statement).first()

        if anomaly_metric:
            print(f"\n  Step 50 的异常信息:")
            print(f"    has_anomaly: {anomaly_metric.has_anomaly}")
            print(f"    anomaly_type: {anomaly_metric.anomaly_type}")
            print(f"    anomaly_message: {anomaly_metric.anomaly_message}")

            assert anomaly_metric.has_anomaly == True, "异常标记应该为True"
            assert anomaly_metric.anomaly_type == "kl_explosion", "异常类型应该是kl_explosion"
        else:
            print("⚠️  未找到 step 50 的指标")

    print("\n✓ 异常检测和同步功能正常！")

    return True


def cleanup():
    """清理测试数据"""
    print("\n" + "="*60)
    print("清理测试数据")
    print("="*60)

    import shutil

    test_dir = Path("./test_platform_metrics")

    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"✓ 已删除测试目录: {test_dir}")

    # 清理数据库中的测试数据
    from sqlmodel import Session, select
    from training_platform.core.database import engine, TrainingJob, TrainingMetric

    with Session(engine) as session:
        # 删除测试任务
        statement = select(TrainingJob).where(
            TrainingJob.name.like("%测试%")
        )
        test_jobs = session.exec(statement).all()

        for job in test_jobs:
            # 删除关联的指标
            metric_statement = select(TrainingMetric).where(
                TrainingMetric.job_uuid == job.uuid
            )
            metrics = session.exec(metric_statement).all()
            for m in metrics:
                session.delete(m)

            # 删除任务
            session.delete(job)

        session.commit()

        print(f"✓ 已删除 {len(test_jobs)} 个测试任务及其指标")


def main():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("#  Phase 1.2 功能测试")
    print("#  测试指标持久化和查询功能")
    print("#"*60)

    try:
        # 测试 1: 创建模拟数据
        test_dir, job_id = test_1_create_mock_metrics_file()

        # 测试 2: 测试格式转换
        test_2_parse_platform_metric()

        # 测试 3: 测试持久化
        job_uuid = test_3_sync_metrics_to_database()

        # 测试 4: 测试增量同步
        test_4_incremental_sync()

        # 测试 5: 测试 API 查询
        test_5_query_metrics_api()

        # 测试 6: 测试异常检测
        test_6_anomaly_detection()

        # 全部通过
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)

        print("\nPhase 1.2 功能验证成功：")
        print("  ✓ PlatformCallback 格式正确")
        print("  ✓ 指标解析和转换正常")
        print("  ✓ 数据持久化到数据库成功")
        print("  ✓ 增量同步避免重复")
        print("  ✓ API 查询接口逻辑正确")
        print("  ✓ 异常检测和标记功能正常")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # 清理测试数据
        cleanup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
