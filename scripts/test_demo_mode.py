#!/usr/bin/env python
"""
Demo模式测试脚本

测试Demo模式的各项功能
"""
import json
import sys


def test_demo_imports():
    """测试Demo模块导入"""
    print("=" * 60)
    print("测试 1: Demo模块导入")
    print("=" * 60)

    try:
        from training_platform.demo import demo_settings, is_demo_mode
        from training_platform.demo.config import set_demo_mode
        from training_platform.demo.mock_data import (
            get_all_demo_jobs,
            get_demo_metrics,
            get_gradient_heatmap,
            get_demo_evaluation_results,
            get_all_demo_pipelines,
            get_demo_compute_result,
        )
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_demo_jobs():
    """测试Demo任务数据"""
    print("\n" + "=" * 60)
    print("测试 2: Demo任务数据")
    print("=" * 60)

    from training_platform.demo.mock_data import get_all_demo_jobs, get_demo_job

    jobs = get_all_demo_jobs()
    print(f"✓ 获取到 {len(jobs)} 个Demo任务")

    for job in jobs:
        status_icon = {"running": "🔄", "completed": "✅", "pending": "⏳"}.get(
            job["status"], "❓"
        )
        print(f"  {status_icon} {job['name']} ({job['algorithm'].upper()}) - {job['status']}")
        if job["status"] == "running":
            print(f"     进度: {job['current_step']}/{job['total_steps']} ({job['progress']}%)")

    # 测试单个任务获取
    job = get_demo_job("demo-grpo-qwen7b-math-002")
    if job:
        print(f"✓ 成功获取GRPO任务详情")
        print(f"  最新指标: reward={job['latest_metrics'].get('reward_mean', 'N/A')}")
    else:
        print("✗ 获取GRPO任务失败")

    return len(jobs) > 0


def test_demo_metrics():
    """测试Demo指标数据"""
    print("\n" + "=" * 60)
    print("测试 3: Demo训练指标")
    print("=" * 60)

    from training_platform.demo.mock_data import get_demo_metrics, get_metrics_summary

    job_id = "demo-grpo-qwen7b-math-002"
    metrics = get_demo_metrics(job_id)
    print(f"✓ 获取到 {len(metrics)} 条指标记录")

    if metrics:
        first = metrics[0]
        last = metrics[-1]
        print(f"  起始 (step {first['step']}): reward={first.get('reward_mean', 'N/A')}")
        print(f"  最新 (step {last['step']}): reward={last.get('reward_mean', 'N/A')}")

    summary = get_metrics_summary(job_id)
    if summary:
        print(f"✓ 指标汇总: {summary}")

    return len(metrics) > 0


def test_gradient_heatmap():
    """测试梯度热力图"""
    print("\n" + "=" * 60)
    print("测试 4: 梯度热力图")
    print("=" * 60)

    from training_platform.demo.mock_data import get_gradient_heatmap, get_gradient_health_report

    job_id = "demo-grpo-qwen7b-math-002"
    heatmap = get_gradient_heatmap(job_id)

    print(f"✓ 热力图层数: {len(heatmap['layers'])}")
    print(f"✓ 热力图步数: {len(heatmap['steps'])}")
    print(f"✓ 数值范围: {heatmap['value_range']}")

    health = get_gradient_health_report(job_id)
    print(f"✓ 梯度健康状态: {health['status']}")
    if health['recommendations']:
        print(f"  建议: {health['recommendations'][0]}")

    return len(heatmap['layers']) > 0


def test_evaluation_comparison():
    """测试评估对比"""
    print("\n" + "=" * 60)
    print("测试 5: 评估对比")
    print("=" * 60)

    from training_platform.demo.mock_data import get_evaluation_comparison

    comparison = get_evaluation_comparison()

    print(f"✓ 对比模型数: {len(comparison['models'])}")
    print(f"✓ Benchmark数: {len(comparison['benchmarks'])}")

    print("\n  模型能力对比:")
    benchmarks = comparison['benchmarks']
    for bm_name, scores in benchmarks.items():
        print(f"  {bm_name}:")
        for model_id, score in scores.items():
            if model_id == "baseline":
                print(f"    基座模型: {score*100:.1f}%")
            elif "grpo" in model_id:
                print(f"    GRPO模型: {score*100:.1f}%")

    return len(comparison['models']) > 0


def test_compute_calculator():
    """测试计算配置"""
    print("\n" + "=" * 60)
    print("测试 6: 智能计算配置")
    print("=" * 60)

    from training_platform.demo.mock_data import get_demo_compute_result

    result = get_demo_compute_result(
        model_size="7B",
        gpu_type="A100-80G",
        gpu_count=8,
        training_type="grpo",
    )

    summary = result["summary"]
    memory = result["memory_estimate"]

    print(f"✓ 模型: {summary['model_size']}")
    print(f"✓ GPU: {summary['gpu_count']}x {summary['gpu_type']}")
    print(f"✓ 批量大小: micro={summary['micro_batch_size']}, global={summary['global_batch_size']}")
    print(f"✓ ZeRO Stage: {summary['zero_stage']}")
    print(f"✓ 内存估算: {memory['per_gpu_gb']:.1f}GB / {memory['available_gpu_memory_gb']}GB")

    if summary['recommendations']:
        print(f"\n  优化建议:")
        for rec in summary['recommendations'][:3]:
            print(f"    - {rec}")

    return "config" in result


def test_pipelines():
    """测试流水线数据"""
    print("\n" + "=" * 60)
    print("测试 7: 训练流水线")
    print("=" * 60)

    from training_platform.demo.mock_data import get_all_demo_pipelines, get_pipeline_templates

    pipelines = get_all_demo_pipelines()
    print(f"✓ 流水线数量: {len(pipelines)}")

    for p in pipelines:
        status_icon = {"running": "🔄", "completed": "✅"}.get(p["status"], "❓")
        print(f"  {status_icon} {p['name']}")
        print(f"     状态: {p['status']}, 进度: {p['progress']}%")
        print(f"     阶段: {p['current_stage']}/{p['total_stages']}")

    templates = get_pipeline_templates()
    print(f"\n✓ 流水线模板: {len(templates)} 个")
    for t in templates:
        print(f"  - {t['name']}")

    return len(pipelines) > 0


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("      Demo模式功能测试")
    print("=" * 60)

    tests = [
        ("模块导入", test_demo_imports),
        ("任务数据", test_demo_jobs),
        ("训练指标", test_demo_metrics),
        ("梯度热力图", test_gradient_heatmap),
        ("评估对比", test_evaluation_comparison),
        ("计算配置", test_compute_calculator),
        ("训练流水线", test_pipelines),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ 测试 '{name}' 异常: {e}")
            results.append((name, False))

    # 汇总
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)

    passed = sum(1 for _, s in results if s)
    total = len(results)

    for name, success in results:
        icon = "✓" if success else "✗"
        print(f"  {icon} {name}")

    print(f"\n结果: {passed}/{total} 测试通过")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
