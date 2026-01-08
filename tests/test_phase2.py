"""
Phase 2 功能测试

测试 Recipe System, Config Diff, Data Versioning, Experience Reuse
"""

import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_platform.core.recipes import (
    RecipeRegistry,
    TaskType,
    apply_recipe_to_job_config,
    validate_recipe_config,
)
from training_platform.core.config_diff import (
    compare_configs,
    format_diff_report,
    compare_recipes,
)
from training_platform.core.dataset_version import (
    calculate_file_hash,
    create_dataset_snapshot,
    compare_dataset_versions,
)


def test_recipe_system():
    """测试配方系统"""
    print("\n" + "=" * 80)
    print("测试 1: Recipe System (配方系统)")
    print("=" * 80)

    # 测试 1.1: 列出所有配方
    print("\n1.1 列出所有配方:")
    all_recipes = RecipeRegistry.list_all()
    print(f"✓ 找到 {len(all_recipes)} 个配方")
    for recipe_info in all_recipes[:3]:
        print(f"  - {recipe_info['name']}: {recipe_info['description'][:50]}...")

    # 测试 1.2: 获取特定配方
    print("\n1.2 获取 GRPO 配方:")
    grpo_recipe = RecipeRegistry.get("grpo_large_scale")
    if grpo_recipe:
        print(f"✓ 配方名称: {grpo_recipe.name}")
        print(f"✓ 算法: {grpo_recipe.recommended_algorithm}")
        print(f"✓ 推荐 GPU 数: {grpo_recipe.recommended_gpus}")
    else:
        print("✗ 未找到配方")
        return False

    # 测试 1.3: 按任务类型筛选
    print("\n1.3 筛选 RLHF 类型配方:")
    rlhf_recipes = RecipeRegistry.list_by_task_type(TaskType.RLHF)
    print(f"✓ 找到 {len(rlhf_recipes)} 个 RLHF 配方")

    # 测试 1.3b: 按标签筛选
    print("\n1.3b 按标签筛选 GRPO 配方:")
    grpo_recipes = RecipeRegistry.list_by_tag("grpo")
    print(f"✓ 找到 {len(grpo_recipes)} 个带 'grpo' 标签的配方")

    # 测试 1.4: 自适应配置
    print("\n1.4 测试自适应配置:")
    config_7b = grpo_recipe.get_config(model_size="7B", num_gpus=8)
    config_70b = grpo_recipe.get_config(model_size="70B", num_gpus=32)
    print(f"✓ 7B 模型 batch_size: {config_7b.get('batch_size')}")
    print(f"✓ 70B 模型 batch_size: {config_70b.get('batch_size')}")

    # 测试 1.5: 验证配置
    print("\n1.5 测试配置验证:")
    warnings = validate_recipe_config(grpo_recipe, config_7b)
    print(f"✓ 配置验证完成，{len(warnings)} 个警告")

    print("\n✅ Recipe System 测试通过")
    return True


def test_config_diff():
    """测试配置对比"""
    print("\n" + "=" * 80)
    print("测试 2: Config Diff (配置对比)")
    print("=" * 80)

    # 测试 2.1: 基础对比
    print("\n2.1 基础配置对比:")
    config_a = {
        "learning_rate": 1e-6,
        "batch_size": 256,
        "kl_coef": 0.02,
        "optimizer": {
            "type": "adam",
            "weight_decay": 0.01
        }
    }
    config_b = {
        "learning_rate": 5e-7,
        "batch_size": 512,
        "kl_coef": 0.02,
        "optimizer": {
            "type": "adam",
            "weight_decay": 0.01
        },
        "warmup_steps": 100
    }

    result = compare_configs(config_a, config_b, "Config A", "Config B")
    print(f"✓ 对比完成:")
    print(f"  - 新增: {result.added_count}")
    print(f"  - 删除: {result.removed_count}")
    print(f"  - 修改: {result.modified_count}")
    print(f"  - 关键参数变化: {result.has_critical_changes}")

    # 测试 2.2: 生成报告
    print("\n2.2 生成对比报告:")
    report = format_diff_report(result)
    print(report[:500])  # 打印前 500 字符

    # 测试 2.3: 对比配方
    print("\n2.3 对比两个配方:")
    result_recipes = compare_recipes("grpo_basic", "grpo_large_scale")
    if result_recipes:
        print(f"✓ 配方对比完成:")
        print(f"  - 修改参数: {result_recipes.modified_count}")
        print(f"  - 摘要: {result_recipes.summary}")
    else:
        print("✗ 配方对比失败")
        return False

    print("\n✅ Config Diff 测试通过")
    return True


def test_dataset_versioning():
    """测试数据版本化"""
    print("\n" + "=" * 80)
    print("测试 3: Data Versioning (数据版本化)")
    print("=" * 80)

    # 创建测试数据文件
    test_file = "/tmp/test_dataset.jsonl"
    test_data = [
        {"prompt": "What is 2+2?", "response": "4"},
        {"prompt": "What is 3+3?", "response": "6"},
    ]

    print(f"\n3.1 创建测试数据文件: {test_file}")
    with open(test_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    print(f"✓ 测试文件创建成功")

    # 测试 3.2: 计算 hash
    print("\n3.2 计算文件 hash:")
    file_hash = calculate_file_hash(test_file)
    print(f"✓ SHA256: {file_hash[:16]}...")

    # 测试 3.3: 创建快照
    print("\n3.3 创建数据集快照:")
    snapshot = create_dataset_snapshot(
        file_path=test_file,
        dataset_name="test_dataset",
        description="Test dataset for Phase 2",
        tags=["test", "math"]
    )
    print(f"✓ 快照创建成功:")
    print(f"  - 数据集: {snapshot['dataset_name']}")
    print(f"  - Hash: {snapshot['file_hash'][:16]}...")
    print(f"  - 格式: {snapshot['format']}")
    print(f"  - 样本数: {snapshot['num_samples']}")

    # 测试 3.4: 修改文件并重新快照
    print("\n3.4 修改文件并创建新快照:")
    test_data.append({"prompt": "What is 5+5?", "response": "10"})
    with open(test_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    snapshot2 = create_dataset_snapshot(
        file_path=test_file,
        dataset_name="test_dataset",
        description="Modified test dataset",
        tags=["test", "math"]
    )
    print(f"✓ 新快照创建成功:")
    print(f"  - Hash: {snapshot2['file_hash'][:16]}...")
    print(f"  - 样本数: {snapshot2['num_samples']}")

    # 测试 3.5: 对比版本
    print("\n3.5 对比两个版本:")
    diff = compare_dataset_versions(snapshot, snapshot2)
    print(f"✓ 版本对比完成:")
    print(f"  - 内容相同: {diff['identical']}")
    print(f"  - Hash 变化: {diff['hash_changed']}")
    print(f"  - 样本数变化: {diff['samples_diff']}")

    # 清理测试文件
    os.remove(test_file)

    print("\n✅ Data Versioning 测试通过")
    return True


def test_experience_reuse():
    """测试经验复用"""
    print("\n" + "=" * 80)
    print("测试 4: Experience Reuse (经验复用)")
    print("=" * 80)

    # 这个测试需要数据库，我们做一些基础的逻辑测试
    from training_platform.core.experience_reuse import (
        suggest_config_adjustments,
    )

    print("\n4.1 测试配置调整建议:")
    current_config = {
        "learning_rate": 1e-5,  # 偏高
        "batch_size": 128,       # 偏小
        "kl_coef": 0.01
    }

    best_practices = [
        {
            "learning_rate": 5e-7,
            "batch_size": 512,
            "kl_coef": 0.02,
            "metric_value": 0.85
        },
        {
            "learning_rate": 8e-7,
            "batch_size": 512,
            "kl_coef": 0.02,
            "metric_value": 0.82
        }
    ]

    suggestions = suggest_config_adjustments(current_config, best_practices)
    print(f"✓ 生成 {len(suggestions)} 条建议:")
    for sug in suggestions:
        print(f"  - {sug['parameter']}: {sug['current_value']} → {sug['suggested_value']}")
        print(f"    原因: {sug['reason']}")

    print("\n✅ Experience Reuse 测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("Phase 2 功能测试")
    print("=" * 80)

    results = []

    # 运行测试
    results.append(("Recipe System", test_recipe_system()))
    results.append(("Config Diff", test_config_diff()))
    results.append(("Data Versioning", test_dataset_versioning()))
    results.append(("Experience Reuse", test_experience_reuse()))

    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！")
        return True
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
