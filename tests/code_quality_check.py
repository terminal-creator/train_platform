"""
代码质量检查脚本

检查 Phase 2 代码的质量问题
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """检查各模块是否可以正常导入"""
    print("\n" + "=" * 80)
    print("1. 检查模块导入")
    print("=" * 80)

    modules = [
        "training_platform.core.recipes",
        "training_platform.core.config_diff",
        "training_platform.core.dataset_version",
        "training_platform.core.experience_reuse",
        "training_platform.api.routers.recipes",
        "training_platform.api.routers.config_diff",
        "training_platform.api.routers.dataset_version",
        "training_platform.api.routers.experience",
    ]

    all_ok = True
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            all_ok = False

    return all_ok


def check_docstrings():
    """检查关键函数是否有文档字符串"""
    print("\n" + "=" * 80)
    print("2. 检查文档字符串")
    print("=" * 80)

    from training_platform.core import recipes, config_diff, dataset_version, experience_reuse

    # 检查核心函数
    functions_to_check = [
        (recipes, "apply_recipe_to_job_config"),
        (recipes, "validate_recipe_config"),
        (config_diff, "compare_configs"),
        (config_diff, "format_diff_report"),
        (dataset_version, "calculate_file_hash"),
        (dataset_version, "create_dataset_snapshot"),
        (experience_reuse, "clone_job_config"),
        (experience_reuse, "recommend_successful_recipes"),
    ]

    all_ok = True
    for module, func_name in functions_to_check:
        func = getattr(module, func_name, None)
        if func and func.__doc__:
            print(f"✓ {module.__name__}.{func_name}: 有文档")
        else:
            print(f"✗ {module.__name__}.{func_name}: 缺少文档")
            all_ok = False

    return all_ok


def check_error_handling():
    """检查错误处理"""
    print("\n" + "=" * 80)
    print("3. 检查错误处理")
    print("=" * 80)

    from training_platform.core.dataset_version import calculate_file_hash
    from training_platform.core.recipes import RecipeRegistry

    # 测试文件不存在的情况
    try:
        calculate_file_hash("/nonexistent/file.txt")
        print("✗ calculate_file_hash 没有正确处理文件不存在的情况")
        return False
    except FileNotFoundError:
        print("✓ calculate_file_hash 正确抛出 FileNotFoundError")

    # 测试获取不存在的配方
    recipe = RecipeRegistry.get("nonexistent_recipe")
    if recipe is None:
        print("✓ RecipeRegistry.get 正确返回 None")
    else:
        print("✗ RecipeRegistry.get 应该返回 None")
        return False

    return True


def check_type_hints():
    """检查类型提示"""
    print("\n" + "=" * 80)
    print("4. 检查类型提示")
    print("=" * 80)

    import inspect
    from training_platform.core import recipes, config_diff

    # 检查关键函数的类型提示
    functions = [
        recipes.apply_recipe_to_job_config,
        recipes.validate_recipe_config,
        config_diff.compare_configs,
    ]

    all_ok = True
    for func in functions:
        sig = inspect.signature(func)
        has_hints = any(param.annotation != inspect.Parameter.empty
                       for param in sig.parameters.values())
        has_return = sig.return_annotation != inspect.Signature.empty

        if has_hints and has_return:
            print(f"✓ {func.__module__}.{func.__name__}: 有完整类型提示")
        else:
            print(f"⚠️  {func.__module__}.{func.__name__}: 缺少部分类型提示")
            # 不算作失败，只是警告

    return all_ok


def check_code_organization():
    """检查代码组织"""
    print("\n" + "=" * 80)
    print("5. 检查代码组织")
    print("=" * 80)

    from training_platform.core.recipes import RecipeRegistry

    # 检查配方数量
    recipes = RecipeRegistry.list_all()
    print(f"✓ 注册了 {len(recipes)} 个配方")

    if len(recipes) >= 9:
        print(f"✓ 配方数量充足")
    else:
        print(f"⚠️  配方数量偏少")

    # 检查配方是否有标签
    recipes_without_tags = [r for r in recipes if not r.get("tags")]
    if recipes_without_tags:
        print(f"⚠️  {len(recipes_without_tags)} 个配方缺少标签")
    else:
        print(f"✓ 所有配方都有标签")

    return True


def check_database_models():
    """检查数据库模型"""
    print("\n" + "=" * 80)
    print("6. 检查数据库模型")
    print("=" * 80)

    from training_platform.core.database import TrainingJob, DatasetVersion

    # 检查 TrainingJob 字段
    job_fields = TrainingJob.model_fields
    if "recipe_id" in job_fields:
        print("✓ TrainingJob 有 recipe_id 字段")
    else:
        print("✗ TrainingJob 缺少 recipe_id 字段")
        return False

    if "dataset_version_hash" in job_fields:
        print("✓ TrainingJob 有 dataset_version_hash 字段")
    else:
        print("✗ TrainingJob 缺少 dataset_version_hash 字段")
        return False

    # 检查 DatasetVersion 表
    version_fields = DatasetVersion.model_fields
    required_fields = ["file_hash", "dataset_name", "file_size", "format"]
    for field in required_fields:
        if field in version_fields:
            print(f"✓ DatasetVersion 有 {field} 字段")
        else:
            print(f"✗ DatasetVersion 缺少 {field} 字段")
            return False

    return True


def check_api_endpoints():
    """检查 API 端点"""
    print("\n" + "=" * 80)
    print("7. 检查 API 端点")
    print("=" * 80)

    from training_platform.api.main import app

    # 统计路由数量
    routes = [r for r in app.routes if hasattr(r, 'methods')]
    print(f"✓ 总共 {len(routes)} 个路由")

    # 检查 Phase 2 相关路由
    phase2_prefixes = ["/api/v1/recipes", "/api/v1/config-diff",
                       "/api/v1/dataset-versions", "/api/v1/experience"]

    for prefix in phase2_prefixes:
        matching_routes = [r for r in routes if hasattr(r, 'path') and r.path.startswith(prefix)]
        if matching_routes:
            print(f"✓ {prefix}: {len(matching_routes)} 个端点")
        else:
            print(f"✗ {prefix}: 没有找到端点")
            return False

    return True


def run_all_checks():
    """运行所有检查"""
    print("\n" + "=" * 80)
    print("Phase 2 代码质量检查")
    print("=" * 80)

    checks = [
        ("模块导入", check_imports),
        ("文档字符串", check_docstrings),
        ("错误处理", check_error_handling),
        ("类型提示", check_type_hints),
        ("代码组织", check_code_organization),
        ("数据库模型", check_database_models),
        ("API 端点", check_api_endpoints),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 检查失败: {e}")
            results.append((name, False))

    # 打印总结
    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 代码质量检查全部通过！")
        return True
    else:
        print(f"\n⚠️  {total - passed} 项检查失败")
        return False


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
