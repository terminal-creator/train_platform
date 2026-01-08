# 测试指南

本文档说明如何运行 Training Platform 的测试套件。

---

## 测试概览

项目包含两个主要测试套件：

1. **功能测试** (`tests/test_phase2.py`)
   - 测试 Phase 2 的所有核心功能
   - 包括 Recipe System, Config Diff, Data Versioning, Experience Reuse

2. **代码质量检查** (`tests/code_quality_check.py`)
   - 验证代码质量和最佳实践
   - 检查模块导入、文档字符串、错误处理等

---

## 快速开始

### 运行所有测试

```bash
# 进入项目目录
cd /path/to/train_platform

# 运行功能测试
python tests/test_phase2.py

# 运行代码质量检查
python tests/code_quality_check.py
```

### 预期输出

**功能测试通过**:
```
================================================================================
测试总结
================================================================================
Recipe System: ✅ 通过
Config Diff: ✅ 通过
Data Versioning: ✅ 通过
Experience Reuse: ✅ 通过

总计: 4/4 通过

🎉 所有测试通过！
```

**代码质量检查通过**:
```
================================================================================
检查总结
================================================================================
模块导入: ✅ 通过
文档字符串: ✅ 通过
错误处理: ✅ 通过
类型提示: ✅ 通过
代码组织: ✅ 通过
数据库模型: ✅ 通过
API 端点: ✅ 通过

总计: 7/7 通过

🎉 代码质量检查全部通过！
```

---

## 详细测试说明

### 1. 功能测试 (`test_phase2.py`)

#### 测试 1: Recipe System (配方系统)

测试配方的注册、获取、筛选和自适应配置功能。

**测试内容**:
- 列出所有配方
- 获取特定配方
- 按任务类型筛选
- 按标签筛选
- 自适应配置 (根据模型大小和 GPU 数量)
- 配置验证

**示例**:
```python
from training_platform.core.recipes import RecipeRegistry, TaskType

# 获取配方
recipe = RecipeRegistry.get("grpo_large_scale")

# 自适应配置
config = recipe.get_config(model_size="7B", num_gpus=8)
```

#### 测试 2: Config Diff (配置对比)

测试配置对比和差异报告生成功能。

**测试内容**:
- 基础配置对比
- 关键参数识别
- 生成人类可读的对比报告
- 对比配方

**示例**:
```python
from training_platform.core.config_diff import compare_configs

config_a = {"learning_rate": 1e-6, "batch_size": 256}
config_b = {"learning_rate": 5e-7, "batch_size": 512}

result = compare_configs(config_a, config_b)
print(result.summary)
```

#### 测试 3: Data Versioning (数据版本化)

测试数据集版本管理和血缘追踪功能。

**测试内容**:
- 计算文件 hash
- 创建数据集快照
- 修改文件并创建新快照
- 对比版本

**示例**:
```python
from training_platform.core.dataset_version import calculate_file_hash, create_dataset_snapshot

# 计算 hash
file_hash = calculate_file_hash("/path/to/dataset.jsonl")

# 创建快照
snapshot = create_dataset_snapshot(
    file_path="/path/to/dataset.jsonl",
    dataset_name="my_dataset",
    description="Test dataset",
    tags=["test"]
)
```

#### 测试 4: Experience Reuse (经验复用)

测试配置调整建议和经验复用功能。

**测试内容**:
- 生成配置调整建议
- 识别参数偏差
- 给出具体理由

**示例**:
```python
from training_platform.core.experience_reuse import suggest_config_adjustments

current_config = {"learning_rate": 1e-5, "batch_size": 128}
best_practices = [
    {"learning_rate": 5e-7, "batch_size": 512, "metric_value": 0.85}
]

suggestions = suggest_config_adjustments(current_config, best_practices)
```

---

### 2. 代码质量检查 (`code_quality_check.py`)

#### 检查 1: 模块导入

验证所有 Phase 2 模块可以正常导入。

**检查的模块**:
- `training_platform.core.recipes`
- `training_platform.core.config_diff`
- `training_platform.core.dataset_version`
- `training_platform.core.experience_reuse`
- API 路由模块

#### 检查 2: 文档字符串

验证关键函数都有完整的文档字符串。

**检查的函数**:
- `apply_recipe_to_job_config`
- `validate_recipe_config`
- `compare_configs`
- `calculate_file_hash`
- 等

#### 检查 3: 错误处理

验证异常情况的正确处理。

**测试场景**:
- 文件不存在
- 配方不存在
- 无效参数

#### 检查 4: 类型提示

验证关键函数有完整的类型提示。

#### 检查 5: 代码组织

验证配方数量、标签等组织结构。

#### 检查 6: 数据库模型

验证 Phase 2 的数据库模型字段。

**检查项目**:
- TrainingJob 有 recipe_id 字段
- TrainingJob 有 dataset_version_hash 字段
- DatasetVersion 表结构完整

#### 检查 7: API 端点

验证所有 Phase 2 API 端点正确注册。

**检查的端点**:
- `/api/v1/recipes` (6 个端点)
- `/api/v1/config-diff` (4 个端点)
- `/api/v1/dataset-versions` (5 个端点)
- `/api/v1/experience` (6 个端点)

---

## 测试环境

### 依赖要求

测试脚本需要以下依赖：

```bash
# 核心依赖
fastapi
sqlmodel
pydantic

# 可选依赖（用于 Parquet 支持）
pyarrow
```

### Python 版本

- Python 3.8+

---

## 常见问题

### Q: 测试失败怎么办？

A: 查看具体的错误信息，通常错误信息会指明问题所在。常见问题包括：
- 缺少依赖包
- 数据库未初始化
- 文件路径不正确

### Q: 如何跳过某些测试？

A: 可以修改测试脚本，注释掉不需要的测试函数调用。

### Q: 测试会修改数据库吗？

A: 功能测试只测试核心逻辑，不会修改数据库。数据库操作的测试需要完整的数据库环境。

### Q: 可以在 CI/CD 中运行这些测试吗？

A: 可以。测试脚本设计为可以在 CI/CD 环境中自动运行，返回值为 0 表示通过，非 0 表示失败。

---

## 扩展测试

### 添加新的测试用例

1. 在 `tests/test_phase2.py` 中添加新的测试函数
2. 在 `run_all_tests()` 中注册新测试
3. 运行测试验证

**示例**:
```python
def test_new_feature():
    """测试新功能"""
    print("\n测试 5: New Feature")
    # 测试代码
    return True

# 在 run_all_tests() 中添加
results.append(("New Feature", test_new_feature()))
```

### 集成测试

对于需要完整数据库和 API 环境的集成测试，建议使用 `pytest` 和 `httpx`:

```python
import pytest
from httpx import AsyncClient
from training_platform.api.main import app

@pytest.mark.asyncio
async def test_recipe_api():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/v1/recipes")
        assert response.status_code == 200
```

---

## 性能测试

### 基准测试

可以使用 `pytest-benchmark` 进行性能测试：

```bash
pip install pytest-benchmark

# 运行基准测试
pytest tests/benchmark_phase2.py --benchmark-only
```

### 负载测试

使用 `locust` 进行 API 负载测试：

```bash
pip install locust

# 启动负载测试
locust -f tests/load_test.py
```

---

## 测试报告

测试完成后，可以查看以下报告：

1. **功能测试报告**: 控制台输出
2. **代码质量报告**: 控制台输出
3. **详细测试报告**: `docs/PHASE2_TEST_REPORT.md`

---

## 持续集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python tests/test_phase2.py
          python tests/code_quality_check.py
```

---

## 贡献测试

欢迎贡献新的测试用例！请遵循以下步骤：

1. Fork 项目
2. 创建测试分支
3. 编写测试用例
4. 确保所有测试通过
5. 提交 Pull Request

---

**最后更新**: 2026-01-08
**测试覆盖率**: ~90%
**维护者**: Training Platform Team
