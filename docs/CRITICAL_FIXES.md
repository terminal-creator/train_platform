# 关键问题修复总结

基于代码复查，修复了三个会在真实生产环境翻车的关键问题：

---

## ✅ 修复 1: 完善 Stage task_id 入库和恢复机制

### 问题描述
- Pipeline root task_id 写入了 DB，但每个 stage 的 `celery_task_id` 没有记录
- Stage 的开始/结束时间没有更新
- Resume 机制不完整，可能重新执行已完成的 stages

### 修复方案

**1. 创建包装 Task (`execute_stage_with_tracking`)**

```python
@app.task(bind=True, name="training_platform.core.pipeline_executor.execute_stage_with_tracking")
def execute_stage_with_tracking(
    self,
    pipeline_uuid: str,
    stage_name: str,
    celery_task_name: str,
    task_params: Dict[str, Any],
):
    """
    包装 task，完整记录 stage 状态：
    1. 记录 celery_task_id (self.request.id)
    2. 更新状态为 RUNNING 并记录 started_at
    3. 执行实际的训练 task
    4. 更新状态为 COMPLETED/FAILED 并记录 completed_at
    """
```

**2. 修改 _create_stage_task()**

```python
# 之前：直接创建 task signature
task_sig = signature(celery_task_name, kwargs=node.params)

# 之后：使用包装 task
task_sig = signature(
    "training_platform.core.pipeline_executor.execute_stage_with_tracking",
    args=(self.pipeline_uuid, stage_name, celery_task_name, node.params),
)
```

**3. 完善 resume() 函数**

```python
def resume(self) -> Dict[str, Any]:
    """
    真正的恢复逻辑：
    1. 从 DB 读取所有 stages（包含 task_name, task_params, depends_on）
    2. 找出已完成的 stages（status == COMPLETED）
    3. 过滤掉已完成的 stages
    4. 调整依赖关系（移除已完成的依赖）
    5. 重新执行剩余 stages
    """
    # 从 DB 重建 stage 配置
    all_stages = []
    for db_stage in db_stages:
        stage_config = {
            "name": db_stage.stage_name,
            "task": db_stage.task_name,  # 从 DB 读取
            "params": db_stage.task_params,  # 从 DB 读取
            "depends_on": db_stage.depends_on,  # 从 DB 读取
        }
        all_stages.append(stage_config)

    # 过滤已完成的 stages
    remaining_stages = [
        stage for stage in all_stages
        if stage["name"] not in completed_stages
    ]

    # 调整依赖关系
    for stage in remaining_stages:
        stage["depends_on"] = [
            dep for dep in stage.get("depends_on", [])
            if dep not in completed_stages
        ]

    # 重新执行
    return self.execute(remaining_stages)
```

### 验证点

可以通过以下方式验证修复：

```python
# 1. 创建一个 pipeline with 并行层
stages = [
    {"name": "A", "task": "train_model", "params": {...}, "depends_on": []},
    {"name": "B", "task": "train_model", "params": {...}, "depends_on": ["A"]},
    {"name": "C", "task": "train_model", "params": {...}, "depends_on": ["A"]},  # 与 B 并行
    {"name": "D", "task": "run_evaluation", "params": {...}, "depends_on": ["B", "C"]},
]

executor = PipelineExecutor("test-pipeline")
executor.execute(stages)

# 2. 检查 DB
with Session(engine) as session:
    repo = PipelineRepository(session)
    stages = repo.get_stages("test-pipeline")

    for stage in stages:
        print(f"{stage.stage_name}:")
        print(f"  celery_task_id: {stage.celery_task_id}")  # 应该有值
        print(f"  started_at: {stage.started_at}")  # 应该有值
        print(f"  completed_at: {stage.completed_at}")  # COMPLETED 应该有值

# 3. 手动让 stage B 失败，然后 resume
executor.resume()

# 应该跳过 A（已完成），重新执行 B, C, D
```

---

## ✅ 修复 2: Metrics Persister 的 print() 和全文件读取

### 问题描述

**问题 A: 大量 print() 而不是 logger**
- 14 处 `print()` 调用
- 高频场景下会拖慢 worker、污染日志
- 不利于集中化日志系统

**问题 B: sync_metrics_from_file 读全文件**
- 每次都从头读取整个文件
- 文件越大越慢（O(n) 复杂度）
- 不适合高频轮询

### 修复方案

**1. 全部改用 logger**

```python
# 之前
print(f"[MetricsPersister] Syncing {len(new_metrics)} new metrics...")

# 之后
logger.info(f"Syncing {len(new_metrics)} new metrics...")
```

**2. 文件增量读取（使用 offset）**

```python
def sync_metrics_from_file(
    job_uuid: str,
    metrics_file: Path,
    session: Session,
    batch_size: int = 100,
    last_offset: int = 0,  # ✨ 新增参数
) -> Dict[str, Any]:
    """
    从 last_offset 开始读取，而不是从头读

    Returns:
        - new_metrics_count: 新增指标数量
        - new_offset: 新的文件 offset  # ✨ 返回新 offset
        - file_size: 当前文件大小
    """
    file_size = metrics_file.stat().st_size

    # 如果文件没有增长，直接返回
    if file_size <= last_offset:
        return {"new_metrics_count": 0, "new_offset": last_offset, ...}

    with open(metrics_file, 'r') as f:
        # ✨ 跳到上次读取的位置
        f.seek(last_offset)

        for line in f:
            # 解析新增的行
            ...

        # ✨ 记录新的 offset
        new_offset = f.tell()

    return {
        "new_metrics_count": len(new_metrics),
        "new_offset": new_offset,  # ✨ 返回新 offset 供下次使用
        "file_size": file_size,
    }
```

**使用示例：**

```python
# 第一次同步
result = sync_metrics_from_file("job-123", Path("metrics.jsonl"), session, last_offset=0)
print(result)
# {"new_metrics_count": 100, "new_offset": 12345, "file_size": 15000}

# 第二次同步（只读取新增部分）
result = sync_metrics_from_file("job-123", Path("metrics.jsonl"), session, last_offset=12345)
print(result)
# {"new_metrics_count": 50, "new_offset": 15000, "file_size": 15000}
```

### 性能对比

| 场景 | 之前（读全文件） | 之后（增量读取） |
|------|----------------|----------------|
| 文件 1MB, 首次读取 | 1MB | 1MB |
| 文件 1MB, 新增 10KB | 1MB | 10KB ✅ |
| 文件 100MB, 新增 10KB | 100MB | 10KB ✅ |

---

## ✅ 修复 3: validate_file_path 的副作用

### 问题描述

```python
# 之前的实现
def validate_file_path(file_path: str) -> str:
    for allowed_dir in ALLOWED_DATASET_DIRS:
        os.makedirs(allowed_dir, exist_ok=True)  # ❌ 副作用！
        allowed_real = os.path.realpath(allowed_dir)
        ...
```

**副作用：**
- 校验函数在访问时创建目录
- 错误配置下会创建一堆目录
- 权限/沙箱边界模糊

### 修复方案

```python
def validate_file_path(file_path: str) -> str:
    """
    **Important**: This function is pure validation - no side effects.
    It does NOT create directories. Allowed directories should be
    created at deployment time.
    """
    for allowed_dir in ALLOWED_DATASET_DIRS:
        # ✅ 只做 expand 和 realpath，不创建目录
        allowed_expanded = os.path.expanduser(allowed_dir)
        allowed_real = os.path.realpath(allowed_expanded)

        # ✅ 使用 commonpath 检查（更健壮）
        try:
            common = os.path.commonpath([allowed_real, real_path])
            if common == allowed_real:
                is_allowed = True
                break
        except ValueError:
            # Paths on different drives or not comparable
            continue
```

### 部署要求

**在部署时预创建 allowed 目录：**

```bash
# Docker entrypoint.sh
mkdir -p ~/train_platform/datasets
mkdir -p ~/datasets
mkdir -p ./datasets
mkdir -p ./data

# 或在 docker-compose.yml
volumes:
  - ./datasets:/app/datasets  # 确保目录存在
```

---

## 📊 修复总结

| 问题 | 影响 | 修复状态 | 验证方式 |
|------|------|---------|---------|
| Stage task_id 未入库 | ❌ Resume 不可靠、状态不可追踪 | ✅ 完全修复 | 检查 DB stage.celery_task_id |
| print() 代替 logger | ⚠️ 高频场景性能差、日志污染 | ✅ 完全修复 | 搜索代码无 print() |
| 全文件读取 | ⚠️ 大文件场景越来越慢 | ✅ 完全修复 | 测试增量读取性能 |
| validate 有副作用 | ⚠️ 可能创建不该有的目录 | ✅ 完全修复 | 验证无 makedirs 调用 |

---

## 🔧 后续建议

### 1. Pipeline 验证测试

建议创建集成测试验证 Pipeline 恢复机制：

```python
def test_pipeline_resume():
    # 创建 pipeline with 4 stages
    # 让 stage 2 失败
    # 调用 resume()
    # 验证只重新执行 stage 2, 3, 4
    # 验证 stage 1 不重复执行
```

### 2. Metrics 性能测试

建议测试大文件场景：

```bash
# 生成 100MB metrics 文件
python generate_test_metrics.py --size=100MB

# 测试增量读取性能
time python test_sync_metrics.py --last-offset=0
time python test_sync_metrics.py --last-offset=90000000  # 90MB
```

### 3. 路径验证测试

建议测试边界情况：

```python
def test_path_validation():
    # 测试 ../ 攻击
    validate_file_path("~/datasets/../../../etc/passwd")  # 应该拒绝

    # 测试 symlink 绕过
    os.symlink("/etc", "~/datasets/etc_link")
    validate_file_path("~/datasets/etc_link/passwd")  # 应该拒绝

    # 测试正常路径
    validate_file_path("~/datasets/train.parquet")  # 应该允许
```

---

## 🎓 工程经验总结

### 关于 Pipeline 状态追踪

**教训：**
- Celery task 的 `task_id` 只有在 task 执行时才能获得（通过 `self.request.id`）
- 不能在提交时预知 task_id，必须在 task 内部记录

**最佳实践：**
- 使用包装 task 统一处理状态更新
- 所有状态字段都记录到 DB（task_id, started_at, completed_at, result, error）
- Resume 逻辑从 DB 重建状态，而不是依赖内存

### 关于文件读取性能

**教训：**
- 训练 metrics 文件会持续增长（几小时训练可能达到几 GB）
- 每次从头读取会导致性能线性下降
- `print()` 在高频场景下会成为瓶颈

**最佳实践：**
- 使用 `f.seek(offset)` 增量读取
- 使用 `logger` 而不是 `print()`
- 返回新 offset 供下次使用

### 关于安全校验

**教训：**
- 校验函数不应该有副作用（创建目录、修改文件等）
- 副作用会让系统行为难以预测

**最佳实践：**
- 校验只做检查，不做修改
- 使用 `os.path.commonpath` 而不是字符串前缀匹配
- 在部署时预创建必要的目录

---

这些修复使系统从"能跑"变成"能在生产环境稳定跑"！✅
