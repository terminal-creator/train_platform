# Celery 细节坑修复总结

基于代码审查，修复了 3 个关键的 Celery 语义层面的坑，确保 Pipeline 运行时参数正确、回调可靠触发、状态准确落库。

---

## ✅ 修复 1: chain 参数传递问题

### 问题描述

```python
# 之前的代码
task_sig = signature(celery_task_name, kwargs=node.params)
return chain(init_stage_sig, task_sig)
```

**Celery 的坑：**
- `chain` 默认会把前一个 task 的返回值作为下一个 task 的**第一个 positional argument**
- 如果下一个 signature 不是 immutable，它会接收这个多余的参数
- 会导致 `TypeError: got multiple values` 或 `takes 0 positional arguments`

### 修复方案

```python
# 修复后的代码
task_sig = sig(
    celery_task_name,
    kwargs=task_params,
    immutable=True,  # ✅ 关键：避免接收 chain 前序结果
).set(
    queue=task_queue,
    link=sig(...),
    link_error=sig(...),
)

return chain(init_stage_sig, task_sig)
```

**关键点：**
- 使用 `immutable=True` (等价于 `.si()`)
- 确保 task 不会吃到前序 task 的返回值
- 只接收自己的 kwargs

---

## ✅ 修复 2: link callback 参数约定

### 问题描述

```python
# 之前的代码
"link": signature(
    "on_stage_success",
    args=(pipeline_uuid, stage_name),
    immutable=True,  # ❌ 矛盾：设置了 immutable
)

def on_stage_success(self, result, pipeline_uuid, stage_name):
    # ❌ 但签名要求 result
```

**Celery 的坑：**
- 如果 link 设置 `immutable=True`，Celery 不会传递前一个 task 的 result
- 但函数签名仍然要求 `result` 参数
- 会导致参数错位或 `TypeError`

### 修复方案

**选择：不设置 immutable，让 Celery 传递 result**

```python
# 修复后的代码
"link": sig(
    "training_platform.core.pipeline_executor.on_stage_success",
    args=(pipeline_uuid, stage_name),
    # ✅ 不设置 immutable，让它接收 task 的 result
)

@app.task(name="training_platform.core.pipeline_executor.on_stage_success")
def on_stage_success(result, pipeline_uuid: str, stage_name: str):
    """
    Celery link callback 会自动传递：
    1. result: 前一个 task 的返回值 (Celery 自动传递)
    2. 我们手动传递的 args: (pipeline_uuid, stage_name)
    """
    # 只更新状态和结果，task_id 已经在 task 开始时记录
    stage.status = PipelineStageStatus.COMPLETED
    stage.result = result
    ...
```

**关键点：**
- 移除 `bind=True`（不需要 `self.request`）
- 移除 `immutable=True`（让 Celery 传递 result）
- 第一个参数是 `result`，后面是我们手动传递的 args

---

## ✅ 修复 3: link_error (errback) 参数约定

### 问题描述

```python
# 之前的代码
"link_error": signature(
    "on_stage_error",
    args=(pipeline_uuid, stage_name),
    immutable=True,
)

def on_stage_error(self, task_id: str, pipeline_uuid, stage_name):
    # ❌ 假设 task_id 是第一个参数
```

**Celery 的坑：**
- Celery errback 的第一个参数是失败 task 的 **UUID**
- 不是 `self.request`，也不是我们以为的 `task_id: str`
- 参数约定：`def errback(uuid, *args)`

### 修复方案

```python
# 修复后的代码
"link_error": sig(
    "training_platform.core.pipeline_executor.on_stage_error",
    args=(pipeline_uuid, stage_name),
    # errback 不需要 immutable
)

@app.task(name="training_platform.core.pipeline_executor.on_stage_error")
def on_stage_error(uuid, pipeline_uuid: str, stage_name: str):
    """
    Celery errback 会自动传递：
    1. uuid: 失败 task 的 UUID (Celery 自动传递)
    2. 我们手动传递的 args: (pipeline_uuid, stage_name)
    """
    # 获取失败 task 的异常信息
    async_result = AsyncResult(uuid, app=app)
    error_message = str(async_result.info) if async_result.info else "Unknown error"

    # 更新状态
    stage.status = PipelineStageStatus.FAILED
    stage.error_message = error_message
    ...
```

**关键点：**
- 移除 `bind=True`
- 第一个参数是 `uuid` (失败 task 的 UUID)
- 使用 `AsyncResult(uuid)` 获取异常信息

---

## ✅ 修复 4: 让 task 自报家门（不靠猜 parent_id）

### 问题描述

```python
# 之前的代码（在回调里猜测）
def on_stage_success(self, result, pipeline_uuid, stage_name):
    # ❌ 猜测 parent_id
    training_task_id = self.request.get('parent_id') or self.request.id
    stage.celery_task_id = training_task_id
```

**问题：**
- 依赖 `parent_id` 不可靠（Celery 版本/worker 配置可能不同）
- 回调不是 bind=True 时无法访问 `self.request`
- 在 chain/group/chord 中 parent_id 可能不是我们想要的

### 修复方案：使用 Celery Signals

**方式 B: Celery Signals（最优雅）**

```python
# celery_config.py
from celery.signals import task_prerun

@task_prerun.connect
def track_pipeline_stage_start(sender=None, task_id=None, task=None, args=None, kwargs=None, **extra):
    """
    在 task 开始执行前自动记录 stage 状态

    通过 Celery signals 自动捕获 task 开始事件，检查 kwargs 中是否有：
    - _pipeline_uuid: Pipeline UUID
    - _stage_name: Stage name

    如果有，则调用 mark_stage_running 记录状态
    """
    if not kwargs:
        return

    pipeline_uuid = kwargs.get('_pipeline_uuid')
    stage_name = kwargs.get('_stage_name')

    if pipeline_uuid and stage_name:
        from .pipeline_executor import mark_stage_running
        mark_stage_running(pipeline_uuid, stage_name, task_id)
```

```python
# pipeline_executor.py
def mark_stage_running(pipeline_uuid: str, stage_name: str, task_id: str):
    """由 task 自己调用（通过 signal 自动触发）"""
    with Session(engine) as session:
        repo = PipelineRepository(session)
        stages = repo.get_stages(pipeline_uuid)
        stage = next((s for s in stages if s.stage_name == stage_name), None)
        if stage:
            # ✅ 记录真实的 task_id
            stage.celery_task_id = task_id
            stage.status = PipelineStageStatus.RUNNING
            stage.started_at = datetime.utcnow()
            repo.update_stage(stage)
```

```python
# _create_stage_task 中注入 pipeline 信息
task_params = dict(node.params)
task_params['_pipeline_uuid'] = self.pipeline_uuid  # ✅ 注入
task_params['_stage_name'] = stage_name  # ✅ 注入

task_sig = sig(celery_task_name, kwargs=task_params, immutable=True)
```

**好处：**
- ✅ 完全自动化（不需要修改任何 training task 的代码）
- ✅ task 自己报告 task_id（通过 signal 的 `task_id` 参数）
- ✅ 不依赖 parent_id 猜测
- ✅ 适用于所有 task（train_model, run_evaluation, preprocess_dataset, cleanup_checkpoints）

---

## 📊 修复前后对比

| 问题 | 之前（有坑） | 之后（修复） |
|------|------------|------------|
| chain 传参 | ❌ task 收到多余参数 | ✅ immutable=True 避免传参 |
| link callback | ❌ immutable+result 矛盾 | ✅ 不设 immutable，正确接收 result |
| link_error | ❌ 参数类型错误 | ✅ 第一个参数是 uuid |
| task_id 记录 | ❌ 猜测 parent_id 不可靠 | ✅ signal 自动记录真实 task_id |

---

## 🎯 完整的 Stage 生命周期

现在的完整流程：

```
1. Pipeline.execute()
   ↓
2. _create_stage_task()
   - 创建 signature(celery_task_name, kwargs={...params, _pipeline_uuid, _stage_name}, immutable=True)
   - 设置 queue, link, link_error
   - 返回 chain(init_stage_sig, task_sig)
   ↓
3. init_stage_status (快速执行)
   - 更新 stage.status = PENDING
   ↓
4. 实际 training task 开始执行
   - Celery signal task_prerun 触发
   - 自动调用 mark_stage_running(pipeline_uuid, stage_name, task_id)
   - 更新 stage.celery_task_id = task_id
   - 更新 stage.status = RUNNING
   - 更新 stage.started_at = now
   ↓
5a. Task 成功 → link callback
    - on_stage_success(result, pipeline_uuid, stage_name)
    - 更新 stage.status = COMPLETED
    - 更新 stage.result = result
    - 更新 stage.completed_at = now

5b. Task 失败 → link_error callback
    - on_stage_error(uuid, pipeline_uuid, stage_name)
    - 获取异常信息 AsyncResult(uuid).info
    - 更新 stage.status = FAILED
    - 更新 stage.error_message = error
    - 标记 pipeline.status = FAILED
```

---

## ✅ 验证方式

### 1. 验证 chain 参数传递

```python
# 创建一个简单的 pipeline
stages = [
    {"name": "A", "task": "train_model", "params": {"job_uuid": "test"}, "depends_on": []},
]

executor = PipelineExecutor("test-pipeline")
executor.execute(stages)

# 观察日志，不应该有参数错误
```

### 2. 验证 link callback 接收 result

```python
# 训练完成后，检查 DB
with Session(engine) as session:
    repo = PipelineRepository(session)
    stages = repo.get_stages("test-pipeline")
    stage = stages[0]

    print(f"status: {stage.status}")  # COMPLETED
    print(f"result: {stage.result}")  # ✅ 应该有训练结果
```

### 3. 验证 link_error 接收 uuid

```python
# 让一个 task 故意失败
# 观察日志，应该看到：
# [on_stage_error] uuid=xxx, pipeline_uuid=xxx, stage_name=xxx
# 不应该有参数错误

# 检查 DB
stage = repo.get_stage("test-pipeline", "A")
print(f"status: {stage.status}")  # FAILED
print(f"error: {stage.error_message}")  # ✅ 应该有错误信息
```

### 4. 验证 task_id 记录

```python
# 检查 stage 的 celery_task_id
with Session(engine) as session:
    repo = PipelineRepository(session)
    stages = repo.get_stages("test-pipeline")

    for stage in stages:
        print(f"{stage.stage_name}:")
        print(f"  celery_task_id: {stage.celery_task_id}")  # ✅ 应该有真实的 task_id
        print(f"  status: {stage.status}")
        print(f"  started_at: {stage.started_at}")  # ✅ 应该有开始时间

        # 验证 task_id 是真实的
        if stage.celery_task_id:
            from celery.result import AsyncResult
            result = AsyncResult(stage.celery_task_id)
            print(f"  task_state: {result.state}")  # SUCCESS/FAILURE/PENDING
```

---

## 🎓 Celery 最佳实践总结

### 1. chain 传参
- **Always** 使用 `immutable=True` 或 `.si()` 来避免接收前序结果
- 除非你**确实需要**前序结果作为参数

### 2. link callback
- **不要混搭** `immutable=True` 和 `result` 参数
- 要么：不设 immutable，接收 result
- 要么：设 immutable，不要 result 参数

### 3. link_error (errback)
- 第一个参数**总是** `uuid` (失败 task 的 UUID)
- 使用 `AsyncResult(uuid)` 获取异常信息
- 不要假设它是 `self.request` 或其他东西

### 4. task_id 记录
- **不要猜测** `parent_id`
- 使用 Celery signals (task_prerun) 自动记录
- 或在 task 开头调用统一的状态更新函数

### 5. 队列隔离
- 使用 `.set(queue="xxx")` 或 `options={"queue": "xxx"}`
- 不同类型的 task 使用不同队列
- training/evaluation/preprocessing/maintenance 分离

---

## 🚀 工程化成果

通过修复这些 Celery 细节坑，我们的 Pipeline 系统达到了：

✅ **参数传递正确**：chain 不会导致参数错位
✅ **回调可靠触发**：link/link_error 正确接收参数
✅ **状态准确落库**：task_id 自动记录，不依赖猜测
✅ **队列完全隔离**：training/evaluation/preprocessing 各走各的队列
✅ **真正的异步**：每个 stage 都是独立的 Celery task

**现在的实现是真正的生产级 Celery Canvas！** 🎉
