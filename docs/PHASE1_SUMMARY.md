# Phase 1 完成总结

## 概览

Phase 1 实现了**真实监控 + 基础诊断**功能，让训练平台能够：
- ✅ 从 verl 训练过程实时采集指标
- ✅ 持久化指标到数据库
- ✅ 通过 WebSocket 实时推送指标到前端
- ✅ 支持历史指标回放
- ✅ 自动检测训练异常（NaN、KL 爆炸、Loss 不下降）

---

## Phase 1.1: verl 集成

### 实现内容

1. **Callback 接口设计**（`docs/CALLBACK_DESIGN.md`）
   - 定义统一的 Callback 接口规范
   - 10 个生命周期 hook（on_train_begin, on_step_end, 等）

2. **PlatformCallback 实现**（`environments/verl/verl/trainer/callbacks/platform_callback.py`）
   - 实时记录训练指标到 JSON Lines 文件
   - 自动检测异常（NaN/Inf、KL 爆炸、Loss 不下降）
   - 维护训练状态文件

3. **verl Trainer 修改**（`environments/verl/verl/trainer/ppo/ray_trainer.py`）
   - 添加 `add_callback()` 和 `_trigger_callbacks()` 方法
   - 在训练循环关键位置插入 callback 调用

4. **使用示例**（`environments/verl/examples/platform_callback_example.py`）

### 输出文件格式

**指标文件**（`{job_id}_metrics.jsonl`）：
```json
{"step": 1, "timestamp": 1704672000.123, "loss": {...}, "reward": {...}, "kl": {...}, ...}
{"step": 2, "timestamp": 1704672002.456, "loss": {...}, "reward": {...}, "kl": {...}, ...}
```

**状态文件**（`{job_id}_status.json`）：
```json
{
    "job_id": "test_job_001",
    "status": "running",
    "current_step": 100,
    "total_steps": 1000,
    "anomaly_detected": false,
    "anomaly_reason": null
}
```

---

## Phase 1.2: 指标存储

### 实现内容

1. **数据库表扩展**（`training_platform/core/database.py`）
   - 扩展 `TrainingMetric` 表，添加 11 个新字段
   - reward_max/min, kl_divergence_max
   - grad_norm_actor/critic
   - tokens_per_second, step_time, gpu_memory_allocated_gib
   - has_anomaly, anomaly_type, anomaly_message

2. **指标持久化逻辑**（`training_platform/core/metrics_persister.py`）
   - `parse_platform_metric()`: JSON → TrainingMetric 转换
   - `sync_metrics_from_file()`: 增量同步到数据库
   - `sync_anomaly_from_status_file()`: 同步异常信息

3. **查询接口**（`training_platform/api/routers/monitoring.py`）
   - `GET /monitoring/{job_id}/metrics` - 查询指标历史
   - `GET /monitoring/{job_id}/metrics/anomalies` - 查询异常
   - `POST /monitoring/{job_id}/metrics/sync` - 手动触发同步

4. **数据库迁移**（`scripts/migrate_db_phase1_2.py`）
   - 自动添加新字段到现有数据库
   - 幂等性：重复执行不会出错

### 使用方法

#### 数据库迁移
```bash
python scripts/migrate_db_phase1_2.py
```

#### 手动同步指标
```bash
curl -X POST http://localhost:8000/api/monitoring/{job_id}/metrics/sync
```

#### 查询历史指标
```bash
curl "http://localhost:8000/api/monitoring/{job_id}/metrics?start_step=0&end_step=100"
```

### 测试

```bash
python tests/test_phase1_2.py
```

测试覆盖：
- ✅ 模拟指标文件创建
- ✅ 指标格式转换
- ✅ 数据库持久化
- ✅ 增量同步（避免重复）
- ✅ API 查询接口
- ✅ 异常检测和同步

---

## Phase 1.3: 实时监控

### 实现内容

1. **指标读取器**（`training_platform/core/metrics_reader.py`）
   - `MetricsReader` 抽象基类
   - `LocalMetricsReader`: 从本地文件系统读取
   - `SSHMetricsReader`: 通过 SSH 从远程服务器读取
   - 支持增量读取（`read_metrics_incremental()`）

2. **WebSocket 实时推送**（`/monitoring/{job_id}/live`）
   - 从真实的指标文件读取数据（替换随机数）
   - 自动判断运行模式（本地 or SSH）
   - 增量推送新增的指标
   - 推送训练状态（running, completed, failed）

3. **历史指标回放**（`/monitoring/{job_id}/playback`）
   - 支持指定起止步骤（start_step, end_step）
   - 支持播放速度控制（0.1x ~ 10x）
   - 发送回放进度（0-1）

### 使用方法

#### 实时监控（前端）

```javascript
// 连接 WebSocket
const ws = new WebSocket(`ws://localhost:8000/api/monitoring/${jobId}/live`);

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);

    if (message.type === 'metric') {
        // 新增指标
        console.log('Step:', message.data.step);
        console.log('Loss:', message.data.loss);
        console.log('Reward:', message.data.reward);
        updateChart(message.data);
    } else if (message.type === 'status') {
        // 状态更新
        console.log('Status:', message.data.status);
    } else if (message.type === 'finished') {
        // 训练完成
        console.log('Training finished:', message.status);
        ws.close();
    }
};
```

#### 历史回放（前端）

```javascript
// 连接回放 WebSocket（2 倍速回放）
const ws = new WebSocket(
    `ws://localhost:8000/api/monitoring/${jobId}/playback?start_step=0&end_step=100&speed=2.0`
);

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);

    if (message.type === 'playback_info') {
        // 回放元信息
        console.log('Total metrics:', message.total_metrics);
        console.log('Speed:', message.speed);
    } else if (message.type === 'metric') {
        // 指标数据
        console.log('Progress:', message.progress);  // 0-1
        updateChart(message.data);
    } else if (message.type === 'playback_complete') {
        // 回放完成
        console.log('Playback complete');
        ws.close();
    }
};
```

### 消息格式

#### 实时监控消息

**指标消息**:
```json
{
    "type": "metric",
    "data": {
        "step": 1,
        "timestamp": "2024-01-08T12:00:00",
        "loss": {"actor_loss": 2.5, "critic_loss": 1.2, "total_loss": 3.7},
        "reward": {"mean": 0.5, "std": 0.1, "max": 0.7, "min": 0.3},
        "kl": {"mean": 0.1, "max": 0.15},
        "gradient": {"actor_norm": 1.5, "critic_norm": 0.8},
        "performance": {"tokens_per_second": 1000, "step_time": 2.5, "gpu_memory_allocated": 40.5}
    }
}
```

**状态消息**:
```json
{
    "type": "status",
    "data": {
        "job_id": "test_job_001",
        "status": "running",
        "current_step": 100,
        "total_steps": 1000,
        "anomaly_detected": false
    }
}
```

#### 回放消息

**回放信息**:
```json
{
    "type": "playback_info",
    "total_metrics": 1000,
    "start_step": 0,
    "end_step": 1000,
    "speed": 2.0
}
```

**指标消息**（带进度）:
```json
{
    "type": "metric",
    "data": {...},
    "progress": 0.5,
    "current_index": 500,
    "total": 1000
}
```

---

## 架构设计

### 数据流

```
训练进程 (verl)
    ↓
PlatformCallback
    ↓
{job_id}_metrics.jsonl  (JSON Lines 文件)
    ↓
MetricsReader (LocalMetricsReader / SSHMetricsReader)
    ↓
WebSocket (/live 或 /playback)
    ↓
前端图表
```

### 持久化流程

```
{job_id}_metrics.jsonl
    ↓
MetricsPersister.sync_metrics_from_file()
    ↓
TrainingMetric 表
    ↓
查询接口 (GET /metrics)
    ↓
前端历史查询
```

---

## 关键设计决策

### 1. 为什么使用 JSON Lines 而不是数据库直写？

- **解耦**：训练进程不依赖数据库连接
- **可靠**：文件写入比网络请求更可靠
- **灵活**：可以离线同步，支持批量导入
- **调试**：文件可读，便于调试和分析

### 2. 为什么需要增量读取？

- **性能**：避免每次都读取整个文件
- **实时性**：只处理新增的数据
- **内存**：不会因为文件过大导致内存溢出

### 3. 为什么需要历史回放？

- **分析**：回看训练过程，定位问题
- **演示**：向他人展示训练效果
- **对比**：同步回放多个实验

### 4. 为什么抽象 MetricsReader？

- **统一接口**：本地和远程使用相同的代码
- **易测试**：可以 mock Reader 进行单元测试
- **易扩展**：未来可能支持 S3、NFS 等存储

---

## 下一步：Phase 1.4 - 基础诊断

虽然 PlatformCallback 已经实现了基础的异常检测，但 Phase 1.4 将在**平台层**增强诊断功能：

- [ ] 1.4.1 实现 NaN/Inf 检测（平台层监控）
- [ ] 1.4.2 实现 KL 散度异常检测（平台层告警）
- [ ] 1.4.3 实现 Loss 不下降检测（平台层分析）
- [ ] 1.4.4 异常时自动标记任务状态

主要区别：
- **训练层**（PlatformCallback）：实时检测，记录到文件
- **平台层**（Phase 1.4）：从数据库分析，触发告警和自动操作

---

## 文件清单

### 新增文件

```
environments/verl/verl/trainer/callbacks/
├── __init__.py
├── base.py                          # Callback 基类
└── platform_callback.py             # Platform 实现

environments/verl/examples/
└── platform_callback_example.py     # 使用示例

training_platform/core/
├── metrics_persister.py             # 指标持久化
└── metrics_reader.py                # 指标读取器

scripts/
└── migrate_db_phase1_2.py           # 数据库迁移

tests/
└── test_phase1_2.py                 # Phase 1.2 测试

docs/
├── CALLBACK_DESIGN.md               # Callback 设计文档
└── PHASE1_SUMMARY.md                # 本文档
```

### 修改文件

```
training_platform/core/database.py          # 扩展 TrainingMetric 表
training_platform/api/routers/monitoring.py # 添加接口和 WebSocket
environments/verl/verl/trainer/ppo/ray_trainer.py  # 添加 callback 支持
TASKS.md                                    # 更新任务清单
```

---

## 验收标准

✅ **Phase 1.1**:
- verl 训练时能生成指标文件和状态文件
- Callback 能检测异常并记录

✅ **Phase 1.2**:
- 指标能正确同步到数据库
- 查询接口能返回正确的数据
- 测试脚本全部通过（6/6）

✅ **Phase 1.3**:
- WebSocket 能实时推送真实指标
- 本地和 SSH 模式都能正常工作
- 历史回放功能正常

⏸️ **Phase 1.4**（待实现）:
- 平台层异常检测和告警
- 自动标记任务状态

---

## 使用建议

### 开发环境

1. 先运行数据库迁移：
   ```bash
   python scripts/migrate_db_phase1_2.py
   ```

2. 启动平台：
   ```bash
   cd training_platform/api
   uvicorn main:app --reload
   ```

3. 测试 WebSocket（使用 websocat 或浏览器）：
   ```bash
   # 安装 websocat
   brew install websocat

   # 连接实时监控
   websocat ws://localhost:8000/api/monitoring/{job_id}/live

   # 连接历史回放
   websocat "ws://localhost:8000/api/monitoring/{job_id}/playback?speed=2.0"
   ```

### 生产环境

1. 确保指标目录存在：
   ```bash
   mkdir -p platform_metrics
   ```

2. 配置环境变量（可选）：
   ```bash
   export PLATFORM_METRICS_DIR=./platform_metrics
   ```

3. 在训练脚本中添加 Callback：
   ```python
   from verl.trainer.callbacks import PlatformCallback

   callback = PlatformCallback(
       job_id="my_job_001",
       output_dir="./platform_metrics",
       enable_anomaly_detection=True,
       kl_threshold=1.0,
       loss_patience=50,
       stop_on_anomaly=False
   )

   trainer.add_callback(callback)
   ```

---

## 故障排查

### 问题：WebSocket 连接失败

**检查**：
1. 任务是否存在：`curl http://localhost:8000/api/jobs/{job_id}`
2. 指标文件是否存在：`ls platform_metrics/{job_id}_metrics.jsonl`
3. 查看日志：`tail -f logs/platform.log`

### 问题：指标没有显示

**检查**：
1. PlatformCallback 是否正确添加到 trainer
2. 指标文件是否有内容：`head platform_metrics/{job_id}_metrics.jsonl`
3. 是否执行了同步：`POST /monitoring/{job_id}/metrics/sync`

### 问题：SSH 模式读取失败

**检查**：
1. SSH 连接是否正常：`ssh user@host`
2. 远程指标文件路径是否正确
3. SSHManager 是否正确初始化

---

**Phase 1 (除 1.4) 已完成！** ✅

下一步：实现 Phase 1.4 基础诊断功能，或者先进行完整的集成测试。
