# 诊断功能使用指南 - Phase 1.4

## 概览

Phase 1.4 实现了平台层的异常检测和诊断功能，提供：
- ✅ 4 种异常检测（NaN/Inf、KL 爆炸、Loss 不下降、Reward 崩溃）
- ✅ 自动标记失败
- ✅ 健康评分系统
- ✅ 诊断建议

---

## 与 PlatformCallback 的区别

| 特性 | PlatformCallback（训练层） | Diagnostics（平台层） |
|------|----------------------------|----------------------|
| 执行时机 | 训练过程中实时 | 定期扫描或手动触发 |
| 数据源 | 训练进程内存 | 数据库持久化数据 |
| 输出 | 写入文件 | API 返回 + 自动操作 |
| 告警 | 仅记录 | 通知 + 自动标记失败 |
| 分析维度 | 当前步骤 | 历史趋势 |

**互补关系**：
- PlatformCallback：快速发现问题，记录到文件
- Diagnostics：深度分析，触发告警和自动操作

---

## 异常检测类型

### 1. NaN/Inf 检测

**检测逻辑**：检查最近 N 个步骤的指标中是否有 NaN/Inf

**检测字段**：
- policy_loss (actor_loss)
- value_loss (critic_loss)
- reward_mean
- kl_divergence
- grad_norm_actor

**严重程度**：CRITICAL（严重）

**诊断建议**：
```
NaN/Inf 通常由以下原因导致：
1. 学习率过高 -> 降低学习率
2. 梯度爆炸 -> 添加梯度裁剪
3. 数值不稳定 -> 检查 reward scaling
4. 数据问题 -> 检查数据预处理
```

**示例**：
```bash
curl -X POST "http://localhost:8000/api/monitoring/{job_id}/diagnose"
```

返回：
```json
{
  "anomalies": [
    {
      "detected": true,
      "anomaly_type": "nan_inf",
      "severity": "critical",
      "message": "NaN/Inf detected in fields: policy_loss, reward_mean at step 150",
      "step": 150,
      "metrics_snapshot": {
        "policy_loss": NaN,
        "value_loss": 1.2,
        "reward_mean": NaN
      },
      "suggestion": "NaN/Inf 通常由以下原因导致：..."
    }
  ]
}
```

### 2. KL 散度爆炸检测

**检测逻辑**：检查 KL 散度是否超过阈值（默认 1.0）

**严重程度**：
- KL > 阈值 * 2.0: CRITICAL
- KL > 阈值 * 1.5: HIGH
- KL > 阈值: MEDIUM

**诊断建议**：
```
KL 散度爆炸的解决方法：
1. 降低学习率（当前可能过高）
2. 增大 KL penalty 系数
3. 减小 PPO clip range
4. 检查 reward scaling 是否合理
```

**配置阈值**：
```python
# 在 verl_adapter 中配置
config = {
    "kl_threshold": 0.5  # 更严格的阈值
}
```

### 3. Loss 不下降检测

**检测逻辑**：检查 Loss 在最近 N 步（默认 50）内的改善幅度

**参数**：
- `patience`: 容忍多少步没有改善（默认 50）
- `min_improvement`: 最小改善幅度（默认 0.01 = 1%）

**严重程度**：
- patience >= 100: HIGH
- patience >= 50: MEDIUM
- patience < 50: LOW

**诊断建议**：
```
Loss 不下降的解决方法：
1. 降低学习率（可能过大错过最优点）
2. 尝试学习率调度（cosine decay, warm restart）
3. 检查是否陷入局部最优
4. 考虑 early stopping
```

### 4. Reward 崩溃检测

**检测逻辑**：检查 Reward 是否突然大幅下降

**参数**：
- `recent_steps`: 检查最近多少步（默认 20）
- `drop_threshold`: 下降幅度阈值（默认 0.5 = 50%）

**严重程度**：
- drop > 70%: CRITICAL
- drop > 50%: HIGH

**诊断建议**：
```
Reward 崩溃的解决方法：
1. 立即停止训练，从之前的检查点恢复
2. 降低学习率
3. 减小 PPO clip range
4. 增大 KL penalty
```

---

## API 接口

### 1. 诊断单个任务

```bash
POST /api/monitoring/{job_id}/diagnose?auto_mark_failed=true
```

**参数**：
- `auto_mark_failed`: 是否自动标记失败（默认 true）

**返回**：
```json
{
  "success": true,
  "job_uuid": "xxx",
  "job_id": "my_job_001",
  "status": "failed",
  "anomalies_count": 2,
  "critical_count": 1,
  "auto_marked_failed": true,
  "anomalies": [
    {
      "detected": true,
      "anomaly_type": "nan_inf",
      "severity": "critical",
      "message": "...",
      "step": 150,
      "suggestion": "..."
    }
  ]
}
```

### 2. 获取检测到的异常（不执行自动操作）

```bash
GET /api/monitoring/{job_id}/anomalies/detected
```

**用途**：仅查询异常，不执行自动标记失败

**返回**：
```json
{
  "job_id": "my_job_001",
  "anomalies_count": 2,
  "anomalies": [...]
}
```

### 3. 诊断所有运行中的任务

```bash
POST /api/monitoring/diagnose-all
```

**用途**：定期扫描（如每分钟一次）实现自动监控

**返回**：
```json
{
  "scanned_jobs": 5,
  "total_anomalies": 3,
  "auto_failed_count": 1,
  "results": [
    {
      "job_uuid": "xxx",
      "anomalies_count": 2,
      ...
    }
  ]
}
```

### 4. 获取任务健康状态

```bash
GET /api/monitoring/{job_id}/health
```

**返回**：
```json
{
  "job_id": "my_job_001",
  "job_status": "running",
  "health_score": 65,
  "health_status": "warning",
  "anomalies_count": 1,
  "anomalies": [...],
  "suggestions": [
    "降低学习率...",
    "添加梯度裁剪..."
  ]
}
```

**健康评分规则**：
- 100: 完全健康，无异常
- 80-99: 有轻微警告（LOW）
- 50-79: 有中等问题（MEDIUM）
- 0-49: 有严重问题（HIGH/CRITICAL）

---

## 使用场景

### 场景 1：手动诊断

用户在前端点击"诊断"按钮：

```javascript
// 前端代码
async function diagnoseJob(jobId) {
    const response = await fetch(
        `/api/monitoring/${jobId}/diagnose?auto_mark_failed=false`,
        { method: 'POST' }
    );
    const result = await response.json();

    if (result.anomalies_count > 0) {
        // 显示异常列表
        showAnomalies(result.anomalies);

        // 询问用户是否标记失败
        if (result.critical_count > 0) {
            if (confirm('检测到严重异常，是否停止训练？')) {
                await stopJob(jobId);
            }
        }
    } else {
        showMessage('任务健康，无异常');
    }
}
```

### 场景 2：自动监控（定时任务）

使用 cron 或定时任务每分钟扫描一次：

```bash
#!/bin/bash
# scripts/auto_diagnose.sh

# 每分钟调用一次
curl -X POST http://localhost:8000/api/monitoring/diagnose-all
```

配置 cron：
```cron
* * * * * /path/to/scripts/auto_diagnose.sh >> /var/log/auto_diagnose.log 2>&1
```

### 场景 3：健康仪表盘

在前端显示所有任务的健康状态：

```javascript
async function updateHealthDashboard() {
    const jobs = await fetchRunningJobs();

    for (const job of jobs) {
        const health = await fetch(`/api/monitoring/${job.id}/health`);
        const data = await health.json();

        // 根据健康评分显示颜色
        const color = data.health_score >= 80 ? 'green' :
                      data.health_score >= 50 ? 'yellow' : 'red';

        updateJobCard(job.id, {
            score: data.health_score,
            status: data.health_status,
            color: color,
            anomalies: data.anomalies_count
        });
    }
}

// 每 30 秒更新一次
setInterval(updateHealthDashboard, 30000);
```

### 场景 4：告警通知

检测到严重异常时发送通知：

```python
# 在 DiagnosticService 中添加通知逻辑
class DiagnosticService:
    def diagnose_job(self, job_uuid, auto_mark_failed=True):
        anomalies = self.detector.detect_all(job_uuid)
        critical_anomalies = [a for a in anomalies if a.severity == AnomalySeverity.CRITICAL]

        if critical_anomalies:
            # 发送邮件通知
            send_email(
                to=job.user_email,
                subject=f"训练任务 {job.job_id} 检测到严重异常",
                body=f"异常类型: {critical_anomalies[0].anomaly_type}\n"
                     f"消息: {critical_anomalies[0].message}\n"
                     f"建议: {critical_anomalies[0].suggestion}"
            )

            # 发送 Webhook
            send_webhook(
                url=job.webhook_url,
                data={
                    "event": "anomaly_detected",
                    "job_id": job.job_id,
                    "anomaly": critical_anomalies[0].to_dict()
                }
            )

        # ... 其他逻辑
```

---

## 配置示例

### 自定义检测阈值

```python
# 在启动训练时配置
diagnostic_config = {
    "kl_threshold": 0.5,        # 更严格的 KL 阈值
    "loss_patience": 100,       # 更长的耐心
    "reward_drop_threshold": 0.3  # 更敏感的 Reward 检测
}

# 存储到任务配置中
job.diagnostic_config = diagnostic_config
```

### 禁用自动标记失败

```bash
# 仅检测，不自动操作
curl -X POST "http://localhost:8000/api/monitoring/{job_id}/diagnose?auto_mark_failed=false"
```

---

## 最佳实践

### 1. 定期扫描

- **频率**：每分钟一次
- **原因**：及时发现问题，减少资源浪费
- **实现**：使用 cron 或 APScheduler

```python
from apscheduler.schedulers.background import BackgroundScheduler

def auto_diagnose():
    # 调用诊断 API
    requests.post("http://localhost:8000/api/monitoring/diagnose-all")

scheduler = BackgroundScheduler()
scheduler.add_job(auto_diagnose, 'interval', minutes=1)
scheduler.start()
```

### 2. 分层告警

- **LOW**: 日志记录
- **MEDIUM**: 前端显示警告
- **HIGH**: 发送邮件通知
- **CRITICAL**: 自动停止 + 紧急通知

### 3. 诊断建议落地

不仅显示建议，还提供快速修复：

```javascript
// 前端显示诊断建议时，提供"应用建议"按钮
function showSuggestion(suggestion) {
    // 解析建议（如"降低学习率"）
    if (suggestion.includes('降低学习率')) {
        // 提供快速调整按钮
        showButton('调整学习率', () => {
            // 自动生成新配置，学习率减半
            const newConfig = {
                ...currentConfig,
                learning_rate: currentConfig.learning_rate * 0.5
            };
            restartJobWithConfig(newConfig);
        });
    }
}
```

### 4. 历史分析

记录每次诊断结果，分析趋势：

```python
# 创建 DiagnosticHistory 表
class DiagnosticHistory(SQLModel, table=True):
    id: int
    job_uuid: str
    timestamp: datetime
    anomalies_count: int
    health_score: int
    anomalies: str  # JSON

# 每次诊断后记录
history = DiagnosticHistory(
    job_uuid=job_uuid,
    timestamp=datetime.utcnow(),
    anomalies_count=len(anomalies),
    health_score=health_score,
    anomalies=json.dumps([a.to_dict() for a in anomalies])
)
session.add(history)
```

---

## 故障排查

### 问题：诊断接口返回"No metrics found"

**原因**：数据库中没有该任务的指标

**检查**：
1. 确认指标已同步：`POST /monitoring/{job_id}/metrics/sync`
2. 查看指标文件：`ls platform_metrics/{job_id}_metrics.jsonl`
3. 检查 PlatformCallback 是否正确添加

### 问题：异常未被检测到

**原因**：阈值设置不合理或数据不足

**检查**：
1. 查看指标数量：`GET /monitoring/{job_id}/metrics`
2. 手动检查指标值：是否确实有异常
3. 调整检测参数（patience, threshold）

### 问题：误报过多

**原因**：阈值过于严格

**解决**：
1. 放宽阈值：增大 `kl_threshold`、`loss_patience`
2. 调整严重程度判断逻辑
3. 禁用某些检测（根据实际情况）

---

## 总结

Phase 1.4 提供了完整的异常检测和诊断功能：

✅ **4 种检测**：NaN/Inf、KL 爆炸、Loss 不下降、Reward 崩溃
✅ **自动化**：自动标记失败、定期扫描
✅ **可配置**：灵活的阈值和参数
✅ **友好**：详细的诊断建议
✅ **健康评分**：直观的健康状态展示

配合 Phase 1.1-1.3 的指标采集和实时监控，现在训练平台具备了完整的监控和诊断能力！
