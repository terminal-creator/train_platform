# Phase 1.5 - 前端优化 测试文档

**完成日期**: 2026-01-08

**目标**: 优化监控页面的用户体验，增加多指标切换、历史回看和异常告警功能

---

## 功能清单

### 1.5.1 多指标图表切换 ✅

**功能描述**:
- 用户可以自定义选择要显示的指标类型
- 支持的指标包括：损失、奖励、KL散度、梯度范数、吞吐量、GPU显存
- 动态生成图表，支持同时显示多个指标
- 图表布局自动适应（2列布局）

**实现文件**:
- `frontend/src/views/MonitoringView.vue`

**关键特性**:
```javascript
const availableMetrics = [
  { id: 'loss', label: '损失曲线', icon: TrendingUp },
  { id: 'reward', label: '奖励曲线', icon: Zap },
  { id: 'kl', label: 'KL 散度', icon: BarChart3 },
  { id: 'gradient', label: '梯度范数', icon: Activity },
  { id: 'tokens_per_second', label: '吞吐量', icon: Clock },
  { id: 'gpu_memory', label: 'GPU 显存', icon: Cpu }
]
```

**UI 操作**:
1. 在监控页面找到"选择要显示的指标"区域
2. 点击指标按钮进行选择/取消选择
3. 选中的指标会以蓝色高亮显示
4. 图表会实时更新显示选中的指标

---

### 1.5.2 指标历史回看 ✅

**功能描述**:
- 支持查看指定步数范围内的历史指标
- 可以设置起始步数和结束步数
- 历史模式下暂停自动刷新和 WebSocket
- 显示当前查询的记录数量

**API 接口**:
```
GET /api/v1/monitoring/{job_id}/metrics?start_step=0&end_step=100&limit=1000
```

**关键参数**:
- `start_step`: 起始步数（可选）
- `end_step`: 结束步数（可选）
- `limit`: 最大返回记录数（默认 1000，最大 10000）

**UI 操作**:
1. 点击右上角的"历史回看"按钮
2. 进入历史模式后，会显示步数范围选择器
3. 输入起始步数和结束步数
4. 系统自动加载该范围内的历史数据
5. 图表显示历史数据
6. 点击"退出历史"返回实时模式

---

### 1.5.3 异常告警提示 ✅

**功能描述**:
- 在页面顶部显示异常告警横幅
- 显示健康评分卡片（0-100 分）
- 异常按严重程度显示不同颜色标签
- 提供诊断建议和解决方案
- 支持手动刷新诊断

**API 接口**:
```
GET /api/v1/monitoring/{job_id}/anomalies/detected  # 获取异常列表
GET /api/v1/monitoring/{job_id}/health              # 获取健康状态
POST /api/v1/monitoring/{job_id}/diagnose           # 手动触发诊断
POST /api/v1/monitoring/{job_id}/metrics/sync       # 同步指标到数据库
```

**异常严重程度**:
- `critical` (红色): 严重异常，训练应该停止
- `high` (橙色): 高危异常，需要立即关注
- `medium` (黄色): 中等异常，建议检查
- `low` (蓝色): 轻微异常，仅供参考

**健康评分**:
- 100: 完全健康（绿色）
- 90-99: 轻微警告（黄色）
- 70-89: 中等问题（橙色）
- 0-69: 严重问题（红色）

**UI 组件**:
1. **异常告警横幅** (顶部红色边框卡片):
   - 显示最多 3 个最新异常
   - 每个异常显示：严重程度标签、消息、步数
   - 可关闭（点击 X 按钮）
   - 操作按钮：查看详情、刷新诊断

2. **健康评分卡片**:
   - 大号健康评分数字（根据分数变色）
   - 健康状态图标（对勾/警告/叉号）
   - 最后检查时间
   - 诊断建议列表

**UI 操作**:
1. 选择一个训练任务后自动加载异常和健康状态
2. 如有异常，顶部会显示红色告警横幅
3. 查看健康评分卡片了解整体状态
4. 点击"同步诊断"按钮手动触发同步和诊断
5. 点击告警横幅的"刷新诊断"重新检测

---

## 数据格式兼容

### 后端 API 返回格式

后端 `/monitoring/{job_id}/metrics` 返回嵌套格式：

```json
{
  "job_id": "xxx",
  "metrics": [
    {
      "step": 1,
      "epoch": 0,
      "timestamp": "2026-01-08T...",
      "loss": {
        "actor_loss": 2.5,
        "critic_loss": 1.2,
        "total_loss": 3.7
      },
      "reward": {
        "mean": 0.5,
        "std": 0.1,
        "max": 0.8,
        "min": 0.2
      },
      "kl": {
        "mean": 0.05,
        "max": 0.08
      },
      "gradient": {
        "actor_norm": 1.5,
        "critic_norm": 0.8
      },
      "performance": {
        "tokens_per_second": 1000,
        "step_time": 2.0,
        "gpu_memory_allocated": 42.5
      },
      "has_anomaly": false,
      "anomaly_type": null,
      "anomaly_message": null
    }
  ]
}
```

### 前端数据访问

前端代码兼容两种格式（嵌套和扁平）：

```javascript
// 使用可选链和备用值
const totalLoss = metric.loss?.total_loss || metric.total_loss
const rewardMean = metric.reward?.mean || metric.reward_mean
const klMean = metric.kl?.mean || metric.kl_divergence
const tokensPerSec = metric.performance?.tokens_per_second || metric.tokens_per_second
```

---

## 测试步骤

### 准备工作

1. 启动后端服务：
```bash
cd /Users/weixiaochen/Desktop/Tutor/S4/train_platform
python -m uvicorn training_platform.api.main:app --reload --port 8000
```

2. 启动前端服务：
```bash
cd frontend
npm run dev
```

3. 准备测试数据（使用 Phase 1 集成测试）：
```bash
python tests/integration_test_phase1.py
```

### 测试用例

#### 测试 1: 多指标图表切换

**步骤**:
1. 打开浏览器访问 `http://localhost:5173`
2. 进入"监控"页面
3. 选择一个有数据的训练任务
4. 找到"选择要显示的指标"区域
5. 点击不同的指标按钮

**预期结果**:
- 默认显示"损失曲线"和"奖励曲线"两个图表
- 点击指标按钮后，图表动态增加/减少
- 选中的指标按钮显示为蓝色背景
- 未选中的指标按钮显示为灰色背景
- 图表以 2 列布局显示
- 每个图表正确显示对应的指标数据

#### 测试 2: 指标历史回看

**步骤**:
1. 在监控页面选择一个任务
2. 点击右上角的"历史回看"按钮
3. 观察页面变化
4. 输入起始步数 "0" 和结束步数 "10"
5. 等待数据加载
6. 查看图表数据
7. 点击"退出历史"

**预期结果**:
- 点击"历史回看"后按钮变为蓝色，文字变为"退出历史"
- 出现步数范围选择器
- 自动刷新复选框变为禁用状态
- 输入步数后自动查询历史数据
- 显示"(共 X 条记录)"
- 图表显示指定范围内的数据
- 点击"退出历史"后恢复实时模式

#### 测试 3: 异常告警提示

**步骤**:
1. 运行集成测试生成异常数据：
```bash
python tests/integration_test_phase1.py
```
2. 在监控页面选择测试任务
3. 观察页面顶部的告警横幅
4. 查看健康评分卡片
5. 点击"同步诊断"按钮
6. 观察告警信息更新

**预期结果**:
- 如有异常，顶部显示红色边框的告警横幅
- 告警横幅显示异常的严重程度标签（颜色编码）
- 显示异常消息和发生的步数
- 健康评分卡片显示评分数字（带颜色）
- 显示对应的健康状态图标
- 如有建议，显示诊断建议列表
- 点击"同步诊断"后数据刷新
- 点击告警横幅的 X 按钮可关闭横幅

#### 测试 4: 实时更新（WebSocket）

**步骤**:
1. 选择一个正在运行的训练任务
2. 确保"自动刷新"已勾选
3. 观察图表是否实时更新
4. 观察指标卡片是否实时更新
5. 如果训练过程中出现异常，观察是否实时弹出告警

**预期结果**:
- 图表数据实时增长
- 当前步数、损失、奖励等指标卡片实时更新
- 如有新异常，自动显示告警横幅
- WebSocket 断开后自动重连

---

## API 测试

### 使用 curl 测试后端接口

1. **获取指标历史**:
```bash
# 获取所有指标
curl -X GET "http://localhost:8000/api/v1/monitoring/{job_uuid}/metrics"

# 获取指定范围
curl -X GET "http://localhost:8000/api/v1/monitoring/{job_uuid}/metrics?start_step=0&end_step=30"
```

2. **获取异常列表**:
```bash
curl -X GET "http://localhost:8000/api/v1/monitoring/{job_uuid}/anomalies/detected"
```

3. **获取健康状态**:
```bash
curl -X GET "http://localhost:8000/api/v1/monitoring/{job_uuid}/health"
```

4. **手动诊断**:
```bash
curl -X POST "http://localhost:8000/api/v1/monitoring/{job_uuid}/diagnose"
```

5. **同步指标**:
```bash
curl -X POST "http://localhost:8000/api/v1/monitoring/{job_uuid}/metrics/sync"
```

---

## 已知问题和限制

1. **历史回看性能**:
   - 当查询的步数范围过大时（>10000 步），可能会有延迟
   - 建议使用分页或限制查询范围

2. **WebSocket 稳定性**:
   - 网络不稳定时可能断开连接
   - 已实现自动重连机制

3. **异常检测延迟**:
   - 平台层异常检测需要手动触发或定期执行
   - 训练层（PlatformCallback）异常检测是实时的

---

## 后续优化建议

1. **多指标图表切换**:
   - 添加图表布局选项（1列、2列、3列）
   - 支持图表拖拽排序
   - 添加图表导出功能（PNG/PDF）

2. **指标历史回看**:
   - 添加时间范围选择（按日期时间）
   - 添加快捷时间范围按钮（最近1小时、最近1天等）
   - 支持指标对比（选择两个时间段对比）

3. **异常告警提示**:
   - 添加邮件/Slack 通知
   - 支持自定义告警规则
   - 添加告警历史记录
   - 支持告警静音功能

---

## 总结

**Phase 1.5 完成度**: 100% ✅

**新增功能**:
- ✅ 多指标图表切换（6 种指标类型）
- ✅ 指标历史回看（支持步数范围查询）
- ✅ 异常告警提示（健康评分 + 异常列表 + 诊断建议）

**修改文件**:
- `frontend/src/api/index.js` - 新增 4 个 API 接口
- `frontend/src/views/MonitoringView.vue` - 完全重写监控页面

**后端支持**:
- ✅ 所有后端 API 已在 Phase 1.4 中实现
- ✅ 数据格式兼容（嵌套结构）

**用户体验提升**:
- 更灵活的指标选择和展示
- 方便的历史数据回看
- 清晰的异常告警和健康状态展示

---

**Phase 1.5 成功完成！** 🎉

现在训练平台的监控功能已经非常完善，提供了强大的可视化和诊断能力！
