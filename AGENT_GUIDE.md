# Long-Running Agent Guide - LLM训练平台

基于Anthropic的"Effective harnesses for long-running agents"方案。

## 项目概述

**目标**: 完成LLM训练平台的三大核心功能
1. **训练算法实现** - 8种算法（SFT, GRPO, DPO, PPO等）
2. **数据工厂** - 清洗、去重、质量评估、格式转换
3. **系统化评估** - Benchmark集成、自动评测、报告生成

**技术栈**:
- 后端: FastAPI + SQLModel + Celery
- 前端: Vue 3 + Pinia + ECharts
- 训练框架: verl (git submodule)

## 关键文件

| 文件 | 用途 |
|------|------|
| `init.sh` | 环境检查脚本 |
| `claude-progress.txt` | 进度日志 |
| `feature_list.json` | 功能列表(22个功能) |
| `DESIGN.md` | 完整设计文档 |
| `TASKS.md` | 历史任务清单 |

## Agent会话协议

### 开始会话

每次会话必须执行:

```bash
# 1. 获取当前目录
pwd

# 2. 读取进度和最近工作
cat claude-progress.txt
git log --oneline -10

# 3. 读取功能列表
cat feature_list.json

# 4. 检查服务状态
bash init.sh

# 5. 在实现新功能前，验证基本功能正常
curl http://localhost:8001/api/v1/monitoring/dashboard
```

### 会话中工作

**关键规则**:

1. **每次只实现一个功能** - 不要同时处理多个功能
2. **充分测试** - 编写测试用例，手动验证
3. **不修改功能描述** - 只修改 `passes` 字段
4. **频繁提交** - 小步快跑，每完成一个步骤就提交

### 结束会话

结束前必须:

1. **提交所有更改**
   ```bash
   git add -A
   git commit -m "feat(ALG-001): 实现SFT监督微调算法"
   ```

2. **更新 claude-progress.txt**
   - 添加新的SESSION条目
   - 记录完成了什么
   - 记录当前状态
   - 列出下一步
   - 记录已知问题

3. **更新 feature_list.json**
   - 只将 `passes: false` 改为 `passes: true`
   - 在 `notes` 中添加备注

## 功能优先级

按以下顺序处理:

### Critical (必须优先)
1. ALG-001: SFT监督微调 - 基础算法
2. ALG-002: GRPO组相对策略优化 - verl核心
3. ALG-003: DPO直接偏好优化 - 主流对齐
4. DATA-001: 数据清洗Pipeline
5. DATA-002: 数据去重检测
6. EVAL-001: GSM8K数学推理评测
7. EVAL-002: MATH高等数学评测

### High (重要)
8. ALG-004: PPO近端策略优化
9. DATA-003: 数据质量评估
10. DATA-004: 数据格式转换
11. EVAL-003: HumanEval代码生成评测
12. EVAL-004: 自动评测触发
13. UI-001: 数据工厂前端界面
14. UI-002: 评估系统前端界面

### Medium/Low (后续迭代)
- ALG-005 ~ ALG-008
- DATA-005, DATA-006
- EVAL-005, EVAL-006

## 代码位置参考

### 训练算法
```
training_platform/core/verl_adapter.py    # 配置生成
training_platform/core/ray_runner.py      # Ray执行
training_platform/api/routers/jobs.py     # 任务API
```

### 数据工厂
```
training_platform/api/routers/training_dataset.py  # 数据集API
training_platform/core/database.py                  # 数据模型
```

### 评估系统
```
training_platform/api/routers/evaluation.py  # 评估API
training_platform/core/inference.py           # 推理
```

### 前端
```
frontend/src/views/           # 页面组件
frontend/src/stores/          # Pinia状态
frontend/src/api/             # API客户端
```

## 测试要求

每个功能完成后:

1. **单元测试**: 核心逻辑有测试覆盖
2. **集成测试**: 使用小模型(Qwen2.5-0.5B)测试
3. **API测试**: curl 或 pytest 验证接口
4. **前端测试**: 浏览器手动验证

## Git提交规范

格式: `type(scope): description`

```
feat(ALG-001): 实现SFT监督微调算法
feat(DATA-001): 添加数据清洗Pipeline
fix(EVAL-001): 修复GSM8K答案提取错误
refactor(api): 重构评估路由
docs: 更新API文档
test: 添加GRPO单元测试
```

## 常见失败模式

| 问题 | 解决方案 |
|------|----------|
| 过早宣布完成 | 检查feature_list.json中所有critical功能是否passing |
| 留下未文档化的bug | 更新progress文件的known issues |
| 过早标记功能完成 | 端到端测试后再标记 |
| 一次做太多 | 每会话只做一个功能 |
| 破坏现有功能 | 先运行init.sh检查服务状态 |
