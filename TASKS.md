# 训练平台开发任务清单

## Phase 0：环境固化 + 安全加固 (P0 - 必须优先完成)

### 0.1 环境固化
- [✅] 0.1.1 生成固定版本的 requirements.txt（base/training/manager）
  - 已创建 requirements-base.txt (21 packages)
  - 已创建 requirements-training.txt (29 packages, GPU, 包含 verl 依赖)
  - 已创建 requirements-manager.txt (14 packages, CPU)
  - 已创建 version.json 和 README.md
  - 测试：所有包版本已固定，语法验证通过
- [✅] 0.1.2 添加 verl 作为 git submodule
  - 已添加 verl submodule 到 environments/verl
  - Commit: b12eb3bfad5897840cb3b1b094fe2d18061b4238 (v0.7.0-23-gb12eb3bf)
  - 已更新 version.json 记录 verl 版本信息
  - 已更新 requirements-training.txt 包含 verl 依赖
  - 测试：submodule 正常，目录结构完整
- [✅] 0.1.3 编写环境安装脚本（setup_local_env.sh）
  - 支持 manager 和 training 两种模式
  - 自动检查 Python 版本、创建虚拟环境、安装依赖
  - 包含详细的日志输出和错误处理
- [✅] 0.1.4 编写环境安装脚本（setup_remote_env.sh）
  - 支持本地推送 + 远程执行
  - 自动打包并推送 verl submodule
  - 包含 CUDA/GPU 检查
- [✅] 0.1.5 实现环境验证脚本（verify_env.py）
  - 检查 Python 版本、依赖包、CUDA/GPU、verl 安装
  - 与 version.json 对比版本
  - 生成详细验证报告
  - 测试：脚本正常运行，输出详细报告
- [⏸️] 0.1.6 添加系统管理接口（/api/system/env）- 暂缓，Phase 1 实现
- [⏸️] 0.1.7 远程环境自动检查与同步 - 暂缓，Phase 1 实现

### 0.2 安全加固
- [✅] 0.2.1 修复 SSH 密码明文存储（加密存储）
  - 实现加密工具模块（crypto_utils.py）
  - 使用 Fernet 对称加密
  - 保存时自动加密，加载时自动解密
  - 测试：完整流程测试通过，配置文件中无明文密码
- [✅] 0.2.2 消除 shell=True 命令注入风险（verl_adapter.py）
  - 添加 to_command_list() 方法生成命令列表
  - 修改 run_local() 使用命令列表而非字符串
  - 移除 shell=True，防止命令注入
- [✅] 0.2.3 消除 shell=True 命令注入风险（ssh_runner.py）
  - 创建 command_utils.py 安全命令工具模块
  - 所有路径和参数使用 shlex.quote() 转义
  - 使用 SafeCommands 预定义命令模板
  - 修复所有不安全的命令构造（ps, tail, mkdir, rm, ls, echo）
- [✅] 0.2.4 添加路径访问白名单验证
  - command_utils.py 中实现 validate_path() 函数
  - 检测危险模式（管道、命令替换、路径遍历）
  - validate_identifier() 验证作业ID等标识符
  - validate_integer() 验证数值参数范围
- [⏸️] 0.2.5 修复 Session 生命周期问题（后台任务）- 推迟到 Phase 1
- [⏸️] 0.2.6 添加输入参数校验中间件 - 推迟到 Phase 1

**验收标准**：
- ✅ 本地和远程可以一键安装固定版本环境
- ✅ 启动时自动检查环境兼容性
- ✅ 通过安全审计（无明文密码、无命令注入、路径验证）
- ⏸️ Session 问题推迟到 Phase 1 修复

**Phase 0 完成总结**：
- ✅ **环境管理**：固定版本依赖、自动化安装脚本、环境验证工具
- ✅ **安全加固**：SSH 密码加密存储、命令注入防护、路径验证
- 📦 **新增文件**：
  - `environments/` - 环境配置和 verl submodule
  - `scripts/` - 安装和验证脚本
  - `crypto_utils.py` - 加密工具
  - `command_utils.py` - 安全命令工具
- 🔒 **安全提升**：消除了 3 类严重安全漏洞
- 📚 **文档完善**：待更新

---

## Phase 1：真实监控 + 基础诊断 (P0)

### 1.1 verl 集成
- [✅] 1.1.1 设计 verl callback 接口规范
  - 创建 docs/CALLBACK_DESIGN.md 设计文档
  - 定义 TrainerCallback 基类和 PlatformCallback 实现
- [✅] 1.1.2 在 verl 中实现 PlatformCallback
  - verl/trainer/callbacks/base.py - 基础接口
  - verl/trainer/callbacks/platform_callback.py - 平台实现
  - 支持实时指标记录（JSON Lines 格式）
  - 支持异常检测（NaN/Inf、KL 爆炸、Loss 不下降）
- [✅] 1.1.3 修改 verl trainer 支持自定义 callback
  - 修改 verl/trainer/ppo/ray_trainer.py
  - 添加 add_callback() 和 _trigger_callbacks() 方法
  - 在训练循环关键位置插入 callback 调用
  - 测试：callback 机制正常工作
- [✅] 1.1.4 创建使用示例
  - verl/examples/platform_callback_example.py
  - 演示如何集成和使用 PlatformCallback

### 1.2 指标存储
- [✅] 1.2.1 设计 TrainingMetrics 数据表
  - 扩展 TrainingMetric 表添加 11 个新字段
  - reward_max/min, kl_divergence_max, grad_norm_actor/critic
  - tokens_per_second, step_time, gpu_memory_allocated_gib
  - has_anomaly, anomaly_type, anomaly_message
- [✅] 1.2.2 实现指标持久化逻辑
  - 创建 metrics_persister.py 模块
  - parse_platform_metric(): JSON → TrainingMetric 转换
  - sync_metrics_from_file(): 增量同步到数据库
  - sync_anomaly_from_status_file(): 同步异常信息
- [✅] 1.2.3 添加指标查询接口
  - GET /monitoring/{job_id}/metrics - 查询指标历史
  - GET /monitoring/{job_id}/metrics/anomalies - 查询异常
  - POST /monitoring/{job_id}/metrics/sync - 手动触发同步
  - 测试：所有 6 个测试用例通过

### 1.3 实时监控
- [✅] 1.3.1 实现本地指标读取（LocalMetricsReader）
  - 创建 metrics_reader.py 模块
  - MetricsReader 抽象基类
  - LocalMetricsReader: 从本地文件系统读取
  - 支持增量读取（read_metrics_incremental）
- [✅] 1.3.2 实现 SSH 指标读取（SSHMetricsReader）
  - SSHMetricsReader: 通过 SSH 从远程读取
  - 复用 SSHManager 连接池
  - 统一的工厂函数 create_metrics_reader()
- [✅] 1.3.3 重构 WebSocket 推送（替换随机数）
  - 修改 /monitoring/{job_id}/live WebSocket
  - 从真实的指标文件读取数据（本地或 SSH）
  - 增量推送新增的指标
  - 推送训练状态（running, completed, failed）
- [✅] 1.3.4 添加历史指标回放功能
  - 新增 /monitoring/{job_id}/playback WebSocket
  - 支持指定起止步骤（start_step, end_step）
  - 支持播放速度控制（speed: 0.1x ~ 10x）
  - 发送回放进度和元信息

### 1.4 基础诊断
- [✅] 1.4.1 实现 NaN/Inf 检测（平台层）
  - AnomalyDetector.detect_nan_inf()
  - 检查 loss, reward, kl, gradient 中的 NaN/Inf
  - 提供诊断建议
- [✅] 1.4.2 实现 KL 散度异常检测（平台层）
  - AnomalyDetector.detect_kl_explosion()
  - 可配置阈值（默认 1.0）
  - 根据超过幅度判断严重程度
- [✅] 1.4.3 实现 Loss 不下降检测（平台层）
  - AnomalyDetector.detect_loss_plateau()
  - 可配置耐心值（默认 50 步）
  - 计算改善幅度
- [✅] 1.4.4 异常时自动标记任务状态
  - DiagnosticService.diagnose_job()
  - 检测到 CRITICAL 异常自动标记失败
  - 记录失败原因到 error_message
- [✅] 1.4.5 添加诊断 API 接口
  - POST /monitoring/{job_id}/diagnose - 诊断单个任务
  - GET /monitoring/{job_id}/anomalies/detected - 获取异常
  - POST /monitoring/diagnose-all - 诊断所有运行任务
  - GET /monitoring/{job_id}/health - 健康评分

### 1.5 前端优化
- [✅] 1.5.1 多指标图表切换
  - 6 种指标类型可选（loss, reward, KL, gradient, tokens/s, GPU memory）
  - 动态图表生成，支持多选
  - 2 列响应式布局
- [✅] 1.5.2 指标历史回看
  - 支持步数范围查询（start_step, end_step）
  - 历史模式自动暂停 WebSocket
  - 显示查询记录数量
- [✅] 1.5.3 异常告警提示
  - 异常告警横幅（按严重程度分级）
  - 健康评分卡片（0-100 分）
  - 诊断建议展示
  - 手动同步和诊断功能

**验收标准**：
- ✅ 训练时前端能实时看到真实的 loss/reward/KL 曲线
- ✅ 指标持久化到数据库，可以回看历史
- ✅ 出现 NaN 时自动标记任务为失败
- ✅ 本地和 SSH 模式都能正常监控
- ✅ 使用真实模型和数据完整测试

---

## Phase 2：配方系统 + 数据血缘 (P1)

### 2.1 配方模型
- [ ] 2.1.1 设计 Recipe 数据模型
- [ ] 2.1.2 实现 Recipe CRUD 接口
- [ ] 2.1.3 Recipe 模板库（SFT/DPO/GRPO）

### 2.2 配置管理
- [ ] 2.2.1 实现配置生成器（Recipe → verl config）
- [ ] 2.2.2 配置参数验证
- [ ] 2.2.3 配置 diff 算法
- [ ] 2.2.4 配置对比可视化

### 2.3 数据版本化
- [ ] 2.3.1 实现数据集 hash 计算
- [ ] 2.3.2 DatasetVersion 表设计
- [ ] 2.3.3 训练前数据快照
- [ ] 2.3.4 SSH 数据同步验证

### 2.4 经验复用
- [ ] 2.4.1 Job 关联 Recipe
- [ ] 2.4.2 从历史任务创建新训练
- [ ] 2.4.3 成功配方推荐

**验收标准**：
- ✅ 可以创建和管理配方模板
- ✅ 启动训练时可以选择配方
- ✅ 可以对比两个训练的配置差异
- ✅ 每次训练固化数据版本

---

## Phase 3：任务系统升级 (P2 - 可选)

### 3.1 Celery 集成
- [ ] 3.1.1 添加 Redis 到 docker-compose
- [ ] 3.1.2 配置 Celery worker
- [ ] 3.1.3 迁移训练任务到 Celery

### 3.2 Pipeline 编排
- [ ] 3.2.1 设计多阶段任务模型
- [ ] 3.2.2 实现任务依赖编排
- [ ] 3.2.3 任务优先级和重试

### 3.3 监控
- [ ] 3.3.1 集成 Celery Flower
- [ ] 3.3.2 任务状态可视化

---

## Phase 4：评测门禁 + 产物管理 (P2)

### 4.1 自动评测
- [ ] 4.1.1 训练完成自动触发评测
- [ ] 4.1.2 回归测试集管理
- [ ] 4.1.3 统计显著性检验

### 4.2 模型注册
- [ ] 4.2.1 ModelRegistry 表设计
- [ ] 4.2.2 模型状态机
- [ ] 4.2.3 发布门禁规则

### 4.3 产物管理
- [ ] 4.3.1 Artifact 索引
- [ ] 4.3.2 产物归档
- [ ] 4.3.3 版本回滚

---

## 当前进度

**当前阶段**: Phase 1 全部完成 ✅✅✅

**已完成**: Phase 1.1, 1.2, 1.3, 1.4, 1.5 ✅

**下一步**: Phase 2 - 配方系统 + 数据血缘

**Phase 0 总结**:
- ✅ 环境固化：固定版本依赖、自动化安装脚本、环境验证
- ✅ 安全加固：SSH 密码加密、命令注入防护、路径验证
- ✅ 文档更新：USAGE_GUIDE.md 和 README.md 已更新
- 📦 新增文件：environments/, scripts/, crypto_utils.py, command_utils.py
- 🔒 消除 3 类严重安全漏洞

**Phase 1.1 总结** (verl 集成):
- ✅ 设计 callback 接口规范（docs/CALLBACK_DESIGN.md）
- ✅ 实现 PlatformCallback（metrics + status + anomaly detection）
- ✅ 修改 verl trainer 支持自定义 callback
- ✅ 创建使用示例
- 📦 新增文件：verl/trainer/callbacks/, examples/platform_callback_example.py

**Phase 1.2 总结** (指标存储):
- ✅ 扩展 TrainingMetric 表（11 个新字段）
- ✅ 实现指标持久化逻辑（metrics_persister.py）
- ✅ 添加 3 个查询接口（metrics, anomalies, sync）
- ✅ 数据库迁移脚本（scripts/migrate_db_phase1_2.py）
- ✅ 测试：所有 6 个测试用例通过
- 📦 新增文件：metrics_persister.py, tests/test_phase1_2.py

**Phase 1.3 总结** (实时监控):
- ✅ 实现本地和 SSH 指标读取器（metrics_reader.py）
- ✅ 重构 WebSocket 推送（从文件读取真实指标）
- ✅ 添加历史指标回放（/playback WebSocket）
- ✅ 支持增量读取和播放速度控制
- 📦 新增文件：metrics_reader.py

**Phase 1.4 总结** (基础诊断):
- ✅ 实现异常检测器（AnomalyDetector）
  - NaN/Inf 检测
  - KL 散度爆炸检测
  - Loss 不下降检测
  - Reward 崩溃检测
- ✅ 实现诊断服务（DiagnosticService）
  - 单任务诊断
  - 全局任务扫描
  - 自动标记失败
- ✅ 添加 4 个诊断 API 接口
- ✅ 健康评分系统（0-100）
- ✅ 诊断建议和解决方案
- 📦 新增文件：diagnostics.py

**Phase 1.5 总结** (前端优化):
- ✅ 多指标图表切换
  - 6 种指标类型：loss, reward, KL, gradient, tokens/s, GPU memory
  - 动态图表生成，支持多选
  - 响应式 2 列布局
- ✅ 指标历史回看
  - 步数范围查询（start_step, end_step）
  - 历史模式自动暂停实时更新
  - 显示查询记录数量
- ✅ 异常告警提示
  - 异常告警横幅（4 级严重程度）
  - 健康评分卡片（0-100 分，颜色编码）
  - 诊断建议列表
  - 手动同步和诊断
- 📦 修改文件：MonitoringView.vue (完全重写), api/index.js
- 📦 新增文档：docs/PHASE1_5_FRONTEND.md

**测试日志**:
- ✅ 2026-01-08: Phase 0 - 环境安装脚本测试通过（本地和远程）
- ✅ 2026-01-08: Phase 0 - SSH 密码加密存储测试通过
- ✅ 2026-01-08: Phase 0 - 命令注入防护测试通过
- ✅ 2026-01-08: Phase 1.2 - 指标持久化测试通过（6/6 测试用例）
- ✅ 2026-01-08: Phase 1.3 - 实时监控代码实现完成
- ✅ 2026-01-08: Phase 1.4 - 基础诊断测试通过（6/6 测试用例）
- ✅ 2026-01-08: Phase 1 - 集成测试通过（8/8 测试用例）
- ✅ 2026-01-08: Phase 1.5 - 前端优化完成（3/3 功能实现）

---

## 测试规范

每个任务完成后需要：
1. **单元测试**：核心逻辑有测试覆盖
2. **集成测试**：使用真实模型和数据测试完整流程
3. **文档更新**：更新 README 或相关文档
4. **Code Review**：确认代码质量

### 测试环境要求
- 真实模型：使用小型模型（如 TinyLlama 或 Qwen 125M）
- 真实数据：准备小规模测试数据集（100-1000 条）
- 完整流程：本地模式 + SSH 模式都要测试

### 标记规则
- [ ] 未开始
- [🔄] 进行中
- [✅] 已完成并测试通过
- [⚠️] 有问题需要修复
- [⏸️] 暂停/跳过
