# 用户训练操作指南 - 完整教程

**适用算法**: SFT (全量/LoRA)、DPO、PPO、GRPO
**平台**: RLHF Training Platform
**前端地址**: `http://localhost:5173`
**API地址**: `http://localhost:8000`

---

## 📋 目录

- [前置准备](#前置准备)
- [算法 1: SFT 全量微调](#算法-1-sft-全量微调)
- [算法 2: SFT LoRA 微调](#算法-2-sft-lora-微调)
- [算法 3: DPO 训练](#算法-3-dpo-训练)
- [算法 4: PPO 训练](#算法-4-ppo-训练)
- [算法 5: GRPO 训练](#算法-5-grpo-训练)
- [常见问题](#常见问题)

---

## 前置准备

### 1. 启动服务

```bash
# 1. 启动后端服务
cd train_platform
python -m training_platform.api.main

# 2. 启动 Celery Worker (新终端)
celery -A training_platform.core.celery_tasks worker --loglevel=info

# 3. 启动前端 (新终端)
cd frontend
npm run dev
```

### 2. 访问前端

浏览器打开：`http://localhost:5173`

### 3. 检查数据集

确保以下数据集文件存在：
```
train_platform/
├── datasets/
│   ├── sales_sft.jsonl      # SFT 数据
│   ├── sales_dpo.jsonl      # DPO 数据
│   ├── ppo_general.json     # PPO 数据
│   ├── sales_grpo.jsonl     # GRPO 数据
│   └── ...
```

---

## 算法 1: SFT 全量微调

**Supervised Fine-Tuning (全量参数更新)**

### Step 1: 打开前端首页

1. 浏览器访问 `http://localhost:5173`
2. 你会看到左侧导航栏，包括：
   - 训练任务 (Jobs)
   - 数据集 (Datasets)
   - 监控 (Monitoring)
   - Pipeline
   - 设置 (Settings)

### Step 2: 进入训练任务页面

1. 点击左侧导航栏的 **"训练任务"** (Jobs)
2. 你会看到任务列表页面，右上角有 **"创建新任务"** 按钮

### Step 3: 点击"创建新任务"

1. 点击右上角的 **"+ 创建新任务"** 按钮
2. 弹出创建任务表单

### Step 4: 填写基本信息

在弹出的表单中填写：

```
任务名称:     SFT Sales Training (Full)
描述:         使用销售数据进行 SFT 全量微调
算法:         SFT ⬅️ 在下拉框中选择
```

### Step 5: 选择模型和数据集

```
模型路径:     Qwen/Qwen2.5-0.5B
训练数据:     ./datasets/sales_sft.jsonl
验证数据:     (留空)
```

💡 **提示**:
- 模型路径可以是 HuggingFace 模型名或本地路径
- 数据集路径相对于项目根目录

### Step 6: 配置训练参数

```
训练轮数 (num_epochs):        1
学习率 (learning_rate):       1e-5
批次大小 (batch_size):        2
上下文长度 (context_length):  512
保存步数 (save_steps):        20
评估步数 (eval_steps):        20
```

### Step 7: 配置 GPU 资源

```
GPU 数量 (num_gpus):          1
GPU 类型:                     A100-80G (或根据实际情况选择)
```

### Step 8: 配置运行模式

**选项 A: 本地运行**
```
运行模式:     Local
```

**选项 B: SSH 远程运行**
```
运行模式:     SSH
SSH 主机:     connect.westc.gpuhub.com
SSH 端口:     27192
SSH 用户名:   root
SSH 密码:     (输入密码)
工作目录:     ~/verl_jobs
```

### Step 9: 确认并创建任务

1. 检查所有配置是否正确
2. 确保 **LoRA 开关关闭**（全量微调不使用 LoRA）
3. 点击 **"创建任务"** 按钮
4. 系统会显示 "任务创建成功" 提示

### Step 10: 监控训练进度

#### 10.1 查看任务列表

- 返回任务列表页面
- 找到刚创建的任务，状态应为 **"PENDING"** 或 **"RUNNING"**

#### 10.2 进入实时监控

1. 点击任务行的 **"监控"** 按钮
2. 或点击左侧导航栏 **"监控"**，选择任务

#### 10.3 查看实时 Metrics

在监控页面你会看到：

**实时图表**:
- Loss 曲线
- Learning Rate 曲线
- 训练进度 (Step/Epoch)

**实时日志**:
- 训练输出
- 错误信息（如果有）

**Checkpoint 列表**:
- 保存的模型检查点
- 每个检查点的 step 和 metrics

#### 10.4 等待训练完成

训练完成后，状态会变为 **"COMPLETED"**

**预计时间**:
- Qwen2.5-0.5B + 1 epoch + RTX 5090: ~10-15 秒
- 更大模型或更多 epoch 会更久

### Step 11: 查看训练结果

1. 返回任务详情页面
2. 查看 **"最终 Metrics"**:
   ```json
   {
     "loss": 2.345,
     "learning_rate": 1e-5,
     "step": 100,
     "epoch": 1
   }
   ```

3. 查看 **"Checkpoint 列表"**:
   - 最新的模型路径
   - 可用于后续推理或继续训练

---

## 算法 2: SFT LoRA 微调

**Supervised Fine-Tuning with LoRA (低秩适配)**

### Step 1-5: 同 SFT 全量微调

按照上面 SFT 全量微调的 Step 1-5 操作

### Step 6: 配置训练参数

```
任务名称:     SFT Sales Training (LoRA)
描述:         使用销售数据进行 SFT LoRA 微调
算法:         SFT

训练轮数:     1
学习率:       1e-4  ⬅️ LoRA 可以用更大的学习率
批次大小:     2
上下文长度:   512
```

### Step 7: 🔥 启用 LoRA 配置

**关键步骤 - 展开 LoRA 配置**

1. 在创建任务表单中，找到 **"LoRA 配置"** 区域
2. 打开 **"启用 LoRA"** 开关 ✅

3. 配置 LoRA 参数：
   ```
   LoRA Rank:     8      ⬅️ 秩，越大参数越多
   LoRA Alpha:    16     ⬅️ 缩放因子，通常是 rank 的 2 倍
   ```

💡 **LoRA 参数说明**:
- **Rank (秩)**: LoRA 矩阵的维度，越大模型容量越大，但训练越慢
  - 推荐值: 4-32
  - 小任务: 4-8
  - 大任务: 16-32

- **Alpha (缩放因子)**: 控制 LoRA 权重的缩放
  - 推荐值: rank * 2
  - 如 rank=8，则 alpha=16

### Step 8: 配置 GPU 和运行模式

```
GPU 数量:     1
运行模式:     Local 或 SSH (根据实际情况)
```

💡 **LoRA 优势**:
- 显存占用更小（约 1/10）
- 训练速度更快
- 可以用更大的学习率
- 适合数据量较小的场景

### Step 9: 创建并监控

1. 点击 **"创建任务"**
2. 进入监控页面
3. 观察训练进度

### Step 10: 对比全量 vs LoRA

训练完成后，对比两种方式：

| 指标 | SFT 全量 | SFT LoRA |
|------|----------|----------|
| 显存占用 | ~10 GB | ~2 GB |
| 训练速度 | 基准 | 快 2-3x |
| 模型大小 | 完整模型 | 仅 LoRA 权重 (~MB) |
| 效果 | 最佳 | 接近全量 (80-95%) |

---

## 算法 3: DPO 训练

**Direct Preference Optimization (偏好对齐)**

### Step 1: 准备 DPO 数据集

DPO 需要偏好对数据，格式如下：

```jsonl
{
  "prompt": "用户问题",
  "chosen": "好的回答",
  "rejected": "差的回答"
}
```

确保数据集 `./datasets/sales_dpo.jsonl` 存在。

### Step 2: 创建 DPO 任务

1. 进入 **"训练任务"** 页面
2. 点击 **"+ 创建新任务"**

### Step 3: 填写基本信息

```
任务名称:     DPO Sales Training
描述:         使用偏好数据进行 DPO 对齐
算法:         DPO ⬅️ 选择 DPO
```

### Step 4: 选择模型和数据

```
模型路径:     Qwen/Qwen2.5-0.5B
训练数据:     ./datasets/sales_dpo.jsonl
```

💡 **重要**:
- DPO 通常需要先进行 SFT 训练
- 模型路径可以指向 SFT 训练后的 checkpoint

### Step 5: 配置 DPO 参数

```
训练轮数:     1
学习率:       5e-6  ⬅️ DPO 通常用更小的学习率
批次大小:     2
上下文长度:   512

Beta (β):     0.1   ⬅️ DPO 的温度参数
```

💡 **DPO Beta 参数**:
- 控制偏好强度
- 越大，模型越倾向于 chosen，但可能过拟合
- 推荐值: 0.1 - 0.5
- 保守: 0.1
- 激进: 0.3-0.5

### Step 6: GPU 配置

```
GPU 数量:     1
运行模式:     Local 或 SSH
```

### Step 7: 高级配置（可选）

展开 **"高级配置"**：

```
KL 系数:      0.001  ⬅️ KL 散度惩罚
参考模型:     (可选) 指向 SFT checkpoint
```

### Step 8: 创建任务

点击 **"创建任务"** 按钮

### Step 9: 监控 DPO 训练

进入监控页面，关注：

**DPO 特有 Metrics**:
- `reward_chosen`: chosen 回答的奖励
- `reward_rejected`: rejected 回答的奖励
- `reward_margin`: 奖励差距（越大越好）
- `accuracy`: 偏好预测准确率

**预期结果**:
- reward_chosen > reward_rejected
- reward_margin 逐渐增大
- accuracy > 0.5 (随机是 0.5)

### Step 10: 评估 DPO 效果

训练完成后：

1. 查看最终 metrics
2. 对比 SFT 和 DPO 模型的生成效果
3. 使用 checkpoint 进行推理测试

---

## 算法 4: PPO 训练

**Proximal Policy Optimization (近端策略优化)**

### Step 1: 准备 PPO 数据

PPO 只需要 prompt 数据：

```json
[
  {"prompt": "用户问题1"},
  {"prompt": "用户问题2"}
]
```

确保 `./datasets/ppo_general.json` 存在。

### Step 2: 配置 Reward Model

PPO 需要 Reward Model 来评估生成质量。

**选项 A: 使用阿里 DashScope API** (推荐)

1. 获取 API Key: https://dashscope.aliyun.com
2. 记录你的 API Key (例如: `sk-xxxxx`)

**选项 B: 使用本地 Reward Model**

准备本地 reward model checkpoint 路径。

### Step 3: 创建 PPO 任务

1. 进入 **"训练任务"** 页面
2. 点击 **"+ 创建新任务"**

### Step 4: 填写基本信息

```
任务名称:     PPO General Training
描述:         使用 PPO 进行强化学习对齐
算法:         PPO ⬅️ 选择 PPO
```

### Step 5: 选择模型和数据

```
模型路径:     Qwen/Qwen2.5-0.5B
训练数据:     ./datasets/ppo_general.json
```

### Step 6: 配置 PPO 参数

```
训练轮数:     1
学习率:       1e-6  ⬅️ PPO 用很小的学习率
批次大小:     2
上下文长度:   512

PPO 专用参数:
├─ KL 系数:        0.001  ⬅️ KL 散度惩罚
├─ Clip Ratio:     0.2    ⬅️ PPO clip 范围
├─ Entropy Coef:   0.0    ⬅️ 熵奖励（鼓励探索）
└─ Rollout N:      5      ⬅️ 每个 prompt 生成几条
```

💡 **PPO 参数解释**:

- **KL 系数**: 防止策略偏离参考模型太远
  - 推荐: 0.001 - 0.01

- **Clip Ratio**: PPO 核心，限制策略更新幅度
  - 推荐: 0.1 - 0.3
  - 保守: 0.1
  - 标准: 0.2

- **Rollout N**: 采样多样性
  - 越大越慢，但探索更充分
  - 推荐: 4 - 8

### Step 7: 🔥 配置 Reward Model

展开 **"Reward Model 配置"**：

**使用阿里 API**:
```
Reward 类型:      API
API Base URL:     https://dashscope.aliyuncs.com/compatible-mode/v1
API Key:          sk-xxxxx  ⬅️ 你的 API Key
API Model:        qwen-plus (或其他模型)
```

**使用本地模型**:
```
Reward 类型:      Model
模型路径:         ./models/reward_model
```

### Step 8: GPU 配置

```
GPU 数量:     1  ⬅️ PPO 显存占用大，建议单卡
运行模式:     SSH (推荐远程 GPU)
```

⚠️ **注意**:
- PPO 需要同时加载 Policy Model + Reward Model
- 显存需求约 2x SFT
- 推荐使用 32GB+ 显存的 GPU

### Step 9: 创建并监控

1. 点击 **"创建任务"**
2. 进入监控页面

### Step 10: 监控 PPO 训练

**PPO 特有 Metrics**:

```
reward:           平均奖励（越高越好）
kl_divergence:    KL 散度（应保持较小）
policy_loss:      策略损失
value_loss:       价值函数损失
approx_kl:        近似 KL（监控策略变化）
clipfrac:         被 clip 的比例
```

**预期曲线**:
- reward 逐渐上升
- kl_divergence 保持稳定（不暴涨）
- policy_loss 下降

**训练时间**:
- PPO 比 SFT 慢 3-5x
- 预计 1 epoch: 1-2 分钟 (Qwen-0.5B)

---

## 算法 5: GRPO 训练

**Group Relative Policy Optimization (组相对策略优化)**

### Step 1: 准备 GRPO 数据

GRPO 用于数学推理，数据格式：

```jsonl
{
  "prompt": "数学问题",
  "solution": "正确答案"
}
```

确保 `./datasets/sales_grpo.jsonl` 存在。

### Step 2: 创建 GRPO 任务

1. 进入 **"训练任务"** 页面
2. 点击 **"+ 创建新任务"**

### Step 3: 填写基本信息

```
任务名称:     GRPO Math Training
描述:         使用 GRPO 进行数学推理训练
算法:         GRPO ⬅️ 选择 GRPO
```

### Step 4: 选择模型和数据

```
模型路径:     Qwen/Qwen2.5-0.5B
训练数据:     ./datasets/sales_grpo.jsonl
```

### Step 5: 配置 GRPO 参数

```
训练轮数:     1
学习率:       1e-6
批次大小:     2
上下文长度:   512

GRPO 专用参数:
├─ Rollout N:          5  ⬅️ 每个问题生成几条推理路径
├─ KL 系数:            0.001
└─ Clip Ratio:         0.2
```

### Step 6: 🔥 配置 Reward Function

GRPO 使用**内置规则**作为 Reward，无需外部 API。

展开 **"Reward Function 配置"**：

```
Reward 类型:       math_verify  ⬅️ 数学验证

验证参数:
├─ 答案提取方法:    boxed        ⬅️ 从 \boxed{} 提取答案
├─ 比较方法:        exact        ⬅️ 精确匹配
└─ 答案字段:        solution     ⬅️ 数据中的答案字段名
```

💡 **Reward Function 类型**:

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| `math_verify` | 数学答案验证 | 数学推理 |
| `format_check` | 格式检查 | 结构化输出 |
| `rule_based` | 自定义规则 | 特定任务 |

**答案提取方法**:
- `boxed`: 从 `\boxed{答案}` 提取
- `last_number`: 提取最后一个数字
- `json`: 从 JSON 中提取

**比较方法**:
- `exact`: 精确匹配
- `numeric`: 数值比较（忽略格式）
- `fuzzy`: 模糊匹配

### Step 7: GPU 配置

```
GPU 数量:     1
运行模式:     Local 或 SSH
```

💡 **GRPO 优势**:
- 无需外部 Reward Model
- 适合有明确答案的任务
- 训练速度比 PPO 快
- 显存占用更小

### Step 8: 创建任务

点击 **"创建任务"** 按钮

### Step 9: 监控 GRPO 训练

**GRPO 特有 Metrics**:

```
reward:             平均奖励
reward_accuracy:    答案正确率（核心指标）
kl_divergence:      KL 散度
group_variance:     组内方差（反映采样多样性）
```

**预期效果**:
- reward_accuracy 从 ~0.2 提升到 ~0.6+
- reward 逐渐上升
- group_variance 保持一定水平（不要太小）

### Step 10: 评估 GRPO 效果

训练完成后：

1. 查看 `reward_accuracy`（最重要）
2. 对比训练前后模型的数学推理能力
3. 测试新的数学问题

**成功标志**:
- reward_accuracy > 0.5
- 模型能够正确推理训练数据中的问题

---

## 📊 算法对比表

| 算法 | 适用场景 | 是否需要 Reward | 显存占用 | 训练速度 | 难度 |
|------|----------|----------------|----------|----------|------|
| **SFT 全量** | 通用微调 | ❌ | 大 | 快 | ⭐ |
| **SFT LoRA** | 小数据微调 | ❌ | 小 | 最快 | ⭐ |
| **DPO** | 偏好对齐 | ❌ (偏好对) | 中 | 中 | ⭐⭐ |
| **PPO** | 强化学习对齐 | ✅ (API/Model) | 最大 | 慢 | ⭐⭐⭐⭐ |
| **GRPO** | 数学推理 | ✅ (规则) | 中 | 较快 | ⭐⭐⭐ |

---

## 🎯 典型训练流程

### 流程 1: 通用对话模型

```
Step 1: SFT 全量        → 获得基础对话能力
Step 2: DPO             → 对齐人类偏好
Step 3: (可选) PPO      → 进一步优化
```

### 流程 2: 特定领域微调（资源受限）

```
Step 1: SFT LoRA        → 快速适配领域数据
Step 2: (可选) DPO      → 领域偏好对齐
```

### 流程 3: 数学推理模型

```
Step 1: SFT             → 基础数学能力
Step 2: GRPO            → 推理能力优化
```

---

## 🔍 监控页面说明

### 主界面

```
┌─────────────────────────────────────────┐
│  训练监控 - [任务名称]                    │
├─────────────────────────────────────────┤
│                                         │
│  📊 实时 Metrics 图表                    │
│  ├─ Loss 曲线                           │
│  ├─ Learning Rate 曲线                  │
│  ├─ Reward 曲线 (PPO/GRPO)              │
│  └─ Accuracy 曲线 (GRPO)                │
│                                         │
├─────────────────────────────────────────┤
│  📝 实时日志                             │
│  ├─ [2024-01-09 14:32] Step 10/100     │
│  ├─ [2024-01-09 14:33] Loss: 2.345     │
│  └─ [2024-01-09 14:34] Saving ckpt...  │
│                                         │
├─────────────────────────────────────────┤
│  💾 Checkpoint 列表                      │
│  ├─ checkpoint-100  (step: 100)         │
│  ├─ checkpoint-50   (step: 50)          │
│  └─ checkpoint-20   (step: 20)          │
│                                         │
└─────────────────────────────────────────┘
```

### 图表说明

**Loss 曲线**:
- X 轴: Step 或 Epoch
- Y 轴: Loss 值
- 预期: 逐渐下降并趋于平稳

**Learning Rate 曲线**:
- 显示学习率变化
- 如果使用 warmup，前期会上升
- 后期可能会下降（如果使用 scheduler）

**Reward 曲线** (仅 PPO/GRPO):
- 显示平均奖励
- 预期: 逐渐上升

**Accuracy 曲线** (仅 GRPO):
- 显示答案正确率
- 预期: 从低到高

### 日志说明

**正常日志**:
```
[INFO] Step 10/100: loss=2.345, lr=1e-5
[INFO] Saving checkpoint at step 20
[INFO] Checkpoint saved to: /path/to/checkpoint-20
```

**警告日志**:
```
[WARNING] High memory usage: 90%
[WARNING] Gradient overflow detected, skipping step
```

**错误日志**:
```
[ERROR] CUDA out of memory
[ERROR] Failed to load checkpoint
```

---

## ⚙️ 设置页面

### 访问设置

1. 点击左侧导航栏 **"设置"** (Settings)
2. 配置默认参数

### 常用设置

**训练默认值**:
```
默认 GPU 数量:    1
默认学习率:       1e-5
默认批次大小:     2
默认运行模式:     Local
```

**SSH 默认配置**:
```
默认 SSH 主机:    connect.westc.gpuhub.com
默认 SSH 端口:    27192
默认工作目录:     ~/verl_jobs
```

**监控配置**:
```
刷新间隔:         5 秒
日志行数:         100 行
自动滚动:         开启
```

---

## ❓ 常见问题

### Q1: 任务一直显示 PENDING，不开始训练？

**可能原因**:
1. Celery Worker 未启动
2. GPU 资源不足
3. 数据集路径错误

**解决方法**:
```bash
# 检查 Celery Worker
ps aux | grep celery

# 如果没有运行，启动 Worker
celery -A training_platform.core.celery_tasks worker --loglevel=info

# 检查日志
tail -f celery.log
```

### Q2: 训练失败，显示 CUDA out of memory？

**原因**: GPU 显存不足

**解决方法**:
1. 减小 batch_size（从 2 降到 1）
2. 减小 context_length（从 512 降到 256）
3. 使用 LoRA（显存占用更小）
4. 使用更大显存的 GPU

### Q3: PPO 训练报错 "Reward model API failed"？

**原因**:
1. API Key 错误
2. API 配额用完
3. 网络连接问题

**解决方法**:
```bash
# 测试 API 连接
curl -X POST "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions" \
  -H "Authorization: Bearer sk-xxxxx" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-plus","messages":[{"role":"user","content":"test"}]}'

# 检查 API Key 是否有效
```

### Q4: SSH 远程训练连接失败？

**检查清单**:
1. ✅ SSH 主机、端口、用户名、密码是否正确
2. ✅ 本地能否 SSH 连接到远程服务器
3. ✅ 远程服务器是否安装了 verl 环境

**测试连接**:
```bash
ssh -p 27192 root@connect.westc.gpuhub.com

# 如果能连接，检查远程环境
which python
python -c "import verl"
```

### Q5: 数据集格式错误？

**检查数据格式**:

**SFT 格式**:
```jsonl
{"prompt": "问题", "response": "回答"}
{"prompt": "问题2", "response": "回答2"}
```

**DPO 格式**:
```jsonl
{"prompt": "问题", "chosen": "好回答", "rejected": "差回答"}
```

**PPO 格式**:
```json
[
  {"prompt": "问题1"},
  {"prompt": "问题2"}
]
```

**GRPO 格式**:
```jsonl
{"prompt": "数学问题", "solution": "答案"}
```

### Q6: 如何查看训练日志？

**方法 1: 前端监控页面**
- 实时日志自动显示

**方法 2: 后端日志文件**
```bash
# API 日志
tail -f logs/api.log

# Celery 日志
tail -f logs/celery.log

# 训练日志（如果使用 SSH）
tail -f ~/verl_jobs/[job_uuid]/logs/train.log
```

### Q7: 训练完成后，模型保存在哪里？

**本地模式**:
```
./checkpoints/[job_uuid]/checkpoint-[step]/
```

**SSH 模式**:
```
远程服务器: ~/verl_jobs/[job_uuid]/checkpoints/checkpoint-[step]/
```

**下载模型**:
```bash
# 从远程下载
scp -P 27192 -r root@connect.westc.gpuhub.com:~/verl_jobs/[job_uuid]/checkpoints/ ./local_checkpoints/
```

### Q8: 如何使用训练好的模型？

**加载 Checkpoint**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained("./checkpoints/[job_uuid]/checkpoint-100/")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/[job_uuid]/checkpoint-100/")

# 推理
inputs = tokenizer("你好", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

**加载 LoRA 权重**:
```python
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "./checkpoints/[job_uuid]/checkpoint-100/")

# 推理
```

### Q9: 训练速度太慢怎么办？

**优化建议**:

1. **使用更强的 GPU**
   - RTX 5090 > A100 > V100 > RTX 3090

2. **减少计算量**
   - 减小 batch_size
   - 减小 context_length
   - 使用 LoRA

3. **使用远程 GPU 服务器**
   - SSH 模式连接到专用 GPU 服务器

4. **优化数据加载**
   - 确保数据集不要太大
   - 使用 SSD 存储数据

### Q10: 如何批量创建多个训练任务？

**方法 1: 使用 Pipeline**
1. 进入 **"Pipeline"** 页面
2. 创建包含多个训练任务的 Pipeline
3. 任务会按顺序或并行执行

**方法 2: 使用 API**
```python
import requests

tasks = [
    {"name": "SFT Task 1", "algorithm": "sft", ...},
    {"name": "SFT Task 2", "algorithm": "sft", ...},
    {"name": "DPO Task", "algorithm": "dpo", ...},
]

for task in tasks:
    response = requests.post("http://localhost:8000/api/v1/jobs/", json=task)
    print(f"Created task: {response.json()['uuid']}")
```

---

## 📚 进阶主题

### 使用 Pipeline 编排训练流程

**场景**: 先 SFT，再 DPO

1. 进入 **"Pipeline"** 页面
2. 点击 **"+ 创建 Pipeline"**
3. 添加 Stage 1: SFT Training
4. 添加 Stage 2: DPO Training
   - 依赖: Stage 1
   - 使用 Stage 1 的 checkpoint
5. 执行 Pipeline

### 使用 API 批量操作

**获取所有任务**:
```bash
curl http://localhost:8000/api/v1/jobs/
```

**创建任务**:
```bash
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "SFT Test",
    "algorithm": "sft",
    "model_path": "Qwen/Qwen2.5-0.5B",
    "train_data_path": "./datasets/sales_sft.jsonl",
    "num_gpus": 1,
    "learning_rate": 1e-5,
    "batch_size": 2,
    "num_epochs": 1
  }'
```

**查看任务详情**:
```bash
curl http://localhost:8000/api/v1/jobs/{job_uuid}
```

**监控 WebSocket**:
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/monitoring/{job_uuid}/live');
ws.onmessage = (event) => {
  console.log(JSON.parse(event.data));
};
```

---

## 🎓 最佳实践

### 1. 训练前检查清单

- [ ] 数据集格式正确
- [ ] 数据集路径存在
- [ ] GPU 可用（本地或远程）
- [ ] Celery Worker 运行中
- [ ] 模型路径可访问

### 2. 参数调优顺序

```
第一步: 使用默认参数跑通流程
第二步: 调整 learning_rate (最重要)
第三步: 调整 batch_size (根据显存)
第四步: 调整算法特定参数 (kl_coef, clip_ratio, etc.)
第五步: 调整 num_epochs (根据收敛情况)
```

### 3. 监控关键指标

| 算法 | 关键指标 | 预期趋势 |
|------|----------|----------|
| SFT | Loss | 下降 |
| DPO | Reward Margin | 上升 |
| PPO | Reward | 上升 |
| GRPO | Reward Accuracy | 上升 |

### 4. 保存重要 Checkpoint

- 训练开始时 (checkpoint-0)
- Loss 最低点
- 训练结束时

### 5. 数据安全

- 定期备份 checkpoint
- 保存训练日志
- 记录训练配置

---

## 📞 获取帮助

### 文档

- [技术架构](./PHASE4_SUMMARY.md)
- [API 文档](http://localhost:8000/docs)
- [算法详解](./ALL_ALGORITHMS_TEST_REPORT.md)

### 问题反馈

1. 查看日志文件
2. 查看本文档 FAQ
3. 提交 GitHub Issue

### 联系方式

- GitHub: https://github.com/terminal-creator/train_platform
- 文档: `./docs/`

---

## 🎉 恭喜！

你已经掌握了使用本平台进行 SFT、DPO、PPO、GRPO 训练的完整流程！

**下一步**:
1. 使用自己的数据集进行训练
2. 尝试不同的参数组合
3. 使用 Pipeline 编排复杂训练流程
4. 部署训练好的模型到生产环境

祝训练顺利！🚀
