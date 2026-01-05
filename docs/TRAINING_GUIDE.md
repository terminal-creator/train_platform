# 训练指南

本指南详细介绍如何使用 Training Platform 进行模型训练，涵盖数据准备、模型下载、训练运行和监控观测。

## 目录

- [1. 支持的训练算法](#1-支持的训练算法)
- [2. 训练数据格式](#2-训练数据格式)
- [3. 模型下载](#3-模型下载)
- [4. 运行训练](#4-运行训练)
- [5. 训练监控](#5-训练监控)
- [6. 常见问题](#6-常见问题)

---

## 1. 支持的训练算法

| 算法 | 全称 | 用途 | 是否需要 Critic |
|------|------|------|----------------|
| **SFT** | Supervised Fine-Tuning | 监督微调，学习指令遵循 | 否 |
| **GRPO** | Group Relative Policy Optimization | 高效 RL，无需 Critic | 否 |
| **PPO** | Proximal Policy Optimization | 完整 RLHF 训练 | 是 |
| **DPO** | Direct Preference Optimization | 直接偏好优化，无需 RM | 否 |
| **GSPO** | Group Self-Play Policy Optimization | 自博弈训练 | 否 |
| **DAPO** | Distribution-Aware Policy Optimization | 分布感知策略优化 | 否 |

### 算法选择建议

```
首次微调 → SFT
↓
需要对齐人类偏好？
├── 有偏好对比数据 → DPO
└── 无偏好数据，只有 Prompt
    ├── 计算资源充足 → PPO
    └── 计算资源有限 → GRPO
```

---

## 2. 训练数据格式

### 2.1 SFT (监督微调)

SFT 需要 **prompt-response 配对数据**，保存为 Parquet 或 JSONL 格式。

#### 对话格式 (推荐)

```jsonl
{"messages": [{"role": "user", "content": "什么是机器学习？"}, {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}]}
{"messages": [{"role": "system", "content": "你是数学专家"}, {"role": "user", "content": "解方程 x^2=4"}, {"role": "assistant", "content": "x=2 或 x=-2"}]}
```

#### 简单格式

```jsonl
{"prompt": "什么是机器学习？", "response": "机器学习是人工智能的一个分支..."}
{"question": "计算 2+2", "answer": "4"}
```

#### 配置示例

```yaml
data:
  train_files: /data/sft_train.parquet
  val_files: /data/sft_val.parquet
  prompt_key: question      # 或 prompt
  response_key: answer      # 或 response
  max_length: 4096
```

---

### 2.2 GRPO (组相对策略优化)

GRPO 只需要 **Prompt 数据**，模型会为每个 prompt 生成多个回复，通过组内比较进行优化。

#### 数据格式

```jsonl
{"prompt": [{"role": "user", "content": "计算 25 * 4 = ?"}], "reward_model": {"style": "rule", "ground_truth": "100"}, "ability": "math"}
{"prompt": [{"role": "user", "content": "写一个快速排序函数"}], "reward_model": {"style": "model"}, "ability": "code"}
```

#### 数据准备脚本示例 (GSM8K)

```python
# verl/examples/data_preprocess/gsm8k.py
from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main")

def process_fn(example, idx):
    question = example["question"] + ' Let\'s think step by step and output the final answer after "####".'
    solution = extract_answer(example["answer"])  # 提取 #### 后的答案

    return {
        "data_source": "gsm8k",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
    }

dataset = dataset.map(process_fn, with_indices=True)
dataset.to_parquet("gsm8k_train.parquet")
```

#### 配置示例

```yaml
data:
  train_files: /data/gsm8k/train.parquet
  train_batch_size: 1024
  max_prompt_length: 512
  max_response_length: 1024

actor_rollout_ref:
  rollout:
    n: 5                    # 每个 prompt 生成 5 个回复
    temperature: 0.7

algorithm:
  adv_estimator: grpo       # 使用 GRPO 优势估计
```

---

### 2.3 PPO (近端策略优化)

PPO 需要 **Prompt 数据** + **奖励模型**。

#### 数据格式

与 GRPO 相同，但需要额外配置奖励模型：

```jsonl
{"prompt": [{"role": "user", "content": "解释什么是深度学习"}], "reward_model": {"style": "model"}, "ability": "qa"}
```

#### 配置示例

```yaml
# PPO 需要 Critic 模型
critic:
  model:
    path: /models/Qwen2.5-7B-Instruct   # Critic 模型路径
  ppo_mini_batch_size: 64

actor_rollout_ref:
  model:
    path: /models/Qwen2.5-7B-Instruct   # Actor 模型路径

reward_model:
  path: /models/reward_model            # 奖励模型路径 (可选，可使用规则奖励)

algorithm:
  adv_estimator: gae         # GAE 优势估计
  gamma: 1.0
  lam: 0.95
```

---

### 2.4 DPO (直接偏好优化)

DPO 需要 **偏好对比数据**：每个 prompt 配一个好回复和一个差回复。

#### 数据格式

```jsonl
{"prompt": "如何学习编程？", "chosen": "学习编程建议从以下几个方面入手...", "rejected": "随便学学就行"}
{"prompt": "解释什么是递归", "chosen": "递归是函数调用自身的编程技术...", "rejected": "就是自己调用自己"}
```

#### 数据准备脚本示例 (HH-RLHF)

```python
# verl/examples/data_preprocess/full_hh_rlhf.py
from datasets import load_dataset

dataset = load_dataset("Dahoas/full-hh-rlhf")

output = {"prompt": [], "chosen": [], "rejected": []}
for data in dataset["train"]:
    output["prompt"].append(data["prompt"])
    output["chosen"].append(data["chosen"])
    output["rejected"].append(data["rejected"])

df = pd.DataFrame(output)
df.to_parquet("dpo_train.parquet")
```

#### 配置示例

```yaml
data:
  train_files: /data/dpo_train.parquet
  max_length: 4096

algorithm:
  beta: 0.1                  # KL 惩罚系数
  loss_type: sigmoid         # sigmoid 或 hinge
```

---

### 2.5 GSPO (组自博弈偏好优化)

GSPO 结合 GRPO 和 DPO 特点，支持两种模式：

#### 模式一：纯 Prompt (自博弈)

```jsonl
{"prompt": [{"role": "user", "content": "设计一个缓存系统"}]}
```

#### 模式二：混合训练

```jsonl
{"prompt": "解释递归", "chosen": "递归是...", "rejected": "就是..."}
{"prompt": [{"role": "user", "content": "设计 API"}], "self_play": true}
```

---

### 数据格式汇总

| 算法 | 必需字段 | 可选字段 |
|------|---------|---------|
| SFT | prompt, response (或 messages) | - |
| GRPO | prompt | reward_model, ability, data_source |
| PPO | prompt | reward_model |
| DPO | prompt, chosen, rejected | - |
| GSPO | prompt | chosen, rejected, self_play |

---

## 3. 模型下载

### 3.1 推荐模型

| 模型系列 | 推荐尺寸 | HuggingFace 路径 |
|---------|---------|-----------------|
| Qwen2.5 | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | `Qwen/Qwen2.5-{size}-Instruct` |
| Qwen3 | 0.6B, 1.7B, 4B, 8B, 14B, 32B, 235B | `Qwen/Qwen3-{size}` |
| DeepSeek | 7B, 67B | `deepseek-ai/deepseek-llm-{size}-chat` |
| Llama3 | 8B, 70B | `meta-llama/Meta-Llama-3-{size}-Instruct` |

### 3.2 下载方式

#### 方式一：使用 HuggingFace Hub

```bash
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /models/Qwen2.5-7B-Instruct
```

#### 方式二：使用 ModelScope (国内推荐)

```bash
# 安装 modelscope
pip install modelscope

# 下载模型
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/models')"
```

#### 方式三：Python 代码下载

```python
from huggingface_hub import snapshot_download

# 下载到指定目录
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/models/Qwen2.5-7B-Instruct",
    local_dir_use_symlinks=False
)
```

### 3.3 模型存储建议

```
/models/
├── Qwen2.5-0.5B-Instruct/    # 小模型用于测试
├── Qwen2.5-7B-Instruct/      # 中等模型
├── Qwen2.5-72B-Instruct/     # 大模型
└── reward_models/
    └── my_reward_model/      # 自训练奖励模型
```

---

## 4. 运行训练

### 4.1 SFT 训练

#### 基础命令

```bash
python -m verl.trainer.sft_trainer \
    model.partial_pretrain=/models/Qwen2.5-7B-Instruct \
    data.train_files=/data/sft_train.parquet \
    data.val_files=/data/sft_val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=4096 \
    trainer.total_epochs=3 \
    trainer.n_gpus_per_node=8 \
    trainer.project_name=my_sft_project \
    trainer.experiment_name=qwen7b_sft \
    trainer.logger='["console","wandb"]'
```

#### LoRA 微调 (节省显存)

```bash
python -m verl.trainer.sft_trainer \
    model.partial_pretrain=/models/Qwen2.5-7B-Instruct \
    model.lora_rank=32 \
    model.lora_alpha=64 \
    model.target_modules=all-linear \
    data.train_files=/data/sft_train.parquet \
    trainer.total_epochs=3 \
    trainer.n_gpus_per_node=4
```

---

### 4.2 GRPO 训练

#### 基础命令

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=/models/Qwen2.5-7B-Instruct \
    data.train_files=/data/gsm8k/train.parquet \
    data.val_files=/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=15 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.project_name=my_grpo_project \
    trainer.logger='["console","wandb"]'
```

#### 完整脚本示例

```bash
#!/bin/bash
# run_grpo.sh

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_gsm8k' \
    trainer.experiment_name='qwen25_7b' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15
```

---

### 4.3 PPO 训练

#### 基础命令

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=/models/Qwen2.5-7B-Instruct \
    critic.model.path=/models/Qwen2.5-7B-Instruct \
    data.train_files=/data/prompts.parquet \
    data.train_batch_size=256 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    critic.ppo_mini_batch_size=64 \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10
```

---

### 4.4 DPO 训练

```bash
python -m verl.trainer.dpo_trainer \
    model.path=/models/Qwen2.5-7B-Instruct \
    data.train_files=/data/dpo_train.parquet \
    data.val_files=/data/dpo_val.parquet \
    algorithm.beta=0.1 \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=8 \
    optim.lr=5e-7
```

---

### 4.5 通过 API 创建训练任务

```python
import requests

# 创建训练任务
response = requests.post("http://localhost:8000/api/jobs", json={
    "name": "qwen7b-grpo-math",
    "description": "Qwen2.5-7B GRPO training on GSM8K",
    "model_path": "Qwen/Qwen2.5-7B-Instruct",
    "model_size": "7B",
    "algorithm": "grpo",
    "train_data_path": "/data/gsm8k/train.parquet",
    "eval_data_path": "/data/gsm8k/test.parquet",
    "num_gpus": 8,
    "learning_rate": 1e-6,
    "batch_size": 1024,
    "num_epochs": 15,
    "context_length": 1536,
    "rollout_n": 5,
    "kl_coef": 0.001
})

job = response.json()
print(f"Job created: {job['uuid']}")

# 启动任务
requests.post(f"http://localhost:8000/api/jobs/{job['uuid']}/start")
```

---

## 5. 训练监控

### 5.0 平台监控 vs Weights & Biases

我们的平台和 W&B 可以**同时使用**，各有优势：

| 功能 | 我们的平台 | Weights & Biases |
|------|-----------|------------------|
| **部署方式** | 本地/私有部署 | 云托管 |
| **实时性** | WebSocket 实时推送 | 轮询更新 (~30s) |
| **GPU 监控** | 原生支持 | 需要额外配置 |
| **任务管理** | 完整生命周期管理 | 仅记录 |
| **模型手术** | 内置合并/选择工具 | 无 |
| **数据隐私** | 数据不出域 | 数据上传云端 |
| **实验对比** | 基础支持 | 强大的对比功能 |
| **团队协作** | 本地数据库 | 团队共享 |
| **Artifact 版本** | 检查点管理 | 完整版本控制 |
| **免费额度** | 无限制 | 有限制 |

**推荐用法**: 同时使用两者
- 平台：任务管理、实时监控、GPU 监控
- W&B：实验对比、长期存档、团队分享

### 5.1 日志监控

#### 控制台日志

训练过程中会输出实时日志：

```
Step 100/10000 | Loss: 0.234 | Reward: 0.856 | KL: 0.0012 | LR: 9.8e-7
Step 200/10000 | Loss: 0.198 | Reward: 0.912 | KL: 0.0015 | LR: 9.6e-7
```

#### Weights & Biases (推荐)

```bash
# 设置 W&B API Key
export WANDB_API_KEY=your_api_key

# 训练时启用 wandb
python -m verl.trainer.main_ppo \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=my_project \
    trainer.experiment_name=exp_001 \
    ...
```

访问 https://wandb.ai 查看训练曲线。

### 5.2 双端监控 (平台 + W&B)

我们提供了 `DualLogger` 工具，可以**同时**向平台和 W&B 发送训练指标。

---

#### 5.2.1 快速开始

**Step 1: 安装 W&B**

```bash
pip install wandb
```

**Step 2: 登录 W&B**

```bash
# 方式一：命令行登录
wandb login

# 方式二：环境变量 (适合服务器)
export WANDB_API_KEY=your_api_key_here
```

获取 API Key: https://wandb.ai/authorize

**Step 3: 在训练脚本中使用**

```python
from training_platform.core.wandb_callback import init_logging

# 一行初始化
logger = init_logging(
    job_uuid="my-job-001",
    wandb_project="llm-training",
)

# 训练时记录
logger.log_metrics(step=100, metrics={"loss": 0.5, "reward": 0.8})

# 结束
logger.finish()
```

---

#### 5.2.2 完整使用示例

**场景一：GRPO 训练脚本**

```python
#!/usr/bin/env python
"""GRPO 训练脚本示例 - 带双端监控"""

import os
from training_platform.core.wandb_callback import DualLogger

# ============ 配置 ============
JOB_UUID = os.environ.get("JOB_UUID", "local-grpo-test")
PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://localhost:8000")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "grpo-math")

# ============ 初始化日志 ============
logger = DualLogger(
    # 平台配置
    job_uuid=JOB_UUID,
    platform_url=PLATFORM_URL,
    platform_enabled=True,

    # W&B 配置
    wandb_project=WANDB_PROJECT,
    wandb_run_name=f"qwen7b-grpo-{JOB_UUID[:8]}",
    wandb_entity=None,  # 你的 W&B 团队名，个人用户可留空
    wandb_config={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "algorithm": "GRPO",
        "dataset": "gsm8k",
        "learning_rate": 1e-6,
        "batch_size": 1024,
        "rollout_n": 5,
        "kl_coef": 0.001,
    },
    wandb_tags=["grpo", "math", "qwen2.5"],
    wandb_enabled=True,
)

# ============ 训练循环 ============
total_steps = 10000
for step in range(total_steps):
    # 模拟训练
    metrics = {
        "policy_loss": 0.5 - step * 0.00003,
        "reward_mean": 0.3 + step * 0.00005,
        "reward_std": 0.2,
        "kl_divergence": 0.01 + step * 0.000001,
        "entropy": 1.5 - step * 0.0001,
    }

    # 记录指标 (同时发送到平台和 W&B)
    logger.log_metrics(
        step=step,
        metrics=metrics,
        epoch=step // 1000,
    )

    # 每 100 步记录 GPU 使用 (仅平台)
    if step % 100 == 0:
        logger.log_gpu_usage(
            gpu_index=0,
            utilization=92.5,
            memory_used=65.0,
            memory_total=80.0,
            temperature=72.0,
        )

    # 每 500 步评估并保存检查点
    if step > 0 and step % 500 == 0:
        eval_score = 0.35 + step * 0.00003  # 模拟评估分数
        checkpoint_path = f"checkpoints/step-{step}"

        logger.log_checkpoint(
            step=step,
            checkpoint_path=checkpoint_path,
            eval_results={"gsm8k": eval_score},
        )
        print(f"Step {step}: GSM8K = {eval_score:.2%}")

# ============ 训练结束 ============
logger.log_summary({
    "best_gsm8k": 0.65,
    "best_step": 8500,
    "final_loss": 0.15,
    "total_tokens": 125_000_000,
})
logger.finish()
print("Training completed!")
```

**场景二：使用 verl 原生训练**

verl 已内置 W&B 支持，只需配置 `trainer.logger`:

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    data.train_files=/data/gsm8k/train.parquet \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='my-grpo-project' \
    trainer.experiment_name='qwen7b-exp1' \
    ...
```

如果还想同时上报到我们的平台，可以设置环境变量让 verl 回调：

```bash
export JOB_UUID="job-$(date +%s)"
export PLATFORM_URL="http://localhost:8000"
export WANDB_PROJECT="my-grpo-project"

python -m verl.trainer.main_ppo \
    trainer.logger='["console","wandb"]' \
    ...
```

**场景三：只用平台，不用 W&B**

```python
from training_platform.core.wandb_callback import DualLogger

logger = DualLogger(
    job_uuid="my-job",
    platform_url="http://localhost:8000",
    wandb_enabled=False,  # 关闭 W&B
)

# 正常使用
logger.log_metrics(step, metrics)
logger.finish()
```

**场景四：只用 W&B，不用平台**

```python
from training_platform.core.wandb_callback import DualLogger

logger = DualLogger(
    job_uuid="my-job",
    platform_enabled=False,  # 关闭平台
    wandb_project="my-project",
)

logger.log_metrics(step, metrics)
logger.finish()
```

---

#### 5.2.3 DualLogger API 参考

| 方法 | 说明 | 平台 | W&B |
|------|------|:----:|:---:|
| `log_metrics(step, metrics)` | 记录训练指标 | ✓ | ✓ |
| `log_gpu_usage(...)` | 记录 GPU 使用 | ✓ | ✓ |
| `log_checkpoint(step, path, eval)` | 记录检查点 | ✓ | ✓ |
| `log_status(status, message)` | 更新任务状态 | ✓ | Alert |
| `log_config(config)` | 记录配置 | - | ✓ |
| `log_summary(summary)` | 记录最终汇总 | - | ✓ |
| `finish()` | 结束日志会话 | ✓ | ✓ |

**参数说明：**

```python
DualLogger(
    # 平台配置
    job_uuid: str,                    # 任务 UUID (必须)
    platform_url: str = "http://localhost:8000",
    platform_enabled: bool = True,

    # W&B 配置
    wandb_project: str = None,        # W&B 项目名
    wandb_run_name: str = None,       # 运行名称 (默认=job_uuid)
    wandb_entity: str = None,         # W&B 团队/用户名
    wandb_config: Dict = None,        # 超参数配置
    wandb_tags: List[str] = None,     # 标签列表
    wandb_enabled: bool = True,
)
```

---

#### 5.2.4 查看效果

训练开始后可以在两个地方查看：

| 平台 | 地址 | 功能 |
|------|------|------|
| **Training Platform** | http://localhost:5173 | 实时监控、任务管理、GPU 监控 |
| **Weights & Biases** | https://wandb.ai | 实验对比、长期存档、团队分享 |

**平台 Dashboard 截图示例:**
- 实时 Loss/Reward 曲线
- GPU 利用率监控
- 任务状态管理

**W&B Dashboard 功能:**
- 多实验对比
- 超参数扫描
- Artifact 版本管理

### 5.3 Prometheus + Grafana 监控

适用于大规模多节点训练。

#### 启动监控服务

```bash
# 设置环境变量
export GF_SERVER_HTTP_PORT=3000
export PROMETHEUS_PORT=9090

# 启动 Grafana
nohup grafana-server \
    --config /tmp/ray/session_latest/metrics/grafana/grafana.ini \
    --homepath /usr/share/grafana \
    web > grafana.log 2>&1 &

# 启动 Prometheus
nohup prometheus \
    --config.file /tmp/ray/session_latest/metrics/prometheus/prometheus.yml \
    --web.enable-lifecycle \
    --web.listen-address=:${PROMETHEUS_PORT} \
    > prometheus.log 2>&1 &
```

#### 训练时启用监控

```bash
python -m verl.trainer.main_ppo \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.prometheus.enable=True \
    ...
```

#### 访问地址

- Grafana: http://localhost:3000 (默认账号 admin/admin)
- Prometheus: http://localhost:9090

### 5.4 关键监控指标

| 指标 | 说明 | 正常范围 |
|------|------|---------|
| `policy_loss` | 策略损失 | 持续下降 |
| `value_loss` | 价值损失 (PPO) | 持续下降 |
| `reward_mean` | 平均奖励 | 持续上升 |
| `kl_divergence` | KL 散度 | < 0.1 |
| `entropy` | 策略熵 | 适度下降 |
| `clip_fraction` | PPO 裁剪比例 | 0.1-0.3 |

### 5.5 API 获取指标

```python
import requests

# 获取训练指标历史
response = requests.get(f"http://localhost:8000/api/jobs/{job_uuid}/metrics")
metrics = response.json()

for m in metrics[-10:]:
    print(f"Step {m['step']}: loss={m['policy_loss']:.4f}, reward={m['reward_mean']:.3f}")
```

### 5.6 查看 Checkpoint

```python
# 获取检查点列表
response = requests.get(f"http://localhost:8000/api/jobs/{job_uuid}/checkpoints")
checkpoints = response.json()

for ckpt in checkpoints:
    print(f"Step {ckpt['step']}: {ckpt['path']}")
    if ckpt.get('eval_results'):
        print(f"  Eval: {ckpt['eval_results']}")
```

---

## 6. 常见问题

### 6.1 OOM (显存不足)

**解决方案：**

```yaml
# 减小 micro batch size
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu: 2

# 启用梯度检查点
actor_rollout_ref.model.enable_gradient_checkpointing: True

# 启用 CPU offload
actor_rollout_ref.actor.fsdp_config.param_offload: True

# 使用 LoRA
model.lora_rank: 32
```

### 6.2 训练不收敛

**可能原因及解决：**

1. **学习率过大** → 减小到 1e-6 或 5e-7
2. **Batch size 过小** → 增大 `train_batch_size`
3. **KL 系数过大** → 减小 `kl_loss_coef` 到 0.001 或更小
4. **数据质量问题** → 检查数据格式，清洗低质量样本

### 6.3 奖励不上升 (GRPO/PPO)

**检查项：**

1. 奖励函数是否正确实现
2. `ground_truth` 是否正确提取
3. 生成的回复是否包含预期格式
4. 增加 `rollout.n` 提高采样多样性

### 6.4 多节点训练

```bash
# 主节点
ray start --head --port=6379

# 工作节点
ray start --address=master_ip:6379

# 提交任务
ray job submit --working-dir . -- python -m verl.trainer.main_ppo \
    trainer.nnodes=4 \
    trainer.n_gpus_per_node=8 \
    ...
```

---

## 附录：配置参数速查

### 数据配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data.train_files` | 训练数据路径 | - |
| `data.val_files` | 验证数据路径 | - |
| `data.train_batch_size` | 全局 batch size | 256 |
| `data.max_prompt_length` | 最大 prompt 长度 | 512 |
| `data.max_response_length` | 最大回复长度 | 1024 |

### 模型配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `actor_rollout_ref.model.path` | 模型路径 | - |
| `model.enable_gradient_checkpointing` | 梯度检查点 | True |
| `model.lora_rank` | LoRA rank (0=禁用) | 0 |

### 训练配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `trainer.total_epochs` | 总 epoch 数 | 15 |
| `trainer.n_gpus_per_node` | 每节点 GPU 数 | 8 |
| `trainer.save_freq` | 保存频率 (step) | -1 |
| `trainer.test_freq` | 验证频率 (step) | -1 |
| `optim.lr` | 学习率 | 1e-6 |

### 算法配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `algorithm.adv_estimator` | 优势估计器 (gae/grpo) | gae |
| `actor_rollout_ref.rollout.n` | 每 prompt 采样数 | 1 |
| `actor_rollout_ref.actor.use_kl_loss` | 使用 KL 损失 | False |
| `actor_rollout_ref.actor.kl_loss_coef` | KL 系数 | 0.001 |
