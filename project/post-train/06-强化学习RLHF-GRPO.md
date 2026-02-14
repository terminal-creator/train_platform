# 第六章：强化学习 - RLHF与GRPO

> **核心目标**：使用强化学习方法精细化对齐模型，达到SOTA效果
>
> **本章目标**：掌握PPO、GRPO等RL方法的原理、实现和调优技巧
>
> **里程碑案例**：DeepSeek-R1使用GRPO实现了纯RL推理涌现

---

## 一、RLHF概述

### 1.1 为什么需要强化学习？

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SFT vs RLHF 的核心区别                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SFT (监督学习)                        RLHF (强化学习)                  │
│  ━━━━━━━━━━━━━━                        ━━━━━━━━━━━━━━━                  │
│                                                                         │
│  学习方式：模仿                        学习方式：探索 + 反馈            │
│                                                                         │
│  "这是正确答案，                       "尝试不同的答案，               │
│   请学习生成它"                         看哪个得分更高"                │
│                                                                         │
│  ┌─────────────┐                      ┌─────────────┐                  │
│  │ 只看到好的   │                      │ 看到好的和坏的│                  │
│  │ 没有负反馈   │                      │ 有正负反馈   │                  │
│  │ 不会"探索"  │                      │ 会尝试新策略  │                  │
│  └─────────────┘                      └─────────────┘                  │
│                                                                         │
│  局限：只能学到训练数据中的模式        优势：可以发现更优的策略         │
│  优势：简单、稳定                      局限：复杂、需要调优             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 RLHF完整流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RLHF三阶段训练流程                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   阶段1: SFT                                                           │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ 输入：指令-回复对                                                │  │
│   │ 输出：SFT Model (能够遵循指令)                                   │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                   │
│                                    ▼                                   │
│   阶段2: Reward Model训练                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ 输入：偏好数据 (chosen vs rejected)                              │  │
│   │ 输出：Reward Model (能够评估回复质量)                            │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                   │
│                                    ▼                                   │
│   阶段3: RL Fine-tuning (PPO/GRPO)                                    │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ Policy Model: 生成回复                                           │  │
│   │ Reward Model: 评估回复质量                                       │  │
│   │ Reference Model: KL约束                                          │  │
│   │                                                                  │  │
│   │ 循环: 生成 → 评分 → 更新策略 → 生成 → ...                        │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 RLHF的数学目标

```python
"""
RLHF的优化目标：
    max E_{x~D, y~π}[r(x,y)] - β * KL(π || π_ref)

分解：
    - E[r(x,y)]: 最大化奖励（让回复更好）
    - KL(π || π_ref): 不要偏离reference太多（保持稳定）
    - β: 平衡两者的系数
"""
import torch
import torch.nn.functional as F

def rlhf_objective(
    rewards: torch.Tensor,           # 奖励分数
    policy_logprobs: torch.Tensor,   # 当前策略的log概率
    ref_logprobs: torch.Tensor,      # reference策略的log概率
    beta: float = 0.1
) -> torch.Tensor:
    """
    RLHF优化目标
    """
    # KL散度
    kl_divergence = policy_logprobs - ref_logprobs

    # 总目标 = 奖励 - β * KL
    objective = rewards - beta * kl_divergence

    return objective.mean()
```

---

## 二、PPO（Proximal Policy Optimization）

### 2.1 PPO核心原理

PPO是RLHF中最常用的算法，核心思想是：**限制每次更新的步长，保证训练稳定**

```python
"""
PPO的核心：Clipped Surrogate Objective

标准Policy Gradient:
    L = E[ratio * A]
    其中 ratio = π(a|s) / π_old(a|s)

PPO Clipped:
    L = E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)]

clip的作用：防止ratio变化太大，保证稳定性
"""
import torch

def ppo_loss(
    logprobs: torch.Tensor,      # 当前策略的log概率
    old_logprobs: torch.Tensor,  # 旧策略的log概率
    advantages: torch.Tensor,     # 优势函数
    clip_epsilon: float = 0.2
) -> torch.Tensor:
    """
    PPO Clipped Loss
    """
    # 计算importance ratio
    ratio = torch.exp(logprobs - old_logprobs)

    # Clipped ratio
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # PPO loss = -min(ratio * A, clipped_ratio * A)
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    return policy_loss
```

### 2.2 PPO在LLM中的实现

```python
"""
PPO for LLM的完整实现框架
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

class LLMPPOTrainer:
    def __init__(
        self,
        policy_model_path: str,
        reward_model_path: str,
        ref_model_path: str
    ):
        # 1. Policy Model (带Value Head)
        self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            policy_model_path,
            torch_dtype=torch.bfloat16
        )

        # 2. Reward Model
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            torch_dtype=torch.bfloat16
        )

        # 3. Reference Model (用于KL约束)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_path,
            torch_dtype=torch.bfloat16
        )

        # PPO配置
        self.ppo_config = PPOConfig(
            learning_rate=1e-6,
            batch_size=16,
            mini_batch_size=4,
            gradient_accumulation_steps=4,
            ppo_epochs=4,           # 每个batch训练几轮
            cliprange=0.2,          # PPO clip范围
            cliprange_value=0.2,    # Value clip范围
            vf_coef=0.1,           # Value loss系数
            kl_penalty="kl",        # KL惩罚类型
            target_kl=0.1,          # 目标KL值
        )

    def train_step(self, prompts: list) -> dict:
        """
        PPO训练的一个step
        """
        # Step 1: 使用当前策略生成回复
        responses = self._generate_responses(prompts)

        # Step 2: 计算奖励
        rewards = self._compute_rewards(prompts, responses)

        # Step 3: 计算优势函数
        advantages, returns = self._compute_advantages(rewards)

        # Step 4: PPO更新
        stats = self._ppo_update(prompts, responses, advantages, returns)

        return stats

    def _generate_responses(self, prompts: list) -> list:
        """生成回复"""
        responses = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.9
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        return responses

    def _compute_rewards(self, prompts: list, responses: list) -> torch.Tensor:
        """使用Reward Model计算奖励"""
        rewards = []
        for prompt, response in zip(prompts, responses):
            text = f"{prompt}\n{response}"
            inputs = self.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                reward = self.reward_model(**inputs).logits.squeeze()
            rewards.append(reward)
        return torch.stack(rewards)

    def _compute_advantages(self, rewards: torch.Tensor):
        """计算优势函数（GAE）"""
        # 简化实现，实际使用GAE (Generalized Advantage Estimation)
        returns = rewards
        advantages = rewards - rewards.mean()
        return advantages, returns

    def _ppo_update(self, prompts, responses, advantages, returns):
        """PPO参数更新"""
        # 实际实现需要多轮mini-batch更新
        pass
```

### 2.3 使用TRL进行PPO训练

```python
"""
使用TRL库进行PPO训练
"""
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import torch

def train_ppo():
    # ============ 1. 配置 ============
    config = PPOConfig(
        model_name="./output/sales_sft",
        learning_rate=1e-6,
        batch_size=16,
        mini_batch_size=4,
        gradient_accumulation_steps=4,
        ppo_epochs=4,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        target_kl=0.1,
        kl_penalty="kl",
        seed=42,
    )

    # ============ 2. 加载模型 ============
    # Policy model with value head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Reference model
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Reward model
    reward_model = load_reward_model("./output/sales_rm")

    # ============ 3. 准备数据 ============
    prompts = load_prompts("./data/prompts/sales_prompts.json")

    # ============ 4. 训练 ============
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    for epoch in range(3):
        for batch_prompts in batch_iterator(prompts, config.batch_size):
            # 生成回复
            query_tensors = [tokenizer.encode(p, return_tensors="pt").squeeze() for p in batch_prompts]
            response_tensors = ppo_trainer.generate(query_tensors, max_new_tokens=256)

            # 计算奖励
            rewards = []
            for query, response in zip(query_tensors, response_tensors):
                text = tokenizer.decode(query) + tokenizer.decode(response)
                reward = reward_model.get_reward(text)
                rewards.append(torch.tensor(reward))

            # PPO更新
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

            print(f"Epoch {epoch}, Mean Reward: {stats['ppo/mean_scores']:.4f}")

    # ============ 5. 保存 ============
    model.save_pretrained("./output/sales_ppo")


def load_reward_model(path: str):
    """加载Reward Model"""
    # 实现加载逻辑
    pass


def load_prompts(path: str) -> list:
    """加载训练prompts"""
    import json
    with open(path) as f:
        return json.load(f)


def batch_iterator(data: list, batch_size: int):
    """批量迭代器"""
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]
```

---

## 三、GRPO（Group Relative Policy Optimization）

### 3.1 GRPO vs PPO

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PPO vs GRPO 对比                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PPO                                   GRPO (DeepSeek)                 │
│  ━━━                                   ━━━━                             │
│                                                                         │
│  需要Critic/Value Model                不需要Critic Model              │
│  ┌─────────────┐                       ┌─────────────┐                 │
│  │ Policy      │                       │ Policy      │                 │
│  │ Critic      │                       │ (仅Policy)  │                 │
│  │ Reference   │                       │ Reference   │                 │
│  │ Reward      │                       │ Reward      │                 │
│  └─────────────┘                       └─────────────┘                 │
│                                                                         │
│  单个回复评分                          组内相对比较                     │
│  reward(y)                             reward(y) - mean(rewards)       │
│                                                                         │
│  优势：                                优势：                           │
│  - 成熟稳定                            - 更简单                         │
│  - 效果验证充分                        - 计算成本低                     │
│                                        - 不需要Critic模型              │
│                                                                         │
│  劣势：                                劣势：                           │
│  - 需要训练Critic                      - 需要采样多个回复               │
│  - 计算成本高                          - 相对较新                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 GRPO核心原理

```python
"""
GRPO的核心思想：
1. 对同一个prompt采样多个回复
2. 用Reward Model给每个回复打分
3. 计算组内相对优势（reward - mean）
4. 用相对优势更新策略

不需要Critic Model，用组内均值代替baseline
"""
import torch
import torch.nn.functional as F

def grpo_loss(
    policy_logprobs: torch.Tensor,    # [batch, num_samples] 当前策略
    old_logprobs: torch.Tensor,       # [batch, num_samples] 采样时的旧策略
    ref_logprobs: torch.Tensor,       # [batch, num_samples] reference策略（用于KL）
    rewards: torch.Tensor,            # [batch, num_samples]
    kl_coef: float = 0.001,
    clip_ratio: float = 10.0
) -> torch.Tensor:
    """
    GRPO Loss

    ⚠️ 重要概念区分（三模型四分布）：
    ┌─────────────────────────────────────────────────────────────────┐
    │  π_current (policy)  : 当前正在更新的策略模型                    │
    │  π_old              : 采样时的策略（用于importance sampling）   │
    │  π_ref (reference)  : 固定的参考策略（用于KL约束，防止跑偏）    │
    │  RM                 : Reward Model（打分用，不参与梯度）        │
    └─────────────────────────────────────────────────────────────────┘

    - ratio = π_current / π_old  ← 用于PPO-style clipping
    - KL = π_current - π_ref     ← 用于约束不偏离reference太远

    Args:
        policy_logprobs: 当前策略的log概率 log π_current(y|x)
        old_logprobs: 采样时旧策略的log概率 log π_old(y|x)
        ref_logprobs: reference策略的log概率 log π_ref(y|x)
        rewards: 每个样本的奖励
        kl_coef: KL散度系数
        clip_ratio: 裁剪比例
    """
    # 计算组内相对优势
    # 每个prompt的多个回复，减去均值
    group_mean = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - group_mean

    # 归一化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ⚠️ 关键：importance ratio = π_current / π_old
    # 这里是 current vs old（采样时的策略），不是 current vs reference！
    ratio = torch.exp(policy_logprobs - old_logprobs)

    # 裁剪ratio（PPO-style clipping）
    clipped_ratio = torch.clamp(ratio, 1/clip_ratio, clip_ratio)

    # Policy loss
    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    # ⚠️ KL penalty 是相对于 reference model，不是 old model
    # 这是两个不同的约束：clipping约束单步更新幅度，KL约束总体偏离
    kl_penalty = (policy_logprobs - ref_logprobs).mean()

    # 总loss
    total_loss = policy_loss + kl_coef * kl_penalty

    return total_loss


# ⚠️ 注意：上述代码仅为原理示意！
# 实际GRPO实现复杂得多，涉及：
# - 高效的Group Rollout（同一prompt采样多个output）
# - 复杂的padding和attention mask处理
# - 分布式采样架构
# - 显存管理与梯度累积
#
# 强烈建议使用成熟框架：
# - OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
# - Verl: https://github.com/volcengine/verl


# DeepSeek-R1的具体参数
deepseek_r1_config = {
    "kl_coef": 0.001,
    "clip_ratio": 10,
    "learning_rate": 3e-6,
    "num_samples_per_prompt": 16,  # 每个prompt采样16个回复
    "max_length": 32768,
    "batch_size": 32  # 32个unique prompts per step
}
```

### 3.3 GRPO完整实现

```python
"""
GRPO训练完整实现
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class GRPOTrainer:
    def __init__(
        self,
        policy_model_path: str,
        reward_model_path: str,
        config: Dict
    ):
        self.config = config

        # 加载Policy Model
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            policy_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # 加载Reference Model（固定不更新）
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            policy_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # 加载Reward Model
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.reward_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(policy_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config["learning_rate"]
        )

    def train_step(self, prompts: List[str]) -> Dict:
        """
        GRPO训练的一个step

        1. 对每个prompt采样多个回复
        2. 计算奖励
        3. 计算GRPO loss
        4. 更新模型
        """
        num_samples = self.config["num_samples_per_prompt"]

        all_responses = []
        all_rewards = []
        all_policy_logprobs = []
        all_ref_logprobs = []

        # Step 1: 采样多个回复
        for prompt in prompts:
            responses = self._sample_responses(prompt, num_samples)
            all_responses.append(responses)

            # Step 2: 计算奖励
            rewards = self._compute_rewards(prompt, responses)
            all_rewards.append(rewards)

            # Step 3: 计算log概率
            policy_logprobs = self._compute_logprobs(self.policy_model, prompt, responses)
            ref_logprobs = self._compute_logprobs(self.ref_model, prompt, responses)

            all_policy_logprobs.append(policy_logprobs)
            all_ref_logprobs.append(ref_logprobs)

        # 转换为tensor
        rewards_tensor = torch.stack([torch.tensor(r) for r in all_rewards])
        policy_logprobs_tensor = torch.stack(all_policy_logprobs)
        ref_logprobs_tensor = torch.stack(all_ref_logprobs)

        # Step 4: 计算GRPO loss
        loss = grpo_loss(
            policy_logprobs_tensor,
            ref_logprobs_tensor,
            rewards_tensor,
            kl_coef=self.config["kl_coef"],
            clip_ratio=self.config["clip_ratio"]
        )

        # Step 5: 更新模型
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
        self.optimizer.step()

        # 统计信息
        stats = {
            "loss": loss.item(),
            "mean_reward": rewards_tensor.mean().item(),
            "reward_std": rewards_tensor.std().item(),
            "kl_divergence": (policy_logprobs_tensor - ref_logprobs_tensor).mean().item()
        }

        return stats

    def _sample_responses(self, prompt: str, num_samples: int) -> List[str]:
        """采样多个回复"""
        responses = []
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.policy_model.device)

        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("max_new_tokens", 256),
                    do_sample=True,
                    temperature=self.config.get("temperature", 1.0),
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].size(1):],
                skip_special_tokens=True
            )
            responses.append(response)

        return responses

    def _compute_rewards(self, prompt: str, responses: List[str]) -> List[float]:
        """计算奖励"""
        rewards = []
        for response in responses:
            text = f"{prompt}\n{response}"
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.reward_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                reward = self.reward_model(**inputs).logits.squeeze().item()
            rewards.append(reward)

        return rewards

    def _compute_logprobs(self, model, prompt: str, responses: List[str]) -> torch.Tensor:
        """计算回复的log概率"""
        logprobs = []

        for response in responses:
            full_text = f"{prompt}{response}"
            inputs = self.tokenizer(full_text, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad() if model == self.ref_model else torch.enable_grad():
                outputs = model(**inputs)
                # 计算response部分的log概率
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = inputs["input_ids"][..., 1:].contiguous()

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

                # 只取response部分
                prompt_len = len(self.tokenizer.encode(prompt))
                response_log_prob = token_log_probs[0, prompt_len-1:].mean()

            logprobs.append(response_log_prob)

        return torch.stack(logprobs)

    def train(self, prompts: List[str], num_epochs: int = 1):
        """完整训练循环"""
        batch_size = self.config["batch_size"]

        for epoch in range(num_epochs):
            # 打乱数据
            import random
            random.shuffle(prompts)

            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                stats = self.train_step(batch_prompts)

                print(f"Epoch {epoch+1}, Step {i//batch_size + 1}: "
                      f"Loss={stats['loss']:.4f}, "
                      f"Mean Reward={stats['mean_reward']:.4f}, "
                      f"KL={stats['kl_divergence']:.4f}")

        # 保存模型
        self.policy_model.save_pretrained("./output/sales_grpo")
        self.tokenizer.save_pretrained("./output/sales_grpo")


# 使用示例
if __name__ == "__main__":
    config = {
        "learning_rate": 3e-6,
        "kl_coef": 0.001,
        "clip_ratio": 10,
        "num_samples_per_prompt": 8,
        "batch_size": 4,
        "max_new_tokens": 256,
        "temperature": 1.0
    }

    trainer = GRPOTrainer(
        policy_model_path="./output/sales_sft",
        reward_model_path="./output/sales_rm",
        config=config
    )

    # 加载训练prompts
    prompts = load_prompts("./data/prompts/sales_prompts.json")

    trainer.train(prompts, num_epochs=1)
```

### 3.4 使用OpenRLHF进行GRPO训练

```bash
# OpenRLHF是一个分布式RLHF框架，支持GRPO
# https://github.com/OpenRLHF/OpenRLHF

# 安装
pip install openrlhf

# GRPO训练
deepspeed --module openrlhf.cli.train_ppo \
   --pretrain ./output/sales_sft \
   --reward_pretrain ./output/sales_rm \
   --save_path ./output/sales_grpo \
   --micro_train_batch_size 2 \
   --train_batch_size 32 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 32 \
   --max_epochs 1 \
   --prompt_max_len 512 \
   --generate_max_len 256 \
   --actor_learning_rate 3e-6 \
   --init_kl_coef 0.001 \
   --use_grpo  # 启用GRPO而非PPO
```

---

## 四、训练稳定性与调优

### 4.1 常见不稳定问题

```python
"""
RL训练常见不稳定问题及解决方案
"""
stability_issues = {
    "Reward Hacking": {
        "症状": "Reward快速上升但生成质量下降",
        "原因": "模型找到了获取高reward的捷径",
        "解决": [
            "增大KL惩罚系数",
            "改进Reward Model",
            "添加多样性奖励"
        ]
    },
    "KL Divergence爆炸": {
        "症状": "KL值快速增长，模型输出变得奇怪",
        "原因": "更新步长太大",
        "解决": [
            "减小学习率",
            "增大KL惩罚系数",
            "减小PPO clip范围"
        ]
    },
    "Entropy Collapse": {
        "症状": "模型输出变得单一，多样性消失",
        "原因": "策略过度收敛",
        "解决": [
            "添加entropy bonus",
            "增加采样温度",
            "减少训练步数"
        ]
    },
    "训练突然崩溃": {
        "症状": "Loss突然变成NaN或极大值",
        "原因": "数值不稳定或梯度爆炸",
        "解决": [
            "使用BF16而非FP16",
            "减小学习率",
            "增加梯度裁剪"
        ]
    }
}
```

### 4.2 KL系数调优

```python
"""
KL系数的动态调整策略
"""
class AdaptiveKLController:
    """
    自适应KL控制器
    根据当前KL值动态调整系数
    """
    def __init__(
        self,
        init_kl_coef: float = 0.001,
        target_kl: float = 0.1,
        horizon: int = 10000
    ):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.horizon = horizon

    def update(self, current_kl: float) -> float:
        """
        根据当前KL值更新系数

        如果KL太大，增加系数（加强约束）
        如果KL太小，减小系数（放松约束）
        """
        proportional_error = current_kl / self.target_kl - 1
        mult = 1 + proportional_error / self.horizon

        self.kl_coef = max(0.0001, self.kl_coef * mult)

        return self.kl_coef


# 使用示例
kl_controller = AdaptiveKLController(init_kl_coef=0.001, target_kl=0.1)

for step in range(num_steps):
    # 训练一步
    stats = trainer.train_step(batch)
    current_kl = stats["kl_divergence"]

    # 更新KL系数
    new_kl_coef = kl_controller.update(current_kl)
    trainer.config["kl_coef"] = new_kl_coef

    print(f"Step {step}, KL: {current_kl:.4f}, KL Coef: {new_kl_coef:.6f}")
```

### 4.3 监控指标

```python
"""
RL训练必须监控的指标
"""
monitoring_metrics = {
    # 核心指标
    "mean_reward": "平均奖励，应该稳步上升",
    "reward_std": "奖励标准差，不应太小（否则缺乏多样性）",
    "kl_divergence": "与reference的KL散度，应在0.01-0.5范围",

    # PPO特定
    "policy_loss": "策略损失",
    "value_loss": "价值损失（PPO）",
    "clipfrac": "被裁剪的比例，不应太高（<0.3）",

    # 稳定性指标
    "entropy": "策略熵，不应太低",
    "approx_kl": "近似KL，用于监控更新幅度",

    # 生成质量
    "response_length": "回复长度，监控是否变得过长或过短",
    "distinct_1": "1-gram多样性",
    "distinct_2": "2-gram多样性"
}

class RLMonitor:
    def __init__(self):
        self.history = {metric: [] for metric in monitoring_metrics}

    def log(self, stats: Dict):
        for metric, value in stats.items():
            if metric in self.history:
                self.history[metric].append(value)

    def check_health(self) -> Dict:
        """检查训练健康状态"""
        issues = []

        # 检查KL
        recent_kl = self.history["kl_divergence"][-10:]
        if recent_kl and sum(recent_kl)/len(recent_kl) > 0.5:
            issues.append("KL过高，建议增大KL惩罚系数")

        # 检查reward趋势
        rewards = self.history["mean_reward"]
        if len(rewards) > 20:
            recent = rewards[-10:]
            earlier = rewards[-20:-10]
            if sum(recent)/len(recent) < sum(earlier)/len(earlier):
                issues.append("Reward开始下降，可能过拟合")

        # 检查多样性
        entropy = self.history.get("entropy", [])
        if entropy and entropy[-1] < 0.5:
            issues.append("Entropy过低，输出多样性不足")

        return {
            "healthy": len(issues) == 0,
            "issues": issues
        }
```

---

## 五、RLVR（可验证奖励的强化学习）

### 5.1 RLVR概述

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RLVR vs RLHF                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RLHF                                  RLVR                            │
│  ━━━━                                  ━━━━                            │
│                                                                         │
│  奖励来源：Reward Model               奖励来源：可验证的规则/结果       │
│  （人类偏好的近似）                    （数学正确性、代码执行结果等）    │
│                                                                         │
│  适用：                                适用：                           │
│  - 开放式任务                          - 数学题                         │
│  - 创意写作                            - 代码生成                        │
│  - 对话质量                            - 逻辑推理                        │
│                                                                         │
│  优点：                                优点：                           │
│  - 适用广泛                            - 奖励信号准确                   │
│  - 处理主观任务                        - Hacking风险显著降低*           │
│                                                                         │
│  缺点：                                缺点：                           │
│  - RM可能有偏差                        - 只适用可验证任务               │
│  - Reward Hacking                      - 仍有Hacking风险（见下文）      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

> ⚠️ **警告：RLVR 仍有 Reward Hacking 风险！**
>
> 虽然可验证奖励比学习的RM更可靠，但模型仍可能学会"钻空子"：
>
> | Hacking类型 | 例子 | 对策 |
> |-------------|------|------|
> | 规则过拟合 | 数学题只学特定题型的pattern | 增加题目多样性、隐藏测试集 |
> | Checker漏洞 | 代码通过测试但不具备泛化性 | 多组测试用例、对抗样本 |
> | 形式投机 | 猜测答案格式而非推理 | 检查推理过程（PRM） |
> | 训练集泄漏 | 记住训练题目的答案 | 严格的训练/验证/测试分离 |
>
> **最佳实践**：
> - 使用隐藏测试集验证泛化能力
> - 定期更换/扩充验证题目
> - 监控训练集vs测试集的pass rate差异
> - 结合PRM检查推理过程质量

### 5.2 销售LLM的可验证奖励

```python
"""
销售LLM的可验证奖励设计
"""
class SalesVerifiableReward:
    def __init__(self):
        # 关键要素检查
        self.required_elements = {
            "greeting": ["感谢", "您好", "欢迎"],
            "empathy": ["理解", "明白", "了解"],
            "question": ["?", "？"],  # 应该提问了解需求
            "value_proposition": ["优势", "价值", "效果", "节省"],
            "call_to_action": ["试用", "演示", "预约", "联系"]
        }

    def compute_reward(self, prompt: str, response: str) -> float:
        """
        计算销售回复的可验证奖励
        """
        rewards = []

        # 1. 格式检查（可验证）
        format_score = self._check_format(response)
        rewards.append(("format", format_score, 0.2))

        # 2. 关键要素检查（可验证）
        elements_score = self._check_elements(response)
        rewards.append(("elements", elements_score, 0.3))

        # 3. 长度检查（可验证）
        length_score = self._check_length(response)
        rewards.append(("length", length_score, 0.1))

        # 4. 禁忌词检查（可验证）
        safety_score = self._check_safety(response)
        rewards.append(("safety", safety_score, 0.2))

        # 5. 对话相关性（需要RM，半可验证）
        # 这部分可以结合RM来评估

        # 加权总分
        total = sum(score * weight for _, score, weight in rewards)

        return total

    def _check_format(self, response: str) -> float:
        """检查格式规范性"""
        score = 0.0

        # 有换行（结构化）
        if "\n" in response:
            score += 0.3

        # 有列表或编号
        if any(marker in response for marker in ["1.", "2.", "-", "•"]):
            score += 0.3

        # 有适当的标点
        if response.count("。") >= 2:
            score += 0.2

        # 有强调（加粗）
        if "**" in response:
            score += 0.2

        return min(1.0, score)

    def _check_elements(self, response: str) -> float:
        """检查关键销售要素"""
        score = 0.0
        for element, keywords in self.required_elements.items():
            if any(kw in response for kw in keywords):
                score += 0.2
        return min(1.0, score)

    def _check_length(self, response: str) -> float:
        """检查长度是否合适"""
        length = len(response)
        if 100 <= length <= 500:
            return 1.0
        elif 50 <= length < 100 or 500 < length <= 800:
            return 0.7
        else:
            return 0.3

    def _check_safety(self, response: str) -> float:
        """检查是否有禁忌内容"""
        forbidden = [
            "保证赚钱", "100%", "绝对没问题",
            "竞品很差", "骗人", "垃圾"
        ]
        for word in forbidden:
            if word in response:
                return 0.0
        return 1.0


# 使用示例
reward_fn = SalesVerifiableReward()

prompt = "客户说太贵了"
response = """我理解您对价格的考虑，这很正常。

让我帮您分析一下投资回报：

1. **成本节省**：使用我们的方案，每年可以节省约30%的运营成本
2. **效率提升**：平均提升工作效率40%
3. **快速回本**：大多数客户在6个月内就看到了回报

您看，我们是否可以安排一次详细的演示，让您亲自体验一下？"""

reward = reward_fn.compute_reward(prompt, response)
print(f"Reward: {reward:.2f}")
```

---

## 六、最佳实践总结

```markdown
## RLHF/GRPO训练检查清单

### 训练前准备
- [ ] SFT模型已经训练好
- [ ] Reward Model已经训练并验证
- [ ] 训练prompts已经准备好
- [ ] 计算资源充足（RL需要更多资源）

### 超参数设置
- [ ] 学习率足够小（1e-6 ~ 5e-6）
- [ ] KL系数合适（0.001 ~ 0.1）
- [ ] Clip范围合理（PPO: 0.2, GRPO: 10）
- [ ] Batch size足够大

### 训练监控
- [ ] 监控mean reward趋势
- [ ] 监控KL divergence
- [ ] 定期检查生成质量
- [ ] 监控entropy/多样性

### 常见问题检查
- [ ] Reward是否持续上升？
- [ ] KL是否在合理范围？
- [ ] 输出是否变得单一？
- [ ] 是否出现Reward Hacking？

### 训练后验证
- [ ] 与SFT模型对比胜率
- [ ] 人工评估生成质量
- [ ] 检查通用能力是否保持
```

---

## 参考资源

### 论文
- [InstructGPT (PPO)](https://arxiv.org/abs/2203.02155)
- [DeepSeek-R1 (GRPO)](https://arxiv.org/abs/2501.12948)
- [PPO原始论文](https://arxiv.org/abs/1707.06347)

### 框架
- [TRL (HuggingFace)](https://github.com/huggingface/trl)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)

### 博客
- [HuggingFace RLHF Blog](https://huggingface.co/blog/rlhf)
- [Nathan Lambert - RLHF Book](https://rlhfbook.com/)

---

> **下一章**：[07-评估与迭代.md](./07-评估与迭代.md) - 学习如何评估模型效果并持续改进
