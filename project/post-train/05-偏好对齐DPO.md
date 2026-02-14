# 第五章：偏好对齐 - DPO（Direct Preference Optimization）

> **核心目标**：无需训练Reward Model，直接使用偏好数据优化模型
>
> **本章目标**：掌握DPO的原理、实现、超参数调优和常见问题
>
> **优势**：更简单、更稳定、计算成本更低

---

## 一、DPO概述

### 1.1 DPO vs RLHF

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      RLHF vs DPO 对比                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RLHF (传统方法)                       DPO (简化方法)                   │
│  ━━━━━━━━━━━━━━━                       ━━━━━━━━━━━━━━                   │
│                                                                         │
│  ┌─────────────┐                       ┌─────────────┐                 │
│  │ 1. SFT      │                       │ 1. SFT      │                 │
│  └──────┬──────┘                       └──────┬──────┘                 │
│         │                                      │                        │
│         ▼                                      │                        │
│  ┌─────────────┐                              │                        │
│  │ 2. 训练RM   │ ◀── 需要额外训练              │                        │
│  └──────┬──────┘                              │                        │
│         │                                      │                        │
│         ▼                                      ▼                        │
│  ┌─────────────┐                       ┌─────────────┐                 │
│  │ 3. PPO优化  │ ◀── 复杂、不稳定      │ 2. DPO优化  │ ◀── 简单直接   │
│  └─────────────┘                       └─────────────┘                 │
│                                                                         │
│  优点：效果上限高                       优点：简单、稳定、成本低        │
│  缺点：复杂、不稳定、成本高             缺点：效果可能略低              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 DPO的核心思想

DPO证明了：**可以跳过Reward Model，直接用偏好数据优化语言模型**

```python
"""
DPO的数学原理

RLHF的目标：
    max E[r(x,y)] - β * KL(π||π_ref)

DPO发现，最优策略可以表示为：
    π*(y|x) = (1/Z(x)) * π_ref(y|x) * exp(r(x,y)/β)

反解reward：
    r(x,y) = β * log(π*(y|x)/π_ref(y|x)) + β * log(Z(x))

代入Bradley-Terry模型，得到DPO Loss：
    L_DPO = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))

其中：
    y_w = chosen (胜出的回复)
    y_l = rejected (失败的回复)
    β = KL散度约束强度
"""
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,   # π(y_w|x)的对数概率
    policy_rejected_logps: torch.Tensor,  # π(y_l|x)的对数概率
    reference_chosen_logps: torch.Tensor,  # π_ref(y_w|x)的对数概率
    reference_rejected_logps: torch.Tensor,  # π_ref(y_l|x)的对数概率
    beta: float = 0.1
) -> torch.Tensor:
    """
    DPO损失函数
    """
    # 计算chosen和rejected的log ratio
    chosen_logratios = policy_chosen_logps - reference_chosen_logps
    rejected_logratios = policy_rejected_logps - reference_rejected_logps

    # DPO loss
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    return loss


# 示例
policy_chosen_logps = torch.tensor([-2.0, -1.5, -2.5])
policy_rejected_logps = torch.tensor([-3.0, -2.5, -3.5])
reference_chosen_logps = torch.tensor([-2.2, -1.7, -2.7])
reference_rejected_logps = torch.tensor([-2.8, -2.3, -3.3])

loss = dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=0.1
)
print(f"DPO Loss: {loss.item():.4f}")
```

### 1.3 DPO的优势

| 优势 | 说明 |
|------|------|
| **无需RM** | 跳过Reward Model训练，减少工作量 |
| **更稳定** | 避免了PPO的不稳定性问题 |
| **计算高效** | 只需要一次前向传播，无需采样 |
| **简单实现** | 代码实现简单，易于调试 |
| **显存友好** | 不需要同时维护多个模型 |

---

## 二、DPO数据准备

### 2.1 数据格式

```json
{
    "prompt": "客户说太贵了，你会怎么回应？",
    "chosen": "我理解您对价格的考虑。让我帮您分析一下投资回报：我们的产品虽然初期投入较高，但从长期来看...",
    "rejected": "我们的价格已经是最低的了，不能再便宜了。"
}
```

### 2.2 偏好数据质量的重要性

> **研究发现**：提高chosen的质量比降低rejected的质量更重要

```python
"""
DPO数据质量分析
"""
class DPODataAnalyzer:
    def __init__(self):
        pass

    def analyze_quality_gap(self, data: list) -> dict:
        """分析chosen和rejected之间的质量差距"""
        gaps = []
        for item in data:
            chosen_score = self._estimate_quality(item["chosen"])
            rejected_score = self._estimate_quality(item["rejected"])
            gaps.append(chosen_score - rejected_score)

        return {
            "mean_gap": sum(gaps) / len(gaps),
            "min_gap": min(gaps),
            "max_gap": max(gaps),
            "low_gap_ratio": sum(1 for g in gaps if g < 0.3) / len(gaps)
        }

    def _estimate_quality(self, text: str) -> float:
        """简单的质量估计"""
        score = 0.5

        # 长度因素
        if 50 < len(text) < 500:
            score += 0.2

        # 结构因素
        if "\n" in text or "1." in text:
            score += 0.1

        # 专业性因素
        professional_words = ["分析", "建议", "方案", "优势"]
        if any(w in text for w in professional_words):
            score += 0.2

        return min(1.0, score)

    def filter_low_quality_pairs(self, data: list, min_gap: float = 0.3) -> list:
        """过滤质量差距太小的数据对"""
        filtered = []
        for item in data:
            chosen_score = self._estimate_quality(item["chosen"])
            rejected_score = self._estimate_quality(item["rejected"])
            if chosen_score - rejected_score >= min_gap:
                filtered.append(item)

        print(f"过滤前: {len(data)}, 过滤后: {len(filtered)}")
        return filtered
```

### 2.3 销售LLM偏好数据构造

```python
"""
销售LLM的DPO数据构造策略
"""
from openai import OpenAI

class SalesDPODataGenerator:
    def __init__(self):
        self.client = OpenAI()

        # 销售场景模板
        self.scenarios = [
            {"type": "price_objection", "prompt_template": "客户说'{objection}'，作为销售你会怎么回应？"},
            {"type": "competitor_comparison", "prompt_template": "客户问'你们和{competitor}比有什么优势？'"},
            {"type": "need_discovery", "prompt_template": "客户说'{need}'，你如何深入挖掘需求？"},
            {"type": "closing", "prompt_template": "客户已经了解了产品，你如何促成成交？"},
        ]

    def generate_chosen(self, prompt: str) -> str:
        """使用强模型生成高质量回复"""
        system_prompt = """你是一个顶级销售顾问，需要生成专业、有效的销售回复。
要求：
1. 表达同理心，理解客户顾虑
2. 提供有价值的信息和分析
3. 引导客户继续对话
4. 专业但不咄咄逼人
5. 长度适中（100-300字）"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    def generate_rejected(self, prompt: str) -> str:
        """生成低质量回复"""
        # 策略1: 使用弱模型
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0
        )
        weak_response = response.choices[0].message.content

        # 策略2: 生成常见错误回复
        bad_responses = [
            "不好意思，这个我不太清楚。",
            "价格就是这样，不能再低了。",
            "你可以自己去官网看看。",
            "买不买随便你。"
        ]

        # 随机选择一种
        import random
        if random.random() > 0.5:
            return weak_response
        else:
            return random.choice(bad_responses)

    def create_preference_pair(self, prompt: str) -> dict:
        """创建一个偏好数据对"""
        chosen = self.generate_chosen(prompt)
        rejected = self.generate_rejected(prompt)

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

    def batch_generate(self, prompts: list) -> list:
        """批量生成偏好数据"""
        data = []
        for i, prompt in enumerate(prompts):
            print(f"生成进度: {i+1}/{len(prompts)}")
            pair = self.create_preference_pair(prompt)
            data.append(pair)
        return data


# 销售场景示例prompts
sales_prompts = [
    "客户说'太贵了'，你会怎么回应？",
    "客户说'我需要再考虑考虑'，你会怎么说？",
    "客户问'你们和XX公司比有什么优势？'",
    "客户说'我们预算有限'，你会如何应对？",
    "客户问'能便宜点吗？'，你会怎么谈价？",
    "客户说'我需要和领导商量'，你下一步怎么做？",
    "客户表示'现在不急'，你如何创造紧迫感？",
    "客户问'有没有成功案例？'，你会怎么介绍？",
]
```

---

## 三、DPO训练实战

### 3.1 使用TRL进行DPO训练

```python
"""
使用TRL库进行DPO训练
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

def train_dpo():
    # ============ 1. 加载模型 ============
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # 或使用SFT后的模型

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Reference model（DPO需要）
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # ============ 2. 添加LoRA ============
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)

    # 使用PEFT时，原始模型可作为reference
    # 这样不需要额外加载ref_model
    ref_model = None  # 设为None，DPOTrainer会自动使用原始权重

    # ============ 3. 加载数据 ============
    dataset = load_dataset("json", data_files="./data/preference/sales_dpo.json")

    # DPO数据格式：需要prompt, chosen, rejected字段
    def format_dataset(example):
        """格式化数据为DPO需要的格式"""
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    dataset = dataset.map(format_dataset)

    # ============ 4. 训练配置 ============
    training_args = DPOConfig(
        output_dir="./output/sales_dpo",

        # DPO特定参数
        beta=0.1,  # KL散度约束强度，核心超参数

        # 训练参数
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,

        # 学习率（DPO通常用较小学习率）
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # 序列长度
        max_length=1024,
        max_prompt_length=512,

        # 其他
        bf16=True,
        logging_steps=10,
        save_steps=200,
        gradient_checkpointing=True,

        # 评估
        eval_strategy="steps",
        eval_steps=200,
    )

    # ============ 5. 训练 ============
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # 如果用PEFT可以设为None
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
    )

    trainer.train()

    # ============ 6. 保存 ============
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    print("DPO训练完成！")


if __name__ == "__main__":
    train_dpo()
```

### 3.2 使用LLaMA-Factory进行DPO

```yaml
# configs/sales_dpo.yaml
# DPO训练配置

### 模型配置
model_name_or_path: ./output/sales_sft  # 使用SFT后的模型
trust_remote_code: true

### 训练方法
stage: dpo
do_train: true
finetuning_type: lora

### LoRA配置
lora_target: all
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

### DPO特定配置
pref_beta: 0.1  # β参数
pref_loss: sigmoid  # 损失函数类型

### 数据配置
dataset: sales_preference
template: qwen
cutoff_len: 1024

### 训练参数
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
num_train_epochs: 1
learning_rate: 5e-6
lr_scheduler_type: cosine
warmup_ratio: 0.1

### 其他
bf16: true
output_dir: ./output/sales_dpo
logging_steps: 10
save_steps: 200
```

```bash
# 启动DPO训练
llamafactory-cli train configs/sales_dpo.yaml
```

### 3.3 β参数调优

```python
"""
β参数的影响和调优策略
"""

# β的作用
beta_explanation = """
β（beta）是DPO中最重要的超参数，控制KL散度约束的强度：

β越小：
- 模型更激进地学习偏好
- 可能与reference model偏离较大
- 学习速度快，但可能不稳定

β越大：
- 模型更保守
- 与reference model保持接近
- 学习速度慢，但更稳定

常用值：0.1（最常用）、0.05、0.2
"""

# 不同场景的β选择
beta_selection = {
    "标准场景": {
        "beta": 0.1,
        "说明": "最常用的默认值"
    },
    "数据质量高": {
        "beta": 0.05,
        "说明": "可以更激进地学习"
    },
    "数据质量不确定": {
        "beta": 0.2,
        "说明": "更保守，减少过拟合风险"
    },
    "追求多样性": {
        "beta": 0.15,
        "说明": "平衡偏好学习和输出多样性"
    }
}

# β调优实验
def beta_sweep(betas: list = [0.05, 0.1, 0.15, 0.2]):
    """
    Beta参数扫描实验
    """
    results = []
    for beta in betas:
        # 训练模型
        model = train_dpo_with_beta(beta)

        # 评估
        metrics = evaluate_model(model)

        results.append({
            "beta": beta,
            "accuracy": metrics["accuracy"],
            "diversity": metrics["diversity"],
            "kl_divergence": metrics["kl_divergence"]
        })

    return results
```

---

## 四、DPO变体

### 4.1 主要DPO变体对比

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DPO变体对比                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  方法        特点                              适用场景                 │
│  ━━━━        ━━━━                              ━━━━━━                   │
│                                                                         │
│  DPO         原始方法，需要reference model     通用                     │
│                                                                         │
│  IPO         添加正则项，更稳定                数据质量不高时           │
│                                                                         │
│  cDPO        添加label smoothing               减少过拟合               │
│                                                                         │
│  KTO         只需要good/bad标签，不需要pair    数据获取困难时           │
│                                                                         │
│  ORPO        不需要reference model             简化训练                 │
│                                                                         │
│  SimPO       简化DPO，效果相当                 追求简单                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 ORPO：无需Reference Model

```python
"""
ORPO (Odds Ratio Preference Optimization)
不需要reference model，更加简化
"""
import torch
import torch.nn.functional as F

def orpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    beta: float = 0.1
):
    """
    ORPO损失函数
    不需要reference model的logprobs
    """
    # 计算odds ratio
    log_odds = policy_chosen_logps - policy_rejected_logps

    # ORPO loss = -log(sigmoid(log_odds)) + SFT_loss
    preference_loss = -F.logsigmoid(beta * log_odds).mean()

    # 可以结合SFT loss
    # total_loss = sft_loss + lambda * preference_loss

    return preference_loss


# TRL中使用ORPO
"""
from trl import ORPOTrainer, ORPOConfig

config = ORPOConfig(
    output_dir="./output/orpo",
    beta=0.1,
    ...
)

trainer = ORPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
    # 注意：不需要ref_model
)
"""
```

### 4.3 SimPO：简化的DPO

```python
"""
SimPO (Simple Preference Optimization)
进一步简化DPO，不需要reference model
使用sequence长度归一化
"""
def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5
):
    """
    SimPO损失函数

    与DPO的区别：
    1. 不需要reference model
    2. 使用长度归一化
    3. 添加target margin
    """
    # 长度归一化
    chosen_rewards = policy_chosen_logps / chosen_lengths
    rejected_rewards = policy_rejected_logps / rejected_lengths

    # 带margin的损失
    logits = beta * (chosen_rewards - rejected_rewards) - gamma
    loss = -F.logsigmoid(logits).mean()

    return loss
```

---

## 五、DPO常见问题与解决方案

### 5.1 训练不收敛

```python
"""
DPO训练不收敛的诊断和解决
"""
training_issues = {
    "Loss不下降": {
        "可能原因": [
            "学习率太小",
            "数据质量差（chosen和rejected差异不明显）",
            "β值设置不当"
        ],
        "解决方案": [
            "增大学习率（从5e-6到1e-5）",
            "检查数据，确保chosen明显优于rejected",
            "尝试减小β值（从0.1到0.05）"
        ]
    },
    "Loss震荡": {
        "可能原因": [
            "学习率太大",
            "batch size太小"
        ],
        "解决方案": [
            "减小学习率",
            "增大gradient accumulation"
        ]
    },
    "模型输出变差": {
        "可能原因": [
            "训练轮数太多",
            "β太小导致偏离reference太多"
        ],
        "解决方案": [
            "减少训练轮数（通常1轮足够）",
            "增大β值"
        ]
    }
}
```

### 5.2 输出多样性下降

```python
"""
DPO导致输出多样性下降的处理
"""
diversity_solutions = {
    "问题": "DPO训练后，模型倾向于生成相似的回复",

    "原因": "模型过度拟合到chosen的模式",

    "解决方案": [
        {
            "方法": "增大β值",
            "说明": "β=0.2或更高，减少偏离reference"
        },
        {
            "方法": "减少训练步数",
            "说明": "可能只需要一半的训练步数"
        },
        {
            "方法": "数据增强",
            "说明": "为同一prompt提供多个不同风格的chosen"
        },
        {
            "方法": "推理时增加temperature",
            "说明": "生成时使用更高的temperature"
        }
    ]
}
```

### 5.3 Verbosity问题

```python
"""
DPO导致回复变长的问题
研究发现DPO训练后模型倾向于生成更长的回复
"""
verbosity_mitigation = {
    "问题": "DPO训练后回复变得冗长",

    "解决方案": [
        {
            "方法": "长度惩罚",
            "代码": """
def length_penalized_dpo_loss(chosen_logps, rejected_logps,
                              chosen_len, rejected_len, beta=0.1, length_penalty=0.01):
    # 标准DPO loss
    loss = -F.logsigmoid(beta * (chosen_logps - rejected_logps)).mean()

    # 添加长度惩罚
    length_diff = torch.abs(chosen_len - rejected_len).float().mean()
    loss += length_penalty * length_diff

    return loss
"""
        },
        {
            "方法": "在数据中包含简洁的chosen",
            "说明": "确保偏好数据中有简洁但高质量的回复"
        },
        {
            "方法": "后处理",
            "说明": "使用模型做摘要或压缩"
        }
    ]
}
```

---

## 六、DPO效果评估

### 6.1 评估指标

```python
"""
DPO训练效果评估
"""
class DPOEvaluator:
    def __init__(self, model_path: str, ref_model_path: str):
        self.model = load_model(model_path)
        self.ref_model = load_model(ref_model_path)
        self.tokenizer = load_tokenizer(model_path)

    def evaluate_preference_accuracy(self, test_data: list) -> float:
        """
        在测试集上评估偏好预测准确率
        模型应该给chosen更高的概率
        """
        correct = 0
        total = len(test_data)

        for item in test_data:
            chosen_logp = self._get_log_prob(item["prompt"], item["chosen"])
            rejected_logp = self._get_log_prob(item["prompt"], item["rejected"])

            if chosen_logp > rejected_logp:
                correct += 1

        return correct / total

    def evaluate_kl_divergence(self, test_prompts: list) -> float:
        """
        评估与reference model的KL散度
        """
        kl_values = []

        for prompt in test_prompts:
            # 获取两个模型的输出分布
            policy_logits = self._get_logits(self.model, prompt)
            ref_logits = self._get_logits(self.ref_model, prompt)

            # 计算KL散度
            kl = F.kl_div(
                F.log_softmax(policy_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction='batchmean'
            )
            kl_values.append(kl.item())

        return sum(kl_values) / len(kl_values)

    def evaluate_win_rate(self, test_prompts: list, judge_model=None) -> float:
        """
        与reference model对比的胜率
        """
        wins = 0
        total = len(test_prompts)

        for prompt in test_prompts:
            policy_response = self._generate(self.model, prompt)
            ref_response = self._generate(self.ref_model, prompt)

            # 使用judge model判断
            winner = judge_model.judge(prompt, policy_response, ref_response)

            if winner == "policy":
                wins += 1

        return wins / total

    def _get_log_prob(self, prompt: str, response: str) -> float:
        # 计算response的对数概率
        pass

    def _get_logits(self, model, prompt: str):
        # 获取模型输出的logits
        pass

    def _generate(self, model, prompt: str) -> str:
        # 生成回复
        pass


# 使用示例
evaluator = DPOEvaluator(
    model_path="./output/sales_dpo",
    ref_model_path="./output/sales_sft"
)

# 评估
test_data = load_json("./data/preference/test.json")
accuracy = evaluator.evaluate_preference_accuracy(test_data)
print(f"偏好准确率: {accuracy:.2%}")
```

### 6.2 人工评估

```python
"""
DPO人工评估框架
"""
def create_human_eval_task(model, ref_model, test_prompts: list) -> list:
    """
    创建人工评估任务
    """
    tasks = []

    for prompt in test_prompts:
        policy_response = model.generate(prompt)
        ref_response = ref_model.generate(prompt)

        # 随机排序避免位置偏见
        import random
        if random.random() > 0.5:
            response_a, response_b = policy_response, ref_response
            mapping = {"A": "policy", "B": "reference"}
        else:
            response_a, response_b = ref_response, policy_response
            mapping = {"A": "reference", "B": "policy"}

        tasks.append({
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "_mapping": mapping  # 内部记录，不展示给标注员
        })

    return tasks
```

---

## 七、DPO最佳实践总结

```markdown
## DPO训练检查清单

### 数据准备
- [ ] 数据格式正确（prompt, chosen, rejected）
- [ ] chosen明显优于rejected
- [ ] 数据量适中（1k-10k对）
- [ ] 覆盖多样化场景

### 模型准备
- [ ] 使用SFT后的模型作为起点
- [ ] 配置LoRA（或准备全参训练资源）
- [ ] 准备reference model（或使用PEFT共享权重）

### 超参数设置
- [ ] β = 0.1（默认，可调整）
- [ ] 学习率 = 1e-6 ~ 5e-6（比SFT小）
- [ ] 训练轮数 = 1（通常足够）
- [ ] 使用梯度累积达到合适的batch size

### 训练监控
- [ ] Loss应该稳定下降
- [ ] 监控chosen/rejected的reward差距
- [ ] 定期Vibe Check生成质量

### 效果验证
- [ ] 偏好准确率 > 70%
- [ ] 与reference的KL散度在合理范围
- [ ] 人工评估胜率 > 55%
- [ ] 输出多样性没有明显下降
```

---

## 参考资源

### 论文
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) - DPO原始论文
- [ORPO](https://arxiv.org/abs/2403.07691) - 无需reference model的方法
- [SimPO](https://arxiv.org/abs/2405.14734) - 简化的DPO

### 博客
- [Cameron Wolfe - DPO深度解析](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
- [HuggingFace - RLHF to DPO](https://huggingface.co/blog/ariG23498/rlhf-to-dpo)

### 工具
- [TRL DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

---

> **下一章**：[06-强化学习RLHF-GRPO.md](./06-强化学习RLHF-GRPO.md) - 学习PPO和GRPO强化学习方法
