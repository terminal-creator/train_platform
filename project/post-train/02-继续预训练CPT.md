# 第二章：继续预训练（Continual Pre-Training, CPT）

> **核心目标**：将领域知识注入基座模型，使其理解专业术语和领域概念
>
> **适用场景**：需要深度领域知识的场景（金融、医疗、法律、销售等）
>
> **本章目标**：掌握CPT的原理、实施方法和避坑技巧

---

## 一、CPT概述

### 1.1 什么是继续预训练？

继续预训练是在预训练基座模型的基础上，使用大量**领域无标注文本**继续进行自监督学习，让模型习得领域专业知识。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CPT在训练流程中的位置                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐   │
│   │   预训练      │         │   继续预训练  │         │   监督微调    │   │
│   │ Pre-Training │  ─────▶ │     CPT      │  ─────▶ │     SFT      │   │
│   └──────────────┘         └──────────────┘         └──────────────┘   │
│         │                        │                        │            │
│         ▼                        ▼                        ▼            │
│   ┌──────────────┐         ┌──────────────┐         ┌──────────────┐   │
│   │ 通用知识      │         │ +领域知识     │         │ +指令遵循     │   │
│   │ 语言理解      │         │ +专业术语     │         │ +对话能力     │   │
│   └──────────────┘         └──────────────┘         └──────────────┘   │
│                                                                         │
│   数据：万亿token            数据：数十GB~TB          数据：数千~数万条   │
│   方法：自监督               方法：自监督             方法：监督学习     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 CPT vs SFT：何时需要CPT？

| 维度 | 继续预训练 (CPT) | 监督微调 (SFT) |
|------|-----------------|----------------|
| **目的** | 注入领域知识 | 学习指令遵循 |
| **数据类型** | 无标注纯文本 | 指令-回复对 |
| **数据量** | 数GB~数TB | 数千~数万条 |
| **计算成本** | 高 | 相对低 |
| **适用场景** | 专业领域、特定语言 | 所有微调场景 |

### 1.3 是否需要CPT的决策流程

```
                    ┌─────────────────────────────┐
                    │   你的任务是否涉及          │
                    │   大量专业领域知识？        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
                   是                            否
                    │                             │
        ┌───────────┴───────────┐               │
        │ 模型是否已掌握这些知识？│           直接SFT
        └───────────┬───────────┘               │
                    │                             │
          ┌────────┴────────┐                   │
          ▼                 ▼                   │
         是                否                   │
          │                 │                   │
          ▼                 ▼                   │
      直接SFT            需要CPT                │
          │                 │                   │
          └────────┬────────┘                   │
                   ▼                             │
              最终方案 ◀──────────────────────────┘
```

**销售LLM的判断**：
- 通用销售技巧：基座模型已有，无需CPT
- 公司专有产品知识：需要CPT或RAG
- 行业专业术语：可能需要轻量CPT
- **结论**：对于销售LLM，如果有大量公司内部知识库，建议做轻量CPT

---

## 二、CPT核心原理

### 2.1 训练目标

CPT的训练目标与预训练相同：**Next Token Prediction**

```python
# 训练目标：最大化下一个token的预测概率
# Loss = -log P(x_t | x_1, x_2, ..., x_{t-1})

import torch
import torch.nn.functional as F

def compute_cpt_loss(logits, labels, ignore_index=-100):
    """
    计算继续预训练的loss
    与预训练完全相同的Causal Language Modeling Loss
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction='mean'
    )
    return loss
```

### 2.2 灾难性遗忘问题

CPT面临的核心挑战是**灾难性遗忘（Catastrophic Forgetting）**：
- 模型在学习新知识时，可能丢失原有的通用能力
- 越激进的训练，遗忘越严重

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         灾难性遗忘的权衡                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│     领域知识获取                              通用能力保持              │
│         ▲                                         ▲                    │
│         │                                         │                    │
│   高    │        ┌─────────────┐                 │   高               │
│         │       │ 理想区域     │                 │                    │
│         │      │  (平衡点)    │                  │                    │
│         │       └─────────────┘                 │                    │
│   低    └──────────────────────────────────────►│   低               │
│                    训练强度                                            │
│                                                                         │
│   训练太弱：领域知识习得不足                                            │
│   训练太强：通用能力严重损失                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 缓解遗忘的策略

#### 策略一：数据混合（Data Replay）

```python
# 核心思想：混合领域数据和通用数据
data_mixture = {
    "domain_data": 0.7,      # 70% 领域数据
    "replay_data": 0.3       # 30% 原始预训练数据（或高质量通用数据）
}

# 实践经验：
# - 贝壳论文推荐 domain:general = 4:1 (即80%:20%)
# - 保守做法可以用 7:3 或 6:4
```

#### 策略二：学习率控制

```python
# CPT的学习率应该远小于预训练
training_config = {
    "pretrain_lr": 1e-4,       # 预训练学习率
    "cpt_lr": 1e-5 ~ 5e-5,     # CPT学习率（降低一个数量级）
    "warmup_ratio": 0.1,       # 预热比例
    "scheduler": "cosine"      # 使用cosine衰减
}
```

#### 策略三：弹性权重固化（EWC）

```python
"""
Elastic Weight Consolidation (EWC)
通过惩罚重要参数的变化来保护原有知识
"""
import torch

class EWCRegularizer:
    def __init__(self, model, fisher_matrix, old_params, lambda_ewc=1000):
        """
        Args:
            model: 当前模型
            fisher_matrix: Fisher信息矩阵（参数重要性）
            old_params: 原始模型参数
            lambda_ewc: 正则化强度
        """
        self.model = model
        self.fisher_matrix = fisher_matrix
        self.old_params = old_params
        self.lambda_ewc = lambda_ewc

    def penalty(self):
        """计算EWC惩罚项"""
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                # 惩罚重要参数的变化
                loss += (self.fisher_matrix[name] *
                        (param - self.old_params[name]).pow(2)).sum()
        return self.lambda_ewc * loss

# 使用示例
def train_step_with_ewc(model, batch, ewc_regularizer):
    # 正常的语言模型loss
    lm_loss = compute_cpt_loss(model(batch))

    # 加上EWC惩罚
    ewc_loss = ewc_regularizer.penalty()

    total_loss = lm_loss + ewc_loss
    return total_loss
```

#### 策略四：模型权重平均

```python
"""
简单而有效：训练完成后，将CPT模型与原始模型做权重平均
"""
def weight_averaging(original_model, cpt_model, alpha=0.5):
    """
    Args:
        original_model: 原始基座模型
        cpt_model: CPT后的模型
        alpha: 原始模型的权重 (0-1)
    """
    averaged_state_dict = {}

    for key in original_model.state_dict():
        averaged_state_dict[key] = (
            alpha * original_model.state_dict()[key] +
            (1 - alpha) * cpt_model.state_dict()[key]
        )

    cpt_model.load_state_dict(averaged_state_dict)
    return cpt_model

# 实践经验：alpha=0.3~0.5 通常效果较好
# 即保留30-50%的原始权重
```

---

## 三、CPT实战：销售LLM领域适配

### 3.1 数据准备

```python
"""
销售LLM的CPT数据准备
"""
import json
from pathlib import Path
from typing import List, Generator
import random

class SalesCPTDataPreparer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_domain_data(self) -> List[str]:
        """准备销售领域数据"""
        domain_texts = []

        # 1. 产品文档
        product_docs = self._load_product_documents()
        domain_texts.extend(product_docs)
        print(f"产品文档: {len(product_docs)} 条")

        # 2. 销售培训材料
        training_materials = self._load_training_materials()
        domain_texts.extend(training_materials)
        print(f"培训材料: {len(training_materials)} 条")

        # 3. 成功案例
        success_cases = self._load_success_cases()
        domain_texts.extend(success_cases)
        print(f"成功案例: {len(success_cases)} 条")

        # 4. 行业报告
        industry_reports = self._load_industry_reports()
        domain_texts.extend(industry_reports)
        print(f"行业报告: {len(industry_reports)} 条")

        return domain_texts

    def prepare_replay_data(self, target_ratio: float = 0.3) -> List[str]:
        """
        准备通用数据用于防止遗忘
        可以使用：
        1. 原始预训练数据子集（如果可获取）
        2. 高质量开源文本（Wikipedia、书籍等）
        3. 通用对话数据
        """
        replay_texts = []

        # 使用Wikipedia文本作为replay数据
        wiki_texts = self._load_wikipedia_sample()
        replay_texts.extend(wiki_texts)

        return replay_texts

    def create_training_file(self,
                            domain_data: List[str],
                            replay_data: List[str],
                            domain_ratio: float = 0.7) -> str:
        """
        创建最终的训练文件
        按比例混合领域数据和通用数据
        """
        # 计算采样数量
        total_domain = len(domain_data)
        total_replay = int(total_domain * (1 - domain_ratio) / domain_ratio)
        total_replay = min(total_replay, len(replay_data))

        # 采样
        sampled_replay = random.sample(replay_data, total_replay)

        # 合并并打乱
        all_data = domain_data + sampled_replay
        random.shuffle(all_data)

        # 保存为训练格式
        output_file = self.output_dir / "cpt_train.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in all_data:
                # 确保每个文档以换行结束
                f.write(text.strip() + '\n\n')

        print(f"领域数据: {total_domain}, 通用数据: {total_replay}")
        print(f"总计: {len(all_data)} 条, 保存至: {output_file}")

        return str(output_file)

    def _load_product_documents(self) -> List[str]:
        """加载产品文档"""
        texts = []
        product_dir = Path("./data/raw/products")

        if product_dir.exists():
            for file in product_dir.glob("*.txt"):
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 按章节分割，每个章节作为一个训练样本
                    sections = content.split('\n\n\n')
                    texts.extend([s.strip() for s in sections if len(s.strip()) > 100])

        return texts

    def _load_training_materials(self) -> List[str]:
        """加载销售培训材料"""
        # 实现加载逻辑...
        return []

    def _load_success_cases(self) -> List[str]:
        """加载成功案例"""
        # 实现加载逻辑...
        return []

    def _load_industry_reports(self) -> List[str]:
        """加载行业报告"""
        # 实现加载逻辑...
        return []

    def _load_wikipedia_sample(self) -> List[str]:
        """加载Wikipedia样本作为通用数据"""
        # 可以使用HuggingFace datasets加载
        # from datasets import load_dataset
        # wiki = load_dataset("wikipedia", "20220301.zh", split="train[:10000]")
        return []


# 使用示例
if __name__ == "__main__":
    preparer = SalesCPTDataPreparer("./data/cpt")

    # 准备数据
    domain_data = preparer.prepare_domain_data()
    replay_data = preparer.prepare_replay_data()

    # 创建训练文件
    train_file = preparer.create_training_file(
        domain_data=domain_data,
        replay_data=replay_data,
        domain_ratio=0.7
    )
```

### 3.2 使用LLaMA-Factory进行CPT

```yaml
# configs/sales_cpt.yaml
# LLaMA-Factory CPT配置文件

### 模型配置
model_name_or_path: Qwen/Qwen2.5-7B  # 使用Base模型，不是Instruct

### 训练方法
stage: pt  # pre-training stage，即CPT
do_train: true
finetuning_type: full  # 全参数训练（CPT通常需要全参）

### 数据配置
dataset: sales_cpt  # 在dataset_info.json中定义
template: default  # 纯文本，无需特殊模板
cutoff_len: 2048  # 序列长度
preprocessing_num_workers: 16

### 训练超参数
per_device_train_batch_size: 4
gradient_accumulation_steps: 8  # 有效batch size = 4 * 8 = 32
learning_rate: 2e-5  # CPT学习率，比预训练低
num_train_epochs: 1  # CPT通常1-2个epoch
lr_scheduler_type: cosine
warmup_ratio: 0.1

### 优化器配置
optim: adamw_torch
weight_decay: 0.01
max_grad_norm: 1.0

### 保存配置
output_dir: ./output/sales_cpt
logging_steps: 10
save_steps: 500
save_total_limit: 3

### 精度配置
bf16: true  # 使用BF16训练

### DeepSpeed配置（多卡训练）
deepspeed: configs/ds_z2_config.json
```

```json
// dataset_info.json - 添加CPT数据集定义
{
  "sales_cpt": {
    "file_name": "data/cpt/cpt_train.txt",
    "file_sha1": null,
    "columns": {
      "prompt": "text"
    }
  }
}
```

```bash
# 启动CPT训练
# 单卡
llamafactory-cli train configs/sales_cpt.yaml

# 多卡（DeepSpeed ZeRO-2）
deepspeed --num_gpus 4 src/train.py configs/sales_cpt.yaml
```

### 3.3 使用Transformers原生训练

```python
"""
使用HuggingFace Transformers进行CPT
更灵活的控制，适合自定义需求
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import os

def train_cpt():
    # ============ 1. 加载模型和分词器 ============
    model_name = "Qwen/Qwen2.5-7B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # 如果显存不够，可以启用gradient checkpointing
        # use_cache=False
    )

    # 启用gradient checkpointing节省显存
    model.gradient_checkpointing_enable()

    # ============ 2. 准备数据集 ============
    def tokenize_function(examples):
        """将文本转换为token ids"""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False,
            return_special_tokens_mask=True
        )

    # 加载数据
    dataset = load_dataset("text", data_files={"train": "./data/cpt/cpt_train.txt"})

    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=dataset["train"].column_names,
    )

    # ============ 3. 数据整理器 ============
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM，不是Masked LM
    )

    # ============ 4. 训练参数 ============
    training_args = TrainingArguments(
        output_dir="./output/sales_cpt",
        overwrite_output_dir=True,

        # 训练参数
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,  # 有效batch size = 32

        # 学习率
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # 优化器
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # 精度
        bf16=True,

        # 日志和保存
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,

        # 其他
        dataloader_num_workers=4,
        remove_unused_columns=False,

        # DeepSpeed（多卡时启用）
        # deepspeed="configs/ds_z2_config.json",
    )

    # ============ 5. 训练 ============
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    print(f"CPT完成，模型保存至: {training_args.output_dir}")


if __name__ == "__main__":
    train_cpt()
```

### 3.4 CPT效果验证

```python
"""
CPT效果验证：测试领域知识习得程度
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CPTEvaluator:
    def __init__(self, original_model_path: str, cpt_model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(cpt_model_path)

        # 加载原始模型
        self.original_model = AutoModelForCausalLM.from_pretrained(
            original_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # 加载CPT后的模型
        self.cpt_model = AutoModelForCausalLM.from_pretrained(
            cpt_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def test_domain_knowledge(self, test_prompts: list):
        """测试领域知识习得"""
        print("=" * 60)
        print("领域知识测试")
        print("=" * 60)

        for prompt in test_prompts:
            print(f"\n【测试问题】{prompt}\n")

            # 原始模型回答
            original_response = self._generate(self.original_model, prompt)
            print(f"【原始模型】{original_response}\n")

            # CPT模型回答
            cpt_response = self._generate(self.cpt_model, prompt)
            print(f"【CPT模型】{cpt_response}\n")

            print("-" * 40)

    def test_general_ability(self, test_prompts: list):
        """测试通用能力保持"""
        print("=" * 60)
        print("通用能力测试（检查是否遗忘）")
        print("=" * 60)

        for prompt in test_prompts:
            print(f"\n【测试问题】{prompt}\n")

            original_response = self._generate(self.original_model, prompt)
            print(f"【原始模型】{original_response}\n")

            cpt_response = self._generate(self.cpt_model, prompt)
            print(f"【CPT模型】{cpt_response}\n")

            print("-" * 40)

    def compute_perplexity(self, texts: list, model_type: str = "cpt"):
        """计算困惑度"""
        model = self.cpt_model if model_type == "cpt" else self.original_model

        total_loss = 0
        total_tokens = 0

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))

        return perplexity.item()

    def _generate(self, model, prompt: str, max_new_tokens: int = 200):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)


# 使用示例
if __name__ == "__main__":
    evaluator = CPTEvaluator(
        original_model_path="Qwen/Qwen2.5-7B",
        cpt_model_path="./output/sales_cpt"
    )

    # 领域知识测试
    domain_prompts = [
        "我们公司的产品相比竞品有哪些优势？",
        "客户说预算有限，应该如何应对？",
        "如何识别客户的真实需求？",
        "销售漏斗的各个阶段应该关注什么？"
    ]

    # 通用能力测试
    general_prompts = [
        "请解释什么是机器学习？",
        "1+1等于几？",
        "写一首关于春天的诗",
        "Python中如何读取文件？"
    ]

    evaluator.test_domain_knowledge(domain_prompts)
    evaluator.test_general_ability(general_prompts)

    # 计算困惑度
    domain_texts = ["销售技巧...", "产品优势..."]
    general_texts = ["今天天气很好...", "机器学习是..."]

    print("\n困惑度对比：")
    print(f"领域文本 - 原始模型: {evaluator.compute_perplexity(domain_texts, 'original'):.2f}")
    print(f"领域文本 - CPT模型: {evaluator.compute_perplexity(domain_texts, 'cpt'):.2f}")
    print(f"通用文本 - 原始模型: {evaluator.compute_perplexity(general_texts, 'original'):.2f}")
    print(f"通用文本 - CPT模型: {evaluator.compute_perplexity(general_texts, 'cpt'):.2f}")
```

---

## 四、CPT进阶技巧

### 4.1 MIP策略（Multi-Task Instruction PreTraining）

**核心思想**：在CPT过程中混入少量SFT数据，让模型在学习知识的同时保持指令遵循能力

```python
# MIP数据混合策略
data_mixture = {
    "domain_text": 0.80,      # 80% 领域纯文本
    "instruction_data": 0.10,  # 10% 指令数据（可以是通用的）
    "replay_data": 0.10        # 10% 通用文本
}

# 好处：
# 1. CPT后可以直接进行对话测试，无需等待SFT
# 2. 减少后续SFT的工作量
# 3. 有助于缓解遗忘
```

### 4.2 渐进式学习率

```python
"""
渐进式学习率策略：
1. 开始时用较小学习率，让模型"热身"
2. 中期提高学习率加速学习
3. 后期降低学习率精细调整
"""
from transformers import get_scheduler

def create_progressive_scheduler(optimizer, num_training_steps):
    """创建渐进式学习率调度器"""
    warmup_steps = int(0.1 * num_training_steps)
    peak_steps = int(0.3 * num_training_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup阶段：线性增加
            return current_step / warmup_steps
        elif current_step < peak_steps:
            # Peak阶段：保持最高
            return 1.0
        else:
            # Decay阶段：余弦衰减
            progress = (current_step - peak_steps) / (num_training_steps - peak_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 4.3 层级冻结策略

```python
"""
层级冻结：只训练部分层，减少遗忘风险
适用于轻量级领域适配
"""
def freeze_layers(model, freeze_ratio: float = 0.5):
    """
    冻结模型的部分层
    Args:
        model: 模型
        freeze_ratio: 冻结的层数比例（从底层开始）
    """
    # 获取所有transformer层
    layers = model.model.layers
    num_layers = len(layers)

    # 计算要冻结的层数
    num_freeze = int(num_layers * freeze_ratio)

    # 冻结底层
    for i, layer in enumerate(layers):
        if i < num_freeze:
            for param in layer.parameters():
                param.requires_grad = False
            print(f"Layer {i}: Frozen")
        else:
            print(f"Layer {i}: Trainable")

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

# 使用示例
# freeze_layers(model, freeze_ratio=0.5)  # 冻结底部50%的层
```

### 4.4 合成数据增强CPT

```python
"""
Synthetic Continued Pretraining
使用LLM生成领域知识的多种表述，增强学习效果
参考ICLR 2025 EntiGraph方法
"""
class SyntheticCPTGenerator:
    def __init__(self):
        self.client = OpenAI()

    def generate_knowledge_variants(self, knowledge: str, num_variants: int = 5) -> list:
        """
        为一条知识生成多种表述
        """
        prompt = f"""请将以下知识内容改写成{num_variants}种不同的表述方式，
保持核心信息不变，但使用不同的：
1. 句式结构
2. 表达角度
3. 详细程度
4. 语言风格

原始知识：
{knowledge}

请返回{num_variants}种改写版本，每个版本用序号标记："""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )

        # 解析返回的多个版本
        variants = response.choices[0].message.content.split('\n\n')
        return [v.strip() for v in variants if v.strip()]

    def create_qa_from_knowledge(self, knowledge: str) -> list:
        """
        将知识转换为问答形式，帮助模型学习
        """
        prompt = f"""基于以下知识内容，生成3-5个问答对，
这些问答应该覆盖知识的不同方面：

知识内容：
{knowledge}

请以JSON格式返回：
[{{"question": "问题1", "answer": "答案1"}}, ...]"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )

        try:
            qa_pairs = json.loads(response.choices[0].message.content)
            # 转换为训练文本
            texts = []
            for qa in qa_pairs:
                texts.append(f"问题：{qa['question']}\n答案：{qa['answer']}")
            return texts
        except:
            return [knowledge]

    def augment_cpt_data(self, original_texts: list) -> list:
        """增强CPT数据"""
        augmented = []
        for text in original_texts:
            # 原始文本
            augmented.append(text)

            # 变体
            variants = self.generate_knowledge_variants(text)
            augmented.extend(variants)

            # 问答形式
            qa_texts = self.create_qa_from_knowledge(text)
            augmented.extend(qa_texts)

        return augmented
```

---

## 五、CPT常见问题与解决方案

### Q1: CPT后模型效果变差了？

**可能原因**：
1. 学习率过高导致遗忘
2. 数据质量差
3. 数据量不足
4. 没有混合通用数据

**解决方案**：
```python
# 1. 降低学习率
learning_rate = 1e-5  # 从2e-5降到1e-5

# 2. 增加通用数据比例
data_mixture = {"domain": 0.6, "general": 0.4}

# 3. 使用权重平均
final_model = weight_averaging(original, cpt_model, alpha=0.4)

# 4. 检查数据质量，清理低质量样本
```

### Q2: 训练显存不足？

**解决方案**：
```python
# 1. 使用gradient checkpointing
model.gradient_checkpointing_enable()

# 2. 减小batch size，增加gradient accumulation
per_device_train_batch_size = 1
gradient_accumulation_steps = 32

# 3. 使用DeepSpeed ZeRO
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,  # 或stage 3
        "offload_optimizer": {"device": "cpu"}
    }
}

# 4. 使用QLoRA（但CPT通常推荐全参）
# 如果资源实在不足，可以考虑LoRA，但效果可能打折扣
```

### Q3: 如何判断CPT是否收敛？

```python
# 监控指标
monitoring_metrics = {
    "training_loss": "应稳定下降",
    "domain_perplexity": "领域文本困惑度应下降",
    "general_perplexity": "通用文本困惑度不应显著上升",
    "validation_loss": "不应出现过拟合（验证集loss上升）"
}

# 早停条件
early_stopping_conditions = [
    "验证集loss连续3次不下降",
    "通用能力测试分数下降超过10%",
    "训练loss不再变化"
]
```

### Q4: 是否应该使用Base模型还是Instruct模型做CPT？

**推荐使用Base模型**：
- Base模型没有对话格式的bias，更适合学习纯知识
- CPT后再做SFT，可以保证格式一致性
- Instruct模型做CPT可能破坏已有的对话能力

---

## 六、CPT vs RAG：如何选择？

| 维度 | CPT | RAG |
|------|-----|-----|
| 知识更新 | 需要重新训练 | 更新文档即可 |
| 推理成本 | 低（无需检索）| 高（需要检索+拼接）|
| 准确性 | 可能产生幻觉 | 可引用原文，更准确 |
| 适用场景 | 核心领域知识 | 实时信息、长尾知识 |
| 实施成本 | 高（需要GPU训练）| 低（只需向量数据库）|

**实践建议**：
- **核心稳定知识**（如销售话术、产品核心卖点）→ **CPT**
- **频繁更新知识**（如最新价格、库存、活动）→ **RAG**
- **最佳实践**：CPT + RAG 结合使用

---

## 参考资源

### 论文
- [Investigating Continual Pretraining in LLMs](https://arxiv.org/abs/2402.17400)
- [Synthetic Continued Pretraining (ICLR 2025)](https://arxiv.org/abs/2409.07431)
- [LLaMA-Pro: Progressive LLaMA](https://arxiv.org/abs/2401.02415)

### 实践指南
- [AMD ROCm CPT Playbook](https://rocm.blogs.amd.com/artificial-intelligence/multilingual-continued-pretraining/README.html)
- [Databricks CPT Guide](https://www.databricks.com/blog/characterizing-datasets-and-building-better-models-continued-pre-training)

---

> **下一章**：[03-监督微调SFT.md](./03-监督微调SFT.md) - 学习如何训练指令遵循能力
