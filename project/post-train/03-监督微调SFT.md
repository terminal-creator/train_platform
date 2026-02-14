# 第三章：监督微调（Supervised Fine-Tuning, SFT）

> **核心目标**：让模型学会遵循指令，掌握对话格式，输出高质量回复
>
> **本章目标**：深入理解SFT的原理、数据准备、训练技巧和效果优化
>
> **销售LLM示例**：训练模型成为专业销售顾问

---

## 一、SFT概述

### 1.1 什么是监督微调？

SFT是使用**人工标注的指令-回复对**对预训练模型进行微调，让模型学会：
1. **理解指令**：明白用户想要什么
2. **遵循格式**：按照期望的格式输出
3. **生成高质量回复**：内容准确、表达清晰

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SFT训练原理                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   输入 (Instruction + Input)              输出 (Response)               │
│   ━━━━━━━━━━━━━━━━━━━━━━━━━               ━━━━━━━━━━━━━━━━              │
│                                                                         │
│   "作为销售顾问，请回答：         ────▶    "感谢您的咨询！              │
│    客户说'太贵了'怎么办？"                 我理解您对价格的考虑...       │
│                                            1. 首先让我帮您分析...       │
│                                            2. 实际上..."                │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                        训练目标                                  │  │
│   │  Loss = -log P(response | instruction, input)                   │  │
│   │  只计算response部分的loss，instruction部分被mask掉               │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 SFT的关键作用

| 能力维度 | SFT前（Base Model） | SFT后（Instruct Model） |
|---------|---------------------|-------------------------|
| 指令理解 | ❌ 无法理解指令意图 | ✅ 准确理解用户需求 |
| 对话格式 | ❌ 只会续写文本 | ✅ 按对话格式回复 |
| 角色扮演 | ❌ 无角色概念 | ✅ 可扮演特定角色 |
| 安全性 | ❌ 可能输出有害内容 | ✅ 初步的安全意识 |
| 格式输出 | ❌ 无法按要求格式化 | ✅ 支持JSON/Markdown等 |

### 1.3 SFT数据的核心原则

> **"样本的精髓在于质量而非数量，少量但精良的样本往往胜过大批中低品质的样本。"**
> — Meta LIMA论文

**黄金法则**：
- **1万条高质量数据 > 10万条低质量数据**
- **数据质量决定模型上限**
- **多样性比数量更重要**

---

## 二、SFT数据格式详解

### 2.1 Chat Template（对话模板）

Chat Template是将对话转换为模型可理解的文本序列的规则。不同模型有不同的模板。

#### Qwen2.5的ChatML格式

```
<|im_start|>system
你是一个专业的销售顾问。<|im_end|>
<|im_start|>user
你们的产品多少钱？<|im_end|>
<|im_start|>assistant
感谢您的咨询！为了给您推荐最合适的方案...<|im_end|>
```

#### Llama 3的格式

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

你是一个专业的销售顾问。<|eot_id|><|start_header_id|>user<|end_header_id|>

你们的产品多少钱？<|eot_id|><|start_header_id|>assistant<|end_header_id|>

感谢您的咨询！...<|eot_id|>
```

#### 通用格式转换代码

```python
"""
Chat Template处理工具
支持主流模型格式
"""
from transformers import AutoTokenizer
from typing import List, Dict

class ChatTemplateProcessor:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def apply_template(self, messages: List[Dict], add_generation_prompt: bool = False) -> str:
        """
        将消息列表转换为模型特定的格式

        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]
            add_generation_prompt: 是否添加生成提示（推理时为True）
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

    def prepare_training_sample(self, messages: List[Dict]) -> Dict:
        """
        准备训练样本，正确设置labels（mask掉instruction部分）
        """
        # 完整对话文本
        full_text = self.apply_template(messages, add_generation_prompt=False)

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=2048,
            return_tensors=None
        )

        # 创建labels（复制input_ids）
        labels = encodings["input_ids"].copy()

        # 找到assistant回复的起始位置，mask掉之前的token
        # 这里需要根据具体模板来实现
        labels = self._mask_instruction_tokens(labels, messages)

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        }

    def _mask_instruction_tokens(self, labels: List[int], messages: List[Dict]) -> List[int]:
        """
        将非assistant部分的token设为-100（忽略loss计算）

        ⚠️ 警告：下面的简化实现仅用于演示原理！
        实际生产中必须使用框架提供的正确实现，否则会导致训练失败！

        推荐方案（按优先级）：
        1. 使用 TRL 的 DataCollatorForCompletionOnlyLM
        2. 使用 LLaMA-Factory 内置的 mask 处理
        3. 基于模板特殊token精确定位（见下方正确实现）
        """
        # ❌ 错误实现（仅展示原理，不可用于训练）：
        # 直接按文本内容长度估算会因为special token、role header等导致错位
        #
        # ✅ 正确实现方案：基于模板的response标记定位
        # 核心思路：找到assistant回复开始的特殊token位置

        # 以Qwen/ChatML格式为例
        response_marker = "<|im_start|>assistant\n"
        response_end_marker = "<|im_end|>"

        # 获取marker的token ids
        marker_ids = self.tokenizer.encode(response_marker, add_special_tokens=False)
        end_marker_ids = self.tokenizer.encode(response_end_marker, add_special_tokens=False)

        # 在labels中找到所有assistant回复区间
        input_ids = labels.copy()
        i = 0
        in_response = False

        while i < len(input_ids):
            # 检查是否匹配response开始marker
            if not in_response and self._match_sequence(input_ids, i, marker_ids):
                i += len(marker_ids)  # 跳过marker本身
                in_response = True
                continue

            # 检查是否匹配response结束marker
            if in_response and self._match_sequence(input_ids, i, end_marker_ids):
                in_response = False
                i += len(end_marker_ids)
                continue

            # 非response区域，mask掉
            if not in_response:
                labels[i] = -100

            i += 1

        return labels

    def _match_sequence(self, ids: List[int], start: int, pattern: List[int]) -> bool:
        """检查ids从start位置是否匹配pattern"""
        if start + len(pattern) > len(ids):
            return False
        return ids[start:start+len(pattern)] == pattern


# 销售LLM示例
def create_sales_training_sample():
    """创建销售场景的训练样本"""
    processor = ChatTemplateProcessor("Qwen/Qwen2.5-7B-Instruct")

    messages = [
        {
            "role": "system",
            "content": "你是一个专业的销售顾问，擅长理解客户需求，提供专业建议。保持友好、专业的态度。"
        },
        {
            "role": "user",
            "content": "你们这个产品多少钱？"
        },
        {
            "role": "assistant",
            "content": """感谢您的咨询！为了给您推荐最适合的方案，我想先了解几个问题：

1. **团队规模**：您的团队大概有多少人会使用？
2. **核心需求**：您主要想解决什么问题？
3. **预算范围**：您的预算大概在什么区间？

了解这些信息后，我可以为您定制最优的解决方案。您方便告诉我吗？"""
        }
    ]

    # 生成ChatML格式
    formatted = processor.apply_template(messages)
    print("=" * 50)
    print("ChatML格式：")
    print(formatted)

    # 准备训练样本
    sample = processor.prepare_training_sample(messages)
    print("\n训练样本token数量：", len(sample["input_ids"]))

    return sample
```

### 2.2 Loss Masking详解

> ⚠️ **关键警告**：Loss Masking是SFT中最容易出错的环节！
> 错误的masking会导致模型学习错误的内容，训练彻底失效。
> **必须在训练前验证masking是否正确**。

**为什么要Mask？**
- 只训练模型生成response的能力
- instruction部分不参与loss计算
- 避免模型"学会"输出instruction

#### 🔴 Masking验证检查清单（必做！）

```python
def validate_loss_masking(tokenizer, sample):
    """
    验证Loss Masking是否正确 —— 训练前必须运行！

    检查标准：
    1. labels中非-100的token必须且只能是assistant回复内容
    2. 所有system/user内容的token必须被mask（=-100）
    3. 特殊token（role header、分隔符）必须被mask
    """
    input_ids = sample["input_ids"]
    labels = sample["labels"]

    print("=" * 60)
    print("Loss Masking 验证报告")
    print("=" * 60)

    # 解码并对齐显示
    for i, (input_id, label) in enumerate(zip(input_ids, labels)):
        token_str = tokenizer.decode([input_id])
        is_trained = label != -100
        marker = "✓ TRAIN" if is_trained else "  skip"
        print(f"{i:4d} | {marker} | {repr(token_str)}")

    # 统计
    train_count = sum(1 for l in labels if l != -100)
    skip_count = sum(1 for l in labels if l == -100)
    print(f"\n训练token数: {train_count}, 跳过token数: {skip_count}")
    print(f"训练比例: {train_count / len(labels) * 100:.1f}%")

    # 人工确认提示
    print("\n⚠️ 请人工检查上述输出：")
    print("   - 标记为TRAIN的token是否都是assistant的回复内容？")
    print("   - system/user的内容是否都被skip？")
    print("   - 特殊token（如<|im_start|>）是否都被skip？")

# 使用示例
# validate_loss_masking(tokenizer, train_dataset[0])
```

#### 推荐的正确实现方式

| 方式 | 推荐度 | 说明 |
|------|--------|------|
| TRL `DataCollatorForCompletionOnlyLM` | ⭐⭐⭐⭐⭐ | 最可靠，自动处理各种模板 |
| LLaMA-Factory 内置处理 | ⭐⭐⭐⭐⭐ | 框架自动处理，无需手动 |
| 基于模板token精确定位 | ⭐⭐⭐ | 需要深入理解模板结构 |
| 按文本长度近似估算 | ❌ | **绝对禁止！会导致错位** |

```python
"""
Loss Masking的三种策略对比
"""
import torch

def demonstrate_loss_masking():
    """
    演示不同的loss masking策略
    """
    # 示例token序列（简化）
    # [SYSTEM] [USER_TURN_1] [ASSISTANT_TURN_1] [USER_TURN_2] [ASSISTANT_TURN_2]
    tokens = ["<sys>", "你好", "<usr>", "产品多少钱", "<ast>", "感谢咨询", "方案如下", "<usr>", "有优惠吗", "<ast>", "当然有"]

    print("=" * 60)
    print("三种Loss Masking策略对比")
    print("=" * 60)

    # 策略1: 完全不Mask（不推荐）
    print("\n策略1: 完全不Mask")
    print("计算Loss的Token: 全部")
    print("问题: 模型会学习输出instruction，不符合预期")

    # 策略2: Mask所有instruction（标准做法）
    print("\n策略2: Mask所有instruction（推荐）")
    labels_2 = [-100, -100, -100, -100, 0, 0, 0, -100, -100, 0, 0]  # -100表示忽略
    print(f"Labels: {labels_2}")
    print("计算Loss的Token: 只有assistant回复")
    print("优点: 模型只学习如何回复")

    # 策略3: 只Mask模板token（研究中）
    print("\n策略3: 只Mask模板token（实验性）")
    labels_3 = [-100, 0, -100, 0, -100, 0, 0, -100, 0, -100, 0]
    print(f"Labels: {labels_3}")
    print("计算Loss的Token: instruction内容 + assistant回复")
    print("研究发现: 某些情况下效果更好，但需要实验验证")


# TRL库中的实现
def trl_style_masking(tokenizer, messages, response_template):
    """
    TRL SFTTrainer风格的masking实现
    """
    from trl import DataCollatorForCompletionOnlyLM

    # 定义response起始标记
    # 对于Qwen: "<|im_start|>assistant\n"
    # 对于Llama3: "<|start_header_id|>assistant<|end_header_id|>\n\n"

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    # collator会自动处理labels，将非response部分设为-100
    return collator
```

### 2.3 多轮对话数据处理

```python
"""
多轮对话的完整处理流程
"""
from typing import List, Dict
import json

class MultiTurnDataProcessor:
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def process_conversation(self, conversation: Dict) -> List[Dict]:
        """
        处理一个多轮对话，生成多个训练样本

        策略选择：
        1. 整体作为一个样本（推荐）
        2. 每轮拆分为独立样本
        3. 滑动窗口方式
        """
        messages = conversation.get("messages", [])

        # 策略1: 整体处理（推荐）
        return [self._process_full_conversation(messages)]

    def _process_full_conversation(self, messages: List[Dict]) -> Dict:
        """
        将完整多轮对话作为一个训练样本
        """
        # 应用chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        encodings = self.tokenizer(
            formatted,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )

        # 创建labels
        labels = self._create_multi_turn_labels(encodings["input_ids"], messages)

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels
        }

    def _create_multi_turn_labels(self, input_ids: List[int], messages: List[Dict]) -> List[int]:
        """
        为多轮对话创建labels
        所有user/system部分mask，所有assistant部分保留
        """
        labels = [-100] * len(input_ids)

        # 找到每个assistant回复的位置
        # 这需要根据具体的chat template来实现
        # 以下是通用逻辑框架

        current_pos = 0
        for msg in messages:
            # 获取这条消息的token
            msg_text = msg["content"]
            msg_tokens = self.tokenizer.encode(msg_text, add_special_tokens=False)

            if msg["role"] == "assistant":
                # 找到这些token在input_ids中的位置
                # 设置labels为实际token值
                start_pos = self._find_subsequence(input_ids, msg_tokens, current_pos)
                if start_pos != -1:
                    for i, token in enumerate(msg_tokens):
                        if start_pos + i < len(labels):
                            labels[start_pos + i] = input_ids[start_pos + i]
                    current_pos = start_pos + len(msg_tokens)

        return labels

    def _find_subsequence(self, sequence: List[int], subsequence: List[int], start: int = 0) -> int:
        """找到子序列在序列中的起始位置"""
        for i in range(start, len(sequence) - len(subsequence) + 1):
            if sequence[i:i+len(subsequence)] == subsequence:
                return i
        return -1


# 销售多轮对话示例
sales_conversation = {
    "conversation_id": "sales_001",
    "messages": [
        {"role": "system", "content": "你是专业销售顾问，帮助客户选择合适的产品方案。"},
        {"role": "user", "content": "你们的CRM系统多少钱？"},
        {"role": "assistant", "content": "感谢咨询！价格取决于您的具体需求。请问：\n1. 团队规模多大？\n2. 主要用于哪些场景？"},
        {"role": "user", "content": "大概30人，主要做客户管理和销售跟进"},
        {"role": "assistant", "content": "了解！30人团队做客户管理，推荐我们的专业版：\n\n**专业版** ¥4,800/月\n- 支持50用户\n- 完整CRM功能\n- 销售漏斗分析\n- 自动化工作流\n\n年付可享8折优惠，合计¥46,080/年。需要我详细介绍某个功能吗？"},
        {"role": "user", "content": "有没有更便宜的方案？"},
        {"role": "assistant", "content": "理解您对成本的考虑。我们还有**基础版** ¥2,400/月：\n\n- 支持30用户\n- 核心CRM功能\n- 基础报表\n\n不过专业版的销售漏斗分析能帮您提升20%转化率，一个月多成交几单就回本了。\n\n您看这样，我申请一下首年7折优惠，专业版实际只要¥40,320，您觉得怎么样？"}
    ],
    "outcome": "won"
}
```

---

## 三、SFT训练实战

### 3.1 使用LLaMA-Factory进行SFT

```yaml
# configs/sales_sft.yaml
# 销售LLM SFT配置

### 模型配置
model_name_or_path: Qwen/Qwen2.5-7B-Instruct  # 或使用CPT后的模型
trust_remote_code: true

### 训练方法
stage: sft
do_train: true
finetuning_type: lora  # 使用LoRA进行高效微调

### LoRA配置
lora_target: all  # 或指定: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lora_rank: 64
lora_alpha: 128  # 通常设为rank的2倍
lora_dropout: 0.05

### 数据配置
dataset: sales_sft
template: qwen  # 使用Qwen的chat template
cutoff_len: 2048
max_samples: 50000  # 最多使用多少样本
preprocessing_num_workers: 16

### 训练超参数
per_device_train_batch_size: 4
gradient_accumulation_steps: 4  # 有效batch size = 16
num_train_epochs: 3
learning_rate: 5e-5
lr_scheduler_type: cosine
warmup_ratio: 0.1

### 优化器
optim: adamw_torch
weight_decay: 0.01
max_grad_norm: 1.0

### 精度
bf16: true

### 保存配置
output_dir: ./output/sales_sft
logging_steps: 10
save_steps: 200
save_total_limit: 5

### 评估配置（可选）
val_size: 0.05
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 200
```

```json
// dataset_info.json - 定义SFT数据集
{
  "sales_sft": {
    "file_name": "data/sft/sales_sharegpt.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  }
}
```

```bash
# 启动训练
# 单卡
llamafactory-cli train configs/sales_sft.yaml

# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train configs/sales_sft.yaml

# 带评估的训练
llamafactory-cli train configs/sales_sft.yaml \
    --val_size 0.05 \
    --eval_strategy steps \
    --eval_steps 200
```

### 3.2 使用TRL库进行SFT

```python
"""
使用HuggingFace TRL进行SFT
更灵活的控制和自定义
"""
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def train_sales_sft():
    # ============ 1. 配置 ============
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    output_dir = "./output/sales_sft_trl"

    # ============ 2. 加载模型 ============
    # 量化配置（QLoRA）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ============ 3. LoRA配置 ============
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,                    # LoRA rank
        lora_alpha=128,          # LoRA alpha
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[         # 目标模块
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ============ 4. 加载数据 ============
    dataset = load_dataset("json", data_files="./data/sft/sales_sharegpt.json")

    def formatting_func(example):
        """将数据格式化为对话"""
        messages = example["conversations"]
        # 转换格式
        formatted_messages = []
        for msg in messages:
            role = "user" if msg["from"] == "human" else "assistant"
            formatted_messages.append({"role": role, "content": msg["value"]})

        # 添加system（如果有）
        if "system" in example and example["system"]:
            formatted_messages.insert(0, {"role": "system", "content": example["system"]})

        return tokenizer.apply_chat_template(formatted_messages, tokenize=False)

    # ============ 5. 数据整理器 ============
    # 设置response template用于loss masking
    # Qwen格式: <|im_start|>assistant\n
    response_template = "<|im_start|>assistant\n"

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    # ============ 6. 训练配置 ============
    training_args = SFTConfig(
        output_dir=output_dir,

        # 训练参数
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        # 学习率
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # 优化
        optim="paged_adamw_8bit",  # 8-bit优化器节省显存
        weight_decay=0.01,
        max_grad_norm=1.0,

        # 精度
        bf16=True,

        # 序列长度
        max_seq_length=2048,
        packing=False,  # 是否打包短序列

        # 日志和保存
        logging_steps=10,
        save_steps=200,
        save_total_limit=5,

        # 评估
        eval_strategy="steps",
        eval_steps=200,
    )

    # ============ 7. 训练 ============
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        formatting_func=formatting_func,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # ============ 8. 保存 ============
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"训练完成，模型保存至: {output_dir}")


if __name__ == "__main__":
    train_sales_sft()
```

### 3.3 使用Unsloth进行高效SFT

```python
"""
使用Unsloth进行2-5倍加速的SFT训练
显存占用更低，训练速度更快
"""
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

def train_with_unsloth():
    # ============ 1. 加载模型（Unsloth优化版本）============
    max_seq_length = 2048
    dtype = None  # 自动检测
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct",  # Unsloth优化版本
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # ============ 2. 添加LoRA适配器 ============
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=128,
        lora_dropout=0,  # Unsloth建议设为0
        bias="none",
        use_gradient_checkpointing="unsloth",  # 使用Unsloth优化的checkpointing
        random_state=42,
    )

    # ============ 3. 准备数据 ============
    dataset = load_dataset("json", data_files="./data/sft/sales_sharegpt.json")

    def formatting_prompts_func(examples):
        """格式化函数"""
        conversations = examples["conversations"]
        texts = []

        for conv in conversations:
            messages = []
            for msg in conv:
                role = "user" if msg["from"] == "human" else "assistant"
                messages.append({"role": role, "content": msg["value"]})

            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # ============ 4. 训练 ============
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=8,
        packing=False,
        args=TrainingArguments(
            output_dir="./output/sales_sft_unsloth",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_steps=200,
            optim="adamw_8bit",
            weight_decay=0.01,
            seed=42,
        ),
    )

    # GPU状态
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name}, 显存: {max_memory}GB, 已用: {start_gpu_memory}GB")

    trainer.train()

    # ============ 5. 保存 ============
    # 保存LoRA适配器
    model.save_pretrained("./output/sales_sft_unsloth/lora")

    # 合并并保存完整模型（可选）
    # model.save_pretrained_merged("./output/sales_sft_unsloth/merged", tokenizer)

    # 保存为GGUF格式（可选，用于llama.cpp）
    # model.save_pretrained_gguf("./output/sales_sft_unsloth/gguf", tokenizer, quantization_method="q4_k_m")


if __name__ == "__main__":
    import torch
    train_with_unsloth()
```

---

## 四、SFT超参数调优

### 4.1 学习率选择

```python
"""
学习率选择指南
"""
learning_rate_guide = {
    "全参微调 (Full Fine-tuning)": {
        "推荐值": "1e-5 ~ 2e-5",
        "说明": "参数量大，需要小学习率防止震荡",
        "示例": 2e-5
    },
    "LoRA微调": {
        "推荐值": "1e-4 ~ 5e-5",
        "说明": "只更新少量参数，可以用稍大学习率",
        "示例": 5e-5
    },
    "QLoRA微调": {
        "推荐值": "1e-4 ~ 2e-4",
        "说明": "4-bit量化后可以用更大学习率",
        "示例": 1e-4
    }
}

# 学习率与batch size的关系
# 经验公式：lr_new = lr_base * sqrt(batch_size_new / batch_size_base)
def adjust_learning_rate(base_lr, base_batch_size, new_batch_size):
    import math
    return base_lr * math.sqrt(new_batch_size / base_batch_size)

# 示例：base_lr=5e-5, base_batch=8, new_batch=32
# 新学习率 = 5e-5 * sqrt(32/8) = 5e-5 * 2 = 1e-4
```

### 4.2 LoRA参数选择

```python
"""
LoRA超参数选择指南
"""
lora_config_guide = {
    "rank (r)": {
        "简单任务（格式调整等）": "r=8~16",
        "中等任务（领域微调）": "r=32~64",
        "复杂任务（大规模微调）": "r=128~256",
        "注意": "QLoRA论文发现r=8和r=256效果差异不大（如果全层都用LoRA）"
    },
    "alpha (α)": {
        "保守设置": "α = r（scale factor = 1）",
        "推荐设置": "α = 2r（scale factor = 2）",
        "激进设置": "α = 4r",
        "说明": "实际缩放因子 = α/r"
    },
    "target_modules": {
        "最小配置": ["q_proj", "v_proj"],
        "推荐配置": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "完整配置": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "说明": "QLoRA论文建议全层都用LoRA效果最好"
    },
    "dropout": {
        "一般设置": 0.05,
        "数据量大": 0.0,  # Unsloth建议
        "数据量小": 0.1
    }
}

# 不同场景的LoRA配置示例
def get_lora_config(scenario: str):
    from peft import LoraConfig

    configs = {
        "快速验证": LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        ),
        "标准生产": LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        ),
        "高质量": LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM"
        )
    }
    return configs.get(scenario)
```

### 4.3 训练轮数选择

```python
"""
训练轮数选择
"""
epoch_guide = {
    "数据量 < 1k": {
        "推荐epochs": "5-10",
        "风险": "容易过拟合",
        "建议": "使用更大dropout，早停"
    },
    "数据量 1k-10k": {
        "推荐epochs": "2-5",
        "说明": "标准配置"
    },
    "数据量 10k-100k": {
        "推荐epochs": "1-3",
        "说明": "数据量充足，不需要太多轮"
    },
    "数据量 > 100k": {
        "推荐epochs": "1-2",
        "说明": "可能1轮就够了"
    }
}

# 早停策略
def should_early_stop(eval_losses: list, patience: int = 3):
    """
    检查是否应该早停
    连续patience次评估loss不下降则停止
    """
    if len(eval_losses) < patience + 1:
        return False

    recent = eval_losses[-patience:]
    return all(recent[i] >= recent[i-1] for i in range(1, len(recent)))
```

---

## 五、SFT常见问题与解决方案

### 5.1 过拟合问题

```python
"""
过拟合的诊断与解决
"""
# 诊断信号
overfitting_signals = [
    "训练loss持续下降，但验证loss开始上升",
    "模型开始逐字复述训练数据",
    "对训练数据外的问题回答质量下降",
    "生成内容多样性降低"
]

# 解决方案
solutions = {
    "减少训练轮数": "从3轮减到1-2轮",
    "增大dropout": "LoRA dropout从0.05增到0.1",
    "减小学习率": "学习率减半",
    "增加数据多样性": "混入更多通用数据",
    "早停": "监控验证loss，连续3次不下降就停止",
    "权重衰减": "增大weight_decay",
}
```

### 5.2 灾难性遗忘

```python
"""
灾难性遗忘的预防
"""
# 预防措施
prevention_measures = {
    "数据混合": {
        "方法": "领域数据70% + 通用数据30%",
        "说明": "保持通用能力的同时学习领域知识"
    },
    "使用LoRA": {
        "方法": "只更新少量参数",
        "说明": "减少对原始权重的干扰"
    },
    "小学习率": {
        "方法": "使用保守的学习率",
        "说明": "避免剧烈更新破坏原有知识"
    },
    "渐进式训练": {
        "方法": "先用通用数据热身，再加入领域数据",
        "说明": "让模型平滑过渡"
    }
}

# 检测遗忘的方法
def check_forgetting(model, tokenizer, general_test_cases: list):
    """
    在通用测试集上评估模型，检查是否遗忘
    """
    results = []
    for case in general_test_cases:
        prompt = case["prompt"]
        expected_keywords = case["expected_keywords"]

        # 生成回复
        response = generate_response(model, tokenizer, prompt)

        # 检查关键词是否出现
        score = sum(1 for kw in expected_keywords if kw in response) / len(expected_keywords)
        results.append({"prompt": prompt, "score": score, "response": response})

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"通用能力保持率: {avg_score:.2%}")

    if avg_score < 0.7:
        print("警告：检测到明显的能力遗忘！")

    return results
```

### 5.3 训练不稳定

```python
"""
训练不稳定的解决方案
"""
stability_tips = {
    "Loss震荡": {
        "症状": "Loss上下剧烈波动",
        "原因": "学习率过大或batch size过小",
        "解决": "减小学习率，增大batch size或gradient accumulation"
    },
    "Loss不下降": {
        "症状": "Loss基本不变",
        "原因": "学习率过小或数据问题",
        "解决": "增大学习率，检查数据格式是否正确"
    },
    "Loss突然爆炸": {
        "症状": "Loss突然变成NaN或极大值",
        "原因": "数值不稳定",
        "解决": "使用BF16代替FP16，减小学习率，增加梯度裁剪"
    },
    "Loss下降后又上升": {
        "症状": "先下降后持续上升",
        "原因": "过拟合或学习率调度问题",
        "解决": "早停，使用cosine衰减"
    }
}

# 稳定训练的配置模板
stable_training_config = {
    "learning_rate": 2e-5,  # 保守学习率
    "warmup_ratio": 0.1,    # 10%预热
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,   # 梯度裁剪
    "bf16": True,           # 使用BF16
    "gradient_accumulation_steps": 8,  # 足够大的有效batch
    "weight_decay": 0.01,   # 轻微权重衰减
}
```

---

## 六、SFT效果评估

### 6.1 快速评估方法

```python
"""
SFT效果快速评估
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SFTEvaluator:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()

    def vibe_check(self, test_cases: list):
        """
        Vibe Check：人工快速检查模型输出质量
        """
        print("=" * 60)
        print("SFT效果 Vibe Check")
        print("=" * 60)

        for i, case in enumerate(test_cases, 1):
            prompt = case["prompt"]
            print(f"\n【测试 {i}】{case.get('description', '')}")
            print(f"输入: {prompt}")

            response = self._generate(prompt)
            print(f"输出: {response}")

            # 检查点
            checks = case.get("checks", [])
            for check in checks:
                passed = check["condition"](response)
                status = "✅" if passed else "❌"
                print(f"  {status} {check['name']}")

            print("-" * 40)

    def evaluate_instruction_following(self, test_cases: list) -> dict:
        """评估指令遵循能力"""
        results = {
            "total": len(test_cases),
            "passed": 0,
            "failed": 0,
            "details": []
        }

        for case in test_cases:
            response = self._generate(case["prompt"])
            passed = case["validator"](response)

            results["passed" if passed else "failed"] += 1
            results["details"].append({
                "prompt": case["prompt"],
                "response": response,
                "passed": passed
            })

        results["accuracy"] = results["passed"] / results["total"]
        return results

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
        return response.strip()


# 销售LLM测试用例
sales_test_cases = [
    {
        "description": "价格咨询处理",
        "prompt": "你们的产品多少钱？",
        "checks": [
            {"name": "不直接报价而是先了解需求", "condition": lambda r: "了解" in r or "需求" in r or "?" in r},
            {"name": "保持专业友好", "condition": lambda r: "感谢" in r or "您好" in r},
        ]
    },
    {
        "description": "异议处理 - 太贵了",
        "prompt": "太贵了，能便宜点吗？",
        "checks": [
            {"name": "理解客户顾虑", "condition": lambda r: "理解" in r or "明白" in r},
            {"name": "提供价值分析", "condition": lambda r: "价值" in r or "效果" in r or "节省" in r},
        ]
    },
    {
        "description": "竞品对比",
        "prompt": "你们和XX公司比有什么优势？",
        "checks": [
            {"name": "不贬低竞品", "condition": lambda r: "差" not in r and "不好" not in r},
            {"name": "突出自身优势", "condition": lambda r: "优势" in r or "特点" in r},
        ]
    },
    {
        "description": "格式输出能力",
        "prompt": "请用JSON格式列出产品的三个主要特点",
        "checks": [
            {"name": "输出包含JSON格式", "condition": lambda r: "{" in r and "}" in r},
        ]
    }
]

# 使用示例
if __name__ == "__main__":
    evaluator = SFTEvaluator("./output/sales_sft")
    evaluator.vibe_check(sales_test_cases)
```

### 6.2 自动化评估指标

```python
"""
SFT自动化评估指标
"""
import json
from typing import List, Dict

class AutoEvaluator:
    def __init__(self, model_path: str):
        # 初始化评估模型（可以用更强的模型来评估）
        self.judge_model = None  # 可以用GPT-4作为judge

    def evaluate_response_quality(self, prompt: str, response: str) -> Dict:
        """
        多维度评估回复质量
        """
        scores = {
            "relevance": self._score_relevance(prompt, response),
            "completeness": self._score_completeness(response),
            "professionalism": self._score_professionalism(response),
            "safety": self._score_safety(response),
            "format": self._score_format(response)
        }

        scores["overall"] = sum(scores.values()) / len(scores)
        return scores

    def _score_relevance(self, prompt: str, response: str) -> float:
        """评估回复与问题的相关性"""
        # 简单实现：检查关键词重叠
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words)
        return min(1.0, overlap / max(len(prompt_words), 1))

    def _score_completeness(self, response: str) -> float:
        """评估回复的完整性"""
        # 基于长度的简单评估
        if len(response) < 50:
            return 0.3
        elif len(response) < 100:
            return 0.6
        elif len(response) < 500:
            return 1.0
        else:
            return 0.8  # 太长可能扣分

    def _score_professionalism(self, response: str) -> float:
        """评估专业性"""
        # 检查是否包含专业表达
        professional_markers = [
            "首先", "其次", "另外", "总结",
            "建议", "方案", "分析", "了解",
            "？", "1.", "2.", "-"  # 结构化表达
        ]
        score = sum(1 for m in professional_markers if m in response)
        return min(1.0, score / 5)

    def _score_safety(self, response: str) -> float:
        """评估安全性"""
        # 检查是否有不当内容
        unsafe_patterns = [
            "保证赚钱", "100%效果", "绝对", "肯定没问题",
            "竞品很差", "其他都是垃圾"
        ]
        for pattern in unsafe_patterns:
            if pattern in response:
                return 0.0
        return 1.0

    def _score_format(self, response: str) -> float:
        """评估格式规范性"""
        # 检查是否有良好的格式
        format_markers = [
            "\n",     # 换行
            "：",     # 冒号
            "。",     # 句号
            "1.",     # 编号
            "- ",     # 列表
        ]
        score = sum(1 for m in format_markers if m in response)
        return min(1.0, score / 3)


def batch_evaluate(evaluator, model, test_data: List[Dict]) -> Dict:
    """批量评估"""
    all_scores = []

    for item in test_data:
        response = model.generate(item["prompt"])
        scores = evaluator.evaluate_response_quality(item["prompt"], response)
        all_scores.append(scores)

    # 计算平均分
    avg_scores = {}
    for key in all_scores[0].keys():
        avg_scores[key] = sum(s[key] for s in all_scores) / len(all_scores)

    return avg_scores
```

---

## 七、总结：SFT最佳实践清单

```markdown
## SFT训练检查清单

### 数据准备
- [ ] 数据格式正确（Alpaca/ShareGPT）
- [ ] Chat Template与模型匹配
- [ ] 数据质量经过检查
- [ ] 数据配比合理（领域:通用）
- [ ] 数据量适中（1k-50k为宜）

### 训练配置
- [ ] 学习率合适（LoRA: 5e-5, 全参: 2e-5）
- [ ] LoRA参数合理（r=64, α=128）
- [ ] 训练轮数适中（1-3轮）
- [ ] 使用BF16精度
- [ ] 梯度裁剪开启

### 训练监控
- [ ] 监控训练Loss下降趋势
- [ ] 监控验证Loss（检查过拟合）
- [ ] 定期Vibe Check生成质量
- [ ] 监控GPU显存使用

### 效果验证
- [ ] 指令遵循测试通过
- [ ] 格式输出正确
- [ ] 领域知识准确
- [ ] 通用能力保持
```

---

## 参考资源

### 工具框架
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 一站式微调框架
- [TRL](https://github.com/huggingface/trl) - HuggingFace官方
- [Unsloth](https://github.com/unslothai/unsloth) - 2-5倍加速
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - 灵活配置

### 论文
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)

### 实践指南
- [Sebastian Raschka - LoRA Tips](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)
- [Unsloth Docs](https://docs.unsloth.ai/)

---

> **下一章**：[04-奖励模型训练.md](./04-奖励模型训练.md) - 学习如何训练Reward Model
