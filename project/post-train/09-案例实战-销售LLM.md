# 09 - 案例实战：从零打造销售LLM

## 9.1 项目背景与目标

### 9.1.1 业务场景

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           智能销售助手应用场景                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   在线客服   │    │   电话销售   │    │  门店导购   │                 │
│  │             │    │             │    │             │                 │
│  │ • 7x24在线  │    │ • 话术辅助  │    │ • 产品推荐  │                 │
│  │ • 即时响应  │    │ • 实时提词  │    │ • 价格查询  │                 │
│  │ • 多轮对话  │    │ • 异议处理  │    │ • 库存查询  │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│           │                 │                 │                        │
│           └─────────────────┼─────────────────┘                        │
│                             ▼                                          │
│                   ┌─────────────────┐                                  │
│                   │    Sales LLM    │                                  │
│                   │  智能销售大模型  │                                  │
│                   └─────────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.1.2 项目目标

| 维度 | 指标 | 目标值 | 基线 |
|------|------|--------|------|
| 准确性 | 产品信息准确率 | >98% | 92% |
| 准确性 | 价格准确率 | >99% | 95% |
| 对话 | 上下文保持率 | >95% | 85% |
| 对话 | 平均对话轮次 | 5-8轮 | 3-4轮 |
| 销售 | 需求挖掘成功率 | >80% | 60% |
| 销售 | 异议处理满意度 | >4.0/5.0 | 3.2/5.0 |
| 效率 | 首次响应时间 | <2s | - |
| 效率 | 并发处理能力 | >100 QPS | - |

### 9.1.3 技术选型

```python
# 技术栈选择
TECH_STACK = {
    "base_model": "Qwen/Qwen2.5-7B-Instruct",  # 基座模型
    "training_framework": "LLaMA-Factory",      # 训练框架
    "distributed": "DeepSpeed ZeRO-2",          # 分布式策略
    "inference": "vLLM",                        # 推理框架
    "hardware": "8x A100 80GB",                 # 训练硬件
}

# 训练路径决策
"""
问：是否需要CPT？
答：销售领域与通用领域差异不大，产品知识可通过RAG补充
    → 不需要CPT，直接SFT

问：选择什么对齐方法？
答：有人工标注的偏好数据，需要精细控制回复风格
    → DPO + 少量GRPO

问：全参数还是LoRA？
答：7B模型，8卡A100可支持全参数
    → 全参数SFT + LoRA DPO（快速迭代）
"""
```

---

## 9.2 数据准备

### 9.2.1 数据收集与处理

```python
import json
import pandas as pd
from typing import List, Dict
from pathlib import Path
import hashlib
from tqdm import tqdm

class SalesDataPipeline:
    """销售数据处理流水线"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_crm_data(self, crm_file: str) -> List[Dict]:
        """处理CRM聊天记录"""

        df = pd.read_csv(crm_file)
        conversations = []

        # 按会话ID分组
        for session_id, group in df.groupby("session_id"):
            messages = []
            group = group.sort_values("timestamp")

            for _, row in group.iterrows():
                role = "user" if row["is_customer"] else "assistant"
                messages.append({
                    "role": role,
                    "content": row["message"]
                })

            # 过滤太短的对话
            if len(messages) >= 4:
                conversations.append({
                    "id": session_id,
                    "messages": messages,
                    "metadata": {
                        "outcome": row.get("outcome", "unknown"),
                        "product_category": row.get("category", ""),
                        "customer_satisfaction": row.get("satisfaction", None)
                    }
                })

        return conversations

    def process_training_scripts(self, scripts_dir: str) -> List[Dict]:
        """处理销售培训话术"""

        scripts = []
        scripts_path = Path(scripts_dir)

        for file in scripts_path.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                # 构造对话格式
                messages = [
                    {"role": "user", "content": item["customer_question"]},
                    {"role": "assistant", "content": item["recommended_response"]}
                ]

                scripts.append({
                    "id": f"script_{hashlib.md5(item['customer_question'].encode()).hexdigest()[:8]}",
                    "messages": messages,
                    "metadata": {
                        "scenario": item.get("scenario", ""),
                        "difficulty": item.get("difficulty", "medium"),
                        "source": "training_script"
                    }
                })

        return scripts

    def create_preference_data(
        self,
        conversations: List[Dict],
        llm_judge
    ) -> List[Dict]:
        """创建偏好数据"""

        preference_data = []

        for conv in tqdm(conversations, desc="Creating preference data"):
            messages = conv["messages"]

            # 找到所有助手回复
            for i, msg in enumerate(messages):
                if msg["role"] == "assistant" and i > 0:
                    # 获取上下文
                    context = messages[:i]
                    user_query = messages[i-1]["content"]
                    original_response = msg["content"]

                    # 生成对比回复
                    alternative = self._generate_alternative(
                        context, user_query, original_response
                    )

                    # 用LLM判断哪个更好
                    comparison = llm_judge.pairwise_comparison(
                        query=user_query,
                        response_a=original_response,
                        response_b=alternative,
                        context=self._format_context(context)
                    )

                    if comparison["winner"] != "tie":
                        chosen = original_response if comparison["winner"] == "A" else alternative
                        rejected = alternative if comparison["winner"] == "A" else original_response

                        preference_data.append({
                            "prompt": user_query,
                            "context": context,
                            "chosen": chosen,
                            "rejected": rejected,
                            "confidence": comparison["confidence"]
                        })

        return preference_data

    def prepare_datasets(
        self,
        crm_file: str,
        scripts_dir: str,
        split_ratio: Dict = {"train": 0.9, "eval": 0.1}
    ):
        """准备所有数据集"""

        # 1. 收集原始数据
        print("Processing CRM data...")
        crm_data = self.process_crm_data(crm_file)
        print(f"  → {len(crm_data)} conversations")

        print("Processing training scripts...")
        scripts_data = self.process_training_scripts(scripts_dir)
        print(f"  → {len(scripts_data)} scripts")

        # 2. 合并并转换为SFT格式
        all_data = crm_data + scripts_data

        sft_data = []
        for item in all_data:
            sft_item = {
                "conversations": item["messages"]
            }
            sft_data.append(sft_item)

        # 3. 划分数据集
        import random
        random.shuffle(sft_data)

        split_idx = int(len(sft_data) * split_ratio["train"])
        train_data = sft_data[:split_idx]
        eval_data = sft_data[split_idx:]

        # 4. 保存数据
        with open(self.output_dir / "train_sft.json", "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(self.output_dir / "eval_sft.json", "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

        print(f"\nDataset prepared:")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Eval: {len(eval_data)} samples")
        print(f"  Saved to: {self.output_dir}")

        return {
            "train": train_data,
            "eval": eval_data
        }


# 执行数据准备
pipeline = SalesDataPipeline("./data/sales_llm")
datasets = pipeline.prepare_datasets(
    crm_file="./raw_data/crm_chats.csv",
    scripts_dir="./raw_data/scripts/",
)
```

### 9.2.2 数据质量检查

```python
class DataQualityChecker:
    """数据质量检查器"""

    def __init__(self):
        self.issues = []

    def check_dataset(self, data: List[Dict]) -> Dict:
        """全面检查数据集"""

        stats = {
            "total_samples": len(data),
            "avg_turns": 0,
            "avg_user_length": 0,
            "avg_assistant_length": 0,
            "issues": []
        }

        turn_counts = []
        user_lengths = []
        assistant_lengths = []

        for i, item in enumerate(data):
            messages = item.get("conversations", item.get("messages", []))

            # 检查对话轮次
            turn_counts.append(len(messages))

            for msg in messages:
                content = msg.get("content", "")

                if msg["role"] == "user":
                    user_lengths.append(len(content))
                else:
                    assistant_lengths.append(len(content))

                # 检查空内容
                if not content or len(content.strip()) == 0:
                    stats["issues"].append({
                        "sample_id": i,
                        "issue": "empty_content",
                        "role": msg["role"]
                    })

                # 检查过长内容
                if len(content) > 2000:
                    stats["issues"].append({
                        "sample_id": i,
                        "issue": "too_long",
                        "length": len(content)
                    })

                # 检查敏感信息
                if self._contains_pii(content):
                    stats["issues"].append({
                        "sample_id": i,
                        "issue": "contains_pii",
                        "role": msg["role"]
                    })

        stats["avg_turns"] = sum(turn_counts) / len(turn_counts)
        stats["avg_user_length"] = sum(user_lengths) / len(user_lengths)
        stats["avg_assistant_length"] = sum(assistant_lengths) / len(assistant_lengths)

        return stats

    def _contains_pii(self, text: str) -> bool:
        """检查是否包含PII"""
        import re

        patterns = [
            r'1[3-9]\d{9}',           # 手机号
            r'\d{17}[\dXx]',          # 身份证
            r'[a-zA-Z0-9._%+-]+@',    # 邮箱
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def generate_report(self, stats: Dict) -> str:
        """生成质量报告"""

        report = f"""
# 数据质量报告

## 基本统计
- 总样本数: {stats['total_samples']}
- 平均对话轮次: {stats['avg_turns']:.1f}
- 用户消息平均长度: {stats['avg_user_length']:.0f}
- 助手消息平均长度: {stats['avg_assistant_length']:.0f}

## 发现的问题
- 总问题数: {len(stats['issues'])}
"""

        # 按问题类型分类
        issue_types = {}
        for issue in stats['issues']:
            issue_type = issue['issue']
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1

        for issue_type, count in issue_types.items():
            report += f"- {issue_type}: {count}条\n"

        return report


# 使用
checker = DataQualityChecker()
with open("./data/sales_llm/train_sft.json", "r") as f:
    train_data = json.load(f)

stats = checker.check_dataset(train_data)
print(checker.generate_report(stats))
```

---

## 9.3 SFT训练

### 9.3.1 训练配置

```yaml
# configs/sales_llm_sft.yaml - LLaMA-Factory配置

### 模型配置
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
trust_remote_code: true

### 数据配置
dataset: sales_sft
template: qwen
cutoff_len: 4096
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16

### 训练方法
stage: sft
do_train: true
finetuning_type: full  # 全参数微调

### 训练超参数
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true

### DeepSpeed配置
deepspeed: configs/ds_z2_config.json

### 输出配置
output_dir: outputs/sales_llm_sft
logging_steps: 10
save_steps: 500
save_total_limit: 3
plot_loss: true

### 评估配置
do_eval: true
eval_strategy: steps
eval_steps: 500
per_device_eval_batch_size: 4
```

```json
// configs/ds_z2_config.json - DeepSpeed ZeRO-2配置
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none"
        },
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": 1.0,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

```python
# data/dataset_info.json - 数据集注册
{
    "sales_sft": {
        "file_name": "sales_llm/train_sft.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }
}
```

### 9.3.2 训练脚本

```bash
#!/bin/bash
# scripts/train_sft.sh

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="sales-llm"
export WANDB_RUN_NAME="sft-v1"

# 训练命令
llamafactory-cli train configs/sales_llm_sft.yaml

# 或使用原生命令
# deepspeed --num_gpus=8 src/train.py configs/sales_llm_sft.yaml
```

```python
# 训练监控脚本
import wandb
from pathlib import Path
import json

class SFTTrainingMonitor:
    """SFT训练监控"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.log_file = self.output_dir / "trainer_state.json"

    def get_training_status(self) -> Dict:
        """获取训练状态"""

        if not self.log_file.exists():
            return {"status": "not_started"}

        with open(self.log_file, "r") as f:
            state = json.load(f)

        current_step = state.get("global_step", 0)
        total_steps = state.get("max_steps", 0)
        best_loss = state.get("best_metric", None)

        # 获取最近的loss
        log_history = state.get("log_history", [])
        recent_losses = [
            entry["loss"] for entry in log_history[-10:]
            if "loss" in entry
        ]

        return {
            "status": "running",
            "current_step": current_step,
            "total_steps": total_steps,
            "progress": current_step / total_steps * 100 if total_steps > 0 else 0,
            "best_loss": best_loss,
            "recent_avg_loss": sum(recent_losses) / len(recent_losses) if recent_losses else None
        }

    def check_convergence(self, patience: int = 5) -> bool:
        """检查是否收敛"""

        status = self.get_training_status()
        log_history = status.get("log_history", [])

        if len(log_history) < patience:
            return False

        # 检查最近N个epoch的loss是否下降
        recent_losses = [entry.get("loss") for entry in log_history[-patience:] if "loss" in entry]

        if len(recent_losses) < patience:
            return False

        # 简单判断：如果最近loss变化小于1%，认为收敛
        loss_change = abs(recent_losses[-1] - recent_losses[0]) / recent_losses[0]
        return loss_change < 0.01
```

### 9.3.3 SFT评估

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import json

class SFTEvaluator:
    """SFT模型评估器"""

    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()

    def evaluate(self, eval_data: List[Dict]) -> Dict:
        """评估模型"""

        results = {
            "total": len(eval_data),
            "samples": []
        }

        for item in tqdm(eval_data, desc="Evaluating"):
            messages = item["conversations"]

            # 获取最后一轮用户输入
            context = messages[:-1]
            expected = messages[-1]["content"]

            # 生成回复
            prompt = self._format_prompt(context)
            generated = self._generate(prompt)

            results["samples"].append({
                "context": context,
                "expected": expected,
                "generated": generated
            })

        return results

    def _format_prompt(self, messages: List[Dict]) -> str:
        """格式化prompt"""

        text = ""
        for msg in messages:
            if msg["role"] == "system":
                text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"

        text += "<|im_start|>assistant\n"
        return text

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """生成回复"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()


# 执行评估
evaluator = SFTEvaluator("outputs/sales_llm_sft/checkpoint-best")

with open("data/sales_llm/eval_sft.json", "r") as f:
    eval_data = json.load(f)

results = evaluator.evaluate(eval_data[:100])

# 保存结果供人工审核
with open("outputs/sft_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

---

## 9.4 DPO对齐

### 9.4.1 偏好数据构建

```python
from typing import List, Dict
import json
from openai import OpenAI

class PreferenceDataBuilder:
    """偏好数据构建器"""

    def __init__(self, sft_model_path: str, judge_model: str = "gpt-4"):
        self.sft_evaluator = SFTEvaluator(sft_model_path)
        self.judge = OpenAI()
        self.judge_model = judge_model

    def build_preference_pairs(
        self,
        eval_data: List[Dict],
        num_samples_per_prompt: int = 4
    ) -> List[Dict]:
        """构建偏好对"""

        preference_pairs = []

        for item in tqdm(eval_data, desc="Building preference pairs"):
            messages = item["conversations"]
            context = messages[:-1]
            prompt = self.sft_evaluator._format_prompt(context)

            # 生成多个候选回复
            candidates = []
            for _ in range(num_samples_per_prompt):
                response = self.sft_evaluator._generate(prompt)
                candidates.append(response)

            # 去重
            candidates = list(set(candidates))
            if len(candidates) < 2:
                continue

            # 用LLM评分
            scored = self._score_candidates(
                context[-1]["content"],  # 用户问题
                candidates
            )

            # 选择最好和最差的组成对
            scored.sort(key=lambda x: x["score"], reverse=True)

            if scored[0]["score"] - scored[-1]["score"] >= 1:  # 分差足够大
                preference_pairs.append({
                    "prompt": self._format_sharegpt_prompt(context),
                    "chosen": scored[0]["response"],
                    "rejected": scored[-1]["response"],
                    "chosen_score": scored[0]["score"],
                    "rejected_score": scored[-1]["score"]
                })

        return preference_pairs

    def _score_candidates(
        self,
        user_query: str,
        candidates: List[str]
    ) -> List[Dict]:
        """给候选回复打分"""

        scored = []

        for candidate in candidates:
            prompt = f"""请评估以下销售回复的质量。

用户问题：{user_query}

销售回复：{candidate}

请从以下维度打分（1-5分）：
1. 准确性：信息是否正确
2. 有效性：是否有效回应需求
3. 专业性：销售技巧是否专业
4. 友好度：语气是否亲切

请只输出一个总分（1-5的数字）："""

            response = self.judge.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )

            try:
                score = float(response.choices[0].message.content.strip())
            except:
                score = 3.0

            scored.append({
                "response": candidate,
                "score": score
            })

        return scored

    def _format_sharegpt_prompt(self, context: List[Dict]) -> str:
        """格式化为ShareGPT格式的prompt"""

        # DPO数据格式：只包含到用户最后一条消息
        return json.dumps(context, ensure_ascii=False)


# 构建DPO数据
builder = PreferenceDataBuilder("outputs/sales_llm_sft/checkpoint-best")

with open("data/sales_llm/eval_sft.json", "r") as f:
    eval_data = json.load(f)

preference_data = builder.build_preference_pairs(eval_data[:500])

# 保存
with open("data/sales_llm/train_dpo.json", "w", encoding="utf-8") as f:
    json.dump(preference_data, f, ensure_ascii=False, indent=2)

print(f"Created {len(preference_data)} preference pairs")
```

### 9.4.2 DPO训练配置

```yaml
# configs/sales_llm_dpo.yaml

### 模型配置
model_name_or_path: outputs/sales_llm_sft/checkpoint-best
trust_remote_code: true

### 数据配置
dataset: sales_dpo
template: qwen
cutoff_len: 4096
preprocessing_num_workers: 16

### 训练方法
stage: dpo
do_train: true
finetuning_type: lora  # DPO用LoRA快速迭代

### LoRA配置
lora_target: all
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

### DPO超参数
pref_beta: 0.1           # β参数，控制偏离参考模型的程度
pref_loss: sigmoid       # sigmoid DPO loss
pref_ftx: 0.0           # SFT正则化系数

### 训练超参数
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 5.0e-6
num_train_epochs: 2
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true

### 输出配置
output_dir: outputs/sales_llm_dpo
logging_steps: 10
save_steps: 200
save_total_limit: 3
```

```json
// data/dataset_info.json 添加DPO数据集
{
    "sales_dpo": {
        "file_name": "sales_llm/train_dpo.json",
        "ranking": true,
        "formatting": "sharegpt",
        "columns": {
            "messages": "prompt",
            "chosen": "chosen",
            "rejected": "rejected"
        }
    }
}
```

```bash
#!/bin/bash
# scripts/train_dpo.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT="sales-llm"
export WANDB_RUN_NAME="dpo-v1"

llamafactory-cli train configs/sales_llm_dpo.yaml
```

### 9.4.3 DPO评估

```python
class DPOEvaluator:
    """DPO模型评估器"""

    def __init__(
        self,
        base_model_path: str,
        dpo_adapter_path: str
    ):
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        # 加载DPO LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, dpo_adapter_path)
        self.model.eval()

    def compare_with_base(
        self,
        test_prompts: List[str],
        base_model_path: str
    ) -> Dict:
        """与基础模型对比"""

        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        results = []

        for prompt in tqdm(test_prompts, desc="Comparing"):
            # DPO模型生成
            dpo_response = self._generate(self.model, prompt)

            # 基础模型生成
            base_response = self._generate(base_model, prompt)

            results.append({
                "prompt": prompt,
                "base_response": base_response,
                "dpo_response": dpo_response
            })

        return results

    def _generate(self, model, prompt: str) -> str:
        """生成回复"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()


# DPO效果对比
evaluator = DPOEvaluator(
    base_model_path="outputs/sales_llm_sft/checkpoint-best",
    dpo_adapter_path="outputs/sales_llm_dpo"
)

test_prompts = [
    "这款产品太贵了，能便宜点吗？",
    "我再考虑考虑，不着急买",
    "你们的产品和竞品比有什么优势？",
]

comparison = evaluator.compare_with_base(
    test_prompts,
    "outputs/sales_llm_sft/checkpoint-best"
)

for item in comparison:
    print(f"Prompt: {item['prompt']}")
    print(f"Base: {item['base_response'][:200]}...")
    print(f"DPO: {item['dpo_response'][:200]}...")
    print("-" * 50)
```

---

## 9.5 模型合并与部署

### 9.5.1 LoRA权重合并

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def merge_lora_weights(
    base_model_path: str,
    lora_adapter_path: str,
    output_path: str
):
    """合并LoRA权重到基础模型"""

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)

    # 保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)

    print("Done!")


# 合并DPO LoRA
merge_lora_weights(
    base_model_path="outputs/sales_llm_sft/checkpoint-best",
    lora_adapter_path="outputs/sales_llm_dpo",
    output_path="outputs/sales_llm_final"
)
```

### 9.5.2 模型量化

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def quantize_model_4bit(model_path: str, output_path: str):
    """4bit量化模型"""

    print("Loading model with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Saving quantized model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # 计算压缩比
    import os
    original_size = sum(
        os.path.getsize(os.path.join(model_path, f))
        for f in os.listdir(model_path) if f.endswith('.safetensors')
    ) / 1e9

    quantized_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path) if f.endswith('.safetensors')
    ) / 1e9

    print(f"Original size: {original_size:.2f} GB")
    print(f"Quantized size: {quantized_size:.2f} GB")
    print(f"Compression ratio: {original_size/quantized_size:.1f}x")


# 量化最终模型
quantize_model_4bit(
    model_path="outputs/sales_llm_final",
    output_path="outputs/sales_llm_final_4bit"
)
```

### 9.5.3 vLLM部署

```python
# deploy/server.py
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="Sales LLM API")

# 全局模型
llm = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class ChatResponse(BaseModel):
    response: str
    usage: dict

# 系统提示词
SYSTEM_PROMPT = """你是一名专业的销售顾问，具备以下特点：
1. 熟悉所有产品信息，能准确回答产品相关问题
2. 善于倾听和理解客户需求，能够主动挖掘深层需求
3. 专业处理客户异议，用同理心和专业知识化解顾虑
4. 语言亲切友好，不过度推销，以客户利益为先
5. 遵守公司规定，不泄露敏感信息，不诋毁竞品

请基于以上原则与客户交流。"""

@app.on_event("startup")
async def startup():
    global llm
    llm = LLM(
        model="outputs/sales_llm_final",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        trust_remote_code=True
    )

@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 构建prompt
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"

    for msg in request.messages:
        if msg.role == "user":
            prompt += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            prompt += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"

    # 生成
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=["<|im_end|>"]
    )

    outputs = llm.generate([prompt], sampling_params)
    response_text = outputs[0].outputs[0].text.strip()

    return ChatResponse(
        response=response_text,
        usage={
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split())
        }
    )

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": llm is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# 启动服务
python deploy/server.py

# 或使用gunicorn
gunicorn deploy.server:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### 9.5.4 Docker部署

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# 安装Python
RUN apt-get update && apt-get install -y python3.10 python3-pip

# 安装依赖
COPY requirements.txt /app/
WORKDIR /app
RUN pip3 install -r requirements.txt

# 复制模型和代码
COPY outputs/sales_llm_final /app/model
COPY deploy /app/deploy

# 环境变量
ENV MODEL_PATH=/app/model
ENV CUDA_VISIBLE_DEVICES=0

# 启动服务
CMD ["python3", "deploy/server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  sales-llm:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 9.6 上线与监控

### 9.6.1 A/B测试配置

```python
import random
from typing import Dict, Optional
from datetime import datetime
import json

class ABTestManager:
    """A/B测试管理器"""

    def __init__(self):
        self.experiments = {}
        self.results = []

    def create_experiment(
        self,
        name: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.5
    ):
        """创建实验"""

        self.experiments[name] = {
            "control": control_model,
            "treatment": treatment_model,
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat(),
            "status": "running"
        }

    def get_variant(self, experiment_name: str, user_id: str) -> str:
        """获取用户分组"""

        exp = self.experiments.get(experiment_name)
        if not exp or exp["status"] != "running":
            return "control"

        # 基于user_id的一致性哈希
        hash_value = hash(user_id + experiment_name) % 100

        if hash_value < exp["traffic_split"] * 100:
            return "treatment"
        else:
            return "control"

    def record_metric(
        self,
        experiment_name: str,
        user_id: str,
        variant: str,
        metrics: Dict
    ):
        """记录指标"""

        self.results.append({
            "experiment": experiment_name,
            "user_id": user_id,
            "variant": variant,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        })

    def analyze_experiment(self, experiment_name: str) -> Dict:
        """分析实验结果"""

        exp_results = [r for r in self.results if r["experiment"] == experiment_name]

        control_metrics = [r["metrics"] for r in exp_results if r["variant"] == "control"]
        treatment_metrics = [r["metrics"] for r in exp_results if r["variant"] == "treatment"]

        analysis = {
            "experiment": experiment_name,
            "sample_sizes": {
                "control": len(control_metrics),
                "treatment": len(treatment_metrics)
            }
        }

        # 计算各指标
        for metric_name in control_metrics[0].keys() if control_metrics else []:
            control_values = [m[metric_name] for m in control_metrics]
            treatment_values = [m[metric_name] for m in treatment_metrics]

            import numpy as np
            from scipy import stats

            control_mean = np.mean(control_values)
            treatment_mean = np.mean(treatment_values)

            # t检验
            t_stat, p_value = stats.ttest_ind(control_values, treatment_values)

            analysis[metric_name] = {
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "lift": (treatment_mean - control_mean) / control_mean * 100,
                "p_value": p_value,
                "is_significant": p_value < 0.05
            }

        return analysis


# 使用示例
ab_manager = ABTestManager()

# 创建实验
ab_manager.create_experiment(
    name="sales_llm_v2_test",
    control_model="sales_llm_v1",
    treatment_model="sales_llm_v2",
    traffic_split=0.2  # 20%流量给新版本
)

# 在请求处理中
def handle_chat_request(user_id: str, messages: List):
    variant = ab_manager.get_variant("sales_llm_v2_test", user_id)

    if variant == "treatment":
        model = load_model("sales_llm_v2")
    else:
        model = load_model("sales_llm_v1")

    response = model.generate(messages)

    # 记录指标（异步）
    ab_manager.record_metric(
        "sales_llm_v2_test",
        user_id,
        variant,
        {"response_time": response.latency, "user_rating": None}  # rating后续收集
    )

    return response
```

### 9.6.2 生产监控

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sales_llm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('sales_llm')

# Prometheus指标
CHAT_REQUESTS = Counter('sales_llm_chat_requests_total', 'Total chat requests')
CHAT_LATENCY = Histogram('sales_llm_chat_latency_seconds', 'Chat latency')
ACTIVE_SESSIONS = Gauge('sales_llm_active_sessions', 'Active chat sessions')
ERROR_COUNT = Counter('sales_llm_errors_total', 'Total errors', ['error_type'])
TOKENS_GENERATED = Counter('sales_llm_tokens_generated_total', 'Tokens generated')

class ProductionMonitor:
    """生产环境监控"""

    def __init__(self, prometheus_port: int = 9090):
        start_http_server(prometheus_port)
        self.session_data = {}

    def on_request_start(self, session_id: str, user_message: str):
        """请求开始"""

        CHAT_REQUESTS.inc()
        ACTIVE_SESSIONS.inc()

        self.session_data[session_id] = {
            "start_time": datetime.now(),
            "user_message": user_message
        }

        logger.info(f"Session {session_id} started: {user_message[:100]}...")

    def on_request_end(
        self,
        session_id: str,
        response: str,
        tokens: int,
        error: Optional[str] = None
    ):
        """请求结束"""

        ACTIVE_SESSIONS.dec()

        if session_id in self.session_data:
            start_time = self.session_data[session_id]["start_time"]
            latency = (datetime.now() - start_time).total_seconds()
            CHAT_LATENCY.observe(latency)

            del self.session_data[session_id]

        if error:
            ERROR_COUNT.labels(error_type=error).inc()
            logger.error(f"Session {session_id} error: {error}")
        else:
            TOKENS_GENERATED.inc(tokens)
            logger.info(f"Session {session_id} completed: latency={latency:.2f}s, tokens={tokens}")

    def log_quality_issue(
        self,
        session_id: str,
        issue_type: str,
        details: str
    ):
        """记录质量问题"""

        logger.warning(f"Quality issue in {session_id}: [{issue_type}] {details}")

        # 这里可以集成告警系统
        if issue_type in ["hallucination", "safety_violation"]:
            self._send_alert(session_id, issue_type, details)

    def _send_alert(self, session_id: str, issue_type: str, details: str):
        """发送告警"""
        # 集成钉钉/Slack/PagerDuty等
        pass


# 集成到服务中
monitor = ProductionMonitor()

@app.post("/v1/chat")
async def chat(request: ChatRequest):
    session_id = str(uuid.uuid4())

    monitor.on_request_start(session_id, request.messages[-1].content)

    try:
        # 生成响应
        response = await generate_response(request)

        # 质量检查
        quality_issues = check_response_quality(response)
        for issue in quality_issues:
            monitor.log_quality_issue(session_id, issue["type"], issue["detail"])

        monitor.on_request_end(session_id, response.text, response.tokens)

        return response

    except Exception as e:
        monitor.on_request_end(session_id, "", 0, error=str(e))
        raise
```

---

## 9.7 持续迭代

### 9.7.1 问题收集与分析

```python
class FeedbackCollector:
    """反馈收集器"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_message TEXT,
                bot_response TEXT,
                rating INTEGER,
                feedback_text TEXT,
                issue_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def record_feedback(
        self,
        session_id: str,
        user_message: str,
        bot_response: str,
        rating: int,
        feedback_text: str = "",
        issue_type: str = ""
    ):
        """记录反馈"""

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO feedback
            (session_id, user_message, bot_response, rating, feedback_text, issue_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, user_message, bot_response, rating, feedback_text, issue_type))

        conn.commit()
        conn.close()

    def get_bad_cases(self, min_samples: int = 100) -> List[Dict]:
        """获取差评案例用于改进"""

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT user_message, bot_response, rating, feedback_text, issue_type
            FROM feedback
            WHERE rating <= 2
            ORDER BY created_at DESC
            LIMIT ?
        ''', (min_samples,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "user_message": row[0],
                "bot_response": row[1],
                "rating": row[2],
                "feedback": row[3],
                "issue_type": row[4]
            }
            for row in rows
        ]

    def generate_improvement_data(self, bad_cases: List[Dict]) -> List[Dict]:
        """从差评案例生成改进数据"""

        improvement_data = []

        for case in bad_cases:
            # 分析问题
            issue_type = case.get("issue_type", "unknown")

            # 生成改进的回复
            improved_response = self._generate_improved_response(
                case["user_message"],
                case["bot_response"],
                case["feedback"]
            )

            if improved_response:
                improvement_data.append({
                    "prompt": case["user_message"],
                    "chosen": improved_response,
                    "rejected": case["bot_response"],
                    "source": "user_feedback"
                })

        return improvement_data


class IterationManager:
    """迭代管理器"""

    def __init__(
        self,
        feedback_collector: FeedbackCollector,
        trainer,
        evaluator
    ):
        self.feedback = feedback_collector
        self.trainer = trainer
        self.evaluator = evaluator
        self.iteration_history = []

    def run_iteration(self) -> Dict:
        """执行一轮迭代"""

        iteration_id = len(self.iteration_history) + 1
        print(f"=== Starting iteration {iteration_id} ===")

        # 1. 收集差评案例
        bad_cases = self.feedback.get_bad_cases(min_samples=200)
        print(f"Collected {len(bad_cases)} bad cases")

        # 2. 生成改进数据
        improvement_data = self.feedback.generate_improvement_data(bad_cases)
        print(f"Generated {len(improvement_data)} improvement samples")

        if len(improvement_data) < 50:
            print("Not enough improvement data, skipping iteration")
            return {"status": "skipped", "reason": "insufficient_data"}

        # 3. 增量DPO训练
        print("Running incremental DPO training...")
        self.trainer.train_dpo(improvement_data)

        # 4. 评估新模型
        print("Evaluating new model...")
        eval_results = self.evaluator.evaluate()

        # 5. 记录结果
        result = {
            "iteration_id": iteration_id,
            "bad_cases_count": len(bad_cases),
            "improvement_samples": len(improvement_data),
            "eval_results": eval_results,
            "timestamp": datetime.now().isoformat()
        }

        self.iteration_history.append(result)

        # 6. 决定是否部署
        if self._should_deploy(eval_results):
            print("New model passed evaluation, deploying...")
            self._deploy_new_model()
            result["deployed"] = True
        else:
            print("New model did not pass evaluation threshold")
            result["deployed"] = False

        return result

    def _should_deploy(self, eval_results: Dict) -> bool:
        """判断是否应该部署"""

        # 检查关键指标是否达标
        thresholds = {
            "accuracy": 0.95,
            "safety_score": 0.98,
            "user_satisfaction": 3.8
        }

        for metric, threshold in thresholds.items():
            if eval_results.get(metric, 0) < threshold:
                return False

        # 检查是否比上一版本有提升
        if self.iteration_history:
            last_results = self.iteration_history[-1].get("eval_results", {})
            for metric in ["accuracy", "user_satisfaction"]:
                if eval_results.get(metric, 0) < last_results.get(metric, 0):
                    return False

        return True
```

### 9.7.2 版本管理

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import json
import shutil
from pathlib import Path

@dataclass
class ModelVersion:
    """模型版本"""
    version: str
    path: str
    created_at: str
    training_config: Dict
    eval_metrics: Dict
    status: str  # dev / staging / production / archived

class ModelRegistry:
    """模型注册表"""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.registry_path / "registry.json"
        self._load_registry()

    def _load_registry(self):
        """加载注册表"""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.versions = json.load(f)
        else:
            self.versions = {}

    def _save_registry(self):
        """保存注册表"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.versions, f, indent=2)

    def register_model(
        self,
        version: str,
        model_path: str,
        training_config: Dict,
        eval_metrics: Dict
    ):
        """注册新模型"""

        # 复制模型到注册表
        dest_path = self.registry_path / version
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(model_path, dest_path)

        # 记录元数据
        self.versions[version] = {
            "path": str(dest_path),
            "created_at": datetime.now().isoformat(),
            "training_config": training_config,
            "eval_metrics": eval_metrics,
            "status": "dev"
        }

        self._save_registry()
        print(f"Registered model version: {version}")

    def promote_to_staging(self, version: str):
        """提升到staging环境"""

        if version not in self.versions:
            raise ValueError(f"Version {version} not found")

        # 将当前staging降级
        for v, meta in self.versions.items():
            if meta["status"] == "staging":
                meta["status"] = "dev"

        self.versions[version]["status"] = "staging"
        self._save_registry()
        print(f"Promoted {version} to staging")

    def promote_to_production(self, version: str):
        """提升到生产环境"""

        if version not in self.versions:
            raise ValueError(f"Version {version} not found")

        if self.versions[version]["status"] != "staging":
            raise ValueError("Must be in staging before production")

        # 将当前production归档
        for v, meta in self.versions.items():
            if meta["status"] == "production":
                meta["status"] = "archived"

        self.versions[version]["status"] = "production"
        self._save_registry()
        print(f"Promoted {version} to production")

    def get_production_model(self) -> Optional[str]:
        """获取生产环境模型路径"""

        for version, meta in self.versions.items():
            if meta["status"] == "production":
                return meta["path"]
        return None

    def rollback(self):
        """回滚到上一个版本"""

        # 找到最近的archived版本
        archived = [
            (v, meta) for v, meta in self.versions.items()
            if meta["status"] == "archived"
        ]

        if not archived:
            raise ValueError("No archived version to rollback to")

        # 按时间排序，取最近的
        archived.sort(key=lambda x: x[1]["created_at"], reverse=True)
        rollback_version = archived[0][0]

        # 将当前production降级
        for v, meta in self.versions.items():
            if meta["status"] == "production":
                meta["status"] = "archived"

        self.versions[rollback_version]["status"] = "production"
        self._save_registry()
        print(f"Rolled back to {rollback_version}")


# 使用示例
registry = ModelRegistry("./model_registry")

# 注册新版本
registry.register_model(
    version="v1.0.0",
    model_path="outputs/sales_llm_final",
    training_config={"learning_rate": 1e-5, "epochs": 3},
    eval_metrics={"accuracy": 0.96, "safety": 0.99}
)

# 提升到staging测试
registry.promote_to_staging("v1.0.0")

# 通过测试后上线
registry.promote_to_production("v1.0.0")
```

---

## 9.8 项目总结

### 9.8.1 完整训练流程回顾

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        销售LLM完整训练流程                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │ 数据准备 │ → │   SFT   │ → │   DPO   │ → │  部署   │             │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘             │
│       │              │              │              │                   │
│       ▼              ▼              ▼              ▼                   │
│  • CRM数据清洗   • 全参数微调   • LoRA训练    • 模型合并              │
│  • 话术脚本处理  • 3 epochs     • β=0.1       • 4bit量化              │
│  • 格式转换      • lr=1e-5     • 2 epochs    • vLLM部署              │
│  • 质量检查      • 多轮对话     • 偏好数据    • 监控告警              │
│                                                                         │
│  数据量: 10万条   训练: 8xA100   训练: 8xA100   推理: 1xA100           │
│  耗时: 2天        耗时: 12小时   耗时: 4小时    QPS: 100+              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.8.2 关键经验总结

| 阶段 | 关键点 | 踩坑经验 |
|------|--------|----------|
| 数据 | 质量 > 数量 | 1000条高质量 > 10000条噪音数据 |
| 数据 | 多轮对话格式 | 确保Loss Mask正确，只学习助手回复 |
| SFT | 学习率选择 | 全参数1e-5，LoRA 5e-5 |
| SFT | 过拟合监控 | eval_loss开始上升就该停了 |
| DPO | β参数调优 | 从0.1开始，太大导致模型僵化 |
| DPO | 数据质量 | chosen和rejected差异要明显 |
| 部署 | 量化选择 | 4bit NF4 效果和速度平衡最好 |
| 部署 | 批处理 | vLLM continuous batching显著提升吞吐 |

### 9.8.3 效果对比

| 指标 | 基座模型 | SFT后 | DPO后 | 目标 |
|------|----------|-------|-------|------|
| 产品准确率 | 82% | 95% | 98% | >98% |
| 上下文保持率 | 75% | 90% | 96% | >95% |
| 需求挖掘率 | 45% | 72% | 85% | >80% |
| 用户满意度 | 3.0 | 3.8 | 4.2 | >4.0 |
| 安全合规率 | 85% | 96% | 99% | >98% |

### 9.8.4 后续优化方向

1. **RAG增强**：接入产品知识库，实时获取最新产品信息
2. **多模态**：支持产品图片理解，视觉问答
3. **个性化**：根据客户画像定制回复风格
4. **实时学习**：Online DPO，从用户反馈持续优化
5. **Agent能力**：接入CRM系统，自动创建工单、查询库存

---

## 本章结语

恭喜你完成了销售LLM的完整训练实战！

从零开始，我们经历了：
- **数据工程**：收集、清洗、格式化
- **SFT训练**：让模型学会销售对话
- **DPO对齐**：让模型学会更好的回复风格
- **工程部署**：量化、服务化、监控
- **持续迭代**：收集反馈、改进模型

记住：**模型训练不是一次性的工作，而是一个持续优化的过程**。

保持迭代，你的模型会越来越好！
