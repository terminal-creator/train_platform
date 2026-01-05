# 训练数据准备指南

本文档详细介绍不同训练算法所需的数据格式和准备方法。

## 目录

- [SFT 监督微调](#sft-监督微调)
- [GRPO 组相对策略优化](#grpo-组相对策略优化)
- [PPO 近端策略优化](#ppo-近端策略优化)
- [DPO 直接偏好优化](#dpo-直接偏好优化)
- [GSPO 组自博弈偏好优化](#gspo-组自博弈偏好优化)
- [数据处理工具](#数据处理工具)

---

## SFT 监督微调

### 数据格式

SFT 需要 prompt-response 配对数据，支持两种格式：

#### 格式一：对话格式 (推荐)

```jsonl
{"messages": [{"role": "user", "content": "什么是机器学习？"}, {"role": "assistant", "content": "机器学习是人工智能的一个分支..."}]}
{"messages": [{"role": "user", "content": "写一首关于春天的诗"}, {"role": "assistant", "content": "春风轻拂柳丝长..."}]}
{"messages": [{"role": "system", "content": "你是一个数学专家"}, {"role": "user", "content": "解方程 x^2 - 5x + 6 = 0"}, {"role": "assistant", "content": "使用因式分解：(x-2)(x-3)=0，所以 x=2 或 x=3"}]}
```

#### 格式二：简单格式

```jsonl
{"prompt": "什么是机器学习？", "response": "机器学习是人工智能的一个分支..."}
{"prompt": "写一首关于春天的诗", "response": "春风轻拂柳丝长..."}
```

### 数据准备步骤

1. **收集原始数据**
   ```python
   # 从各种来源收集数据
   # - 开源数据集 (Alpaca, ShareGPT, etc.)
   # - 自有标注数据
   # - 合成数据
   ```

2. **数据清洗**
   ```python
   def clean_sft_data(item):
       # 去除空白
       item['prompt'] = item['prompt'].strip()
       item['response'] = item['response'].strip()

       # 过滤过短/过长样本
       if len(item['prompt']) < 10 or len(item['response']) < 10:
           return None
       if len(item['prompt']) > 4096 or len(item['response']) > 4096:
           return None

       return item
   ```

3. **格式转换**
   ```python
   def convert_to_messages(item):
       return {
           "messages": [
               {"role": "user", "content": item["prompt"]},
               {"role": "assistant", "content": item["response"]}
           ]
       }
   ```

### 训练配置

```yaml
# sft_config.yaml
data:
  train_files: ["/data/sft_train.jsonl"]
  val_files: ["/data/sft_val.jsonl"]
  max_length: 4096

trainer:
  total_epochs: 3
  save_freq: 500
```

---

## GRPO 组相对策略优化

### 数据格式

GRPO 只需要 **prompt 数据**，模型会自己生成多个回复并通过奖励模型评分。

```jsonl
{"prompt": "解决这个数学问题：如果 x + 5 = 12，求 x 的值。"}
{"prompt": "写一个 Python 函数计算斐波那契数列的第 n 项。"}
{"prompt": "解释量子纠缠的原理。"}
```

### 带奖励函数的格式 (可选)

如果有自定义奖励函数，可以在数据中指定：

```jsonl
{"prompt": "计算 25 * 4 = ?", "reward_type": "math_accuracy", "ground_truth": "100"}
{"prompt": "def fibonacci(n):", "reward_type": "code_execution", "test_cases": ["fibonacci(10) == 55"]}
```

### 数据准备步骤

1. **收集 Prompt 数据**
   ```python
   # 数学推理任务
   math_prompts = [
       {"prompt": f"Calculate: {a} + {b} = ?"}
       for a, b in generate_math_problems()
   ]

   # 代码生成任务
   code_prompts = [
       {"prompt": f"Write a Python function to {task}"}
       for task in coding_tasks
   ]
   ```

2. **添加难度标签 (可选)**
   ```python
   def add_difficulty(item):
       # 根据问题复杂度添加标签
       item['difficulty'] = estimate_difficulty(item['prompt'])
       return item
   ```

3. **课程学习排序 (可选)**
   ```python
   # 按难度排序，先简单后困难
   prompts = sorted(prompts, key=lambda x: x.get('difficulty', 0))
   ```

### 奖励函数配置

```python
# reward_functions.py
def math_reward(prompt, response, ground_truth):
    """数学问题奖励函数"""
    try:
        # 提取答案
        answer = extract_answer(response)
        # 检查正确性
        if answer == ground_truth:
            return 1.0
        return 0.0
    except:
        return 0.0

def code_reward(prompt, response, test_cases):
    """代码执行奖励函数"""
    try:
        # 执行代码
        exec_result = safe_execute(response)
        # 运行测试
        passed = sum(1 for tc in test_cases if run_test(exec_result, tc))
        return passed / len(test_cases)
    except:
        return 0.0
```

### 训练配置

```yaml
# grpo_config.yaml
data:
  train_files: ["/data/grpo_prompts.jsonl"]
  max_length: 4096

rollout:
  n: 8  # 每个 prompt 生成 8 个回复
  temperature: 0.7
  top_p: 0.9

algorithm:
  kl_coef: 0.02
  clip_ratio: 0.2
```

---

## PPO 近端策略优化

### 数据格式

PPO 需要 **prompt 数据** + **奖励模型**。

```jsonl
{"prompt": "写一篇关于人工智能发展的文章。"}
{"prompt": "帮我解释什么是神经网络。"}
{"prompt": "设计一个高效的排序算法。"}
```

### 奖励模型准备

PPO 需要预训练的奖励模型：

```python
# 训练奖励模型
from transformers import AutoModelForSequenceClassification

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "your-base-model",
    num_labels=1
)

# 使用偏好数据训练
# 数据格式: {"prompt": "...", "chosen": "好回复", "rejected": "差回复"}
```

### 数据准备步骤

1. **准备 Prompt 数据**
   ```python
   prompts = load_jsonl("prompts.jsonl")
   ```

2. **准备奖励模型训练数据**
   ```jsonl
   {"prompt": "介绍Python", "chosen": "Python是一种广泛使用的高级编程语言...", "rejected": "Python就是个语言"}
   ```

3. **训练奖励模型**
   ```bash
   python train_reward_model.py \
     --model_name your-base-model \
     --train_data preference_data.jsonl \
     --output_dir ./reward_model
   ```

### 训练配置

```yaml
# ppo_config.yaml
model:
  path: "/models/Qwen2-7B"

reward_model:
  path: "/models/reward_model"

data:
  train_files: ["/data/ppo_prompts.jsonl"]

rollout:
  n: 1  # PPO 每个 prompt 只生成一个回复

algorithm:
  kl_coef: 0.02
  entropy_coef: 0.01
  clip_ratio: 0.2

critic:
  # PPO 需要 Critic 模型
  strategy: "fsdp"
```

---

## DPO 直接偏好优化

### 数据格式

DPO 需要 **偏好对比数据**：每个 prompt 配一个好回复和一个差回复。

```jsonl
{"prompt": "如何学习编程？", "chosen": "学习编程建议从以下几个方面入手：1. 选择一门入门语言...", "rejected": "随便学学就行"}
{"prompt": "解释什么是深度学习", "chosen": "深度学习是机器学习的一个分支，使用多层神经网络...", "rejected": "就是很深的学习"}
{"prompt": "Python和Java哪个好？", "chosen": "这取决于你的使用场景。Python适合数据科学和快速开发...", "rejected": "Python最好，Java垃圾"}
```

### 数据准备步骤

1. **收集偏好数据**

   方法一：人工标注
   ```python
   # 展示两个回复，让标注员选择更好的
   annotation_task = {
       "prompt": prompt,
       "response_a": response1,
       "response_b": response2,
       "instruction": "选择更好的回复"
   }
   ```

   方法二：使用强模型打分
   ```python
   def generate_preference_pair(prompt, model, judge_model):
       # 生成多个回复
       responses = [model.generate(prompt) for _ in range(4)]

       # 用 GPT-4 等强模型打分
       scores = [judge_model.score(prompt, r) for r in responses]

       # 选择最好和最差的
       best_idx = scores.index(max(scores))
       worst_idx = scores.index(min(scores))

       return {
           "prompt": prompt,
           "chosen": responses[best_idx],
           "rejected": responses[worst_idx]
       }
   ```

2. **数据质量过滤**
   ```python
   def filter_dpo_data(item):
       # 确保 chosen 和 rejected 有明显区别
       if len(item['chosen']) < 50 or len(item['rejected']) < 20:
           return False

       # 避免太相似的对比
       similarity = compute_similarity(item['chosen'], item['rejected'])
       if similarity > 0.9:
           return False

       return True
   ```

3. **平衡数据分布**
   ```python
   # 确保各类任务分布均衡
   categories = ['math', 'code', 'writing', 'qa']
   balanced_data = balance_by_category(data, categories)
   ```

### 训练配置

```yaml
# dpo_config.yaml
data:
  train_files: ["/data/dpo_train.jsonl"]
  val_files: ["/data/dpo_val.jsonl"]
  max_length: 4096

algorithm:
  beta: 0.1  # KL 惩罚系数
  loss_type: "sigmoid"  # 或 "hinge"

trainer:
  total_epochs: 1  # DPO 通常只需要 1 epoch
  learning_rate: 5e-7
```

---

## GSPO 组自博弈偏好优化

### 数据格式

GSPO 结合了 GRPO 和 DPO 的特点，支持两种模式：

#### 模式一：纯 Prompt (自博弈)

```jsonl
{"prompt": "设计一个缓存系统的架构"}
{"prompt": "优化这段代码的性能：def fib(n): return fib(n-1) + fib(n-2) if n > 1 else n"}
```

#### 模式二：带偏好数据 (混合训练)

```jsonl
{"prompt": "解释递归", "chosen": "递归是函数调用自身的编程技术...", "rejected": "递归就是自己调用自己"}
{"prompt": "设计一个 API", "self_play": true}
```

### 数据准备步骤

1. **准备基础 Prompt**
   ```python
   # 收集高质量任务 prompt
   prompts = [
       {"prompt": task, "category": cat}
       for task, cat in task_list
   ]
   ```

2. **生成自博弈数据 (可选预生成)**
   ```python
   def generate_self_play_pairs(prompt, model, n=4):
       """生成多个回复，用于自博弈"""
       responses = []
       for _ in range(n):
           response = model.generate(prompt, temperature=0.8)
           responses.append(response)
       return {"prompt": prompt, "responses": responses}
   ```

3. **混合偏好数据**
   ```python
   # 混合自博弈数据和人工偏好数据
   dataset = self_play_data + human_preference_data
   random.shuffle(dataset)
   ```

### 训练配置

```yaml
# gspo_config.yaml
data:
  train_files: ["/data/gspo_train.jsonl"]
  max_length: 4096

rollout:
  n: 4  # 组内生成数量
  temperature: 0.8

algorithm:
  self_play_ratio: 0.7  # 70% 自博弈，30% 监督
  kl_coef: 0.02
```

---

## 数据处理工具

### 数据转换脚本

```python
# scripts/convert_data.py
import json
import argparse

def convert_alpaca_to_messages(input_file, output_file):
    """将 Alpaca 格式转换为 messages 格式"""
    with open(input_file, 'r') as f:
        data = json.load(f)

    converted = []
    for item in data:
        messages = []
        if item.get('instruction'):
            prompt = item['instruction']
            if item.get('input'):
                prompt += f"\n\n{item['input']}"
            messages.append({"role": "user", "content": prompt})
        if item.get('output'):
            messages.append({"role": "assistant", "content": item['output']})

        if messages:
            converted.append({"messages": messages})

    with open(output_file, 'w') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def convert_sharegpt_to_messages(input_file, output_file):
    """将 ShareGPT 格式转换为 messages 格式"""
    with open(input_file, 'r') as f:
        data = json.load(f)

    converted = []
    for conv in data:
        messages = []
        for turn in conv.get('conversations', []):
            role = 'user' if turn['from'] == 'human' else 'assistant'
            messages.append({"role": role, "content": turn['value']})

        if messages:
            converted.append({"messages": messages})

    with open(output_file, 'w') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["alpaca", "sharegpt"], required=True)
    args = parser.parse_args()

    if args.format == "alpaca":
        convert_alpaca_to_messages(args.input, args.output)
    else:
        convert_sharegpt_to_messages(args.input, args.output)
```

### 数据验证脚本

```python
# scripts/validate_data.py
import json
import argparse
from collections import Counter

def validate_sft_data(file_path):
    """验证 SFT 数据格式"""
    errors = []
    stats = Counter()

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)

                if 'messages' in item:
                    if not isinstance(item['messages'], list):
                        errors.append(f"Line {i+1}: messages should be a list")
                    else:
                        stats['total'] += 1
                        stats['turns'] += len(item['messages'])
                elif 'prompt' in item and 'response' in item:
                    stats['total'] += 1
                else:
                    errors.append(f"Line {i+1}: missing required fields")

            except json.JSONDecodeError as e:
                errors.append(f"Line {i+1}: JSON parse error - {e}")

    return errors, stats

def validate_dpo_data(file_path):
    """验证 DPO 数据格式"""
    errors = []
    stats = Counter()

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)

                if 'prompt' not in item:
                    errors.append(f"Line {i+1}: missing 'prompt'")
                if 'chosen' not in item:
                    errors.append(f"Line {i+1}: missing 'chosen'")
                if 'rejected' not in item:
                    errors.append(f"Line {i+1}: missing 'rejected'")

                stats['total'] += 1
                stats['avg_chosen_len'] += len(item.get('chosen', ''))
                stats['avg_rejected_len'] += len(item.get('rejected', ''))

            except json.JSONDecodeError as e:
                errors.append(f"Line {i+1}: JSON parse error - {e}")

    if stats['total'] > 0:
        stats['avg_chosen_len'] //= stats['total']
        stats['avg_rejected_len'] //= stats['total']

    return errors, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--type", choices=["sft", "dpo", "grpo"], required=True)
    args = parser.parse_args()

    if args.type == "sft":
        errors, stats = validate_sft_data(args.file)
    elif args.type == "dpo":
        errors, stats = validate_dpo_data(args.file)
    else:
        # GRPO 只需要 prompt
        errors, stats = [], Counter()

    print(f"Stats: {dict(stats)}")
    if errors:
        print(f"Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  - {e}")
```

### 数据统计脚本

```python
# scripts/data_stats.py
import json
import argparse
from collections import Counter
import numpy as np

def compute_stats(file_path):
    """计算数据集统计信息"""
    lengths = []
    categories = Counter()

    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)

            # 计算长度
            if 'messages' in item:
                text = ' '.join(m['content'] for m in item['messages'])
            elif 'prompt' in item:
                text = item['prompt'] + item.get('response', '')
            else:
                text = str(item)

            lengths.append(len(text))

            # 统计类别
            if 'category' in item:
                categories[item['category']] += 1

    print(f"Total samples: {len(lengths)}")
    print(f"Length - Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.1f}, Median: {np.median(lengths):.1f}")
    print(f"Length percentiles - 90%: {np.percentile(lengths, 90):.0f}, 95%: {np.percentile(lengths, 95):.0f}, 99%: {np.percentile(lengths, 99):.0f}")

    if categories:
        print(f"\nCategories:")
        for cat, count in categories.most_common():
            print(f"  {cat}: {count} ({count/len(lengths)*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    compute_stats(args.file)
```

---

## 数据质量检查清单

### 通用检查

- [ ] JSON 格式正确，每行一个 JSON 对象
- [ ] 无空白或过短样本
- [ ] 无重复数据
- [ ] 编码统一 (UTF-8)
- [ ] 无敏感/有害内容

### SFT 数据检查

- [ ] prompt 和 response 配对完整
- [ ] response 质量高、信息准确
- [ ] 覆盖目标任务类型

### GRPO/PPO 数据检查

- [ ] prompt 多样性足够
- [ ] prompt 难度分布合理
- [ ] 奖励函数设计正确

### DPO 数据检查

- [ ] chosen 明显优于 rejected
- [ ] 偏好一致性高
- [ ] 无标注错误

---

## 推荐数据规模

| 训练类型 | 最小规模 | 推荐规模 | 说明 |
|---------|---------|---------|------|
| SFT | 1K | 10K-100K | 质量优先 |
| GRPO | 5K | 50K-500K | 多样性优先 |
| PPO | 10K | 100K+ | 需要足够探索 |
| DPO | 5K | 20K-100K | 偏好对质量要求高 |
| GSPO | 10K | 50K-200K | 自博弈需要迭代 |
