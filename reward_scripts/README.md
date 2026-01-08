# 奖励脚本接口规范

本目录包含奖励函数/奖励模型的脚本模板和示例。

## 接口规范

所有奖励脚本必须实现以下接口：

### 输入格式

脚本通过 **标准输入 (stdin)** 接收 JSON 格式的数据：

```json
{
  "prompts": ["问题1", "问题2", ...],
  "responses": ["回答1", "回答2", ...],
  "metadata": {
    "solutions": ["正确答案1", "正确答案2", ...],
    "data_sources": ["math", "math", ...],
    "custom_field": [...]
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `prompts` | List[str] | 原始问题列表 |
| `responses` | List[str] | 模型生成的回答列表 |
| `metadata` | Dict | 额外信息（来自数据集的其他字段） |

### 输出格式

脚本必须通过 **标准输出 (stdout)** 返回 JSON 格式的奖励值：

```json
{
  "rewards": [0.8, 1.0, 0.0, 0.5, ...],
  "details": [
    {"correct": true, "extracted": "42", "expected": "42"},
    {"correct": true, "extracted": "100", "expected": "100"},
    ...
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `rewards` | List[float] | 奖励值列表，与输入一一对应，范围通常为 [0, 1] 或 [-1, 1] |
| `details` | List[Dict] | 可选，每个样本的详细信息（用于调试） |

## 脚本模板

```python
#!/usr/bin/env python3
"""
奖励脚本模板
"""
import sys
import json

def compute_rewards(prompts, responses, metadata):
    """
    计算奖励值

    Args:
        prompts: List[str] - 问题列表
        responses: List[str] - 回答列表
        metadata: Dict - 额外信息

    Returns:
        rewards: List[float] - 奖励值列表
        details: List[Dict] - 详细信息（可选）
    """
    rewards = []
    details = []

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        # TODO: 实现你的奖励逻辑
        reward = 0.0
        detail = {}

        rewards.append(reward)
        details.append(detail)

    return rewards, details

def main():
    # 从 stdin 读取输入
    input_data = json.load(sys.stdin)

    prompts = input_data["prompts"]
    responses = input_data["responses"]
    metadata = input_data.get("metadata", {})

    # 计算奖励
    rewards, details = compute_rewards(prompts, responses, metadata)

    # 输出结果到 stdout
    output = {
        "rewards": rewards,
        "details": details
    }
    json.dump(output, sys.stdout)

if __name__ == "__main__":
    main()
```

## 使用方法

### 在平台中配置

1. 将脚本放在 `reward_scripts/` 目录下
2. 在创建训练任务时选择 **自定义脚本**
3. 填写脚本路径（相对于 reward_scripts/ 或绝对路径）

### 测试脚本

```bash
# 测试脚本
echo '{"prompts": ["1+1=?"], "responses": ["2"], "metadata": {"solutions": ["2"]}}' | python reward_scripts/rule_math.py
```

## 示例脚本

| 脚本 | 说明 | 适用场景 |
|------|------|----------|
| `rule_math.py` | 数学规则验证 | GRPO 数学任务 |
| `rule_format.py` | 格式检查 | 检查输出格式规范 |
| `api_reward.py` | 调用外部 API | 使用 GPT-4 等作为评判 |
| `model_reward.py` | 本地奖励模型 | PPO 使用本地 RM |

## 注意事项

1. 脚本必须是可执行的：`chmod +x your_script.py`
2. 脚本应在 5 秒内返回结果（批量处理时）
3. 错误信息输出到 stderr，不要污染 stdout
4. 奖励值建议归一化到 [0, 1] 或 [-1, 1]
