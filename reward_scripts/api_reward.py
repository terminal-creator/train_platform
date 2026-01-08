#!/usr/bin/env python3
"""
API 奖励脚本

适用于：使用外部 API（如 GPT-4、Claude）作为奖励评判
功能：调用 LLM API 对回答进行评分

支持的 API：
1. OpenAI (GPT-4, GPT-3.5)
2. Anthropic (Claude)
3. 阿里云 DashScope (Qwen)
4. 自定义 OpenAI 兼容 API
"""
import sys
import json
import os
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 尝试导入 OpenAI 库
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# 评分 Prompt 模板
SCORING_PROMPT = """你是一个回答质量评估专家。请评估以下回答的质量。

问题：{prompt}

回答：{response}

{reference_section}

请从以下几个维度评分（每项 0-10 分）：
1. 正确性：回答是否正确、准确
2. 完整性：回答是否完整、全面
3. 清晰度：回答是否清晰、易懂
4. 相关性：回答是否与问题相关

请直接返回 JSON 格式的评分结果：
{{"correctness": 分数, "completeness": 分数, "clarity": 分数, "relevance": 分数, "overall": 总分(0-10)}}
"""


def create_client(api_type: str, api_key: str, base_url: str = None):
    """创建 API 客户端"""
    if not HAS_OPENAI:
        raise ImportError("请安装 openai 库: pip install openai")

    if api_type == "openai":
        return OpenAI(api_key=api_key)
    elif api_type == "dashscope":
        return OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    elif api_type == "custom":
        return OpenAI(api_key=api_key, base_url=base_url)
    else:
        return OpenAI(api_key=api_key, base_url=base_url)


def call_api_for_score(
    client,
    model: str,
    prompt: str,
    response: str,
    reference: str = None
) -> Dict:
    """调用 API 获取评分"""
    reference_section = ""
    if reference:
        reference_section = f"参考答案：{reference}\n"

    scoring_prompt = SCORING_PROMPT.format(
        prompt=prompt,
        response=response,
        reference_section=reference_section
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个专业的回答质量评估助手。请严格按照 JSON 格式返回评分。"},
                {"role": "user", "content": scoring_prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )

        result_text = completion.choices[0].message.content.strip()

        # 尝试解析 JSON
        # 处理可能的 markdown 代码块
        if "```" in result_text:
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        scores = json.loads(result_text)
        return scores

    except Exception as e:
        print(f"API 调用失败: {e}", file=sys.stderr)
        return {"overall": 5.0, "error": str(e)}


def compute_rewards(
    prompts: List[str],
    responses: List[str],
    metadata: Dict
) -> Tuple[List[float], List[Dict]]:
    """
    使用 API 计算奖励

    Args:
        prompts: 问题列表
        responses: 回答列表
        metadata: 包含 api_type, api_key, model, references 等

    Returns:
        rewards: 奖励值列表 (归一化到 0-1)
        details: 详细评分信息
    """
    # 获取配置
    api_type = metadata.get("api_type", "openai")
    api_key = metadata.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    base_url = metadata.get("base_url")
    model = metadata.get("model", "gpt-3.5-turbo")
    references = metadata.get("references", [])
    max_workers = metadata.get("max_workers", 4)

    if not api_key:
        raise ValueError("未提供 API Key，请在 metadata 中设置 api_key 或设置环境变量")

    # 创建客户端
    client = create_client(api_type, api_key, base_url)

    rewards = []
    details = []

    # 并行调用 API
    def process_item(i):
        prompt = prompts[i]
        response = responses[i]
        reference = references[i] if i < len(references) else None

        scores = call_api_for_score(client, model, prompt, response, reference)
        overall = scores.get("overall", 5.0)
        reward = overall / 10.0  # 归一化到 0-1

        return i, reward, scores

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, i): i for i in range(len(prompts))}
        results = [None] * len(prompts)

        for future in as_completed(futures):
            i, reward, scores = future.result()
            results[i] = (reward, scores)

    for reward, scores in results:
        rewards.append(reward)
        details.append(scores)

    return rewards, details


def main():
    # 从 stdin 读取输入
    input_data = json.load(sys.stdin)

    prompts = input_data["prompts"]
    responses = input_data["responses"]
    metadata = input_data.get("metadata", {})

    # 计算奖励
    rewards, details = compute_rewards(prompts, responses, metadata)

    # 输出结果
    output = {
        "rewards": rewards,
        "details": details
    }
    json.dump(output, sys.stdout)


if __name__ == "__main__":
    main()
