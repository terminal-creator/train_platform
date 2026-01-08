#!/usr/bin/env python3
"""
数学规则奖励函数

适用于：GRPO 数学任务
功能：验证模型回答的数学答案是否正确

支持的答案提取方式：
1. boxed: 从 \boxed{} 中提取
2. last_number: 提取最后出现的数字
3. equals: 提取等号后的值
"""
import sys
import json
import re
from typing import List, Dict, Tuple, Optional


def extract_answer_boxed(response: str) -> Optional[str]:
    """从 \\boxed{} 中提取答案"""
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, response)
    if matches:
        return matches[-1].strip()
    return None


def extract_answer_last_number(response: str) -> Optional[str]:
    """提取最后出现的数字"""
    pattern = r'[-+]?\d*\.?\d+'
    matches = re.findall(pattern, response)
    if matches:
        return matches[-1]
    return None


def extract_answer_equals(response: str) -> Optional[str]:
    """提取等号后的答案"""
    pattern = r'=\s*([-+]?\d*\.?\d+)'
    matches = re.findall(pattern, response)
    if matches:
        return matches[-1]
    return None


def extract_answer(response: str, method: str = "auto") -> Optional[str]:
    """根据方法提取答案"""
    if method == "boxed":
        return extract_answer_boxed(response)
    elif method == "last_number":
        return extract_answer_last_number(response)
    elif method == "equals":
        return extract_answer_equals(response)
    elif method == "auto":
        # 自动尝试多种方法
        answer = extract_answer_boxed(response)
        if answer:
            return answer
        answer = extract_answer_equals(response)
        if answer:
            return answer
        return extract_answer_last_number(response)
    return None


def normalize_number(s: str) -> Optional[float]:
    """将字符串归一化为数字"""
    try:
        # 移除空格和常见符号
        s = s.strip().replace(',', '').replace(' ', '')
        # 处理分数
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        return float(s)
    except:
        return None


def compare_answers(extracted: str, expected: str, method: str = "numeric") -> bool:
    """比较答案"""
    if method == "exact":
        return extracted.strip().lower() == expected.strip().lower()
    elif method == "numeric":
        ext_num = normalize_number(extracted)
        exp_num = normalize_number(expected)
        if ext_num is not None and exp_num is not None:
            return abs(ext_num - exp_num) < 1e-6
        return extracted.strip() == expected.strip()
    elif method == "fuzzy":
        ext_num = normalize_number(extracted)
        exp_num = normalize_number(expected)
        if ext_num is not None and exp_num is not None:
            # 允许 1% 的误差
            return abs(ext_num - exp_num) <= abs(exp_num) * 0.01 + 1e-6
        return extracted.strip().lower() == expected.strip().lower()
    return False


def compute_rewards(
    prompts: List[str],
    responses: List[str],
    metadata: Dict
) -> Tuple[List[float], List[Dict]]:
    """
    计算数学奖励

    Args:
        prompts: 问题列表
        responses: 回答列表
        metadata: 包含 solutions(正确答案), extract_method, compare_method

    Returns:
        rewards: 奖励值列表 (0.0 或 1.0)
        details: 详细信息
    """
    solutions = metadata.get("solutions", [])
    extract_method = metadata.get("extract_method", "auto")
    compare_method = metadata.get("compare_method", "numeric")

    rewards = []
    details = []

    for i, response in enumerate(responses):
        # 提取模型答案
        extracted = extract_answer(response, extract_method)

        # 获取正确答案
        expected = solutions[i] if i < len(solutions) else None

        # 计算奖励
        if extracted is None:
            reward = 0.0
            correct = False
        elif expected is None:
            reward = 0.5  # 无法验证时给中等分数
            correct = None
        else:
            correct = compare_answers(extracted, str(expected), compare_method)
            reward = 1.0 if correct else 0.0

        rewards.append(reward)
        details.append({
            "extracted": extracted,
            "expected": expected,
            "correct": correct,
            "extract_method": extract_method,
            "compare_method": compare_method
        })

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
