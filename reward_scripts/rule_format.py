#!/usr/bin/env python3
"""
格式规则奖励函数

适用于：检查模型输出是否符合特定格式要求
功能：根据格式规则给予奖励

支持的格式检查：
1. json: 检查是否为有效 JSON
2. markdown: 检查是否包含 markdown 结构
3. code: 检查是否包含代码块
4. length: 检查长度是否在范围内
5. keywords: 检查是否包含/排除特定关键词
6. custom: 自定义正则表达式
"""
import sys
import json
import re
from typing import List, Dict, Tuple


def check_json_format(response: str) -> Tuple[bool, str]:
    """检查是否为有效 JSON"""
    try:
        # 尝试提取 JSON 部分
        if "```json" in response:
            match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                json.loads(match.group(1))
                return True, "valid_json_in_codeblock"
        elif "```" in response:
            match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if match:
                json.loads(match.group(1))
                return True, "valid_json_in_codeblock"

        # 直接尝试解析
        json.loads(response.strip())
        return True, "valid_json"
    except:
        return False, "invalid_json"


def check_markdown_format(response: str) -> Tuple[bool, str]:
    """检查是否包含 markdown 结构"""
    has_headers = bool(re.search(r'^#+\s+', response, re.MULTILINE))
    has_lists = bool(re.search(r'^[\*\-\d]+[\.\)]\s+', response, re.MULTILINE))
    has_code = bool(re.search(r'```', response))
    has_bold = bool(re.search(r'\*\*.*?\*\*', response))

    score = sum([has_headers, has_lists, has_code, has_bold])
    if score >= 2:
        return True, f"markdown_score_{score}"
    return False, f"markdown_score_{score}"


def check_code_format(response: str, language: str = None) -> Tuple[bool, str]:
    """检查是否包含代码块"""
    if language:
        pattern = rf'```{language}\s*\n.*?\n```'
    else:
        pattern = r'```\w*\s*\n.*?\n```'

    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return True, f"has_code_blocks_{len(matches)}"
    return False, "no_code_blocks"


def check_length(response: str, min_len: int = 0, max_len: int = float('inf')) -> Tuple[bool, str]:
    """检查长度是否在范围内"""
    length = len(response)
    if min_len <= length <= max_len:
        return True, f"length_{length}_in_range"
    return False, f"length_{length}_out_of_range"


def check_keywords(
    response: str,
    required: List[str] = None,
    forbidden: List[str] = None
) -> Tuple[bool, str]:
    """检查关键词"""
    response_lower = response.lower()

    if required:
        missing = [kw for kw in required if kw.lower() not in response_lower]
        if missing:
            return False, f"missing_keywords_{missing}"

    if forbidden:
        found = [kw for kw in forbidden if kw.lower() in response_lower]
        if found:
            return False, f"forbidden_keywords_{found}"

    return True, "keywords_ok"


def check_custom_regex(response: str, pattern: str, should_match: bool = True) -> Tuple[bool, str]:
    """自定义正则检查"""
    try:
        matches = bool(re.search(pattern, response, re.DOTALL | re.MULTILINE))
        if matches == should_match:
            return True, f"regex_{'matched' if matches else 'not_matched'}_as_expected"
        return False, f"regex_{'matched' if matches else 'not_matched'}_unexpected"
    except re.error as e:
        return False, f"regex_error_{e}"


def compute_rewards(
    prompts: List[str],
    responses: List[str],
    metadata: Dict
) -> Tuple[List[float], List[Dict]]:
    """
    计算格式奖励

    Args:
        prompts: 问题列表
        responses: 回答列表
        metadata: 包含 checks 配置列表

    Returns:
        rewards: 奖励值列表
        details: 详细信息

    metadata.checks 示例：
    [
        {"type": "json"},
        {"type": "length", "min": 100, "max": 1000},
        {"type": "keywords", "required": ["答案"], "forbidden": ["不知道"]},
        {"type": "regex", "pattern": r"\d+", "should_match": true}
    ]
    """
    checks = metadata.get("checks", [{"type": "length", "min": 1}])
    weights = metadata.get("weights", None)  # 各检查项的权重

    if weights is None:
        weights = [1.0] * len(checks)

    rewards = []
    details = []

    for response in responses:
        check_results = []
        total_weight = sum(weights)

        for i, check in enumerate(checks):
            check_type = check.get("type", "length")
            weight = weights[i] if i < len(weights) else 1.0

            if check_type == "json":
                passed, msg = check_json_format(response)
            elif check_type == "markdown":
                passed, msg = check_markdown_format(response)
            elif check_type == "code":
                passed, msg = check_code_format(response, check.get("language"))
            elif check_type == "length":
                passed, msg = check_length(
                    response,
                    check.get("min", 0),
                    check.get("max", float('inf'))
                )
            elif check_type == "keywords":
                passed, msg = check_keywords(
                    response,
                    check.get("required", []),
                    check.get("forbidden", [])
                )
            elif check_type == "regex":
                passed, msg = check_custom_regex(
                    response,
                    check.get("pattern", ".*"),
                    check.get("should_match", True)
                )
            else:
                passed, msg = True, "unknown_check_type"

            check_results.append({
                "type": check_type,
                "passed": passed,
                "message": msg,
                "weight": weight
            })

        # 计算加权奖励
        reward = sum(
            r["weight"] * (1.0 if r["passed"] else 0.0)
            for r in check_results
        ) / total_weight

        rewards.append(reward)
        details.append({
            "checks": check_results,
            "total_passed": sum(1 for r in check_results if r["passed"]),
            "total_checks": len(check_results)
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
