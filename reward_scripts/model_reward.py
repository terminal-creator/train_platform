#!/usr/bin/env python3
"""
本地奖励模型脚本

适用于：PPO 训练，使用本地部署的奖励模型
功能：加载本地 Reward Model 对回答进行评分

支持的模型类型：
1. HuggingFace 奖励模型（AutoModelForSequenceClassification）
2. 自定义奖励模型
"""
import sys
import json
import os
from typing import List, Dict, Tuple

# 尝试导入必要的库
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class RewardModel:
    """本地奖励模型封装"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str = "auto"
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("请安装 transformers: pip install transformers torch")

        # 确定设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 确定数据类型
        if torch_dtype == "auto":
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float32

        print(f"加载奖励模型: {model_path}", file=sys.stderr)
        print(f"设备: {self.device}, 数据类型: {self.torch_dtype}", file=sys.stderr)

        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # 设置 padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def score(
        self,
        prompts: List[str],
        responses: List[str],
        batch_size: int = 8
    ) -> List[float]:
        """
        计算奖励分数

        Args:
            prompts: 问题列表
            responses: 回答列表
            batch_size: 批次大小

        Returns:
            scores: 奖励分数列表
        """
        # 构建输入文本
        texts = []
        for prompt, response in zip(prompts, responses):
            # 常见的奖励模型输入格式
            text = f"Human: {prompt}\n\nAssistant: {response}"
            texts.append(text)

        scores = []

        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)

            # 前向传播
            outputs = self.model(**inputs)

            # 获取分数（取 logits 的第一个值作为奖励分数）
            if hasattr(outputs, 'logits'):
                batch_scores = outputs.logits.squeeze(-1).tolist()
            else:
                batch_scores = outputs[0].squeeze(-1).tolist()

            # 确保是列表
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]

            scores.extend(batch_scores)

        return scores


def normalize_scores(scores: List[float], method: str = "sigmoid") -> List[float]:
    """归一化分数到 [0, 1]"""
    import math

    if method == "sigmoid":
        return [1 / (1 + math.exp(-s)) for s in scores]
    elif method == "minmax":
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            return [0.5] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]
    elif method == "none":
        return scores
    else:
        return scores


def compute_rewards(
    prompts: List[str],
    responses: List[str],
    metadata: Dict
) -> Tuple[List[float], List[Dict]]:
    """
    使用本地奖励模型计算奖励

    Args:
        prompts: 问题列表
        responses: 回答列表
        metadata: 包含 model_path, device, batch_size, normalize 等

    Returns:
        rewards: 奖励值列表
        details: 详细信息
    """
    # 获取配置
    model_path = metadata.get("model_path")
    device = metadata.get("device", "auto")
    torch_dtype = metadata.get("torch_dtype", "auto")
    batch_size = metadata.get("batch_size", 8)
    normalize = metadata.get("normalize", "sigmoid")

    if not model_path:
        raise ValueError("未指定奖励模型路径，请在 metadata 中设置 model_path")

    # 加载模型
    rm = RewardModel(model_path, device, torch_dtype)

    # 计算分数
    raw_scores = rm.score(prompts, responses, batch_size)

    # 归一化
    rewards = normalize_scores(raw_scores, normalize)

    # 构建详细信息
    details = [
        {"raw_score": raw, "normalized": norm}
        for raw, norm in zip(raw_scores, rewards)
    ]

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
