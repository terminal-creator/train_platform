"""
Data Format Converter

Supports conversion between different training data formats:
- SFT: instruction-response / prompt-response
- DPO: prompt-chosen-rejected (preference pairs)
- GRPO/PPO: prompt-only with optional metadata
- OpenAI Messages format
- Alpaca format
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataFormatDetector:
    """Auto-detect data format from file content."""

    FORMATS = {
        "sft_prompt_response": {"required": ["prompt", "response"]},
        "sft_instruction": {"required": ["instruction", "output"]},
        "sft_alpaca": {"required": ["instruction", "input", "output"]},
        "dpo_preference": {"required": ["prompt", "chosen", "rejected"]},
        "grpo_prompt": {"required": ["prompt"], "optional": ["data_source", "solution"]},
        "openai_messages": {"required": ["messages"]},
        "sharegpt": {"required": ["conversations"]},
    }

    @staticmethod
    def detect(data: List[Dict]) -> Tuple[str, float]:
        """Detect data format. Returns (format_name, confidence)."""
        if not data:
            return "unknown", 0.0

        sample = data[0]
        keys = set(sample.keys())

        # Check each format
        best_format = "unknown"
        best_score = 0.0

        for fmt_name, fmt_spec in DataFormatDetector.FORMATS.items():
            required = set(fmt_spec["required"])
            if required.issubset(keys):
                # Score: required match + bonus for optional
                score = len(required) / max(len(keys), 1)
                optional = set(fmt_spec.get("optional", []))
                bonus = len(optional.intersection(keys)) * 0.1
                score += bonus

                if score > best_score:
                    best_score = score
                    best_format = fmt_name

        # Check for OpenAI messages format (nested structure)
        if "messages" in keys:
            msgs = sample.get("messages", [])
            if isinstance(msgs, list) and len(msgs) > 0:
                if isinstance(msgs[0], dict) and "role" in msgs[0]:
                    return "openai_messages", 0.95

        return best_format, min(best_score, 1.0)

    @staticmethod
    def detect_from_file(file_path: str) -> Tuple[str, float]:
        """Detect format from a file."""
        path = Path(file_path)
        data = _load_data(str(path))
        return DataFormatDetector.detect(data[:10])


def _load_data(file_path: str) -> List[Dict]:
    """Load data from various file formats."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
        if isinstance(content, list):
            return content
        return [content]
    elif suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    elif suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _save_data(data: List[Dict], file_path: str):
    """Save data to file."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    elif suffix == ".json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    elif suffix == ".parquet":
        import pandas as pd
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def convert_to_sft(data: List[Dict], source_format: str = None) -> List[Dict]:
    """Convert data to SFT format (prompt-response)."""
    if not source_format:
        source_format, _ = DataFormatDetector.detect(data[:10])

    result = []
    for item in data:
        if source_format == "sft_prompt_response":
            result.append({"prompt": item["prompt"], "response": item["response"]})
        elif source_format == "sft_instruction":
            result.append({"prompt": item["instruction"], "response": item["output"]})
        elif source_format == "sft_alpaca":
            prompt = item["instruction"]
            if item.get("input"):
                prompt = f"{prompt}\n\n{item['input']}"
            result.append({"prompt": prompt, "response": item["output"]})
        elif source_format == "openai_messages":
            messages = item.get("messages", [])
            prompt_parts = []
            response = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("system", "user"):
                    prompt_parts.append(content)
                elif role == "assistant":
                    response = content
            result.append({
                "prompt": "\n".join(prompt_parts),
                "response": response,
            })
        elif source_format == "sharegpt":
            convs = item.get("conversations", [])
            prompt_parts = []
            response = ""
            for conv in convs:
                role = conv.get("from", conv.get("role", ""))
                value = conv.get("value", conv.get("content", ""))
                if role in ("system", "human", "user"):
                    prompt_parts.append(value)
                elif role in ("gpt", "assistant"):
                    response = value
            result.append({
                "prompt": "\n".join(prompt_parts),
                "response": response,
            })
        elif source_format == "dpo_preference":
            # Use chosen as the response
            result.append({"prompt": item["prompt"], "response": item["chosen"]})
        else:
            # Try to extract prompt/response from any format
            prompt = item.get("prompt", item.get("instruction", item.get("question", "")))
            response = item.get("response", item.get("output", item.get("answer", "")))
            if prompt:
                result.append({"prompt": str(prompt), "response": str(response)})

    return result


def convert_to_dpo(data: List[Dict], source_format: str = None) -> List[Dict]:
    """Convert data to DPO format (prompt-chosen-rejected)."""
    if not source_format:
        source_format, _ = DataFormatDetector.detect(data[:10])

    result = []
    for item in data:
        if source_format == "dpo_preference":
            result.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })
        elif source_format in ("sft_prompt_response", "sft_instruction", "sft_alpaca"):
            # For SFT data, the existing response is "chosen" but we have no "rejected"
            sft_item = convert_to_sft([item], source_format)[0]
            result.append({
                "prompt": sft_item["prompt"],
                "chosen": sft_item["response"],
                "rejected": "",  # Needs to be filled by user
            })
            logger.warning("Converting SFT to DPO: 'rejected' field is empty, needs manual annotation")
        elif source_format == "openai_messages":
            sft_item = convert_to_sft([item], source_format)[0]
            result.append({
                "prompt": sft_item["prompt"],
                "chosen": sft_item["response"],
                "rejected": "",
            })

    return result


def convert_to_grpo(data: List[Dict], source_format: str = None) -> List[Dict]:
    """Convert data to GRPO format (prompt with optional answer)."""
    if not source_format:
        source_format, _ = DataFormatDetector.detect(data[:10])

    result = []
    for item in data:
        if source_format == "grpo_prompt":
            entry = {"prompt": item["prompt"]}
            if "data_source" in item:
                entry["data_source"] = item["data_source"]
            if "solution" in item:
                entry["solution"] = item["solution"]
            result.append(entry)
        elif source_format in ("sft_prompt_response", "sft_instruction", "sft_alpaca"):
            sft_item = convert_to_sft([item], source_format)[0]
            result.append({
                "prompt": sft_item["prompt"],
                "solution": sft_item["response"],
                "data_source": "converted",
            })
        elif source_format == "openai_messages":
            sft_item = convert_to_sft([item], source_format)[0]
            result.append({
                "prompt": sft_item["prompt"],
                "solution": sft_item["response"],
                "data_source": "converted",
            })
        elif source_format == "dpo_preference":
            result.append({
                "prompt": item["prompt"],
                "data_source": "converted",
            })
        else:
            prompt = item.get("prompt", item.get("instruction", item.get("question", "")))
            if prompt:
                entry = {"prompt": str(prompt), "data_source": "converted"}
                answer = item.get("solution", item.get("answer", item.get("response", "")))
                if answer:
                    entry["solution"] = str(answer)
                result.append(entry)

    return result


def convert_to_openai_messages(data: List[Dict], source_format: str = None) -> List[Dict]:
    """Convert data to OpenAI Messages format."""
    if not source_format:
        source_format, _ = DataFormatDetector.detect(data[:10])

    result = []
    for item in data:
        if source_format == "openai_messages":
            result.append(item)
        elif source_format in ("sft_prompt_response", "sft_instruction", "sft_alpaca"):
            sft_item = convert_to_sft([item], source_format)[0]
            result.append({
                "messages": [
                    {"role": "user", "content": sft_item["prompt"]},
                    {"role": "assistant", "content": sft_item["response"]},
                ]
            })
        elif source_format == "dpo_preference":
            result.append({
                "messages": [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["chosen"]},
                ]
            })
        else:
            prompt = item.get("prompt", item.get("instruction", ""))
            response = item.get("response", item.get("output", ""))
            if prompt:
                result.append({
                    "messages": [
                        {"role": "user", "content": str(prompt)},
                        {"role": "assistant", "content": str(response)},
                    ]
                })

    return result


def convert_file(
    input_path: str,
    output_path: str,
    target_format: str,
    source_format: str = None,
) -> Dict[str, Any]:
    """
    Convert a data file from one format to another.

    Args:
        input_path: Path to input file
        output_path: Path to output file
        target_format: One of 'sft', 'dpo', 'grpo', 'openai_messages'
        source_format: Optional source format (auto-detected if not provided)

    Returns:
        Conversion statistics
    """
    data = _load_data(input_path)

    if not source_format:
        source_format, confidence = DataFormatDetector.detect(data[:10])
        logger.info(f"Auto-detected format: {source_format} (confidence: {confidence:.2f})")

    converters = {
        "sft": convert_to_sft,
        "dpo": convert_to_dpo,
        "grpo": convert_to_grpo,
        "ppo": convert_to_grpo,  # PPO uses same format as GRPO
        "openai_messages": convert_to_openai_messages,
    }

    if target_format not in converters:
        raise ValueError(f"Unsupported target format: {target_format}. Supported: {list(converters.keys())}")

    converted = converters[target_format](data, source_format)
    _save_data(converted, output_path)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "source_format": source_format,
        "target_format": target_format,
        "input_count": len(data),
        "output_count": len(converted),
        "skipped": len(data) - len(converted),
    }
