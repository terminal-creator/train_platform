"""
Data Cleaning Pipeline

Provides configurable cleaning operations for training data:
- Length filtering (min/max tokens)
- Format validation (JSON schema)
- Empty/null value handling
- Anomaly detection
- Progress callbacks
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for data cleaning operations."""
    # Length filtering
    min_prompt_length: int = 1
    max_prompt_length: int = 10000
    min_response_length: int = 1
    max_response_length: int = 50000

    # Token-based length filtering (approximate: 1 token ≈ 4 chars for English, 2 chars for Chinese)
    min_prompt_tokens: Optional[int] = None
    max_prompt_tokens: Optional[int] = None
    min_response_tokens: Optional[int] = None
    max_response_tokens: Optional[int] = None

    # Content filtering
    remove_empty: bool = True
    remove_duplicates: bool = False  # Use deduplication module for advanced dedup
    strip_whitespace: bool = True
    remove_html_tags: bool = False
    remove_urls: bool = False

    # Format validation
    required_fields: List[str] = field(default_factory=list)
    validate_json_fields: List[str] = field(default_factory=list)

    # Quality thresholds
    min_unique_chars_ratio: float = 0.0  # Minimum ratio of unique characters
    max_repetition_ratio: float = 1.0  # Maximum ratio of repeated n-grams


@dataclass
class CleaningStats:
    """Statistics from a cleaning run."""
    total_input: int = 0
    total_output: int = 0
    removed_empty: int = 0
    removed_too_short: int = 0
    removed_too_long: int = 0
    removed_invalid_format: int = 0
    removed_low_quality: int = 0
    removed_duplicates: int = 0
    warnings: List[str] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return self.total_input - self.total_output

    @property
    def removal_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.total_removed / self.total_input

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "total_removed": self.total_removed,
            "removal_rate": round(self.removal_rate * 100, 2),
            "breakdown": {
                "empty": self.removed_empty,
                "too_short": self.removed_too_short,
                "too_long": self.removed_too_long,
                "invalid_format": self.removed_invalid_format,
                "low_quality": self.removed_low_quality,
                "duplicates": self.removed_duplicates,
            },
            "warnings": self.warnings,
        }


class DataCleaningPipeline:
    """
    Configurable data cleaning pipeline.

    Usage:
        config = CleaningConfig(min_prompt_length=10, max_response_length=5000)
        pipeline = DataCleaningPipeline(config)
        cleaned, stats = pipeline.clean(data)
    """

    def __init__(self, config: CleaningConfig = None):
        self.config = config or CleaningConfig()
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set progress callback: callback(current, total, message)"""
        self._progress_callback = callback

    def _report_progress(self, current: int, total: int, message: str = ""):
        if self._progress_callback:
            self._progress_callback(current, total, message)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: mix of Chinese and English."""
        if not text:
            return 0
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return chinese_chars + other_chars // 4

    def _check_repetition(self, text: str, n: int = 3) -> float:
        """Check n-gram repetition ratio."""
        if len(text) < n:
            return 0.0
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        if not ngrams:
            return 0.0
        unique = len(set(ngrams))
        return 1.0 - (unique / len(ngrams))

    def _clean_text(self, text: str) -> str:
        """Apply text cleaning operations."""
        if not text:
            return text

        if self.config.strip_whitespace:
            text = text.strip()
            # Normalize multiple spaces/newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)

        if self.config.remove_html_tags:
            text = re.sub(r'<[^>]+>', '', text)

        if self.config.remove_urls:
            text = re.sub(r'https?://\S+', '', text)

        return text

    def clean_item(self, item: Dict[str, Any], stats: CleaningStats) -> Optional[Dict[str, Any]]:
        """Clean a single data item. Returns None if item should be removed."""
        cfg = self.config

        # Check required fields
        if cfg.required_fields:
            for field_name in cfg.required_fields:
                if field_name not in item or item[field_name] is None:
                    stats.removed_invalid_format += 1
                    return None

        # Get text fields
        prompt = str(item.get("prompt", item.get("instruction", "")))
        response = str(item.get("response", item.get("output", item.get("chosen", ""))))

        # Clean text
        prompt = self._clean_text(prompt)
        response = self._clean_text(response)

        # Check empty
        if cfg.remove_empty:
            if not prompt.strip():
                stats.removed_empty += 1
                return None
            # Response can be empty for GRPO/PPO format
            if "response" in item or "output" in item:
                if not response.strip():
                    stats.removed_empty += 1
                    return None

        # Check character length
        if len(prompt) < cfg.min_prompt_length or len(prompt) > cfg.max_prompt_length:
            if len(prompt) < cfg.min_prompt_length:
                stats.removed_too_short += 1
            else:
                stats.removed_too_long += 1
            return None

        if response and ("response" in item or "output" in item):
            if len(response) < cfg.min_response_length or len(response) > cfg.max_response_length:
                if len(response) < cfg.min_response_length:
                    stats.removed_too_short += 1
                else:
                    stats.removed_too_long += 1
                return None

        # Check token length
        if cfg.min_prompt_tokens or cfg.max_prompt_tokens:
            tokens = self._estimate_tokens(prompt)
            if cfg.min_prompt_tokens and tokens < cfg.min_prompt_tokens:
                stats.removed_too_short += 1
                return None
            if cfg.max_prompt_tokens and tokens > cfg.max_prompt_tokens:
                stats.removed_too_long += 1
                return None

        if cfg.min_response_tokens or cfg.max_response_tokens:
            tokens = self._estimate_tokens(response)
            if cfg.min_response_tokens and tokens < cfg.min_response_tokens:
                stats.removed_too_short += 1
                return None
            if cfg.max_response_tokens and tokens > cfg.max_response_tokens:
                stats.removed_too_long += 1
                return None

        # Check quality
        if cfg.min_unique_chars_ratio > 0:
            for text in [prompt, response]:
                if text:
                    unique_ratio = len(set(text)) / len(text)
                    if unique_ratio < cfg.min_unique_chars_ratio:
                        stats.removed_low_quality += 1
                        return None

        if cfg.max_repetition_ratio < 1.0:
            for text in [prompt, response]:
                if text and len(text) > 20:
                    rep_ratio = self._check_repetition(text)
                    if rep_ratio > cfg.max_repetition_ratio:
                        stats.removed_low_quality += 1
                        return None

        # Validate JSON fields
        for field_name in cfg.validate_json_fields:
            val = item.get(field_name)
            if val and isinstance(val, str):
                try:
                    json.loads(val)
                except json.JSONDecodeError:
                    stats.removed_invalid_format += 1
                    return None

        # Return cleaned item
        cleaned = dict(item)
        if "prompt" in cleaned:
            cleaned["prompt"] = prompt
        if "instruction" in cleaned:
            cleaned["instruction"] = prompt
        if "response" in cleaned:
            cleaned["response"] = response
        if "output" in cleaned:
            cleaned["output"] = response

        return cleaned

    def clean(
        self,
        data: List[Dict[str, Any]],
    ) -> tuple:
        """
        Clean a list of data items.

        Returns:
            (cleaned_data, stats)
        """
        stats = CleaningStats(total_input=len(data))
        cleaned = []
        seen_hashes = set() if self.config.remove_duplicates else None

        for i, item in enumerate(data):
            if i % 1000 == 0:
                self._report_progress(i, len(data), f"Cleaning item {i}/{len(data)}")

            result = self.clean_item(item, stats)
            if result is None:
                continue

            # Simple exact dedup
            if seen_hashes is not None:
                key = json.dumps(result, sort_keys=True, ensure_ascii=False)
                h = hash(key)
                if h in seen_hashes:
                    stats.removed_duplicates += 1
                    continue
                seen_hashes.add(h)

            cleaned.append(result)

        stats.total_output = len(cleaned)
        self._report_progress(len(data), len(data), "Cleaning complete")

        if stats.removal_rate > 0.5:
            stats.warnings.append(
                f"High removal rate: {stats.removal_rate*100:.1f}%. Consider adjusting cleaning config."
            )

        logger.info(
            f"Cleaning complete: {stats.total_input} -> {stats.total_output} "
            f"(removed {stats.total_removed}, {stats.removal_rate*100:.1f}%)"
        )

        return cleaned, stats

    def clean_file(
        self,
        input_path: str,
        output_path: str,
    ) -> Dict[str, Any]:
        """Clean a data file and save results."""
        from .data_converter import _load_data, _save_data

        data = _load_data(input_path)
        cleaned, stats = self.clean(data)
        _save_data(cleaned, output_path)

        return {
            "input_path": input_path,
            "output_path": output_path,
            **stats.to_dict(),
        }
