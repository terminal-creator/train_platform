"""
Data Quality Assessment

Provides automatic quality scoring for training data:
- Text quality metrics (length, diversity, repetition)
- Instruction-response relevance detection
- Format conformity checks
- Quality report generation
"""

import json
import logging
import math
import re
from collections import Counter
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QualityConfig:
    """Configuration for quality assessment."""
    # Scoring weights
    length_weight: float = 0.2
    diversity_weight: float = 0.2
    repetition_weight: float = 0.2
    relevance_weight: float = 0.2
    format_weight: float = 0.2

    # Thresholds
    ideal_prompt_length: tuple = (50, 500)  # chars
    ideal_response_length: tuple = (100, 2000)  # chars
    min_vocabulary_ratio: float = 0.3  # unique words / total words
    max_ngram_repetition: float = 0.3  # 3-gram repetition ratio

    # Text fields
    prompt_field: str = "prompt"
    response_field: str = "response"


@dataclass
class ItemQuality:
    """Quality score for a single item."""
    overall: float = 0.0
    length_score: float = 0.0
    diversity_score: float = 0.0
    repetition_score: float = 0.0
    relevance_score: float = 0.0
    format_score: float = 0.0
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": round(self.overall, 3),
            "length_score": round(self.length_score, 3),
            "diversity_score": round(self.diversity_score, 3),
            "repetition_score": round(self.repetition_score, 3),
            "relevance_score": round(self.relevance_score, 3),
            "format_score": round(self.format_score, 3),
            "issues": self.issues,
        }


@dataclass
class QualityReport:
    """Quality assessment report for a dataset."""
    total_items: int = 0
    average_score: float = 0.0
    score_distribution: Dict[str, int] = field(default_factory=dict)
    dimension_averages: Dict[str, float] = field(default_factory=dict)
    common_issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    low_quality_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "average_score": round(self.average_score, 3),
            "score_distribution": self.score_distribution,
            "dimension_averages": {
                k: round(v, 3) for k, v in self.dimension_averages.items()
            },
            "common_issues": self.common_issues,
            "recommendations": self.recommendations,
            "low_quality_count": len(self.low_quality_indices),
        }


class QualityAssessor:
    """Assess quality of training data."""

    def __init__(self, config: QualityConfig = None):
        self.config = config or QualityConfig()

    def _score_length(self, text: str, ideal_range: tuple) -> float:
        """Score text length relative to ideal range."""
        length = len(text)
        min_ideal, max_ideal = ideal_range

        if min_ideal <= length <= max_ideal:
            return 1.0
        elif length < min_ideal:
            return max(0.0, length / min_ideal)
        else:
            # Gentle decay for longer texts
            excess = length - max_ideal
            return max(0.0, 1.0 - excess / (max_ideal * 2))

    def _score_diversity(self, text: str) -> float:
        """Score vocabulary diversity."""
        words = text.lower().split()
        if not words:
            return 0.0
        unique_words = len(set(words))
        ratio = unique_words / len(words)
        return min(1.0, ratio / 0.5)  # Normalize: 0.5 ratio = 1.0 score

    def _score_repetition(self, text: str, n: int = 3) -> float:
        """Score text repetition (lower repetition = higher score)."""
        if len(text) < n * 2:
            return 1.0

        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        if not ngrams:
            return 1.0

        unique_ngrams = len(set(ngrams))
        repetition_ratio = 1.0 - (unique_ngrams / len(ngrams))

        return max(0.0, 1.0 - repetition_ratio * 2)

    def _score_relevance(self, prompt: str, response: str) -> float:
        """Score relevance between prompt and response."""
        if not prompt or not response:
            return 0.5  # Neutral score

        # Word overlap heuristic
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        if not prompt_words:
            return 0.5

        # Some overlap is expected but not too much (copy)
        overlap = len(prompt_words.intersection(response_words))
        overlap_ratio = overlap / len(prompt_words)

        if overlap_ratio < 0.05:
            return 0.3  # Very low overlap, might be irrelevant
        elif overlap_ratio > 0.8:
            return 0.5  # Too much overlap, might be copying
        else:
            return min(1.0, 0.5 + overlap_ratio)

    def _score_format(self, item: Dict) -> float:
        """Score format conformity."""
        score = 1.0
        issues = []

        prompt = str(item.get(self.config.prompt_field, ""))
        response = str(item.get(self.config.response_field, ""))

        # Check for obvious issues
        if not prompt.strip():
            score -= 0.5
            issues.append("empty_prompt")

        if self.config.response_field in item and not response.strip():
            score -= 0.5
            issues.append("empty_response")

        # Check for encoding issues
        if '\x00' in prompt or '\x00' in response:
            score -= 0.3
            issues.append("null_bytes")

        # Check for very long single lines (potential data corruption)
        for line in prompt.split('\n'):
            if len(line) > 5000:
                score -= 0.2
                issues.append("very_long_line")
                break

        return max(0.0, score), issues

    def assess_item(self, item: Dict) -> ItemQuality:
        """Assess quality of a single data item."""
        cfg = self.config
        prompt = str(item.get(cfg.prompt_field, ""))
        response = str(item.get(cfg.response_field, ""))

        quality = ItemQuality()

        # Length score
        prompt_len_score = self._score_length(prompt, cfg.ideal_prompt_length)
        response_len_score = self._score_length(response, cfg.ideal_response_length) if response else 1.0
        quality.length_score = (prompt_len_score + response_len_score) / 2

        # Diversity score
        prompt_div = self._score_diversity(prompt)
        response_div = self._score_diversity(response) if response else 1.0
        quality.diversity_score = (prompt_div + response_div) / 2

        # Repetition score
        prompt_rep = self._score_repetition(prompt)
        response_rep = self._score_repetition(response) if response else 1.0
        quality.repetition_score = (prompt_rep + response_rep) / 2

        # Relevance score
        quality.relevance_score = self._score_relevance(prompt, response)

        # Format score
        quality.format_score, format_issues = self._score_format(item)
        quality.issues.extend(format_issues)

        # Add quality issues
        if quality.length_score < 0.5:
            quality.issues.append("suboptimal_length")
        if quality.diversity_score < 0.3:
            quality.issues.append("low_diversity")
        if quality.repetition_score < 0.5:
            quality.issues.append("high_repetition")
        if quality.relevance_score < 0.3:
            quality.issues.append("low_relevance")

        # Overall score
        quality.overall = (
            cfg.length_weight * quality.length_score +
            cfg.diversity_weight * quality.diversity_score +
            cfg.repetition_weight * quality.repetition_score +
            cfg.relevance_weight * quality.relevance_score +
            cfg.format_weight * quality.format_score
        )

        return quality

    def assess_dataset(self, data: List[Dict]) -> QualityReport:
        """Assess quality of an entire dataset."""
        report = QualityReport(total_items=len(data))

        if not data:
            return report

        scores = []
        all_issues = Counter()
        dimension_sums = {
            "length": 0.0,
            "diversity": 0.0,
            "repetition": 0.0,
            "relevance": 0.0,
            "format": 0.0,
        }

        for i, item in enumerate(data):
            quality = self.assess_item(item)
            scores.append(quality.overall)

            dimension_sums["length"] += quality.length_score
            dimension_sums["diversity"] += quality.diversity_score
            dimension_sums["repetition"] += quality.repetition_score
            dimension_sums["relevance"] += quality.relevance_score
            dimension_sums["format"] += quality.format_score

            for issue in quality.issues:
                all_issues[issue] += 1

            if quality.overall < 0.5:
                report.low_quality_indices.append(i)

        n = len(data)
        report.average_score = sum(scores) / n

        # Score distribution
        report.score_distribution = {
            "excellent (0.8-1.0)": sum(1 for s in scores if s >= 0.8),
            "good (0.6-0.8)": sum(1 for s in scores if 0.6 <= s < 0.8),
            "fair (0.4-0.6)": sum(1 for s in scores if 0.4 <= s < 0.6),
            "poor (0.0-0.4)": sum(1 for s in scores if s < 0.4),
        }

        # Dimension averages
        report.dimension_averages = {
            k: v / n for k, v in dimension_sums.items()
        }

        # Common issues
        report.common_issues = [
            {"issue": issue, "count": count, "percentage": round(count / n * 100, 1)}
            for issue, count in all_issues.most_common(10)
        ]

        # Recommendations
        if report.dimension_averages.get("length", 0) < 0.5:
            report.recommendations.append("Many items have suboptimal length. Consider filtering or adjusting length requirements.")
        if report.dimension_averages.get("diversity", 0) < 0.5:
            report.recommendations.append("Low vocabulary diversity detected. Data may be too repetitive.")
        if report.dimension_averages.get("repetition", 0) < 0.5:
            report.recommendations.append("High text repetition detected. Consider deduplication or data augmentation.")
        if report.dimension_averages.get("relevance", 0) < 0.5:
            report.recommendations.append("Low prompt-response relevance. Check data quality and pairing.")
        if len(report.low_quality_indices) > n * 0.2:
            report.recommendations.append(f"{len(report.low_quality_indices)} items ({len(report.low_quality_indices)/n*100:.0f}%) below quality threshold. Consider cleaning.")

        return report
