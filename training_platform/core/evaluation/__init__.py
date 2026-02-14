"""
Evaluation Framework

Provides benchmark evaluation for LLMs:
- GSM8K: Math reasoning
- MATH: Advanced math
- HumanEval: Code generation
- MMLU: Multi-subject QA
- Auto-trigger evaluation after training
- Report generation
"""

from .benchmarks import (
    GSM8KEvaluator,
    MATHEvaluator,
    HumanEvalEvaluator,
    MMLUEvaluator,
    get_evaluator,
    list_benchmarks,
)
from .auto_trigger import EvalTrigger
from .report_generator import generate_report

__all__ = [
    "GSM8KEvaluator",
    "MATHEvaluator",
    "HumanEvalEvaluator",
    "MMLUEvaluator",
    "get_evaluator",
    "list_benchmarks",
    "EvalTrigger",
    "generate_report",
]
